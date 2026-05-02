"""
main.py
────────
Basketball analytics pipeline — SAM2 chunked tracking.

Usage
─────
  # Basic (no homography calibration):
  python main.py --video data/game.mp4 --output runs/detect/out.mp4

  # Full analytics (heatmaps + real-world distances/speeds):
  python main.py --video data/game.mp4 \\
                 --homography config/homography.npz \\
                 --meters-per-pixel 0.0264

  # Reduce chunk size if laptop crashes:
  python main.py --video data/game.mp4 --chunk-size 30

Analytics outputs (always generated)
──────────────────────────────────────
  runs/detect/distance_report.csv   — total distance per player
  runs/detect/speed_report.csv      — avg / max speed per player

Additional outputs (require --homography)
──────────────────────────────────────────
  runs/detect/heatmap_team_a.png
  runs/detect/heatmap_team_b.png
  Mini-court overlay drawn live in the bottom-left of each video frame.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from detection.ball_tracker import BallTracker
from detection.detector import BasketballDetector
from team_clustering.clusterer import (
    TeamClusterer, CLASS_PLAYER, CLASS_REF, TEAM_A, TEAM_B, TEAM_COLORS,
)
from tracking.interpolator import TrajectoryInterpolator
from tracking.sam2_tracker import SAM2Tracker, REF_ID_OFFSET

# Analytics — all optional imports so the pipeline still runs without them
try:
    from analytics.homography import HomographyTransformer
    from analytics.heatmap import HeatmapBuilder
    from analytics.distance import build_report as build_distance_report
    from analytics.distance import export_csv as export_distance_csv
    from analytics.speed import build_speed_report, export_csv as export_speed_csv
    _ANALYTICS_AVAILABLE = True
except ImportError as _e:
    _ANALYTICS_AVAILABLE = False
    print(f"[Main] Analytics modules not found ({_e}). "
          f"Distance/speed/heatmap will be skipped.")


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "models/RT-DETR/RT-DETR.pt"
DEFAULT_SAM2   = "models/sam2/sam2.1_hiera_small.pt"
DEFAULT_VIDEO  = "data/test_3.mp4"
DEFAULT_OUTPUT = "runs/detect/output.mp4"
DEFAULT_TRAJ   = "runs/detect/trajectories.json"

CONF_THRESHOLD    = 0.30
IOU_THRESHOLD     = 0.45
IMGSZ             = 640       # RT-DETR native resolution
BALL_CONF         = 0.10

SHOW_TRAILS       = True
SHOW_IDS          = True
SHOW_TEAMS        = True
SHOW_MASKS        = True
MASK_ALPHA        = 0.55

CHUNK_SIZE        = 60
CHUNK_OVERLAP     = 10
REPROMPT_INTERVAL = 30
GPU_SCALE         = 0.5

# Mini-court overlay size (fraction of video width)
MINIMAP_WIDTH_FRAC = 0.22
MINIMAP_ALPHA      = 0.82    # blend opacity


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--sam2",       default=DEFAULT_SAM2)
    p.add_argument("--sam2-size",  default="small", choices=["small","base","large"])
    p.add_argument("--video",      default=DEFAULT_VIDEO)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--traj",       default=DEFAULT_TRAJ)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--no-tiling",  action="store_true")
    p.add_argument("--no-trails",  action="store_true")
    p.add_argument("--no-masks",   action="store_true")
    p.add_argument("--device",     default="cuda")
    # ── Analytics ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--homography", default=None, metavar="PATH",
        help="Path to saved homography .npz file (enables heatmaps + "
             "real-world distance/speed). Generate with scripts/calibrate.py.",
    )
    p.add_argument(
        "--meters-per-pixel", type=float, default=None, metavar="SCALE",
        help="Real-world scale factor (e.g. 0.0264 for 1280-px wide frame). "
             "Enables m/s and metre columns in the CSV reports.",
    )
    p.add_argument(
        "--no-analytics", action="store_true",
        help="Skip all analytics (distance, speed, heatmap).",
    )
    return p.parse_args()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_video(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    ram_mb = len(frames) * width * height * 3 / 1024**2
    print(f"[Main] Loaded {len(frames)} frames  "
          f"({width}×{height} @ {fps:.1f}fps)  RAM: ~{ram_mb:.0f} MB")
    return frames, fps, width, height


def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def draw_hud(frame, frame_idx, total, fps, n_players, n_refs, clusterer):
    rosters = clusterer.get_team_rosters()
    lines   = [
        f"Frame  : {frame_idx:5d} / {total}",
        f"FPS    : {fps:5.1f}",
        f"Players: {n_players}",
        f"Refs   : {n_refs}",
        f"Team A : {len(rosters.get(0, []))}",
        f"Team B : {len(rosters.get(1, []))}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (8, y-th-2), (8+tw+4, y+4), (0,0,0), -1)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        y += th + 10
    return frame


def save_trajectories(trajectories, path):
    out = {"players": {}, "referees": {}}
    for tid, pts in trajectories.items():
        s = [[round(cx,2), round(cy,2), fi] for cx,cy,fi in pts]
        if tid >= REF_ID_OFFSET:
            out["referees"][str(tid - REF_ID_OFFSET)] = s
        else:
            out["players"][str(tid)] = s
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Main] Trajectories → {path}")


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _trajectories_for_analytics(
    filled: dict[int, list[tuple]],
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Convert the internal filled-trajectory dict to the format expected by
    distance.py and speed.py: {str_id: [[x, y, frame], ...]}.

    Returns (player_trajs, ref_trajs).
    """
    players: dict[str, list] = {}
    refs:    dict[str, list] = {}
    for tid, pts in filled.items():
        data = [[round(cx, 2), round(cy, 2), fi] for cx, cy, fi in pts]
        if tid >= REF_ID_OFFSET:
            refs[str(tid - REF_ID_OFFSET)] = data
        else:
            players[str(tid)] = data
    return players, refs


def _draw_minimap(
    frame:       np.ndarray,
    transformer: "HomographyTransformer",
    frame_dets:  list[dict],
    map_w:       int,
    map_h:       int,
) -> np.ndarray:
    """
    Draw a scaled-down top-down court with live player dots into the
    bottom-left corner of the frame.
    """
    canvas = transformer.make_court_canvas()

    # Collect players already transformed to court coords
    players_on_court = []
    for det in frame_dets:
        if det.get("class_id") != CLASS_PLAYER:
            continue
        tid     = det.get("track_id", -1)
        team_id = det.get("team_id", -1)
        try:
            cx, cy = transformer.transform_bbox_foot(det["bbox"])
        except Exception:
            continue
        if not transformer.is_on_court((cx, cy), margin=20):
            continue
        players_on_court.append({
            "court_pos": (cx, cy),
            "team_id":   team_id,
            "track_id":  tid,
        })

    canvas = transformer.draw_players_on_canvas(
        canvas, players_on_court, team_colors=TEAM_COLORS,
    )

    # Scale to minimap size
    mini = cv2.resize(canvas, (map_w, map_h), interpolation=cv2.INTER_AREA)

    # Paste into bottom-left with alpha blending
    fh, fw = frame.shape[:2]
    y1, y2 = fh - map_h - 8, fh - 8
    x1, x2 = 8, 8 + map_w

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = (mini.astype(np.float32) * MINIMAP_ALPHA +
               roi * (1.0 - MINIMAP_ALPHA)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"RT-DETR weights not found: {args.model}")
    if not os.path.exists(args.sam2):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {args.sam2}\n"
            "Download: wget -P models/sam2 "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        )

    run_analytics = _ANALYTICS_AVAILABLE and not args.no_analytics

    print("\n" + "═" * 60)
    print("  BASKETBALL ANALYTICS — SAM2 PIPELINE (Chunked)")
    print("═" * 60)

    rtdetr_device = "0" if args.device == "cuda" else args.device

    # ── 1. Load video ─────────────────────────────────────────────────────────
    frames, fps, width, height = load_video(args.video)
    total_frames = len(frames)

    # ── 2. Analytics setup ────────────────────────────────────────────────────
    transformer:     "HomographyTransformer | None" = None
    heatmap_builder: "HeatmapBuilder | None"        = None

    if run_analytics:
        out_dir = os.path.dirname(args.output) or "runs/detect"

        # Homography (optional — enables heatmap + real-world metrics)
        if args.homography and os.path.exists(args.homography):
            try:
                transformer = HomographyTransformer.load(args.homography)
                heatmap_builder = HeatmapBuilder(transformer)
                print(f"[Analytics] Homography loaded — heatmaps enabled")
            except Exception as e:
                print(f"[Analytics] Warning: could not load homography ({e}). "
                      f"Heatmaps disabled.")
        elif args.homography:
            print(f"[Analytics] Warning: homography file not found: {args.homography}. "
                  f"Run scripts/calibrate.py first.")

        if transformer is None:
            print("[Analytics] No homography — heatmaps and minimap disabled. "
                  "Distance/speed will use pixel units.")

        # Pre-compute minimap dimensions
        minimap_w = int(width * MINIMAP_WIDTH_FRAC)
        minimap_h = int(minimap_w * (
            transformer.court_height_px / transformer.court_width_px
            if transformer else 0.53
        ))
    else:
        minimap_w = minimap_h = 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — RT-DETR detects all anchor frames then is deleted from GPU
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[Phase 1] RT-DETR detecting…")
    detector = BasketballDetector(
        model_path=args.model, conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD, imgsz=IMGSZ, device=rtdetr_device,
    )
    detector.warmup()

    frame0_dets = detector.parse(detector.detect(frames[0]))
    n_p = sum(1 for d in frame0_dets if d["class_id"] == CLASS_PLAYER)
    n_r = sum(1 for d in frame0_dets if d["class_id"] == CLASS_REF)
    print(f"[Phase 1] Frame 0: {n_p} players, {n_r} refs")

    anchor_dets: dict[int, list[dict]] = {}
    for fi in range(REPROMPT_INTERVAL, total_frames, REPROMPT_INTERVAL):
        dets = detector.parse(detector.detect(frames[fi]))
        anchor_dets[fi] = [d for d in dets
                           if d["class_id"] in (CLASS_PLAYER, CLASS_REF)]
    print(f"[Phase 1] Pre-detected {len(anchor_dets)} anchor frames")

    del detector
    free_memory()
    print("[Phase 1] RT-DETR removed from GPU\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — SAM2 chunked tracking
    # ══════════════════════════════════════════════════════════════════════════
    sam2_tracker = SAM2Tracker(
        checkpoint        = args.sam2,
        model_size        = args.sam2_size,
        device            = args.device,
        chunk_size        = args.chunk_size,
        chunk_overlap     = CHUNK_OVERLAP,
        reprompt_interval = REPROMPT_INTERVAL,
        gpu_scale         = GPU_SCALE,
    )

    tracking_results = sam2_tracker.process_video(
        frames      = frames,
        frame0_dets = frame0_dets,
        anchor_dets = anchor_dets,
    )

    total_players = sam2_tracker.total_player_tracks
    total_refs    = sam2_tracker.total_ref_tracks
    draw_fn       = sam2_tracker.draw_tracks

    del sam2_tracker
    free_memory()
    print("[Phase 2] SAM2 removed from GPU\n")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Ball + Team clustering + Analytics + write video
    # ══════════════════════════════════════════════════════════════════════════
    print("[Phase 3] Writing output video…")
    ball_tracker = BallTracker(
        rtdetr_path = args.model,
        device      = rtdetr_device,
    )
    clusterer    = TeamClusterer(warm_up_frames=60)
    interpolator = TrajectoryInterpolator(max_gap=15, use_parabolic=True)
    writer       = make_writer(args.output, fps, width, height)
    trajectories: dict[int, list[tuple]] = defaultdict(list)
    t_prev       = time.perf_counter()
    fps_display  = 0.0

    for frame_idx, frame in enumerate(frames):
        frame_dets = tracking_results.get(frame_idx, [])

        ball_tracker.update(frame, frame_idx)

        # Team clustering — must happen before heatmap so team_id is set
        clusterer.update(frame, frame_dets)
        for det in frame_dets:
            tid = det["track_id"]
            if tid != -1:
                det["team_id"] = clusterer.get_team(tid)

        # Trajectory accumulation
        for det in frame_dets:
            tid = det["track_id"]
            if tid != -1:
                cx, cy = det["center"]
                trajectories[tid].append((cx, cy, frame_idx))

        # ── Heatmap accumulation (per-frame, needs team_id already set) ──────
        if heatmap_builder is not None:
            heatmap_builder.add_frame(frame_dets)

        # ── Render ────────────────────────────────────────────────────────────
        vis = draw_fn(
            frame, frame_dets,
            show_trails = SHOW_TRAILS and not args.no_trails,
            show_ids    = SHOW_IDS,
            show_teams  = SHOW_TEAMS,
            show_masks  = SHOW_MASKS and not args.no_masks,
            mask_alpha  = MASK_ALPHA,
        )
        vis = ball_tracker.draw(vis, frame_idx, trail_len=0)

        # ── Mini-court overlay ────────────────────────────────────────────────
        if transformer is not None and minimap_w > 0:
            vis = _draw_minimap(
                vis, transformer, frame_dets, minimap_w, minimap_h,
            )

        t_now       = time.perf_counter()
        fps_display = 0.9 * fps_display + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        vis = draw_hud(vis, frame_idx, total_frames, fps_display,
                       total_players, total_refs, clusterer)
        writer.write(vis)

        if frame_idx % 30 == 0:
            pct = 100.0 * frame_idx / max(total_frames, 1)
            print(f"\r[Phase 3] {frame_idx:5d}/{total_frames} ({pct:.1f}%)"
                  f"  {fps_display:.1f}fps", end="", flush=True)

    writer.release()
    print(f"\n[Main] Video → {args.output}")

    # ══════════════════════════════════════════════════════════════════════════
    # POST-LOOP — refine, interpolate, analytics
    # ══════════════════════════════════════════════════════════════════════════
    print("[Main] Refining team clustering…")
    clusterer.refine()

    print("[Main] Interpolating trajectories…")
    filled = interpolator.fill_all(dict(trajectories))
    filled = interpolator.stitch_out_of_bounds(filled, frame_width=width)
    save_trajectories(filled, args.traj)

    # ── Analytics reports ─────────────────────────────────────────────────────
    if run_analytics:
        out_dir = os.path.dirname(args.output) or "runs/detect"
        mpp     = args.meters_per_pixel   # may be None

        player_trajs, _ = _trajectories_for_analytics(filled)

        # Distance report
        print("[Analytics] Computing distance report…")
        dist_rows = build_distance_report(player_trajs, meters_per_pixel=mpp)
        export_distance_csv(dist_rows, os.path.join(out_dir, "distance_report.csv"))

        # Speed report
        print("[Analytics] Computing speed report…")
        speed_rows = build_speed_report(player_trajs, fps=fps, meters_per_pixel=mpp)
        export_speed_csv(speed_rows, os.path.join(out_dir, "speed_report.csv"))

        # Heatmaps
        if heatmap_builder is not None:
            print("[Analytics] Saving heatmaps…")
            for team_id, name in ((TEAM_A, "team_a"), (TEAM_B, "team_b")):
                path = os.path.join(out_dir, f"heatmap_{name}.png")
                saved = heatmap_builder.save_team_heatmap(team_id, path)
                if saved:
                    print(f"[Analytics] Heatmap → {path}")
                else:
                    print(f"[Analytics] No data for {name} — heatmap skipped")

    print(f"\n{'═'*60}")
    print(f"  DONE  |  {args.output}")
    print(f"  Players: {total_players}  Refs: {total_refs}")
    if run_analytics:
        unit = "m" if args.meters_per_pixel else "px"
        print(f"  Analytics: distance + speed ({unit}) saved to "
              f"{os.path.dirname(args.output) or 'runs/detect'}/")
        if heatmap_builder:
            print(f"  Heatmaps:  team_a + team_b saved")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run_pipeline(parse_args())