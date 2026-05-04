"""
main.py
────────
Basketball analytics pipeline — OC-SORT tracking + RT-DETR detection.

Usage
─────
  python main.py --video data/game.mp4

  # With real-world analytics:
  python main.py --video data/game.mp4 \\
                 --homography config/homography.npz \\
                 --meters-per-pixel 0.0264

  # With GUI-calibrated team assignment:
  python main.py --video data/game.mp4 --team-a-cluster 0

Pipeline
────────
  RT-DETR  →  OC-SORT (with Re-ID gallery)  →  Team clustering
           →  Ball tracking  →  Analytics  →  Output video

Outputs (runs/detect/)
──────────────────────
  output.mp4            annotated video
  trajectories.json     per-player centre-point timeline
  distance_report.csv   total distance per player
  speed_report.csv      avg/max speed per player
  heatmap_team_a.png    occupancy heatmap — Team A   (requires --homography)
  heatmap_team_b.png    occupancy heatmap — Team B   (requires --homography)
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

from detection.ball_tracker import BallTracker
from detection.detector import BasketballDetector
from team_clustering.clusterer import (
    TeamClusterer, CLASS_PLAYER, CLASS_REF, TEAM_A, TEAM_B, TEAM_COLORS,
)
from tracking.interpolator import TrajectoryInterpolator
from tracking.ocsort_tracker import OCSortTracker, REF_ID_OFFSET

# Analytics — optional; pipeline runs without them
try:
    from analytics.homography import HomographyTransformer
    from analytics.heatmap import HeatmapBuilder
    from analytics.distance import build_report as build_distance_report
    from analytics.distance import export_csv as export_distance_csv
    from analytics.speed import build_speed_report, export_csv as export_speed_csv
    _ANALYTICS = True
except ImportError as _e:
    _ANALYTICS = False
    print(f"[Main] Analytics not available ({_e}) — "
          f"distance/speed/heatmap will be skipped.")


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "models/RT-DETR/RT-DETR.pt"
DEFAULT_VIDEO  = "data/test_3.mp4"
DEFAULT_OUTPUT = "runs/detect/output.mp4"
DEFAULT_TRAJ   = "runs/detect/trajectories.json"

CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45
IMGSZ          = 640

SHOW_TRAILS    = True
SHOW_IDS       = True
SHOW_TEAMS     = True

# OC-SORT defaults
OCSORT_MAX_AGE         = 90    # frames absent before gallery archival (~3s)
OCSORT_MIN_HITS        = 3     # detections before track appears in output
OCSORT_IOU_THR         = 0.30  # primary association threshold
OCSORT_IOU_THR_2       = 0.10  # secondary (low-conf) association threshold
OCSORT_DELTA_T         = 3     # OCM velocity lookback
OCSORT_REID_MAX_AGE    = 300   # gallery entry lifetime (~10 s at 30 fps)
OCSORT_REID_THR        = 0.55  # Re-ID combined cost threshold (permissive)

# Mini-court overlay
MINIMAP_WIDTH_FRAC = 0.22
MINIMAP_ALPHA      = 0.82


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Basketball analytics — OC-SORT + RT-DETR")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help="RT-DETR weights path")
    p.add_argument("--video",       default=DEFAULT_VIDEO)
    p.add_argument("--output",      default=DEFAULT_OUTPUT)
    p.add_argument("--traj",        default=DEFAULT_TRAJ)
    p.add_argument("--no-trails",   action="store_true")
    p.add_argument("--device",      default="cuda")

    # Analytics
    p.add_argument("--homography",        default=None, metavar="PATH",
                   help="Path to homography .npz (enables heatmap + minimap)")
    p.add_argument("--meters-per-pixel",  type=float, default=None,
                   metavar="SCALE")
    p.add_argument("--no-analytics",      action="store_true")

    # Team calibration (set by GUI)
    p.add_argument("--team-a-cluster",    type=int, default=None,
                   choices=[0, 1],
                   help="Which K-Means cluster is Team A (from GUI calibration)")
    return p.parse_args()


def free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_video(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    mb = len(frames) * width * height * 3 / 1024 ** 2
    print(f"[Main] Loaded {len(frames)} frames  "
          f"({width}×{height} @ {fps:.1f} fps)  RAM ≈ {mb:.0f} MB")
    return frames, fps, width, height


def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def draw_hud(frame, frame_idx, total, disp_fps, tracker, clusterer):
    rosters = clusterer.get_team_rosters()
    lines = [
        f"Frame  : {frame_idx:5d} / {total}",
        f"FPS    : {disp_fps:5.1f}",
        f"Players: {tracker.total_player_tracks}",
        f"Refs   : {tracker.total_ref_tracks}",
        f"Gallery: {tracker.gallery_size}",
        f"Team A : {len(rosters.get(TEAM_A, []))}",
        f"Team B : {len(rosters.get(TEAM_B, []))}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (8, y - th - 2), (8 + tw + 4, y + 4),
                      (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        y += th + 10
    return frame


def save_trajectories(trajectories: dict, path: str, ref_ids: set) -> None:
    out: dict = {"players": {}, "referees": {}}
    for tid, pts in trajectories.items():
        data = [[float(round(cx, 2)), float(round(cy, 2)), int(fi)]
                for cx, cy, fi in pts]
        if tid in ref_ids:
            out["referees"][str(tid)] = data
        else:
            out["players"][str(tid)] = data
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Main] Trajectories → {path}")


def _traj_for_analytics(
    filled: dict[int, list],
) -> tuple[dict[str, list], dict[str, list]]:
    players: dict[str, list] = {}
    refs:    dict[str, list] = {}
    for tid, pts in filled.items():
        data = [[round(cx, 2), round(cy, 2), fi] for cx, cy, fi in pts]
        if tid > REF_ID_OFFSET:
            refs[str(tid - REF_ID_OFFSET)] = data
        else:
            players[str(tid)] = data
    return players, refs


def draw_minimap(
    frame:       np.ndarray,
    transformer: "HomographyTransformer",
    frame_dets:  list[dict],
    map_w:       int,
    map_h:       int,
) -> np.ndarray:
    """Draw scaled top-down court with live player dots in the bottom-left."""
    canvas = transformer.make_court_canvas()
    players_on_court = []
    for det in frame_dets:
        if det.get("class_id") != CLASS_PLAYER:
            continue
        try:
            cx, cy = transformer.transform_bbox_foot(det["bbox"])
        except Exception:
            continue
        if not transformer.is_on_court((cx, cy), margin=20):
            continue
        players_on_court.append({
            "court_pos": (cx, cy),
            "team_id":   det.get("team_id", -1),
            "track_id":  det["track_id"],
        })
    canvas = transformer.draw_players_on_canvas(
        canvas, players_on_court, team_colors=TEAM_COLORS)

    mini    = cv2.resize(canvas, (map_w, map_h), interpolation=cv2.INTER_AREA)
    fh, fw  = frame.shape[:2]
    y1, y2  = fh - map_h - 8, fh - 8
    x1, x2  = 8, 8 + map_w
    roi     = frame[y1:y2, x1:x2].astype(np.float32)
    blended = (mini.astype(np.float32) * MINIMAP_ALPHA +
               roi * (1.0 - MINIMAP_ALPHA)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> None:
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"RT-DETR weights not found: {args.model}")

    run_analytics = _ANALYTICS and not args.no_analytics

    print("\n" + "═" * 60)
    print("  BASKETBALL ANALYTICS  —  OC-SORT + RT-DETR")
    print("═" * 60)

    device = "0" if args.device == "cuda" else args.device

    # ── Load video ────────────────────────────────────────────────────────────
    frames, fps, width, height = load_video(args.video)
    total_frames = len(frames)

    # ── Analytics setup ───────────────────────────────────────────────────────
    transformer:     "HomographyTransformer | None" = None
    heatmap_builder: "HeatmapBuilder | None"        = None
    minimap_w = minimap_h = 0

    if run_analytics:
        out_dir = os.path.dirname(args.output) or "runs/detect"
        if args.homography and os.path.exists(args.homography):
            try:
                transformer     = HomographyTransformer.load(args.homography)
                heatmap_builder = HeatmapBuilder(transformer)
                minimap_w = int(width * MINIMAP_WIDTH_FRAC)
                minimap_h = int(minimap_w * (
                    transformer.court_height_px / transformer.court_width_px))
                print(f"[Analytics] Homography loaded — heatmaps enabled")
            except Exception as e:
                print(f"[Analytics] Warning: could not load homography ({e})")
        elif args.homography:
            print(f"[Analytics] Warning: homography file not found "
                  f"({args.homography}) — heatmaps disabled")

    # ── Initialise detector ───────────────────────────────────────────────────
    print("\n[Main] Loading RT-DETR detector…")
    detector = BasketballDetector(
        model_path=args.model,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        device=device,
    )
    detector.warmup()
    print("[Main] RT-DETR ready\n")

    # ── Detect frame 0 to count players and refs ─────────────────────────────
    print("[Main] Detecting frame 0 for player/ref count…")
    det0_result = detector.detect(frames[0])
    det0_parsed = detector.parse(det0_result)
    n_players_f0 = sum(1 for d in det0_parsed if d["class_id"] == CLASS_PLAYER)
    n_refs_f0    = sum(1 for d in det0_parsed if d["class_id"] == CLASS_REF)
    # Add a small buffer in case a player is briefly off-screen in frame 0
    max_players = max(n_players_f0, 10)
    max_refs    = max(n_refs_f0,    3)
    print(f"[Main] Frame 0: {n_players_f0} players, {n_refs_f0} refs  "
          f"→ caps: {max_players} players, {max_refs} refs")

    # ── Initialise OC-SORT tracker ────────────────────────────────────────────
    tracker = OCSortTracker(
        max_age         = OCSORT_MAX_AGE,
        min_hits        = OCSORT_MIN_HITS,
        iou_threshold   = OCSORT_IOU_THR,
        iou_threshold_2 = OCSORT_IOU_THR_2,
        delta_t         = OCSORT_DELTA_T,
        reid_max_age    = OCSORT_REID_MAX_AGE,
        reid_threshold  = OCSORT_REID_THR,
        max_players     = max_players,
        max_refs        = max_refs,
    )

    # ── Initialise ancillary modules ──────────────────────────────────────────
    clusterer    = TeamClusterer(warm_up_frames=60)
    interpolator = TrajectoryInterpolator(max_gap=15, use_parabolic=True)
    ball_tracker = BallTracker(
        rtdetr_path=args.model,
        device=device,
    )

    # Apply GUI team calibration if provided
    if getattr(args, "team_a_cluster", None) is not None:
        clusterer.set_user_label_map(
            team_a_is_group0=(args.team_a_cluster == 0))
        print(f"[Main] Team assignment locked — "
              f"cluster {args.team_a_cluster} = Team A")

    # ── Main processing loop ──────────────────────────────────────────────────
    print("[Main] Processing frames…\n")
    writer       = make_writer(args.output, fps, width, height)
    trajectories: dict[int, list[tuple]] = defaultdict(list)
    t_prev        = time.perf_counter()
    fps_display   = 0.0

    for frame_idx, frame in enumerate(frames):

        # 1. RT-DETR detection
        result   = detector.detect(frame)
        raw_dets = detector.parse(result)

        # 2. OC-SORT tracking
        frame_dets = tracker.update(frame, raw_dets, frame_idx)

        # 3. Ball tracking
        ball_tracker.update(frame, frame_idx)

        # 4. Team assignment — must happen before heatmap
        clusterer.update(frame, frame_dets)
        for det in frame_dets:
            tid = det["track_id"]
            det["team_id"] = clusterer.get_team(tid)

        # 5. Trajectory accumulation
        for det in frame_dets:
            tid        = det["track_id"]
            cx, cy     = det["center"]
            trajectories[tid].append((cx, cy, frame_idx))

        # 6. Heatmap accumulation
        if heatmap_builder is not None:
            heatmap_builder.add_frame(frame_dets)

        # 7. Render
        vis = tracker.draw_tracks(
            frame, frame_dets,
            show_trails = SHOW_TRAILS and not args.no_trails,
            show_ids    = SHOW_IDS,
            show_teams  = SHOW_TEAMS,
        )
        vis = ball_tracker.draw(vis, frame_idx, trail_len=0)

        # 8. Mini-court overlay
        if transformer is not None and minimap_w > 0:
            vis = draw_minimap(vis, transformer, frame_dets,
                               minimap_w, minimap_h)

        # 9. HUD
        t_now        = time.perf_counter()
        fps_display  = 0.9 * fps_display + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev       = t_now
        vis = draw_hud(vis, frame_idx, total_frames, fps_display,
                       tracker, clusterer)
        writer.write(vis)

        if frame_idx % 30 == 0:
            pct = 100.0 * frame_idx / max(total_frames, 1)
            print(f"\r[Main] {frame_idx:5d}/{total_frames} "
                  f"({pct:.1f}%)  "
                  f"{fps_display:.1f} fps  "
                  f"active={tracker.n_active}  "
                  f"gallery={tracker.gallery_size}",
                  end="", flush=True)

    writer.release()
    print(f"\n[Main] Video → {args.output}")

    # ── Post-loop ─────────────────────────────────────────────────────────────
    print("[Main] Refining team clustering…")
    clusterer.refine()

    print("[Main] Interpolating trajectories…")
    filled = interpolator.fill_all(dict(trajectories))
    filled = interpolator.stitch_out_of_bounds(filled, frame_width=width)
    ref_ids = set(clusterer.get_team_rosters().get(2, []))  # TEAM_REF = 2
    save_trajectories(filled, args.traj, ref_ids)

    # ── Analytics reports ─────────────────────────────────────────────────────
    if run_analytics:
        out_dir = os.path.dirname(args.output) or "runs/detect"
        mpp     = args.meters_per_pixel

        player_trajs, _ = _traj_for_analytics(filled)

        print("[Analytics] Distance report…")
        dist_rows = build_distance_report(player_trajs, meters_per_pixel=mpp)
        export_distance_csv(dist_rows,
                            os.path.join(out_dir, "distance_report.csv"))

        print("[Analytics] Speed report…")
        speed_rows = build_speed_report(player_trajs, fps=fps,
                                        meters_per_pixel=mpp)
        export_speed_csv(speed_rows,
                         os.path.join(out_dir, "speed_report.csv"))

        if heatmap_builder is not None:
            print("[Analytics] Saving heatmaps…")
            for team_id, name in ((TEAM_A, "team_a"), (TEAM_B, "team_b")):
                p = os.path.join(out_dir, f"heatmap_{name}.png")
                if heatmap_builder.save_team_heatmap(team_id, p):
                    print(f"[Analytics] Heatmap → {p}")
                else:
                    print(f"[Analytics] No data for {name}")

    print(f"\n{'═' * 60}")
    print(f"  DONE  |  {args.output}")
    print(f"  Players tracked : {tracker.total_player_tracks}")
    print(f"  Refs tracked    : {tracker.total_ref_tracks}")
    if run_analytics:
        unit = "m" if args.meters_per_pixel else "px"
        print(f"  Analytics (dist/speed in {unit}) → "
              f"{os.path.dirname(args.output) or 'runs/detect'}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    run_pipeline(parse_args())