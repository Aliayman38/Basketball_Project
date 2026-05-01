"""
main.py
────────
Basketball analytics pipeline using SAM2 tracking.

Usage
─────
  python main.py --video data/game.mp4 --output runs/detect/out.mp4

SAM2 setup (one-time)
──────────────────────
  pip install sam2
  mkdir -p models/sam2
  wget -P models/sam2 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from src.detection.ball_tracker import BallTracker
from src.detection.detector import BasketballDetector
from src.team_clustering.clusterer import (
    TeamClusterer, TEAM_UNKNOWN, TEAM_REF, CLASS_PLAYER, CLASS_REF,
)
from src.tracking.interpolator import TrajectoryInterpolator
from src.tracking.sam2_tracker import SAM2Tracker, REF_ID_OFFSET


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "models/weights/best.pt"
DEFAULT_SAM2   = "models/sam2/sam2.1_hiera_small.pt"
DEFAULT_VIDEO  = "data/test_3.mp4"
DEFAULT_OUTPUT = "runs/detect/output.mp4"
DEFAULT_TRAJ   = "runs/detect/trajectories.json"

CONF_THRESHOLD = 0.30
IOU_THRESHOLD  = 0.45
IMGSZ          = 1280
BALL_CONF      = 0.15

SHOW_TRAILS    = True
SHOW_IDS       = True
SHOW_TEAMS     = True
SHOW_MASKS     = True
MASK_ALPHA     = 0.25

# Re-prompt every N frames — lower = more stable IDs but slower init
REPROMPT_INTERVAL = 30


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=DEFAULT_MODEL)
    p.add_argument("--sam2",       default=DEFAULT_SAM2)
    p.add_argument("--sam2-size",  default="small", choices=["small", "base", "large"])
    p.add_argument("--video",      default=DEFAULT_VIDEO)
    p.add_argument("--output",     default=DEFAULT_OUTPUT)
    p.add_argument("--traj",       default=DEFAULT_TRAJ)
    p.add_argument("--no-tiling",  action="store_true")
    p.add_argument("--no-trails",  action="store_true")
    p.add_argument("--no-masks",   action="store_true")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
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
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[Main] Loaded {len(frames)} frames  ({width}×{height} @ {fps:.1f}fps)")
    return frames, fps, width, height


def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def draw_hud(frame, frame_idx, total, fps, tracker, clusterer):
    rosters = clusterer.get_team_rosters()
    lines   = [
        f"Frame  : {frame_idx:5d} / {total}",
        f"FPS    : {fps:5.1f}",
        f"Players: {tracker.total_player_tracks}",
        f"Refs   : {tracker.total_ref_tracks}",
        f"Team A : {len(rosters.get(0, []))}",
        f"Team B : {len(rosters.get(1, []))}",
    ]
    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (8, y-th-2), (8+tw+4, y+4), (0, 0, 0), -1)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += th + 10
    return frame


def save_trajectories(trajectories, path):
    out = {"players": {}, "referees": {}}
    for tid, pts in trajectories.items():
        s = [[round(cx, 2), round(cy, 2), fi] for cx, cy, fi in pts]
        if tid >= REF_ID_OFFSET:
            out["referees"][str(tid - REF_ID_OFFSET)] = s
        else:
            out["players"][str(tid)] = s
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Main] Trajectories → {path}")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"YOLO weights not found: {args.model}")
    if not os.path.exists(args.sam2):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {args.sam2}\n"
            f"Download: wget -P models/sam2 "
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        )

    print("\n" + "═" * 60)
    print("  BASKETBALL ANALYTICS — SAM2 PIPELINE")
    print("═" * 60)

    # ── 1. Load video ─────────────────────────────────────────────────────────
    frames, fps, width, height = load_video(args.video)
    total_frames = len(frames)

    # YOLO device string ("0" for GPU, "cpu")
    yolo_device = "0" if args.device == "cuda" else args.device

    # ── 2. Init components ────────────────────────────────────────────────────
    detector = BasketballDetector(
        model_path = args.model,
        conf       = CONF_THRESHOLD,
        iou        = IOU_THRESHOLD,
        imgsz      = IMGSZ,
        device     = yolo_device,
    )
    detector.warmup()

    ball_tracker = BallTracker(
        model_path = args.model,
        ball_conf  = BALL_CONF,
        imgsz      = IMGSZ,
        use_tiling = not args.no_tiling,
        device     = yolo_device,
    )

    sam2_tracker = SAM2Tracker(
        checkpoint        = args.sam2,
        model_size        = args.sam2_size,
        device            = args.device,
        reprompt_interval = REPROMPT_INTERVAL,
    )

    clusterer    = TeamClusterer(warm_up_frames=60)
    interpolator = TrajectoryInterpolator(max_gap=15, use_parabolic=True)

    # ── 3. Detect frame 0, init SAM2 with multi-frame re-prompting ────────────
    print("\n[Main] Detecting on frame 0…")
    frame0_dets = detector.parse(detector.detect(frames[0]))
    n_p = sum(1 for d in frame0_dets if d["class_id"] == CLASS_PLAYER)
    n_r = sum(1 for d in frame0_dets if d["class_id"] == CLASS_REF)
    print(f"[Main] Frame 0: {n_p} players, {n_r} refs")

    # Pass detector so SAM2 can add anchor prompts every REPROMPT_INTERVAL frames
    sam2_tracker.initialize(frames, frame0_dets, detector=detector)

    # ── 4. SAM2 propagate ─────────────────────────────────────────────────────
    print("\n[Main] SAM2 propagating…")
    tracking_results = sam2_tracker.propagate(frames)
    print(f"[Main] Tracked {sam2_tracker.total_player_tracks} players, "
          f"{sam2_tracker.total_ref_tracks} refs")

    # ── 5. Team clustering + write video ──────────────────────────────────────
    print("\n[Main] Team clustering and writing video…")
    writer       = make_writer(args.output, fps, width, height)
    trajectories: dict[int, list[tuple]] = defaultdict(list)
    t_prev       = time.perf_counter()
    fps_display  = 0.0

    for frame_idx, frame in enumerate(frames):

        frame_dets = tracking_results.get(frame_idx, [])

        # Ball tracking — update position, no trail drawn
        ball_tracker.update(frame, frame_idx)

        # Team clustering
        clusterer.update(frame, frame_dets)
        for det in frame_dets:
            tid = det["track_id"]
            if tid != -1:
                det["team_id"] = clusterer.get_team(tid)

        # Collect trajectories
        for det in frame_dets:
            tid = det["track_id"]
            if tid != -1:
                cx, cy = det["center"]
                trajectories[tid].append((cx, cy, frame_idx))

        # Draw players/refs/masks
        vis = sam2_tracker.draw_tracks(
            frame, frame_dets,
            show_trails = SHOW_TRAILS and not args.no_trails,
            show_ids    = SHOW_IDS,
            show_teams  = SHOW_TEAMS,
            show_masks  = SHOW_MASKS and not args.no_masks,
            mask_alpha  = MASK_ALPHA,
        )

        # Draw ball dot (no trail)
        vis = ball_tracker.draw_ball(vis, frame_idx, trail_length=0)

        t_now       = time.perf_counter()
        fps_display = 0.9 * fps_display + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        vis = draw_hud(vis, frame_idx, total_frames, fps_display,
                       sam2_tracker, clusterer)
        writer.write(vis)

        if frame_idx % 30 == 0:
            pct = 100.0 * frame_idx / max(total_frames, 1)
            print(f"\r[Main] {frame_idx:5d}/{total_frames} ({pct:.1f}%)"
                  f"  {fps_display:.1f}fps",
                  end="", flush=True)

    writer.release()
    print(f"\n[Main] Video written → {args.output}")

    # ── 6. Post-loop ──────────────────────────────────────────────────────────
    print("[Main] Refining team clustering…")
    clusterer.refine()
    clusterer.print_roster()

    print("[Main] Interpolating trajectories…")
    filled = interpolator.fill_all(dict(trajectories))
    filled = interpolator.stitch_out_of_bounds(filled, frame_width=width)

    save_trajectories(filled, args.traj)

    print(f"\n{'═'*60}")
    print(f"  DONE  |  {args.output}")
    print(f"  Players: {sam2_tracker.total_player_tracks}  "
          f"Refs: {sam2_tracker.total_ref_tracks}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(parse_args())