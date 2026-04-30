"""
main.py
────────
Basketball analytics pipeline.

Usage
─────
  python main.py
  python main.py --video data/game.mp4 --output runs/detect/out.mp4
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
from src.tracking.tracker import PlayerTracker, REF_ID_OFFSET


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "models/weights/best.pt"
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


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--video",     default=DEFAULT_VIDEO)
    p.add_argument("--output",    default=DEFAULT_OUTPUT)
    p.add_argument("--traj",      default=DEFAULT_TRAJ)
    p.add_argument("--no-tiling", action="store_true")
    p.add_argument("--no-trails", action="store_true")
    p.add_argument("--device",    default="0")
    return p.parse_args()


def build_pipeline(args):
    print("\n" + "═" * 60)
    print("  BASKETBALL ANALYTICS — PIPELINE INIT")
    print("═" * 60)

    ball_tracker = BallTracker(
        model_path = args.model,
        ball_conf  = BALL_CONF,
        imgsz      = IMGSZ,
        use_tiling = not args.no_tiling,
        device     = args.device,
    )

    tracker = PlayerTracker(
        model_path = args.model,
        conf       = CONF_THRESHOLD,
        iou        = IOU_THRESHOLD,
        imgsz      = IMGSZ,
        device     = args.device,
    )

    clusterer    = TeamClusterer(warm_up_frames=60)
    interpolator = TrajectoryInterpolator(max_gap=15, use_parabolic=True)

    print("═" * 60 + "\n")
    return {
        "ball_tracker": ball_tracker,
        "tracker":      tracker,
        "clusterer":    clusterer,
        "interpolator": interpolator,
    }


def open_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {path}")
    return cap


def make_writer(path, fps, w, h):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def draw_hud(frame, frame_idx, total, fps, tracker, clusterer):
    rosters = clusterer.get_team_rosters()
    lines = [
        f"Frame : {frame_idx:5d} / {total}",
        f"FPS   : {fps:5.1f}",
        f"Players : {tracker.total_player_tracks}",
        f"Refs    : {tracker.total_ref_tracks}",
        f"Team A  : {len(rosters.get(0, []))}",
        f"Team B  : {len(rosters.get(1, []))}",
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
        s = [[round(cx,2), round(cy,2), fi] for cx,cy,fi in pts]
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
        raise FileNotFoundError(f"Weights not found: {args.model}")

    pipe         = build_pipeline(args)
    ball_tracker = pipe["ball_tracker"]
    tracker      = pipe["tracker"]
    clusterer    = pipe["clusterer"]
    interpolator = pipe["interpolator"]

    cap          = open_video(args.video)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer       = make_writer(args.output, fps, width, height)

    print(f"[Main] Video  : {args.video}  ({width}×{height} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"[Main] Output : {args.output}\n")

    trajectories: dict[int, list[tuple]] = defaultdict(list)
    frame_idx   = 0
    t_prev      = time.perf_counter()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Ball
            ball_tracker.update(frame, frame_idx)

            # 2. Track
            tracked_dets = tracker.update(frame)

            # 3. Team clustering
            clusterer.update(frame, tracked_dets)
            for det in tracked_dets:
                tid = det["track_id"]
                if tid != -1:
                    det["team_id"] = clusterer.get_team(tid)

            # 4. Collect trajectories
            for det in tracked_dets:
                tid = det["track_id"]
                if tid != -1:
                    cx, cy = det["center"]
                    trajectories[tid].append((cx, cy, frame_idx))

            # 5. Draw
            vis = tracker.draw_tracks(
                frame, tracked_dets,
                show_trails = SHOW_TRAILS and not args.no_trails,
                show_ids    = SHOW_IDS,
                show_teams  = SHOW_TEAMS,
            )
            vis = ball_tracker.draw_ball(vis, frame_idx)

            t_now       = time.perf_counter()
            fps_display = 0.9 * fps_display + 0.1 / max(t_now - t_prev, 1e-6)
            t_prev      = t_now

            vis = draw_hud(vis, frame_idx, total_frames, fps_display, tracker, clusterer)
            writer.write(vis)

            if frame_idx % 30 == 0:
                pct = 100.0 * frame_idx / max(total_frames, 1)
                print(f"\r[Main] {frame_idx:5d}/{total_frames} ({pct:.1f}%)"
                      f"  {fps_display:.1f}fps"
                      f"  players={tracker.total_player_tracks}"
                      f"  refs={tracker.total_ref_tracks}",
                      end="", flush=True)

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print(f"\n\n[Main] Done — {frame_idx} frames.")

    print("[Main] Refining clustering…")
    clusterer.refine()
    clusterer.print_roster()

    print("[Main] Interpolating…")
    filled = interpolator.fill_all(dict(trajectories))
    filled = interpolator.stitch_out_of_bounds(filled, frame_width=width)

    save_trajectories(filled, args.traj)

    print(f"\n{'═'*60}")
    print(f"  DONE  |  {args.output}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run_pipeline(parse_args())