"""
main.py
────────
Full basketball analytics pipeline.

Stage order per frame
─────────────────────
  1. BasketballDetector  → raw detection dicts
  2. BallTracker         → dedicated ball position (low-conf + SAHI)
  3. PlayerTracker       → persistent player/ref IDs via ByteTrack
  4. TeamClusterer       → jersey-colour KMeans assigns TEAM_A / TEAM_B
  5. Visualise & write   → annotated output video

Post-loop
─────────
  6. TeamClusterer.refine()        → re-fit on all collected colour data
  7. TrajectoryInterpolator        → fill jump/occlusion gaps in tracks
  8. Save trajectories (JSON)      → input for Rababah's analytics module

Usage
─────
  python main.py
  python main.py --video data/game2.mp4 --output runs/detect/game2.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.detection.ball_tracker import BallTracker
from src.detection.detector import BasketballDetector
from src.team_clustering.clusterer import (
    TEAM_UNKNOWN,
    TeamClusterer,
)
from src.tracking.interpolator import TrajectoryInterpolator
from src.tracking.tracker import PlayerTracker, REF_ID_OFFSET


# ── Pipeline config ───────────────────────────────────────────────────────────

DEFAULT_MODEL   = "models/weights/best.pt"
DEFAULT_VIDEO   = "data/test_3.mp4"
DEFAULT_OUTPUT  = "runs/detect/output.mp4"
DEFAULT_TRAJ    = "runs/detect/trajectories.json"   # for analytics module

# Detection
CONF_THRESHOLD  = 0.30
IOU_THRESHOLD   = 0.45
IMGSZ           = 1280

# Tracking
TRACK_THRESH    = 0.25
TRACK_BUFFER    = 30
MATCH_THRESH    = 0.80

# Ball
BALL_CONF       = 0.15
USE_TILING      = True

# Display
SHOW_TRAILS     = True
SHOW_IDS        = True
SHOW_TEAMS      = True
SHOW_BALL_TRAIL = True


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basketball Analytics Pipeline")
    p.add_argument("--model",   default=DEFAULT_MODEL,  help="Path to YOLO .pt weights")
    p.add_argument("--video",   default=DEFAULT_VIDEO,  help="Input video path")
    p.add_argument("--output",  default=DEFAULT_OUTPUT, help="Output video path")
    p.add_argument("--traj",    default=DEFAULT_TRAJ,   help="Trajectory JSON output path")
    p.add_argument("--no-tiling",  action="store_true", help="Disable SAHI ball tiling")
    p.add_argument("--no-trails",  action="store_true", help="Disable motion trails in output")
    p.add_argument("--device",  default="0",            help="Torch device ('0' or 'cpu')")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(args: argparse.Namespace):
    """Initialise all pipeline components and return them as a dict."""
    print("\n" + "═" * 60)
    print("  BASKETBALL ANALYTICS — PIPELINE INIT")
    print("═" * 60)

    detector = BasketballDetector(
        model_path = args.model,
        conf       = CONF_THRESHOLD,
        iou        = IOU_THRESHOLD,
        imgsz      = IMGSZ,
        device     = args.device,
    )
    detector.warmup()

    ball_tracker = BallTracker(
        model_path  = args.model,
        ball_conf   = BALL_CONF,
        imgsz       = IMGSZ,
        use_tiling  = not args.no_tiling,
        device      = args.device,
    )

    tracker = PlayerTracker(
        track_thresh = TRACK_THRESH,
        track_buffer = TRACK_BUFFER,
        match_thresh = MATCH_THRESH,
    )

    clusterer = TeamClusterer(
        warm_up_frames = 60,
        method         = "kmeans",
    )

    interpolator = TrajectoryInterpolator(
        max_gap       = 15,
        use_parabolic = True,
    )

    print("═" * 60 + "\n")

    return {
        "detector":     detector,
        "ball_tracker": ball_tracker,
        "tracker":      tracker,
        "clusterer":    clusterer,
        "interpolator": interpolator,
    }


# ─────────────────────────────────────────────────────────────────────────────
def open_video(video_path: str) -> cv2.VideoCapture:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def make_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer at: {output_path}")
    return writer


# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(
    frame:        np.ndarray,
    frame_idx:    int,
    total_frames: int,
    fps_actual:   float,
    tracker:      PlayerTracker,
    clusterer:    TeamClusterer,
) -> np.ndarray:
    """
    Overlay a heads-up display: frame counter, FPS, track count, team sizes.
    Drawn in the top-left corner.
    """
    vis   = frame
    rosters = clusterer.get_team_rosters()
    lines = [
        f"Frame: {frame_idx:5d} / {total_frames}",
        f"FPS  : {fps_actual:5.1f}",
        f"Players tracked: {tracker.total_player_tracks}",
        f"Refs   tracked : {tracker.total_ref_tracks}",
        f"Team A : {len(rosters.get(0, []))} players",
        f"Team B : {len(rosters.get(1, []))} players",
    ]

    y = 24
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (8, y - th - 2), (8 + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += th + 10

    return vis


# ─────────────────────────────────────────────────────────────────────────────
def save_trajectories(
    trajectories:  dict[int, list[tuple]],
    output_path:   str,
    ref_id_offset: int = REF_ID_OFFSET,
) -> None:
    """
    Serialise trajectories to JSON for the analytics module.

    Format
    ──────
    {
      "players": {
        "<track_id>": [[cx, cy, frame_idx], ...]
      },
      "referees": {
        "<display_id>": [[cx, cy, frame_idx], ...]
      }
    }

    `display_id` for referees = track_id - REF_ID_OFFSET (e.g. 10003 → 3).
    """
    out: dict = {"players": {}, "referees": {}}

    for tid, pts in trajectories.items():
        serialised = [[round(cx, 2), round(cy, 2), fidx] for cx, cy, fidx in pts]
        if tid >= ref_id_offset:
            out["referees"][str(tid - ref_id_offset)] = serialised
        else:
            out["players"][str(tid)] = serialised

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    total = len(out["players"]) + len(out["referees"])
    print(f"[Main] Trajectories saved → {output_path}  ({total} tracks)")


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> None:
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    # ── Init components ───────────────────────────────────────────────────────
    pipe = build_pipeline(args)
    detector     = pipe["detector"]
    ball_tracker = pipe["ball_tracker"]
    tracker      = pipe["tracker"]
    clusterer    = pipe["clusterer"]
    interpolator = pipe["interpolator"]

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap          = open_video(args.video)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = make_writer(args.output, fps, width, height)

    print(f"[Main] Video  : {args.video}  ({width}×{height} @ {fps:.1f}fps,  {total_frames} frames)")
    print(f"[Main] Output : {args.output}")
    print()

    # ── Trajectory buffer  {track_id: [(cx, cy, frame_idx), ...]} ─────────────
    # Collected live during the loop; fed into TrajectoryInterpolator after.
    trajectories: dict[int, list[tuple]] = defaultdict(list)

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_idx   = 0
    t_prev      = time.perf_counter()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. Detection ──────────────────────────────────────────────────
            raw_result = detector.detect(frame)
            raw_dets   = detector.parse(raw_result)

            # ── 2. Ball tracking (dedicated, low-conf + SAHI) ─────────────────
            ball_tracker.update(frame, frame_idx)

            # ── 3. Player + referee tracking (ByteTrack) ─────────────────────
            # tracked_dets = raw_dets + {track_id, team_id} keys
            tracked_dets = tracker.update(raw_dets)

            # ── 4. Team clustering (jersey colour KMeans) ─────────────────────
            # clusterer.update() buffers jersey colours per track_id.
            # After warm_up_frames it fits KMeans and begins assigning labels.
            clusterer.update(frame, tracked_dets)

            # ── 5. Write team labels back into dets ───────────────────────────
            # tracker.update() seeds team_id = TEAM_UNKNOWN (-1).
            # Now that clusterer has run, fill in the real assignment.
            for det in tracked_dets:
                tid = det["track_id"]
                if tid != -1:
                    det["team_id"] = clusterer.get_team(tid)

            # ── 6. Collect trajectories for post-loop interpolation ───────────
            for det in tracked_dets:
                tid = det["track_id"]
                if tid != -1:
                    cx, cy = det["center"]
                    trajectories[tid].append((cx, cy, frame_idx))

            # ── 7. Visualise ──────────────────────────────────────────────────
            vis = tracker.draw_tracks(
                frame,
                tracked_dets,
                show_trails = SHOW_TRAILS and not args.no_trails,
                show_ids    = SHOW_IDS,
                show_teams  = SHOW_TEAMS,
            )
            vis = ball_tracker.draw_ball(vis, frame_idx)

            # FPS counter (rolling average over last frame)
            t_now       = time.perf_counter()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev      = t_now

            vis = draw_hud(vis, frame_idx, total_frames, fps_display, tracker, clusterer)

            writer.write(vis)

            # ── 8. Console progress every 30 frames ───────────────────────────
            if frame_idx % 30 == 0:
                pct = 100.0 * frame_idx / max(total_frames, 1)
                print(
                    f"\r[Main] {frame_idx:5d}/{total_frames}  ({pct:5.1f}%)  "
                    f"{fps_display:5.1f}fps  "
                    f"players={tracker.total_player_tracks}  "
                    f"refs={tracker.total_ref_tracks}",
                    end="", flush=True,
                )

            frame_idx += 1

    finally:
        # Always release resources, even if an exception is raised mid-loop
        cap.release()
        writer.release()

    print(f"\n\n[Main] Frame loop complete — {frame_idx} frames processed.")

    # ── Post-loop: refine team clustering on full data ────────────────────────
    print("\n[Main] Refining team clustering…")
    clusterer.refine()
    clusterer.print_roster()

    # ── Post-loop: fill trajectory gaps (jumps, occlusions) ──────────────────
    print("[Main] Running trajectory interpolation…")
    filled_trajectories = interpolator.fill_all(dict(trajectories))

    # Optional: print gap report for debugging
    gap_report = interpolator.get_gap_report(dict(trajectories))
    _print_gap_summary(gap_report)

    # ── Save trajectories for analytics module ────────────────────────────────
    save_trajectories(filled_trajectories, args.traj)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("═" * 60)
    print(f"  Output video  : {args.output}")
    print(f"  Trajectories  : {args.traj}")
    print(f"  Frames        : {frame_idx}")
    print(f"  {tracker}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def _print_gap_summary(gap_report: dict[int, list[dict]]) -> None:
    """Print a concise gap-fill summary to stdout."""
    total_gaps  = sum(len(v) for v in gap_report.values())
    jump_gaps   = sum(
        1 for gaps in gap_report.values()
        for g in gaps if g["type"] == "jump"
    )
    filled_gaps = sum(
        1 for gaps in gap_report.values()
        for g in gaps if g["filled"]
    )

    print(
        f"[Main] Gap report — tracks with gaps: {len(gap_report)}  |  "
        f"total gaps: {total_gaps}  |  "
        f"filled: {filled_gaps}  |  "
        f"jump-type: {jump_gaps}  |  "
        f"skipped (too long): {total_gaps - filled_gaps}"
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)