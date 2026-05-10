"""
src/analytics/court_detection/topdown_view.py
───────────────────────────────────────────────
Renders a top-down basketball court visualization with players
projected from camera pixels to court meters using per-frame H.

Inputs:
  - The original tracking video (with bboxes/IDs/teams already drawn)
  - per-frame keypoints from the trained court model
  - trajectories.json (player foot positions in pixels per frame)

Outputs:
  - A side-by-side video: original camera view | top-down minimap
  - JSON file with per-player real-world (X_m, Y_m) trajectories
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .court_template import (
    KP_TO_WORLD, COURT_LENGTH_M, COURT_WIDTH_M, CY_CENTER,
    PAINT_DEPTH_M, PAINT_TOP_Y, PAINT_BOT_Y,
)
from .homography import HomographyTracker, project_point


# ─────────────────────────────────────────────────────────────────────────────
#  Top-down court rendering
# ─────────────────────────────────────────────────────────────────────────────
def make_court_canvas(
    out_w_px: int = 560,    # canvas width in pixels (visual size)
    out_h_px: int = 300,    # canvas height in pixels
    pad_px:   int = 20,
) -> tuple[np.ndarray, float, float, int]:
    """
    Create a top-down basketball court image (drawn lines on a wood
    background). Returns the canvas plus the meters-to-pixels scale
    factors for plotting players on top of it.

    Returns
    -------
    canvas, scale_x, scale_y, pad_px
        canvas    : np.ndarray (H, W, 3)
        scale_x   : pixels per meter along court length
        scale_y   : pixels per meter along court width
        pad_px    : how much border the canvas reserves
    """
    canvas = np.full((out_h_px, out_w_px, 3), (60, 90, 130), dtype=np.uint8)  # wood-tone

    inner_w = out_w_px - 2 * pad_px
    inner_h = out_h_px - 2 * pad_px
    scale_x = inner_w / COURT_LENGTH_M
    scale_y = inner_h / COURT_WIDTH_M

    def m_to_px(x_m: float, y_m: float) -> tuple[int, int]:
        return (int(pad_px + x_m * scale_x), int(pad_px + y_m * scale_y))

    line_color = (255, 255, 255)
    th = 2

    # Outer boundary
    cv2.rectangle(canvas, m_to_px(0, 0), m_to_px(COURT_LENGTH_M, COURT_WIDTH_M),
                  line_color, th)

    # Center line
    cv2.line(canvas, m_to_px(COURT_LENGTH_M/2, 0),
                     m_to_px(COURT_LENGTH_M/2, COURT_WIDTH_M),
             line_color, th)

    # Center circle
    cx, cy = m_to_px(COURT_LENGTH_M/2, CY_CENTER)
    r_px = int(1.8 * (scale_x + scale_y) / 2)
    cv2.circle(canvas, (cx, cy), r_px, line_color, th)

    # Left paint
    cv2.rectangle(canvas, m_to_px(0, PAINT_TOP_Y),
                          m_to_px(PAINT_DEPTH_M, PAINT_BOT_Y),
                  line_color, th)
    # Left FT circle
    fx, fy = m_to_px(PAINT_DEPTH_M, CY_CENTER)
    cv2.circle(canvas, (fx, fy), int(1.8 * (scale_x + scale_y) / 2),
               line_color, th)

    # Right paint
    cv2.rectangle(canvas, m_to_px(COURT_LENGTH_M - PAINT_DEPTH_M, PAINT_TOP_Y),
                          m_to_px(COURT_LENGTH_M, PAINT_BOT_Y),
                  line_color, th)
    # Right FT circle
    fx2, fy2 = m_to_px(COURT_LENGTH_M - PAINT_DEPTH_M, CY_CENTER)
    cv2.circle(canvas, (fx2, fy2), int(1.8 * (scale_x + scale_y) / 2),
               line_color, th)

    return canvas, scale_x, scale_y, pad_px


def draw_player_dot(
    canvas:  np.ndarray,
    world_xy: tuple[float, float],
    scale_x: float,
    scale_y: float,
    pad_px:  int,
    color:   tuple[int, int, int] = (0, 255, 255),
    label:   str | None = None,
) -> np.ndarray:
    """Plot one player on the top-down canvas."""
    x_m, y_m = world_xy

    # Skip if outside the drawn court area (projection went haywire)
    if not (-2 < x_m < COURT_LENGTH_M + 2 and -2 < y_m < COURT_WIDTH_M + 2):
        return canvas

    px = int(pad_px + x_m * scale_x)
    py = int(pad_px + y_m * scale_y)
    cv2.circle(canvas, (px, py), 6, color,        -1)
    cv2.circle(canvas, (px, py), 7, (0, 0, 0),     1)
    if label:
        cv2.putText(canvas, label, (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,     1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  Main render function
# ─────────────────────────────────────────────────────────────────────────────
def render_topdown_video(
    input_video_path:  str,
    output_video_path: str,
    keypoints_json:    str,
    trajectories_json: Optional[str] = None,
    canvas_w:          int = 560,
    canvas_h:          int = 300,
) -> dict:
    """
    Produce a side-by-side video: input video on the left, top-down
    court minimap on the right.

    Parameters
    ----------
    input_video_path   : the rendered tracking video (e.g. final_with_landmarks.mp4)
    output_video_path  : where to write the side-by-side result
    keypoints_json     : path to court_keypoints.json (from landmarks_overlay)
    trajectories_json  : path to trajectories.json (optional, draws players)
    canvas_w, canvas_h : top-down minimap size in pixels

    Returns
    -------
    summary dict with homography stats and total processing time
    """
    # Load per-frame keypoints from JSON
    with open(keypoints_json) as f:
        kp_data = json.load(f)
    per_frame_kps = kp_data["per_frame_keypoints"]
    print(f"[Topdown] Loaded {len(per_frame_kps)} frames of keypoints")

    # Load player trajectories if provided
    trajectories = {}
    frame_to_players: dict[int, list[dict]] = {}   # frame_idx -> [{id, foot_pixel, team}, ...]
    if trajectories_json and os.path.exists(trajectories_json):
        with open(trajectories_json) as f:
            traj = json.load(f)
        # Trajectories format: {"players": {pid: [{"frame":N, "center":[x,y], "team":...}, ...]}}
        for pid, recs in traj.get("players", {}).items():
            for rec in recs:
                fi = int(rec["frame"])
                cx, cy = rec["center"]
                # bbox bottom-center is the foot — approximate from "center"
                foot = (float(cx), float(cy))
                frame_to_players.setdefault(fi, []).append({
                    "id":   str(pid),
                    "foot": foot,
                    "team": rec.get("team", ""),
                })
        print(f"[Topdown] Loaded trajectories for {len(traj.get('players', {}))} players")

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # The output is the input video on the left + top-down minimap on the right
    # Resize minimap to match input video height
    mini_h = in_h
    mini_w = int(canvas_w * (mini_h / canvas_h))
    out_w = in_w + mini_w
    out_h = in_h

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output_video_path}")

    print(f"[Topdown] Input: {in_w}x{in_h} @ {fps:.1f} fps, {n_total} frames")
    print(f"[Topdown] Output: {out_w}x{out_h}")

    # Precompute the static court canvas (it doesn't change per frame)
    base_canvas, scale_x, scale_y, pad_px = make_court_canvas(canvas_w, canvas_h)

    # Initialize the homography tracker
    tracker = HomographyTracker(KP_TO_WORLD, min_correspondences=6)

    # Per-frame processing
    fi = 0
    t0 = time.time()
    world_trajectories: dict[str, list[dict]] = {}   # for JSON export

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Get this frame's keypoints (rebuild from JSON tuples)
            if fi < len(per_frame_kps):
                raw = per_frame_kps[fi]
                keypoints = [(int(kp[0]), float(kp[1]), float(kp[2]), float(kp[3]))
                             for kp in raw]
            else:
                keypoints = []

            # Compute / update homography
            H = tracker.update(keypoints)

            # Build the minimap canvas for this frame (copy of the static base)
            mini = base_canvas.copy()

            # Project players onto the minimap, if we have H and trajectories
            if H is not None and fi in frame_to_players:
                for player in frame_to_players[fi]:
                    foot_pix = player["foot"]
                    try:
                        world_xy = project_point(foot_pix, H)
                    except Exception:
                        continue
                    color = _team_color(player["team"])
                    draw_player_dot(mini, world_xy, scale_x, scale_y, pad_px,
                                    color=color, label=player["id"])
                    # Save for JSON export
                    world_trajectories.setdefault(player["id"], []).append({
                        "frame": fi,
                        "x_m":   world_xy[0],
                        "y_m":   world_xy[1],
                        "team":  player["team"],
                    })

            # HUD on the minimap
            status = "FRESH" if (tracker.frames_since_fresh == 0 and H is not None) \
                     else ("REUSED" if H is not None else "NO H")
            cv2.putText(mini, f"H: {status}", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(mini, f"H: {status}", (10, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Resize minimap to match input video height
            mini_resized = cv2.resize(mini, (mini_w, mini_h), interpolation=cv2.INTER_AREA)

            # Concatenate: input frame on left, minimap on right
            combined = np.hstack([frame, mini_resized])
            writer.write(combined)

            fi += 1
            if fi % 30 == 0:
                elapsed = time.time() - t0
                rate = fi / max(elapsed, 1e-3)
                print(f"   topdown: {fi}/{n_total}  ({rate:.1f} fps)")
    finally:
        cap.release()
        writer.release()

    # Save world-coordinate trajectories
    if world_trajectories:
        out_dir = Path(output_video_path).parent
        traj_out = out_dir / "trajectories_world.json"
        with open(traj_out, "w") as f:
            json.dump({"world_trajectories": world_trajectories}, f, indent=2)
        print(f"[Topdown]   World trajectories → {traj_out}")

    elapsed = time.time() - t0
    print(f"\n[Topdown] ✓ {fi} frames in {elapsed:.1f}s")
    summary = tracker.summary()
    summary["total_frames"]    = fi
    summary["elapsed_seconds"] = elapsed
    print(f"[Topdown]   Fresh H frames:  {summary['frames_fresh']}")
    print(f"[Topdown]   Reused H frames: {summary['frames_reused']}")
    print(f"[Topdown]   No-H frames:     {summary['frames_no_h']}")
    print(f"[Topdown]   Output:          {output_video_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _team_color(team_name: str) -> tuple[int, int, int]:
    """Map team string to a BGR color for the minimap dot."""
    if not team_name:
        return (200, 200, 200)
    t = team_name.lower()
    if "yellow" in t:    return (0, 255, 255)        # yellow
    if "blue" in t:      return (255, 100,   0)      # dark blue
    if "white" in t:     return (240, 240, 240)
    if "dark" in t:      return (100,  60,  30)
    return (180, 180, 180)
