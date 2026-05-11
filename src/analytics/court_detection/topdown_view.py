"""
src/analytics/court_detection/topdown_view.py
───────────────────────────────────────────────
Renders the top-down basketball court minimap with players projected
from camera pixels to canvas pixels using per-frame H.

Inspired by abdullahtarek/basketball_analysis tactical_view drawer.

Inputs:
  - Camera-view video (e.g. final_with_landmarks.mp4)
  - per-frame keypoints JSON (from landmarks_overlay)
  - trajectories.json (player pixel positions per frame)

Outputs:
  - Side-by-side video: camera view | top-down canvas with players
  - JSON file with each player's canvas-pixel trajectory over time
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
    KP_TO_WORLD,
    CANVAS_W, CANVAS_H,
    PX_PER_M_X, PX_PER_M_Y,
    CY, HALF_X, HOOP_X,
    PAINT_DEPTH_PX, PAINT_TOP_Y, PAINT_BOT_Y,
    FT_CIRC_PX,
    COURT_LENGTH_M, COURT_WIDTH_M,
)
from .homography import HomographyTracker, project_point


# ─────────────────────────────────────────────────────────────────────────────
#  Court canvas drawing
# ─────────────────────────────────────────────────────────────────────────────
def make_court_canvas() -> np.ndarray:
    """Build the static top-down basketball court image (background)."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), (60, 95, 145), dtype=np.uint8)  # wood color
    line = (255, 255, 255)
    th = 2

    # Outer rectangle
    cv2.rectangle(canvas, (0, 0), (CANVAS_W - 1, CANVAS_H - 1), line, th)

    # Half-court line
    cv2.line(canvas, (HALF_X, 0), (HALF_X, CANVAS_H), line, th)

    # Center circle
    cv2.circle(canvas, (HALF_X, CY), FT_CIRC_PX, line, th)

    # Left paint
    cv2.rectangle(canvas, (0, PAINT_TOP_Y), (PAINT_DEPTH_PX, PAINT_BOT_Y), line, th)
    # Left FT circle
    cv2.circle(canvas, (PAINT_DEPTH_PX, CY), FT_CIRC_PX, line, th)
    # Left basket (small filled circle at hoop position)
    cv2.circle(canvas, (HOOP_X, CY), 4, (0, 0, 200), -1)

    # Right paint (mirrored)
    right_paint_start = CANVAS_W - PAINT_DEPTH_PX
    right_hoop_x = CANVAS_W - HOOP_X
    cv2.rectangle(canvas, (right_paint_start, PAINT_TOP_Y),
                          (CANVAS_W, PAINT_BOT_Y), line, th)
    cv2.circle(canvas, (right_paint_start, CY), FT_CIRC_PX, line, th)
    cv2.circle(canvas, (right_hoop_x, CY), 4, (0, 0, 200), -1)

    return canvas


def draw_player_dot(
    canvas:  np.ndarray,
    canvas_xy: tuple[float, float],
    color:   tuple[int, int, int] = (0, 255, 255),
    label:   Optional[str] = None,
    radius:  int = 7,
) -> np.ndarray:
    """Plot one player on the canvas, with bounds checking."""
    x, y = canvas_xy

    # Soft clamp to canvas (small margin for visibility)
    margin = 5
    if not (-margin < x < CANVAS_W + margin and -margin < y < CANVAS_H + margin):
        return canvas

    x_i = int(max(0, min(CANVAS_W - 1, x)))
    y_i = int(max(0, min(CANVAS_H - 1, y)))

    cv2.circle(canvas, (x_i, y_i), radius,     color,     -1)
    cv2.circle(canvas, (x_i, y_i), radius + 1, (0, 0, 0),  1)
    if label:
        cv2.putText(canvas, label, (x_i + radius + 2, y_i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x_i + radius + 2, y_i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color,    1, cv2.LINE_AA)
    return canvas


def _team_color(team_name: str) -> tuple[int, int, int]:
    """Map team string to BGR color for the minimap dot."""
    if not team_name:
        return (200, 200, 200)
    t = team_name.lower()
    if "yellow" in t:  return (0, 255, 255)
    if "blue"   in t:  return (255, 100, 0)
    if "white"  in t:  return (240, 240, 240)
    if "red"    in t:  return (0, 0, 220)
    if "dark"   in t:  return (100, 60, 30)
    return (180, 180, 180)


# ─────────────────────────────────────────────────────────────────────────────
#  Main render function
# ─────────────────────────────────────────────────────────────────────────────
def render_topdown_video(
    input_video_path:  str,
    output_video_path: str,
    keypoints_json:    str,
    trajectories_json: Optional[str] = None,
) -> dict:
    """
    Produce a side-by-side video: input video on left, top-down minimap on right.

    Parameters
    ----------
    input_video_path   : the camera-view video (e.g. final_with_landmarks.mp4)
    output_video_path  : where to write the side-by-side result
    keypoints_json     : path to court_keypoints.json (from landmarks_overlay)
    trajectories_json  : path to trajectories.json (player tracking output)

    Returns
    -------
    summary dict with homography stats and processing time.
    """
    # Load per-frame keypoints
    with open(keypoints_json) as f:
        kp_data = json.load(f)
    per_frame_kps = kp_data["per_frame_keypoints"]
    print(f"[Topdown] Loaded {len(per_frame_kps)} frames of keypoints")

    # Load player trajectories
    frame_to_players: dict[int, list[dict]] = {}
    if trajectories_json and os.path.exists(trajectories_json):
        with open(trajectories_json) as f:
            traj = json.load(f)
        for pid, recs in traj.get("players", {}).items():
            for rec in recs:
                fi = int(rec["frame"])
                cx, cy = rec["center"]
                # Use the BOTTOM-CENTER of the bbox as the foot position
                # (the "center" field from trajectories is bbox center, so
                # we approximate foot as a point slightly below center)
                # If a 'bbox' field is available we'd use that — for now,
                # center is a reasonable approximation for the foot point
                foot = (float(cx), float(cy))
                frame_to_players.setdefault(fi, []).append({
                    "id":   str(pid),
                    "foot": foot,
                    "team": rec.get("team", ""),
                })
        print(f"[Topdown] Loaded trajectories for "
              f"{len(traj.get('players', {}))} players")

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Side-by-side output: scale minimap to match video height
    mini_h = in_h
    mini_w = int(CANVAS_W * (mini_h / CANVAS_H))
    out_w  = in_w + mini_w

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".",
                exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, in_h)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output_video_path}")

    print(f"[Topdown] Input: {in_w}x{in_h} @ {fps:.1f} fps, {n_total} frames")
    print(f"[Topdown] Output: {out_w}x{in_h}")

    # Precompute the static canvas (court schematic doesn't change)
    base_canvas = make_court_canvas()

    # Init the homography tracker
    tracker = HomographyTracker(KP_TO_WORLD, min_correspondences=4,
                                 ransac_threshold_px=10.0, validate=True)

    fi = 0
    t0 = time.time()
    world_trajectories: dict[str, list[dict]] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # This frame's keypoints
            if fi < len(per_frame_kps):
                raw = per_frame_kps[fi]
                keypoints = [(int(kp[0]), float(kp[1]), float(kp[2]),
                              float(kp[3])) for kp in raw]
            else:
                keypoints = []

            H = tracker.update(keypoints)

            # Build minimap
            mini = base_canvas.copy()

            if H is not None and fi in frame_to_players:
                for player in frame_to_players[fi]:
                    foot_pix = player["foot"]
                    try:
                        canvas_xy = project_point(foot_pix, H)
                    except Exception:
                        continue
                    color = _team_color(player["team"])
                    draw_player_dot(mini, canvas_xy,
                                    color=color, label=player["id"])
                    # Save for export
                    world_trajectories.setdefault(player["id"], []).append({
                        "frame":  fi,
                        "x_canvas": canvas_xy[0],
                        "y_canvas": canvas_xy[1],
                        "team":   player["team"],
                    })

            # HUD on minimap
            status = "FRESH" if (tracker.frames_since_fresh == 0 and H is not None) \
                     else ("REUSED" if H is not None else "NO H")
            cv2.rectangle(mini, (0, 0), (CANVAS_W, 24), (20, 20, 20), -1)
            cv2.putText(mini, "TOP-DOWN VIEW", (8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(mini, f"H: {status}", (CANVAS_W - 110, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if status == "FRESH":
                cv2.circle(mini, (CANVAS_W - 15, 12), 4, (0, 255, 0), -1)
            elif status == "REUSED":
                cv2.circle(mini, (CANVAS_W - 15, 12), 4, (0, 165, 255), -1)
            else:
                cv2.circle(mini, (CANVAS_W - 15, 12), 4, (0, 0, 255), -1)

            # Resize minimap to match input video height
            mini_resized = cv2.resize(mini, (mini_w, mini_h),
                                     interpolation=cv2.INTER_AREA)

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

    # Save canvas-coordinate trajectories
    if world_trajectories:
        out_dir = Path(output_video_path).parent
        traj_out = out_dir / "trajectories_world.json"
        with open(traj_out, "w") as f:
            json.dump({
                "canvas_w": CANVAS_W,
                "canvas_h": CANVAS_H,
                "court_length_m": COURT_LENGTH_M,
                "court_width_m":  COURT_WIDTH_M,
                "px_per_m_x": PX_PER_M_X,
                "px_per_m_y": PX_PER_M_Y,
                "trajectories": world_trajectories,
            }, f, indent=2)
        print(f"[Topdown]   Canvas trajectories → {traj_out}")

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
