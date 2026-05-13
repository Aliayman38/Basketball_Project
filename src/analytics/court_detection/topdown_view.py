"""
src/analytics/court_detection/topdown_view.py
───────────────────────────────────────────────
Renders side-by-side video: camera view | top-down court with players.
Uses roboflow/sports library for court drawing and ViewTransformer.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from sports.basketball import CourtConfiguration, League, draw_court, draw_points_on_court
    from sports import MeasurementUnit
    HAS_SPORTS_LIB = True
except ImportError:
    HAS_SPORTS_LIB = False

from .court_template import KP_TO_WORLD, VERTICES, COURT_LENGTH_FT, COURT_WIDTH_FT
from .homography import HomographyTracker, project_points


def _make_fallback_court(width=800, height=425) -> np.ndarray:
    canvas = np.full((height, width, 3), (60, 95, 145), dtype=np.uint8)
    line = (255, 255, 255)
    sx = width / COURT_LENGTH_FT
    sy = height / COURT_WIDTH_FT
    def to_px(x_ft, y_ft):
        return (int(x_ft * sx), int(y_ft * sy))
    cv2.rectangle(canvas, to_px(0, 0), to_px(94, 50), line, 2)
    cv2.line(canvas, to_px(47, 0), to_px(47, 50), line, 2)
    cv2.circle(canvas, to_px(47, 25), int(6 * (sx + sy) / 2), line, 2)
    cv2.rectangle(canvas, to_px(0, 17), to_px(19, 33), line, 2)
    cv2.rectangle(canvas, to_px(75, 17), to_px(94, 33), line, 2)
    cv2.circle(canvas, to_px(19, 25), int(6 * (sx + sy) / 2), line, 2)
    cv2.circle(canvas, to_px(75, 25), int(6 * (sx + sy) / 2), line, 2)
    cv2.circle(canvas, to_px(5.25, 25), 4, (0, 0, 200), -1)
    cv2.circle(canvas, to_px(88.75, 25), 4, (0, 0, 200), -1)
    return canvas


def _team_color_bgr(team_name: str) -> tuple:
    if not team_name:
        return (200, 200, 200)
    t = team_name.strip().lower()
    if t in ("t1", "team 0", "team0"):  return (0, 255, 255)    # yellow
    if t in ("t2", "team 1", "team1"):  return (0, 0, 255)      # red
    if "yellow" in t: return (0, 255, 255)
    if "blue" in t:   return (255, 100, 0)
    if "white" in t:  return (240, 240, 240)
    if "red" in t:    return (0, 0, 220)
    if "dark" in t:   return (100, 60, 30)
    return (180, 180, 180)


def render_topdown_video(
    input_video_path:  str,
    output_video_path: str,
    keypoints_json:    str,
    trajectories_json: Optional[str] = None,
) -> dict:
    # ── Load per-frame keypoints ────────────────────────────────────────
    with open(keypoints_json) as f:
        kp_data = json.load(f)
    per_frame_kps = kp_data.get("per_frame_keypoints", kp_data.get("per_frame", []))
    print(f"[Topdown] Loaded {len(per_frame_kps)} frames of keypoints")

    # ── Load player trajectories ────────────────────────────────────────
    frame_to_players: dict[int, list[dict]] = {}
    if trajectories_json and os.path.exists(trajectories_json):
        with open(trajectories_json) as f:
            traj = json.load(f)
        for pid, recs in traj.get("players", {}).items():
            for rec in recs:
                fi = int(rec["frame"])
                cx, cy = rec["center"]
                frame_to_players.setdefault(fi, []).append({
                    "id":   str(pid),
                    "foot": (float(cx), float(cy)),
                    "team": rec.get("team", ""),
                })
        print(f"[Topdown] Loaded trajectories for "
              f"{len(traj.get('players', {}))} players, "
              f"{len(frame_to_players)} frames with data")

    # ── Build court image ───────────────────────────────────────────────
    if HAS_SPORTS_LIB:
        config = CourtConfiguration(
            league=League.NBA, measurement_unit=MeasurementUnit.FEET)
        base_court = draw_court(config=config)
        print(f"[Topdown] Court image: {base_court.shape} (roboflow/sports)")
    else:
        base_court = _make_fallback_court()
        print(f"[Topdown] Court image: {base_court.shape} (fallback)")

    court_h, court_w = base_court.shape[:2]
    court_scale_x = court_w / COURT_LENGTH_FT
    court_scale_y = court_h / COURT_WIDTH_FT

    # ── Open input video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mini_h = in_h
    mini_w = int(court_w * (mini_h / court_h))
    out_w  = in_w + mini_w

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".",
                exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, in_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open writer for: {output_video_path}")

    print(f"[Topdown] Input: {in_w}x{in_h} @ {fps:.1f} fps, {n_total} frames")
    print(f"[Topdown] Output: {out_w}x{in_h}")

    # ── Homography tracker ──────────────────────────────────────────────
    tracker = HomographyTracker(
        vertices=VERTICES, conf_threshold=0.5,
        min_correspondences=4, max_stale_frames=60)

    fi = 0
    t0 = time.time()
    world_trajectories: dict[str, list[dict]] = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Parse keypoints (handle both dict and list format)
            if fi < len(per_frame_kps):
                raw = per_frame_kps[fi]
                keypoints = []
                for kp in raw:
                    if isinstance(kp, dict):
                        keypoints.append((int(kp['idx']), float(kp['x']),
                                          float(kp['y']), float(kp['conf'])))
                    else:
                        keypoints.append((int(kp[0]), float(kp[1]),
                                          float(kp[2]), float(kp[3])))
            else:
                keypoints = []

            transformer = tracker.update(keypoints)

            # Build court minimap
            court = base_court.copy()

            if transformer is not None and fi in frame_to_players:
                players = frame_to_players[fi]
                foot_pixels = np.array(
                    [p["foot"] for p in players], dtype=np.float32)

                if len(foot_pixels) > 0:
                    try:
                        court_xy = project_points(foot_pixels, transformer)

                        for p, (cx, cy) in zip(players, court_xy):
                            px = int(cx * court_scale_x)
                            py = int(cy * court_scale_y)

                            if -20 < px < court_w + 20 and -20 < py < court_h + 20:
                                px = max(0, min(court_w - 1, px))
                                py = max(0, min(court_h - 1, py))
                                color = _team_color_bgr(p["team"])

                                cv2.circle(court, (px, py), 20, color, -1)
                                cv2.circle(court, (px, py), 22, (0, 0, 0), 2)
                                cv2.putText(
                                    court, p["id"], (px + 24, py - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                                cv2.putText(
                                    court, p["id"], (px + 24, py - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    color, 1, cv2.LINE_AA)

                            world_trajectories.setdefault(p["id"], []).append({
                                "frame": fi,
                                "x_ft": float(cx),
                                "y_ft": float(cy),
                                "team": p["team"],
                            })
                    except Exception:
                        pass

            # HUD
            status = "FRESH" if (tracker.frames_since_fresh == 0 and transformer is not None) \
                     else ("REUSED" if transformer is not None else "NO H")
            cv2.rectangle(court, (0, 0), (court_w, 30), (20, 20, 20), -1)
            cv2.putText(court, f"TOP-DOWN VIEW   H: {status}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            mini_resized = cv2.resize(court, (mini_w, mini_h),
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

    # Save world trajectories
    if world_trajectories:
        out_dir = Path(output_video_path).parent
        traj_out = out_dir / "analytics" / "trajectories_world.json"
        traj_out.parent.mkdir(parents=True, exist_ok=True)
        with open(traj_out, "w") as f:
            json.dump({
                "court_length_ft": COURT_LENGTH_FT,
                "court_width_ft":  COURT_WIDTH_FT,
                "trajectories":    world_trajectories,
            }, f, indent=2)
        print(f"[Topdown]   World trajectories → {traj_out}")

    elapsed = time.time() - t0
    summary = tracker.summary()
    summary["total_frames"] = fi
    summary["elapsed_seconds"] = elapsed
    print(f"\n[Topdown] ✓ {fi} frames in {elapsed:.1f}s")
    print(f"[Topdown]   Fresh H: {summary['frames_fresh']}")
    print(f"[Topdown]   Reused:  {summary['frames_reused']}")
    print(f"[Topdown]   No-H:    {summary['frames_no_h']}")
    print(f"[Topdown]   Output:  {output_video_path}")
    return summary
