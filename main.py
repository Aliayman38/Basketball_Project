"""
main.py
Orchestration layer: detection → tracking → analytics → dashboard visualization.
"""

from __future__ import annotations

import cv2
import json
import time
import torch
from pathlib import Path

from tracking.tracker import BasketballTracker, TEAM_NAMES_SHORT
from src.analytics.dashboard import render_video

# ── Analytics ─────────────────────────────────────────────────────────────────
import sys
_src = Path(__file__).parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from src.analytics.distance import build_report as build_distance_report, export_csv as export_distance_csv
from src.analytics.speed  import build_speed_report, export_csv as export_speed_csv
from src.analytics.shot_detector import detect_shots, load_trajectory, ShotResult
from src.analytics.possession import (
    build_possession_report,
    export_possession_csv,
    export_possession_json,
    print_possession_summary,
    get_possession_by_frame,
)
from src.analytics.possession_overlay import render_possession_video
from src.analytics.court_detection.landmarks_overlay import run_landmarks


# ═════════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════════

VIDEO_PATH        = 'data/shooting3.mp4'
MODEL_PATH        = 'models/weights/last.pt'
OUTPUT_PATH       = 'runs/bot-sort tracking/tracking_botsort3.mp4'
TRAJECTORIES_PATH = 'runs/bot-sort tracking/analytics/trajectories.json'
REID_PATH         = 'osnet_x0_25_msmt17.pt'
DEVICE            = torch.device('cuda:0')

METERS_PER_PIXEL = 0.0264
INCLUDE_REFEREES = False

TEAM_0_DESC = "a basketball player wearing a yellow jersey"
TEAM_1_DESC = "a basketball player wearing a dark blue jersey"

# ── Possession Config ─────────────────────────────────────────────────────────
BALL_POSSESSION_THRESHOLD = 80.0      # pixels: max distance ball→player center
POSSESSION_MIN_FRAMES     = 3         # debounce: frames to confirm change

# ── Landmark Config ───────────────────────────────────────────────────────────
LANDMARKS_WEIGHTS = "models/weights/court_kp.pt"
LANDMARKS_CONF    = 0.30


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def save_trajectories(trajectories: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f'   Trajectories → {path}')


def run_analytics(trajectories: dict, fps: float, analytics_dir: Path,
                  meters_per_pixel: float | None = None,
                  include_referees: bool = False):
    """Compute and export distance & speed reports."""

    # Distance
    flat = {pid: pts for pid, pts in trajectories.get("players", {}).items()}
    if include_referees:
        for rid, pts in trajectories.get("referees", {}).items():
            flat[f"ref_{rid}"] = pts
    export_distance_csv(
        build_distance_report(flat, meters_per_pixel),
        analytics_dir / "distance_report.csv"
    )

    # Speed
    speed_input = {
        pid: [[r["center"][0], r["center"][1], float(r["frame"])] for r in recs]
        for pid, recs in trajectories.get("players", {}).items()
    }
    export_speed_csv(
        build_speed_report(speed_input, fps=fps, meters_per_pixel=meters_per_pixel),
        analytics_dir / "speed_report.csv"
    )
    print(f"\n📊 Analytics saved to {analytics_dir}/")


def run_shot_detection(trajectories_path: str, analytics_dir: Path) -> list[ShotResult]:
    """Detect made shots from ball trajectory."""
    points, hoop = load_trajectory(trajectories_path)
    shots = detect_shots(points, hoop)

    records = [
        {
            "shot": i,
            "frames": f"{s.arc_start_frame}–{s.arc_end_frame}",
            "apex_frame": s.apex_frame,
            "entry": {"x": s.entry_x, "y": s.entry_y},
            "confidence": s.confidence,
        }
        for i, s in enumerate(shots, 1)
    ]

    shots_path = analytics_dir / "shots.json"
    with shots_path.open("w", encoding="utf-8") as f:
        json.dump({"made_shots": records, "hoop": {"x": hoop.x, "y": hoop.y, "radius": hoop.radius}}, f, indent=2)

    print(f"\n🏀 Shot Detection: {len(shots)} made shot(s)")
    for rec in records:
        print(f"   Shot {rec['shot']}: frames {rec['frames']}  apex@{rec['apex_frame']}  conf={rec['confidence']:.2f}")
    return shots


def assign_shot_teams(shots: list[ShotResult], trajectories: dict, clip) -> list[tuple[int, int]]:
    """Assign each shot to a team and return (frame, team_idx) events."""
    events = []
    for shot in shots:
        best_team, best_dist = 0, float('inf')
        for pid_str, records in trajectories.get("players", {}).items():
            for rec in records:
                if abs(rec["frame"] - shot.apex_frame) < 5:
                    px, py = rec["center"]
                    dist = ((px - shot.entry_x) ** 2 + (py - shot.entry_y) ** 2) ** 0.5
                    if dist < best_dist and rec.get("team"):
                        best_dist = dist
                        best_team = 0 if rec["team"] == clip.TEAM_NAMES[0] else 1
        events.append((shot.arc_end_frame, best_team))
        print(f"   Shot at frame {shot.arc_end_frame} → Team {TEAM_NAMES_SHORT[best_team]}")
    return events


def run_possession(trajectories: dict, fps: float, analytics_dir: Path,
                   team_names: tuple[str, str] = ("Team 0", "Team 1")) -> dict:
    """
    Compute and export ball possession analytics.
    Returns possession_by_frame dict for visualization.
    """
    print("\n🏀 Running Possession Analysis...")

    report = build_possession_report(
        trajectories,
        ball_threshold=BALL_POSSESSION_THRESHOLD,
        min_consecutive_frames=POSSESSION_MIN_FRAMES,
        team_names=team_names,
    )

    # Export
    export_possession_csv(report, analytics_dir / "possession_report.csv")
    export_possession_json(report, analytics_dir / "possession_report.json")

    # Console summary
    print_possession_summary(report, fps=fps)

    # Return frame lookup for visualization
    return get_possession_by_frame(report)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('🚀 Starting Basketball Analysis Pipeline...')

    # Init tracker
    tracker = BasketballTracker(
        model_path=MODEL_PATH,
        reid_path=REID_PATH,
        device=DEVICE,
        team_0_desc=TEAM_0_DESC,
        team_1_desc=TEAM_1_DESC,
    )

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Process frames
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process_frame(frame)
        writer.write(frame)

        if tracker.frame_count % 50 == 0:
            print(f'Frame {tracker.frame_count}  |  '
                  f'{tracker.frame_count / (time.time() - t0):.1f} FPS')

    cap.release()
    writer.release()

    # Save trajectories
    trajectories = tracker.get_trajectories()
    save_trajectories(trajectories, TRAJECTORIES_PATH)

    # Analytics pipeline
    analytics_dir = Path(TRAJECTORIES_PATH).parent

    shots = run_shot_detection(TRAJECTORIES_PATH, analytics_dir)
    shot_events = assign_shot_teams(shots, trajectories, tracker.clip)
    run_analytics(trajectories, fps, analytics_dir, METERS_PER_PIXEL, INCLUDE_REFEREES)

    # ── Ball Possession ─────────────────────────────────────────────────────
    team_names = (
        tracker.clip.TEAM_NAMES[0] if hasattr(tracker.clip, 'TEAM_NAMES') else "Team 0",
        tracker.clip.TEAM_NAMES[1] if hasattr(tracker.clip, 'TEAM_NAMES') else "Team 1",
    )
    possession_by_frame = run_possession(trajectories, fps, analytics_dir, team_names=team_names)

    # ── Possession Visualization ────────────────────────────────────────────
    print("\n🎨 Rendering possession highlights...")
    possession_video_path = str(Path(OUTPUT_PATH).parent / "tracking_possession.mp4")
    render_possession_video(
        input_video_path=OUTPUT_PATH,
        output_video_path=possession_video_path,
        trajectories=trajectories,
        possession_by_frame=possession_by_frame,
        fps=fps,
    )

    # ── Court Landmarks ─────────────────────────────────────────────────────
    print("\n🏀 Running Court Landmark Detection...")
    landmarks_video_path = str(Path(OUTPUT_PATH).parent / "tracking_landmarks.mp4")
    run_landmarks(
        input_video_path=possession_video_path,
        output_video_path=landmarks_video_path,
        analytics_dir=analytics_dir,
        weights_path=LANDMARKS_WEIGHTS,
        conf_threshold=LANDMARKS_CONF,
        log_every=30,
    )

    # Visualization (dashboard + score banner + shot flashes)
    final_output = str(Path(OUTPUT_PATH).parent / "final_output1.mp4")
    render_video(
        video_path=landmarks_video_path,  # Use landmarks video as input
        traj_path=TRAJECTORIES_PATH,
        dist_csv=analytics_dir / "distance_report.csv",
        speed_csv=analytics_dir / "speed_report.csv",
        out_path=final_output,
        shots=shots,
        shot_events=shot_events,
    )

    elapsed = time.time() - t0
    print(f'\n✅ Done — {tracker.frame_count} frames in {elapsed:.1f}s '
          f'({tracker.frame_count / elapsed:.1f} FPS avg)')
    print(f'   Tracked video      → {OUTPUT_PATH}')
    print(f'   Possession video   → {possession_video_path}')
    print(f'   Landmarks video    → {landmarks_video_path}')
    print(f'   Final video        → {final_output}')


if __name__ == '__main__':
    main()
