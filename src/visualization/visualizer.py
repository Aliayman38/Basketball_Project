"""
overlay.py
Overlay tracked player statistics (distance & speed) onto a video.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict[str, dict[str, str]]:
    """Parse a CSV into {player_id: {column: value}} dict."""
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("player_id", "").strip()
            if pid:
                result[pid] = {k.strip(): v.strip() for k, v in row.items()}
    return result


def load_frame_index(
    path: str,
    gap_threshold: int = 30,
) -> dict[int, dict[str, tuple[int, int]]]:
    """Build frame-indexed lookup with linear interpolation across tracking gaps.

    For each player, consecutive detections separated by ≤ gap_threshold frames
    are filled with linearly interpolated (x, y) so the stat overlay stays
    attached during brief occlusions or pose-induced tracker drops.

    Args:
        path:          Path to trajectories.json.
        gap_threshold: Maximum gap (in frames) to interpolate across.
                       Gaps larger than this are left empty (player truly lost).
    """
    with open(path) as f:
        players: dict[str, list] = json.load(f)["players"]

    frames: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)

    for pid, points in players.items():
        # Collect and sort valid detections
        detections: list[tuple[int, int, int]] = []
        for point in points:
            try:

                frame = int(point["frame"])

                x, y = point["center"]
                x = int(x)
                y = int(y)

                detections.append((frame, x, y))
            except (TypeError, IndexError, ValueError):
                continue

        if not detections:
            continue

        detections.sort()

        # Write real detections
        for frame, x, y in detections:
            frames[frame][pid] = (x, y)

        # Fill gaps between consecutive detections via linear interpolation
        for (f0, x0, y0), (f1, x1, y1) in zip(detections, detections[1:]):
            gap = f1 - f0
            if gap <= 1 or gap > gap_threshold:
                continue  # no gap, or gap too large to interpolate safely
            for step in range(1, gap):
                t = step / gap
                xi = round(x0 + t * (x1 - x0))
                yi = round(y0 + t * (y1 - y0))
                frames[f0 + step].setdefault(pid, (xi, yi))  # don't overwrite real detections

    return frames


# Explicit map: column name → display unit string.
# Avoids brittle split("_")[-1] parsing (e.g. "avg_speed_m_s" → "s", not "m/s").
_UNIT_MAP: dict[str, str] = {
    "total_distance_m":  "m",
    "total_distance_px": "px",
    "avg_speed_m_s":     "m/s",
    "avg_speed_px_s":    "px/s",
}


def pick(row: dict[str, str], metric_key: str, pixel_key: str) -> tuple[str, str]:
    """Return (value, display_unit) preferring metric, falling back to pixels."""
    if metric_key in row and row[metric_key]:
        return row[metric_key], _UNIT_MAP.get(metric_key, "m")
    if pixel_key in row and row[pixel_key]:
        return row[pixel_key], _UNIT_MAP.get(pixel_key, "px")
    return "0", "px"


def draw_label(frame, x: int, y: int, text: str) -> None:
    """Draw clear stats label above player."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6

    # move label higher above the player center
    x1 = max(0, x - tw // 2 - pad)
    y1 = max(0, y - 70 - th - pad * 2)
    x2 = min(frame.shape[1] - 1, x + tw // 2 + pad)
    y2 = min(frame.shape[0] - 1, y - 70)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        frame,
        text,
        (x1 + pad, y2 - pad),
        font,
        scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_video(args: argparse.Namespace) -> None:
    frames_data = load_frame_index(args.trajectories)
    dist_data   = load_csv(args.distance_csv)
    speed_data  = load_csv(args.speed_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for pid, (x, y) in frames_data.get(frame_idx, {}).items():
            print("FRAME:", frame_idx, "PID:", pid, "POS:", x, y)
            d_row = dist_data.get(pid, {})
            s_row = speed_data.get(pid, {})

            dist,  d_unit = pick(d_row, "total_distance_m",  "total_distance_px")
            speed, s_unit = pick(s_row, "avg_speed_m_s",     "avg_speed_px_s")
            print("DIST:", dist, d_unit, "SPEED:", speed, s_unit)

            text = f"D: {float(dist):.1f}{d_unit} | S: {float(speed):.2f}{s_unit}"
            label_x = max(5, min(x - 120, frame.shape[1] - 350))
            label_y = max(30, y - 60)

            cv2.putText(
                frame,
                text,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done — {frame_idx} frames written → {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Overlay player stats onto video.")
    p.add_argument("--video",        default="Basketball_Project/runs/bot-sort tracking/tracking_botsort.mp4",                       help="Input video file")
    p.add_argument("--trajectories", default="Basketball_Project/runs/bot-sort tracking/analytics/trajectories.json", help="Trajectories JSON")
    p.add_argument("--distance-csv", default="runs/detect/distance_report.csv", help="Distance CSV")
    p.add_argument("--speed-csv",    default="runs/detect/speed_report.csv",  help="Speed CSV")
    p.add_argument("--output",       default="final_output.mp4",              help="Output video path")
    process_video(p.parse_args())


if __name__ == "__main__":
    main()