"""
src/analytics/speed.py
Compute per-player speed statistics from trajectory data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trajectories(source: Union[str, Path, dict]) -> dict[str, list[list[float]]]:
    """Load trajectories from a dict or a JSON file path.

    Expected format: {"players": {"<id>": [[x, y, frame], ...], ...}}
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {path}")
        with path.open() as f:
            data = json.load(f)

    if "players" not in data:
        raise ValueError("JSON must contain a top-level 'players' key.")
    return data["players"]


def export_csv(records: list[dict], output: Union[str, Path]) -> None:
    """Write speed report records to a CSV file."""
    if not records:
        print("No records to export.", file=sys.stderr)
        return

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Speed report saved → {output}")


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_speed(
    points: list[list[float]],
    fps: float,
    meters_per_pixel: Optional[float] = None,
) -> Optional[dict]:
    """Compute speed statistics for a single player's trajectory.

    Args:
        points: List of [x, y, frame_idx] entries (unsorted allowed).
        fps: Frames per second of the source video.
        meters_per_pixel: Optional scale factor for real-world units.

    Returns:
        Dict with speed stats, or None if fewer than 2 valid points.
    """
    # Validate and sort by frame index
    valid: list[tuple[float, float, float]] = []
    for p in points:
        try:
            x, y, frame = float(p[0]), float(p[1]), float(p[2])
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(frame):
                valid.append((x, y, frame))
        except (TypeError, IndexError, ValueError):
            continue

    if len(valid) < 2:
        return None

    valid.sort(key=lambda p: p[2])

    speeds_px_s: list[float] = []

    for i in range(1, len(valid)):
        x0, y0, f0 = valid[i - 1]
        x1, y1, f1 = valid[i]

        frame_diff = f1 - f0
        if frame_diff <= 0:
            continue  # skip duplicate or out-of-order frames

        dist_px = math.hypot(x1 - x0, y1 - y0)
        time_s = frame_diff / fps
        speeds_px_s.append(dist_px / time_s)

    if not speeds_px_s:
        return None

    result: dict = {
        "avg_speed_px_s": round(sum(speeds_px_s) / len(speeds_px_s), 4),
        "max_speed_px_s": round(max(speeds_px_s), 4),
    }

    if meters_per_pixel is not None:
        result["avg_speed_m_s"] = round(result["avg_speed_px_s"] * meters_per_pixel, 4)
        result["max_speed_m_s"] = round(result["max_speed_px_s"] * meters_per_pixel, 4)

    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_speed_report(
    trajectories: dict[str, list],
    fps: float = 30.0,
    meters_per_pixel: Optional[float] = None,
) -> list[dict]:
    """Build a speed report for all players.

    Args:
        trajectories: Mapping of player_id → list of [x, y, frame] points.
        fps: Frames per second.
        meters_per_pixel: Optional real-world scale factor.

    Returns:
        List of per-player speed stat dicts, sorted by player_id.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if meters_per_pixel is not None and meters_per_pixel <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel}")

    report: list[dict] = []

    for player_id, points in sorted(trajectories.items(), key=lambda kv: kv[0]):
        stats = compute_speed(points, fps, meters_per_pixel)
        if stats is None:
            continue
        report.append({"player_id": player_id, **stats})

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute player speed from trajectories.")
    parser.add_argument("--input", required=True, help="Path to trajectories.json")
    parser.add_argument("--output", required=True, help="Path for output CSV")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30)")
    parser.add_argument("--meters-per-pixel", type=float, default=None,
                        help="Scale factor for real-world speed (e.g. 0.0264)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        trajectories = load_trajectories(args.input)
        report = build_speed_report(trajectories, fps=args.fps,
                                    meters_per_pixel=args.meters_per_pixel)
        export_csv(report, args.output)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()