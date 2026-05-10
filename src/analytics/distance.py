"""
src/analytics/distance.py
─────────────────────────
Compute total distance covered by each tracked player from trajectory data.

CLI:
    python src/analytics/distance.py \\
        --input  runs/detect/trajectories.json \\
        --output runs/detect/distance_report.csv \\
        [--meters-per-pixel 0.0264] \\
        [--include-referees]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


# ── types ─────────────────────────────────────────────────────────────────────

Point = tuple[float, float, int]                     # (x, y, frame_idx)
Row   = dict[str, Any]


# ── core ──────────────────────────────────────────────────────────────────────

def _sanitize_trajectory(raw: list[Any]) -> list[Point]:
    """Filter invalid entries and sort by frame index."""

    pts: list[Point] = []

    for p in raw:
        try:
            if isinstance(p, dict):
                x, y = p["center"]
                f = p["frame"]
            else:
                x, y, f = p

            x, y, f = float(x), float(y), float(f)

            if math.isfinite(x) and math.isfinite(y) and math.isfinite(f):
                pts.append((x, y, int(f)))

        except (TypeError, ValueError, IndexError, KeyError):
            continue

    pts.sort(key=lambda p: p[2])

    return pts


def _gaps(frames: list[int]) -> int:
    """Count non-consecutive frame jumps."""
    return sum(b - a > 1 for a, b in zip(frames, frames[1:]))


def compute_distance(
    player_id: str,
    raw: list[Any],
    meters_per_pixel: float | None = None,
) -> Row | None:
    """Return a stats dict for one player, or None if trajectory is empty."""
    if meters_per_pixel is not None and meters_per_pixel <= 0:
        raise ValueError(f"meters_per_pixel must be positive, got {meters_per_pixel!r}.")

    pts = _sanitize_trajectory(raw)
    if len(pts) < 5:
        return None

    dist_px = sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1, f1), (x2, y2, f2) in zip(pts, pts[1:])
        if f2 - f1 <= 15
    )
    frames = [p[2] for p in pts]

    row: Row = {
        "player_id":          player_id,
        "total_distance_px":  round(dist_px, 4),
        "frames_tracked":     len(pts),
        "first_frame":        frames[0],
        "last_frame":         frames[-1],
        "missing_frame_gaps": _gaps(frames),
    }
    if meters_per_pixel is not None:
        row["total_distance_m"] = round(dist_px * meters_per_pixel, 4)
    return row


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_trajectories(
    path: str | Path,
    include_referees: bool = False,
) -> dict[str, list[Any]]:
    """
    Load trajectories.json written by main.py.

    Returns a flat {player_id: [[x, y, frame], ...]} dict.
    Referee IDs are prefixed with "ref_" when included.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Any]] = {}
    for pid, pts in data.get("players", {}).items():
        out[pid] = pts
    if include_referees:
        for rid, pts in data.get("referees", {}).items():
            out[f"ref_{rid}"] = pts
    return out


def build_report(
    trajectories: dict[str, list[Any]],
    meters_per_pixel: float | None = None,
) -> list[Row]:
    """Compute distance stats for all players; sort by player id."""
    rows = [
        row for pid, pts in trajectories.items()
        if (row := compute_distance(pid, pts, meters_per_pixel)) is not None
    ]
    rows.sort(key=lambda r: str(r["player_id"]).lstrip("ref_").zfill(10))
    return rows


def export_csv(rows: list[Row], path: str | Path) -> None:
    """Write the report to a CSV file."""
    if not rows:
        print("[distance] Warning: empty report, nothing to write.", file=sys.stderr)
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    has_m = any("total_distance_m" in r for r in rows)
    fieldnames = [
        "player_id", "total_distance_px",
        *( ["total_distance_m"] if has_m else [] ),
        "frames_tracked", "first_frame", "last_frame", "missing_frame_gaps",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"[distance] Report saved → {out.resolve()}  ({len(rows)} players)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Player distance reporter.")
    p.add_argument("--input",  "-i", required=True,  metavar="PATH")
    p.add_argument("--output", "-o", required=True,  metavar="PATH")
    p.add_argument("--meters-per-pixel", "-m", type=float, default=None, metavar="SCALE")
    p.add_argument("--include-referees", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        trajectories = load_trajectories(args.input, args.include_referees)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        sys.exit(f"[distance] Error loading trajectories: {exc}")

    report = build_report(trajectories, args.meters_per_pixel)
    export_csv(report, args.output)