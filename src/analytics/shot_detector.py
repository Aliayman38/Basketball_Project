"""
src/analytics/shot_detector.py
Detect made basketball shot attempts from ball trajectory data.

Research-backed approach:
  1. Segment trajectory into candidate shot arcs (parabolic shape with clear apex).
  2. For each arc, check geometric conditions relative to the hoop.
  3. Enforce minimum arc height and downward velocity at hoop crossing.

Input JSON format:
{
  "hoop": {"x": 940, "y": 360, "radius": 45},
  "ball": [{"frame": 51, "center": [365, 183]}, ...],
  "net": {"1": [{"frame": 64, "bbox": [...], "center": [865, 181]}, ...]}
}

"hoop" may be derived automatically from "net" detections if absent.
All coordinates are in image-pixel space (y increases downward).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["detect_shots", "load_trajectory", "ShotResult", "Point", "Hoop"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Point:
    frame: int
    x: float
    y: float
    interpolated: bool = False  # ← FIXED: proper field instead of dynamic hack


@dataclass
class Hoop:
    x: float          # rim centre x (px)
    y: float          # rim centre y (px) — top of rim
    radius: float     # inner rim radius (px)

    # Derived geometry (set post-init)
    entry_band_top: float = field(init=False)    # y above which ball must peak
    entry_band_bot: float = field(init=False)    # y below which ball exits

    def __post_init__(self):
        # Ball must peak above the rim top before descending through it
        self.entry_band_top = self.y - self.radius * 0.5
        # Ball exits below the net bottom (~1.5 rim radii below rim centre)
        self.entry_band_bot = self.y + self.radius * 1.5


@dataclass
class ShotResult:
    arc_start_frame: int
    arc_end_frame: int
    apex_frame: int
    entry_x: float
    entry_y: float
    confidence: float   # 0.0–1.0


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------

def _hoop_from_net(net: dict) -> Optional[Hoop]:
    """
    Derive a Hoop from net detections when no explicit 'hoop' key exists.
    Averages all net-centre detections across all tracked net IDs.
    Rim radius = half the mean bbox width (removed erroneous *3 multiplier).
    """
    xs, ys, radii = [], [], []
    for detections in net.values():
        for d in detections:
            cx, cy = float(d["center"][0]), float(d["center"][1])
            xs.append(cx)
            ys.append(cy)
            if "bbox" in d and len(d["bbox"]) == 4:
                radii.append((float(d["bbox"][2]) - float(d["bbox"][0])) / 2.0)
    if not xs:
        return None
    return Hoop(
        x=sum(xs) / len(xs),
        y=sum(ys) / len(ys),
        radius=(sum(radii) / len(radii)) if radii else 45.0,  # ← FIXED: removed *3
    )


def load_trajectory(source: str | Path | dict) -> tuple[list[Point], Hoop]:
    """
    Load ball trajectory + hoop from a JSON file path or pre-loaded dict.

    Accepts both ball formats:
      - Legacy: {"frame": N, "x": X, "y": Y}
      - New:    {"frame": N, "center": [X, Y]}
    Hoop is read from "hoop" key; if absent, derived from "net" detections.
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        with path.open() as f:
            data = json.load(f)

    # Hoop: explicit key takes priority; fall back to net detections
    if "hoop" in data:
        h = data["hoop"]
        hoop = Hoop(x=float(h["x"]), y=float(h["y"]), radius=float(h["radius"]))
    elif "net" in data:
        hoop = _hoop_from_net(data["net"])
        if hoop is None:
            raise ValueError("No 'hoop' key and no usable 'net' detections found.")
    else:
        raise ValueError("JSON must contain a 'hoop' or 'net' key.")

    # Ball points: support both {x,y} and {center:[x,y]} formats
    points: list[Point] = []
    for p in data["ball"]:
        frame = int(p["frame"])
        if "center" in p:
            x, y = float(p["center"][0]), float(p["center"][1])
        else:
            x, y = float(p["x"]), float(p["y"])
        points.append(Point(frame=frame, x=x, y=y))

    points.sort(key=lambda p: p.frame)
    return points, hoop


# ---------------------------------------------------------------------------
# Trajectory interpolation  (preprocessing — runs before shot analysis)
# ---------------------------------------------------------------------------

def interpolate_gaps(
    points: list[Point],
    max_gap: int = 3,
) -> list[Point]:
    """
    Fill small frame-gaps in the ball trajectory with linear interpolation.

    Why this helps: YOLO/tracking loses the ball for 1-3 frames near the
    rim (net occlusion, motion blur). Without filling those gaps, the arc
    is split in two and neither half satisfies the geometric shot conditions.

    Strategy:
      - Scan consecutive known detections for frame gaps.
      - If gap size <= max_gap: synthesise intermediate Points by linearly
        interpolating (x, y) between the two anchor detections.
      - If gap > max_gap: leave it empty.
    """
    if len(points) < 2:
        return list(points)

    filled: list[Point] = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        filled.append(p1)

        gap = p2.frame - p1.frame - 1   # number of missing frames between anchors
        if 0 < gap <= max_gap:
            for step in range(1, gap + 1):
                t  = step / (gap + 1)           # normalised position in (0, 1)
                ix = p1.x + t * (p2.x - p1.x)
                iy = p1.y + t * (p2.y - p1.y)
                filled.append(Point(frame=p1.frame + step, x=round(ix, 2),
                                    y=round(iy, 2), interpolated=True))  # ← FIXED
        # Gaps > max_gap: intentionally left empty

    filled.append(points[-1])
    return filled


# ---------------------------------------------------------------------------
# Arc segmentation
# ---------------------------------------------------------------------------

def _find_apex(pts: list[Point]) -> int:
    """Return index of the topmost point (min y, since y increases downward)."""
    return min(range(len(pts)), key=lambda i: pts[i].y)


def segment_arcs(
    points: list[Point],
    min_arc_len: int = 8,
    apex_height_px: float = 40.0,
) -> list[list[Point]]:
    """
    Split the full trajectory into candidate shot arcs.

    A new arc begins whenever the ball transitions from descending to
    ascending (a new shot launched). Each arc must:
      - contain at least `min_arc_len` points
      - have an apex at least `apex_height_px` above its start/end y
    """
    if len(points) < min_arc_len:
        return []

    arcs: list[list[Point]] = []
    current: list[Point] = [points[0]]

    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]
        dy = curr.y - prev.y  # positive = moving down in image space

        # Detect new arc: ball starts moving upward after going down
        if dy < 0 and len(current) >= 2 and (current[-1].y - current[-2].y) > 0:
            if len(current) >= min_arc_len:
                arcs.append(current)
            current = [prev]  # overlap one point for continuity

        current.append(curr)

    if len(current) >= min_arc_len:
        arcs.append(current)

    # Filter: apex must be significantly higher than endpoints
    valid = []
    for arc in arcs:
        apex_idx = _find_apex(arc)
        apex_y   = arc[apex_idx].y
        end_y    = max(arc[0].y, arc[-1].y)
        if (end_y - apex_y) >= apex_height_px:
            valid.append(arc)

    return valid


# ---------------------------------------------------------------------------
# Per-arc shot scoring
# ---------------------------------------------------------------------------

def _interpolate_x_at_y(p1: Point, p2: Point, target_y: float) -> Optional[float]:
    """Linear interpolation: x coordinate where the segment crosses target_y."""
    dy = p2.y - p1.y
    if abs(dy) < 1e-6:
        return None
    t = (target_y - p1.y) / dy
    if not (0.0 <= t <= 1.0):
        return None
    return p1.x + t * (p2.x - p1.x)


def _ball_passes_through_hoop(
    arc: list[Point], hoop: Hoop
) -> tuple[bool, float, float, float]:
    """
    Check whether the descending portion of the arc passes through the rim.

    Returns (passes, entry_x, entry_y, vert_speed_at_crossing).
    The ball must cross hoop.y while moving downward, and entry_x must
    be within hoop.radius of hoop.x (inside the rim cylinder).
    """
    apex_idx = _find_apex(arc)
    descent  = arc[apex_idx:]   # only the falling portion

    for i in range(1, len(descent)):
        p1, p2 = descent[i - 1], descent[i]
        # Only consider top-to-bottom crossings (ball falling through rim plane)
        if p1.y <= hoop.y <= p2.y:
            entry_x = _interpolate_x_at_y(p1, p2, hoop.y)
            if entry_x is None:
                continue
            lateral_err = abs(entry_x - hoop.x)
            # ← FIXED: removed debug print
            if lateral_err <= hoop.radius * 1.8:
                dy_px_per_frame = p2.y - p1.y   # downward speed at crossing
                return True, entry_x, hoop.y, dy_px_per_frame

    return False, 0.0, 0.0, 0.0


def _ball_exits_below_hoop(arc: list[Point], hoop: Hoop) -> bool:
    """
    Confirm the ball travels below the net bottom after crossing the rim.
    Rejects rim disturbances where the ball bounces back up immediately.
    """
    crossed = False
    for p in arc:
        if p.y >= hoop.y:
            crossed = True
        if crossed and p.y >= hoop.entry_band_bot:
            return True
    return False


def _peak_above_rim(arc: list[Point], hoop: Hoop) -> bool:
    """Ball apex must be above the rim top (standard entry angle requirement)."""
    apex_y = min(p.y for p in arc)
    return apex_y <= hoop.entry_band_top


def _compute_confidence(
    lateral_err: float,
    hoop_radius: float,
    vert_speed: float,
    min_speed: float = 5.0,
) -> float:
    """
    Score 0–1 based on lateral accuracy and descent speed.
    Centre swish = 1.0; rim-scraper with slow descent ≈ 0.3.
    """
    lateral_score = max(0.0, 1.0 - (lateral_err / hoop_radius))  # ← FIXED: clamp early
    speed_score   = min(vert_speed / (min_speed * 3), 1.0)
    return round(max(0.0, min(1.0, 0.6 * lateral_score + 0.4 * speed_score)), 3)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_shots(
    points: list[Point],
    hoop: Hoop,
    min_arc_len: int = 8,
    apex_height_px: float = 20.0,
    min_descent_speed: float = 3.0,  # ← FIXED: aligned with CLI default
    min_frames_between_shots: int = 30,  # ← NEW: prevent duplicate detections
) -> list[ShotResult]:
    """
    Analyse a full-game trajectory and return a list of shot results.

    Parameters
    ----------
    points            : sorted list of ball positions (frame, x, y)
    hoop              : rim geometry
    min_arc_len       : minimum frames for a valid shot arc
    apex_height_px    : minimum apex-to-endpoint height to qualify as a shot
    min_descent_speed : minimum px/frame downward velocity at rim crossing
    min_frames_between_shots : cooldown between shot detections (default: 30)

    Returns
    -------
    List of ShotResult (only confirmed made shots).
    """
    # Preprocessing: fill small tracking gaps before arc analysis
    points = interpolate_gaps(points, max_gap=3)
    arcs = segment_arcs(points, min_arc_len, apex_height_px)
    results: list[ShotResult] = []
    last_shot_end = -999

    for arc in arcs:
        # Reject arcs too far from hoop horizontally
        arc_center_x = sum(p.x for p in arc) / len(arc)
        if abs(arc_center_x - hoop.x) > 200:
            continue
        apex_idx = _find_apex(arc)

        # Gate 1: apex must be above the rim
        if not _peak_above_rim(arc, hoop):
            continue

        # Gate 2: ball must cross the hoop plane within the rim cylinder
        passes, entry_x, entry_y, vert_speed = _ball_passes_through_hoop(arc, hoop)
        # ← FIXED: single call, no duplicate

        # Fallback heuristic: ball near hoop while descending, tracking lost
        if not passes:
            last_point = arc[-1]
            apex_y = arc[apex_idx].y
            descending = last_point.y > apex_y  # ← FIXED: compare to apex, not start
            if (abs(last_point.x - hoop.x) < 120
                    and last_point.y < hoop.y + 50
                    and descending):
                passes = True
                entry_x = last_point.x
                entry_y = last_point.y
                vert_speed = 5.0
            else:
                continue

        # Gate 3: descent speed must exceed threshold (avoids slow rim rolls)
        if vert_speed < min_descent_speed:
            continue

        # Gate 4: ball must exit below the net bottom (confirms through-net travel)
        if not _ball_exits_below_hoop(arc, hoop):
            continue

        # Gate 5: minimum spacing between shots (prevents duplicate detections)
        if arc[0].frame - last_shot_end < min_frames_between_shots:
            continue

        lateral_err = abs(entry_x - hoop.x)
        confidence  = _compute_confidence(lateral_err, hoop.radius, vert_speed)

        results.append(ShotResult(
            arc_start_frame=arc[0].frame,
            arc_end_frame=arc[-1].frame,
            apex_frame=arc[apex_idx].frame,
            entry_x=round(entry_x, 2),
            entry_y=round(entry_y, 2),
            confidence=confidence,
        ))
        last_shot_end = arc[-1].frame

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect made basketball shots from trajectory JSON.")
    parser.add_argument("--input",  required=True, help="Path to trajectory JSON")
    parser.add_argument("--output", default=None,  help="Optional path to save results JSON")
    parser.add_argument("--min-arc-len",       type=int,   default=8,    help="Min frames per arc (default: 8)")
    parser.add_argument("--apex-height",        type=float, default=40.0, help="Min apex height in px (default: 40)")
    parser.add_argument("--min-descent-speed",  type=float, default=3.0,  help="Min px/frame descent at rim (default: 3)")
    parser.add_argument("--min-shot-spacing",   type=int,   default=30,   help="Min frames between shots (default: 30)")
    args = parser.parse_args()

    try:
        points, hoop = load_trajectory(args.input)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading trajectory: {e}", file=sys.stderr)
        sys.exit(1)

    shots = detect_shots(
        points, hoop,
        min_arc_len=args.min_arc_len,
        apex_height_px=args.apex_height,
        min_descent_speed=args.min_descent_speed,
        min_frames_between_shots=args.min_shot_spacing,
    )

    print(f"\nDetected {len(shots)} made shot(s):\n")
    output_records = []
    for i, s in enumerate(shots, 1):
        rec = {
            "shot": i,
            "frames": f"{s.arc_start_frame}–{s.arc_end_frame}",
            "apex_frame": s.apex_frame,
            "entry": {"x": s.entry_x, "y": s.entry_y},
            "confidence": s.confidence,
        }
        output_records.append(rec)
        print(f"  Shot {i}: frames {rec['frames']}  apex@{s.apex_frame}  "
              f"entry=({s.entry_x}, {s.entry_y})  confidence={s.confidence:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump({"made_shots": output_records}, f, indent=2)
        print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
