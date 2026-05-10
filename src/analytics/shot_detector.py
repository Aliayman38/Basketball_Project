"""
src/analytics/shot_detector.py
Detect made basketball shot attempts from ball trajectory data.

Approach: hoop-region crossing — no full arc segmentation required.
  1. Interpolate small ball-tracking gaps (1-5 frames).
  2. Define a hoop ROI around the rim.
  3. Detect a made shot when the ball crosses the rim y-level downward,
     lands within the rim x-range, and later exits below the net.
  4. Cooldown prevents duplicate detections.

Input JSON:
  {"hoop": {"x":940,"y":360,"radius":45},
   "ball": [{"frame":51,"center":[365,183]}, ...],
   "net":  {"1":[{"frame":64,"bbox":[...],"center":[865,181]}]}}

"hoop" is derived from "net" if absent.
"""

from __future__ import annotations

import json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = ["detect_shots", "load_trajectory", "ShotResult", "Point", "Hoop"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Point:
    frame: int
    x: float
    y: float
    interpolated: bool = False


@dataclass
class Hoop:
    x: float       # rim centre x (px)
    y: float       # rim centre y (px)
    radius: float  # inner rim radius (px)


@dataclass
class ShotResult:
    arc_start_frame: int
    arc_end_frame: int
    apex_frame: int
    entry_x: float
    entry_y: float
    confidence: float


# ── I/O ───────────────────────────────────────────────────────────────────────

def _hoop_from_net(net: dict) -> Optional[Hoop]:
    """Average all net-centre detections to estimate hoop position."""
    xs, ys, radii = [], [], []
    for dets in net.values():
        for d in dets:
            xs.append(float(d["center"][0])); ys.append(float(d["center"][1]))
            if "bbox" in d and len(d["bbox"]) == 4:
                radii.append((float(d["bbox"][2]) - float(d["bbox"][0])) / 2.0)
    if not xs:
        return None
    return Hoop(x=sum(xs)/len(xs), y=sum(ys)/len(ys),
                radius=sum(radii)/len(radii) if radii else 45.0)


def load_trajectory(source: str | Path | dict) -> tuple[list[Point], Hoop]:
    """Load ball trajectory + hoop from a JSON file or dict."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text())

    if "hoop" in data:
        h = data["hoop"]
        hoop = Hoop(x=float(h["x"]), y=float(h["y"]), radius=float(h["radius"]))
    elif "net" in data:
        hoop = _hoop_from_net(data["net"])
        if hoop is None:
            raise ValueError("No usable hoop or net detections found.")
    else:
        raise ValueError("JSON must contain a 'hoop' or 'net' key.")

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


# ── Interpolation ─────────────────────────────────────────────────────────────

def interpolate_gaps(points: list[Point], max_gap: int = 5) -> list[Point]:
    """Fill small frame-gaps (<=max_gap) with linear interpolation."""
    if len(points) < 2:
        return list(points)
    filled: list[Point] = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        filled.append(p1)
        gap = p2.frame - p1.frame - 1
        if 0 < gap <= max_gap:
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                filled.append(Point(
                    frame=p1.frame + step,
                    x=round(p1.x + t * (p2.x - p1.x), 2),
                    y=round(p1.y + t * (p2.y - p1.y), 2),
                    interpolated=True,
                ))
    filled.append(points[-1])
    return filled


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_shots(
    points: list[Point],
    hoop: Hoop,
    min_descent_speed: float = 3.0,
    min_frames_between_shots: int = 45,
    max_gap: int = 5,
) -> list[ShotResult]:
    """
    Detect made shots using hoop-ROI crossing logic.

    ROI:  x in [hoop.x ± 1.5*radius],  y in [hoop.y-60 .. hoop.y+140]
    Made shot conditions (all must hold):
      1. Ball enters ROI from above (y < hoop.y) while moving downward.
      2. Ball x is within 1.5*radius of hoop.x at rim crossing.
      3. Ball later appears below hoop.y + 140 (exits net region).
      4. Minimum cooldown between detections.
    """
    points = interpolate_gaps(points, max_gap)
    if not points:
        return []

    roi_x_tol = 1.5 * hoop.radius   # lateral tolerance
    roi_y_top = hoop.y - 60         # ROI top (ball must come from above)
    roi_y_bot = hoop.y + 140        # ROI / net bottom

    results: list[ShotResult] = []
    last_shot_frame = -999
    in_roi = False
    roi_entry_frame = -1
    apex_y, apex_frame = float("inf"), -1

    for i in range(1, len(points)):
        p_prev, p = points[i - 1], points[i]
        dy = p.y - p_prev.y   # positive = moving down

        # Track apex (highest point before potential rim crossing)
        if p.y < apex_y:
            apex_y, apex_frame = p.y, p.frame

        # Detect entry into ROI from above
        if not in_roi and roi_y_top <= p.y <= hoop.y and abs(p.x - hoop.x) <= roi_x_tol:
            in_roi = True
            roi_entry_frame = p.frame
            apex_y, apex_frame = p_prev.y, p_prev.frame  # reset apex to just before ROI

        if not in_roi:
            continue

        # Check rim-plane crossing: ball moves from above to below hoop.y
        if p_prev.y <= hoop.y <= p.y and dy >= min_descent_speed:
            # Interpolate exact x at rim level
            t = (hoop.y - p_prev.y) / (p.y - p_prev.y)
            entry_x = p_prev.x + t * (p.x - p_prev.x)

            if abs(entry_x - hoop.x) > roi_x_tol:
                in_roi = False; apex_y = float("inf")
                continue

            # Look ahead: confirm ball exits below net bottom
            exited = any(q.y >= roi_y_bot for q in points[i:i + 30])
            if not exited:
                in_roi = False; apex_y = float("inf")
                continue

            # Cooldown check
            if roi_entry_frame - last_shot_frame < min_frames_between_shots:
                in_roi = False; apex_y = float("inf")
                continue

            lateral_err = abs(entry_x - hoop.x)
            speed_score = min(dy / (min_descent_speed * 3), 1.0)
            lat_score   = max(0.0, 1.0 - lateral_err / hoop.radius)
            confidence  = round(0.6 * lat_score + 0.4 * speed_score, 3)

            # Find arc end: first frame after entry that exits ROI or data ends
            arc_end = points[min(i + 30, len(points) - 1)].frame

            results.append(ShotResult(
                arc_start_frame=roi_entry_frame,
                arc_end_frame=arc_end,
                apex_frame=apex_frame,
                entry_x=round(entry_x, 2),
                entry_y=round(hoop.y, 2),
                confidence=confidence,
            ))
            last_shot_frame = roi_entry_frame
            in_roi = False
            apex_y = float("inf")

        # Ball left ROI without a valid crossing — reset
        elif p.y > roi_y_bot or abs(p.x - hoop.x) > roi_x_tol * 2:
            in_roi = False
            apex_y = float("inf")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Detect made basketball shots from trajectory JSON.")
    parser.add_argument("--input",  required=True, help="Path to trajectory JSON")
    parser.add_argument("--output", default=None,  help="Optional path to save results JSON")
    parser.add_argument("--min-arc-len",       type=int,   default=8,   help="(unused, kept for CLI compat)")
    parser.add_argument("--apex-height",        type=float, default=40.0,help="(unused, kept for CLI compat)")
    parser.add_argument("--min-descent-speed",  type=float, default=3.0, help="Min px/frame descent at rim")
    parser.add_argument("--min-shot-spacing",   type=int,   default=45,  help="Min frames between shots")
    args = parser.parse_args()

    try:
        points, hoop = load_trajectory(args.input)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading trajectory: {e}", file=sys.stderr); sys.exit(1)

    shots = detect_shots(points, hoop,
                         min_descent_speed=args.min_descent_speed,
                         min_frames_between_shots=args.min_shot_spacing)

    print(f"\nDetected {len(shots)} made shot(s):\n")
    output_records = []
    for i, s in enumerate(shots, 1):
        rec = {"shot": i, "frames": f"{s.arc_start_frame}\u2013{s.arc_end_frame}",
               "apex_frame": s.apex_frame,
               "entry": {"x": s.entry_x, "y": s.entry_y}, "confidence": s.confidence}
        output_records.append(rec)
        print(f"  Shot {i}: frames {rec['frames']}  apex@{s.apex_frame}  "
              f"entry=({s.entry_x}, {s.entry_y})  confidence={s.confidence:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"made_shots": output_records}, indent=2))
        print(f"\nResults saved \u2192 {out_path}")


if __name__ == "__main__":
    main()