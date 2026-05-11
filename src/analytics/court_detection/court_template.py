"""
src/analytics/court_detection/court_template.py
─────────────────────────────────────────────────
Maps keypoint indices (from the trained YOLO-pose model) to pixel
coordinates on a TOP-DOWN COURT CANVAS.

Approach inspired by abdullahtarek/basketball_analysis:
  - Destination space is canvas PIXELS (not meters)
  - Fixed 800×400 canvas representing the full court
  - Each keypoint index → exact pixel on the canvas
  - The mapping describes the LEFT HALF-COURT only (which is what
    the model produces in broadcast frames). Right-half indices
    mirror the same geometry.

This is far simpler and less error-prone than meter coordinates,
because we directly control where things should appear on the
top-down rendering.

OBSERVED MODEL SCHEMA (from inspected Lakers vs Thunder frame):
   KP 0   top of left paint, near 3pt arc top intersection
   KP 1   adjacent to 0, top paint area
   KP 2   top-left paint corner near baseline
   KP 3   left sideline midpoint (near-camera)
   KP 4   top of free-throw circle
   KP 5   top FT-line × paint corner
   KP 6   center of free-throw circle (FT line midpoint)
   KP 7   top FT line edge (near 5)
   KP 8   bottom-left CORNER of court (sideline × baseline)
   KP 9   bottom of free-throw circle
   KP 10  bottom FT-line × paint corner
   KP 11  bottom-left paint corner near baseline
   KP 12  top of half-court line (center line × top sideline)
   KP 13  bottom of half-court line / mid-court area
"""

# ─────────────────────────────────────────────────────────────────────────────
# Top-down canvas dimensions (in pixels)
# ─────────────────────────────────────────────────────────────────────────────
CANVAS_W = 800   # canvas width  (long side of court, 28m in real life)
CANVAS_H = 400   # canvas height (short side, 15m in real life)

# Court geometry on the canvas (computed from real FIBA proportions)
# FIBA: court is 28m × 15m, paint is 5.8m deep × 4.9m wide,
#       FT circle radius 1.8m, basket center 1.575m from baseline
COURT_LENGTH_M = 28.0   # for converting back to real meters if needed
COURT_WIDTH_M  = 15.0

# Pixel scale on the canvas
PX_PER_M_X = CANVAS_W / COURT_LENGTH_M    # ≈ 28.57 px/m
PX_PER_M_Y = CANVAS_H / COURT_WIDTH_M     # ≈ 26.67 px/m

# Centre of court
CY = CANVAS_H // 2                        # 200

# Paint dimensions in pixels (left paint)
PAINT_DEPTH_PX = int(5.8  * PX_PER_M_X)   # 165
PAINT_HALF_PX  = int(2.45 * PX_PER_M_Y)   # 65   (4.9/2 m × px/m)
PAINT_TOP_Y    = CY - PAINT_HALF_PX       # 135
PAINT_BOT_Y    = CY + PAINT_HALF_PX       # 265

# Basket position
HOOP_X     = int(1.575 * PX_PER_M_X)      # 45
FT_CIRC_PX = int(1.8   * (PX_PER_M_X + PX_PER_M_Y) / 2)  # 50 (approx radius in px)

# Half-court x-coordinate
HALF_X = CANVAS_W // 2                    # 400


# ─────────────────────────────────────────────────────────────────────────────
# KP_TO_WORLD — keypoint index → (x_canvas_px, y_canvas_px)
# ─────────────────────────────────────────────────────────────────────────────
# Built from the observed schema. Confidently-mapped keypoints only.
# Indices NOT in this dict are filtered out before homography.

KP_TO_WORLD = {
    # ── Left baseline corners ──────────────────────────────────────────
    # KP 8 — bottom-left corner of court (sideline × baseline, lower)
     8: (0,                  CANVAS_H),

    # ── Left paint baseline corners ────────────────────────────────────
    # KP 2 — top-left paint corner near baseline
     2: (HOOP_X,              PAINT_TOP_Y),
    # KP 11 — bottom-left paint corner near baseline
    11: (HOOP_X,              PAINT_BOT_Y),

    # ── Left paint FT-line corners ─────────────────────────────────────
    # KP 5 — top FT-line × paint corner
     5: (PAINT_DEPTH_PX,      PAINT_TOP_Y),
    # KP 10 — bottom FT-line × paint corner
    10: (PAINT_DEPTH_PX,      PAINT_BOT_Y),

    # ── Free-throw line / circle ───────────────────────────────────────
    # KP 6 — center of FT line (FT circle center)
     6: (PAINT_DEPTH_PX,      CY),
    # KP 4 — top of FT circle (away from baseline)
     4: (PAINT_DEPTH_PX + FT_CIRC_PX, CY),
    # KP 9 — bottom of FT circle (toward baseline)
     9: (PAINT_DEPTH_PX - FT_CIRC_PX, CY),

    # ── Half-court line ────────────────────────────────────────────────
    # KP 12 — center line × top sideline
    12: (HALF_X,              0),
    # KP 13 — center line × bottom sideline
    13: (HALF_X,              CANVAS_H),
}


def get_world_point(kp_index: int):
    """Look up the canvas pixel (x, y) for a keypoint index."""
    return KP_TO_WORLD.get(kp_index)


def num_known_keypoints() -> int:
    """Number of keypoint indices in the mapping."""
    return len(KP_TO_WORLD)
