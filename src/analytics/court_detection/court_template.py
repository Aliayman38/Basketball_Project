"""
src/analytics/court_detection/court_template.py
─────────────────────────────────────────────────
Maps keypoint indices (from the trained YOLO26-pose model) to real-world
court coordinates in meters.

FIBA court dimensions (used for international and Jordan league play):
    Length:           28.0 m  (along X axis, basket-to-basket)
    Width:            15.0 m  (along Y axis)
    3-point line:     6.75 m  from center of basket
    Free-throw line:  4.6 m   from baseline (center of basket is 1.575 m
                              from baseline, so FT line is 4.6 - 1.575 =
                              3.025 m from basket center)
    Free-throw circle radius: 1.8 m
    Restricted area radius:   1.25 m
    Paint (key) width:        4.9 m
    Center circle radius:     1.8 m

Coordinate convention used here:
    Origin (0, 0)        = top-left corner of the court
    X axis               = along the long side, increases going right
    Y axis               = along the short side, increases going down
    So the court occupies (0, 0) to (28, 15) in meters.

──────────────────────────────────────────────────────────────────────
                          NOTE ABOUT THIS MAPPING
──────────────────────────────────────────────────────────────────────
The Roboflow `basketball-court-detection-2` schema does not publish a
formal index-to-landmark legend. This file contains a best-effort
mapping built by inspecting the test outputs from your trained model
(NBA Lakers vs Thunder broadcast).

If the homography output looks wrong (players projected to crazy
positions), the most likely cause is one or two indices in this dict
mapped to the wrong landmark. To verify or fix:

  1. Run scripts/visualize_court_schema.py on a clear broadcast frame
  2. Look at where each numbered keypoint lands on the actual court
  3. Update the (x_m, y_m) values below to match real court positions

Indices NOT present in this dict are assumed to be unused (or noisy)
and will be filtered out before homography computation.
"""

# Court dimensions in meters (FIBA standard)
COURT_LENGTH_M = 28.0
COURT_WIDTH_M  = 15.0

# Free-throw line distance from baseline
FT_LINE_FROM_BASELINE = 5.8        # FIBA: 5.80 m to far edge of FT line
PAINT_WIDTH_M         = 4.9        # FIBA paint is 4.9 m wide
PAINT_DEPTH_M         = 5.8        # paint extends 5.80 m from baseline

# Derived center positions for FIBA court
CY_CENTER     = COURT_WIDTH_M / 2          # 7.5 m  (center along Y)
PAINT_TOP_Y   = CY_CENTER - PAINT_WIDTH_M/2  # 5.05 m
PAINT_BOT_Y   = CY_CENTER + PAINT_WIDTH_M/2  # 9.95 m

# 3-point arc distance from baseline (where the arc begins, before curve)
THREE_PT_FROM_BASELINE = 0.9       # the straight portion before arc curve
THREE_PT_FROM_BASKET   = 6.75      # arc radius from basket center


# ─────────────────────────────────────────────────────────────────────────────
#  Keypoint index → (x_meters, y_meters) on FIBA court
# ─────────────────────────────────────────────────────────────────────────────
# Layout convention:
#   - Indices 0-15 are LEFT half of court (X = 0 to 14)
#   - Indices 16-31 are RIGHT half of court (X = 14 to 28)
#   - Within each half: top-to-bottom, then sideline-to-baseline
#
# This mapping is the educated default. It will likely be ~80-90% correct
# out of the box; some indices may need tweaking.

KP_TO_WORLD = {
    # ── LEFT HALF (left baseline + left paint + left 3pt) ───────────────
     0: (0.0,                  0.0),                  # top-left corner
     1: (0.0,                  CY_CENTER),            # left sideline midpoint
     2: (0.0,                  COURT_WIDTH_M),        # bottom-left corner
     3: (THREE_PT_FROM_BASELINE,  0.0),               # 3pt arc start (top)
     4: (THREE_PT_FROM_BASELINE,  COURT_WIDTH_M),     # 3pt arc start (bottom)
     5: (PAINT_DEPTH_M,        PAINT_TOP_Y),          # left FT-line top corner
     6: (PAINT_DEPTH_M,        CY_CENTER),            # left FT-line center
     7: (PAINT_DEPTH_M,        PAINT_BOT_Y),          # left FT-line bottom corner
     8: (1.575,                PAINT_TOP_Y),          # left paint baseline corner (top)
     9: (1.575,                PAINT_BOT_Y),          # left paint baseline corner (bottom)
    10: (THREE_PT_FROM_BASKET, CY_CENTER - 0.5),      # 3pt arc apex (left, slightly above center)
    11: (THREE_PT_FROM_BASKET, CY_CENTER + 0.5),      # 3pt arc apex (left, slightly below center)
    12: (PAINT_DEPTH_M + 1.8,  CY_CENTER),            # top of left FT circle
    13: (PAINT_DEPTH_M - 1.8,  CY_CENTER),            # bottom of left FT circle
    # 14, 15: reserved for additional left-side landmarks if used

    # ── CENTER ──────────────────────────────────────────────────────────
    14: (COURT_LENGTH_M/2,     0.0),                  # center line top
    15: (COURT_LENGTH_M/2,     COURT_WIDTH_M),        # center line bottom

    # ── RIGHT HALF (mirror of left) ─────────────────────────────────────
    16: (COURT_LENGTH_M,                       0.0),                 # top-right corner
    17: (COURT_LENGTH_M,                       CY_CENTER),           # right sideline mid
    18: (COURT_LENGTH_M,                       COURT_WIDTH_M),       # bottom-right corner
    19: (COURT_LENGTH_M - THREE_PT_FROM_BASELINE,  0.0),              # 3pt arc start (top)
    20: (COURT_LENGTH_M - THREE_PT_FROM_BASELINE,  COURT_WIDTH_M),    # 3pt arc start (bottom)
    21: (COURT_LENGTH_M - PAINT_DEPTH_M,       PAINT_TOP_Y),          # right FT-line top
    22: (COURT_LENGTH_M - PAINT_DEPTH_M,       CY_CENTER),            # right FT-line center
    23: (COURT_LENGTH_M - PAINT_DEPTH_M,       PAINT_BOT_Y),          # right FT-line bottom
    24: (COURT_LENGTH_M - 1.575,               PAINT_TOP_Y),          # right paint baseline (top)
    25: (COURT_LENGTH_M - 1.575,               PAINT_BOT_Y),          # right paint baseline (bottom)
    26: (COURT_LENGTH_M - THREE_PT_FROM_BASKET, CY_CENTER - 0.5),     # 3pt apex (right, top)
    27: (COURT_LENGTH_M - THREE_PT_FROM_BASKET, CY_CENTER + 0.5),     # 3pt apex (right, bottom)
    28: (COURT_LENGTH_M - PAINT_DEPTH_M - 1.8, CY_CENTER),            # top of right FT circle
    29: (COURT_LENGTH_M - PAINT_DEPTH_M + 1.8, CY_CENTER),            # bottom of right FT circle
    30: (COURT_LENGTH_M/2,     CY_CENTER - 1.8),     # center circle top
    31: (COURT_LENGTH_M/2,     CY_CENTER + 1.8),     # center circle bottom
}


def get_world_point(kp_index: int) -> tuple[float, float] | None:
    """Look up the real-world (x, y) in meters for a keypoint index.
    Returns None if the index is not in the mapping."""
    return KP_TO_WORLD.get(kp_index)


def num_known_keypoints() -> int:
    """Count of keypoint indices we have world coordinates for."""
    return len(KP_TO_WORLD)
