"""
src/analytics/court_detection/court_template.py
─────────────────────────────────────────────────
Keypoint-to-court mapping for the Roboflow basketball-court-detection-2 dataset.

The mapping comes directly from the roboflow/sports library:
    from sports.basketball import CourtConfiguration, League
    config = CourtConfiguration(league=League.NBA)
    config.vertices  # → 33 (x, y) pairs in FEET

Coordinate system (NBA court, in FEET):
    X: 0 → 94  (left baseline → right baseline)
    Y: 0 → 50  (top sideline → bottom sideline)
    Origin (0, 0) = top-left corner of the court
"""
from __future__ import annotations

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT  = 50.0

KP_TO_WORLD = {
     0: (  0.00,   0.00),
     1: (  0.00,   2.99),
     2: (  0.00,  17.00),
     3: (  0.00,  33.01),
     4: (  0.00,  47.02),
     5: (  0.00,  50.00),
     6: (  5.25,  25.00),
     7: ( 13.92,   2.99),
     8: ( 13.92,  47.02),
     9: ( 19.00,  17.00),
    10: ( 19.00,  25.00),
    11: ( 19.00,  33.01),
    12: ( 27.40,   0.00),
    13: ( 29.01,  25.00),
    14: ( 27.40,  50.00),
    15: ( 46.99,   0.00),
    16: ( 46.99,  25.00),
    17: ( 46.99,  50.00),
    18: ( 66.61,   0.00),
    19: ( 65.00,  25.00),
    20: ( 66.61,  50.00),
    21: ( 75.00,  17.00),
    22: ( 75.00,  25.00),
    23: ( 75.00,  33.01),
    24: ( 80.09,   2.99),
    25: ( 80.09,  47.02),
    26: ( 88.75,  25.00),
    27: ( 94.00,   0.00),
    28: ( 94.00,   2.99),
    29: ( 94.00,  17.00),
    30: ( 94.00,  33.01),
    31: ( 94.00,  47.02),
    32: ( 94.00,  50.00),
}

VERTICES = [KP_TO_WORLD[i] for i in range(len(KP_TO_WORLD))]


def get_world_point(kp_index: int):
    return KP_TO_WORLD.get(kp_index)

def num_known_keypoints() -> int:
    return len(KP_TO_WORLD)
