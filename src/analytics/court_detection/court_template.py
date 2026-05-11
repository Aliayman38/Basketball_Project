"""
src/analytics/court_detection/court_template.py
─────────────────────────────────────────────────
Maps keypoint indices (from the trained YOLO26-pose model) to real-world
court coordinates in meters.

DESIGN CHOICE: Conservative mapping
──────────────────────────────────────────────────────────────────────
This file ONLY includes keypoint indices whose court-feature meaning is
unambiguous from inspecting the trained model's output. Ambiguous indices
(hash marks, mid-paint markers, FT-circle quadrants) are deliberately
LEFT OUT — they will be filtered before homography solving, so they
don't contribute wrong world coordinates.

This is intentional: a homography needs only 4 correct correspondences.
Including 14 mappings with 5 wrong ones gives WORSE results than
including 7 confidently-correct ones, because RANSAC has to discard the
wrong ones anyway and the more wrong points there are, the higher the
chance RANSAC chooses a bad consensus.

The certain-ones mapping (below) gives clean, predictable results.

Coordinate convention (FIBA 28m × 15m):
    Origin (0, 0)        = top-left corner of the court (TV-side)
    X axis               = court LENGTH, increases left-to-right (0 → 28 m)
    Y axis               = court WIDTH, increases top-to-bottom (0 → 15 m)
    Left basket center   ≈ (1.575, 7.5)
    Right basket center  ≈ (26.425, 7.5)
"""

# ── Court dimensions in meters (FIBA standard) ──────────────────────────────
COURT_LENGTH_M  = 28.0
COURT_WIDTH_M   = 15.0
PAINT_WIDTH_M   = 4.9
PAINT_DEPTH_M   = 5.8
FT_CIRCLE_R     = 1.8
HOOP_FROM_BASE  = 1.575

CY_CENTER       = COURT_WIDTH_M / 2          # 7.5
PAINT_TOP_Y     = CY_CENTER - PAINT_WIDTH_M/2  # 5.05
PAINT_BOT_Y     = CY_CENTER + PAINT_WIDTH_M/2  # 9.95


# ─────────────────────────────────────────────────────────────────────────────
#  Keypoint index → (x_meters, y_meters)
# ─────────────────────────────────────────────────────────────────────────────
# Only the keypoints whose court-feature meaning is CERTAIN from the
# observed frame. Confidently-mapped keypoints are far more valuable
# than guessed ones.
#
# All entries here describe the LEFT half-court (X = 0 to 14 meters)
# because that's what the model produced in the inspected frame.
# Right-half indices (16-31) follow the same schema mirrored — added
# below where the meaning is unambiguous.

KP_TO_WORLD = {
    # ── LEFT HALF-COURT, certain mappings ──────────────────────────────
    # KP 2:  top-left paint corner near baseline
    #        (corner where paint meets baseline, top side)
     2: (HOOP_FROM_BASE,       PAINT_TOP_Y),

    # KP 5:  top FT-line × paint corner
    #        (corner at the top of the paint where FT line meets paint edge)
     5: (PAINT_DEPTH_M,        PAINT_TOP_Y),

    # KP 6:  middle of free-throw line (or paint midline)
    #        Located at the FT-line midpoint horizontally
     6: (PAINT_DEPTH_M,        CY_CENTER),

    # KP 8:  bottom-left CORNER of court
    #        (sideline meets baseline, lower / camera-near side)
     8: (0.0,                  COURT_WIDTH_M),

    # KP 10: bottom FT-line × paint corner (mirror of KP 5)
    10: (PAINT_DEPTH_M,        PAINT_BOT_Y),

    # KP 11: bottom-left paint corner near baseline (mirror of KP 2)
    11: (HOOP_FROM_BASE,       PAINT_BOT_Y),

    # KP 12: top of half-court line (center line × top sideline)
    12: (COURT_LENGTH_M / 2,   0.0),

    # KP 13: bottom of half-court line (center line × bottom sideline)
    13: (COURT_LENGTH_M / 2,   COURT_WIDTH_M),

    # ── RIGHT HALF-COURT (mirrored — for frames showing right basket) ─────
    # If the model produces right-half keypoints, they follow the same
    # geometric meaning mirrored across center. Same indices, mirrored
    # X coordinates (X' = COURT_LENGTH_M - X).
    #
    # We don't add separate right-half indices here because the model
    # uses ONE schema regardless of which end is shown — the model
    # detects whichever features are visible. If you see right-side
    # frames being plotted incorrectly, this is the place to extend.
}


def get_world_point(kp_index: int):
    """Look up real-world (x, y) in meters for a keypoint index."""
    return KP_TO_WORLD.get(kp_index)


def num_known_keypoints() -> int:
    """Count of keypoint indices we have world coordinates for."""
    return len(KP_TO_WORLD)
