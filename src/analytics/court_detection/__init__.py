"""
src/analytics/court_detection
──────────────────────────────
Court landmark detection + homography-based top-down projection.
"""
from .landmarks_overlay import run_landmarks

from .court_template import KP_TO_WORLD, VERTICES, COURT_LENGTH_FT, COURT_WIDTH_FT

from .homography import (
    build_transformer,
    project_point,
    project_points,
    HomographyTracker,
)

from .topdown_view import render_topdown_video

__all__ = [
    "run_landmarks",
    "KP_TO_WORLD",
    "VERTICES",
    "COURT_LENGTH_FT",
    "COURT_WIDTH_FT",
    "build_transformer",
    "project_point",
    "project_points",
    "HomographyTracker",
    "render_topdown_video",
]
