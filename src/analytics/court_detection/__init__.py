"""
src/analytics/court_detection
──────────────────────────────
Roboflow-based basketball court keypoint detection.

This subpackage is Rashed's contribution to the project. It is fully
optional: the rest of the pipeline (detection, tracking, clustering,
analytics) does not depend on it. If Roboflow is unavailable or no
API key is set, the rest of the project keeps working unchanged.

Public API
──────────
    from src.analytics.court_detection import (
        CourtKeypointDetector,
        draw_keypoints,
        load_keypoints_from_inference_result,
    )

Typical use (offline post-processing on a rendered video):
    See scripts/add_landmarks_overlay.py

Typical use (per-frame inside main.py — when ready):
    detector = CourtKeypointDetector(api_key=os.environ["ROBOFLOW_API_KEY"])
    keypoints = detector.detect(frame)        # list[(idx, x, y, conf)]
    draw_keypoints(frame, keypoints)          # in-place
"""
from .detector import CourtKeypointDetector, load_keypoints_from_inference_result
from .overlay  import draw_keypoints

__all__ = [
    "CourtKeypointDetector",
    "load_keypoints_from_inference_result",
    "draw_keypoints",
]
