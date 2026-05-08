"""
src/analytics/court_detection/detector.py
──────────────────────────────────────────
Wrapper around the Roboflow `basketball-court-detection-2/13` keypoint
model (YOLOv11-pose). Outputs court landmark keypoints with confidence
scores from a single frame.

The model is hosted on Roboflow Universe:
    https://universe.roboflow.com/samet-mmrat/basketball-court-detection-2-axedc

Why a class and not just a function:
  - The model object is large; we want to load weights ONCE and call
    `detect()` per frame.
  - The Roboflow SDK's response shape varies across versions (Pydantic
    object vs. dict). The class encapsulates that detail so callers
    don't have to think about it.

This class uses the local `inference` backend, which downloads the
model weights once (cached afterwards) and runs offline. No internet
is required after the first call.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

# Type alias: one keypoint = (model_index, x_pixel, y_pixel, confidence)
Keypoint = Tuple[int, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
class CourtKeypointDetector:
    """
    Detects basketball court landmarks in a single frame.

    Parameters
    ----------
    model_id        : Roboflow model identifier. Default targets the
                      `basketball-court-detection-2/13` weights, which
                      were validated to work on broadcast footage.
    api_key         : Roboflow API key. Falls back to the
                      ROBOFLOW_API_KEY environment variable if None.
    conf_threshold  : Default minimum confidence for a keypoint to be
                      returned. Can be overridden per call.

    Examples
    --------
    >>> detector = CourtKeypointDetector(api_key="rf_...")
    >>> kps = detector.detect(frame)
    >>> for idx, x, y, c in kps:
    ...     print(f"keypoint {idx}: ({x:.0f}, {y:.0f}) conf={c:.2f}")
    """

    DEFAULT_MODEL_ID = "basketball-court-detection-2/13"
    DEFAULT_CONF     = 0.30

    def __init__(
        self,
        model_id:       str = DEFAULT_MODEL_ID,
        api_key:        Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF,
    ) -> None:
        self.model_id       = model_id
        self.conf_threshold = conf_threshold

        api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if api_key is None:
            raise ValueError(
                "ROBOFLOW_API_KEY not provided. Pass `api_key=...` or set the "
                "ROBOFLOW_API_KEY environment variable."
            )

        # Defer the import so the rest of the project doesn't fail to
        # import this file when `inference` isn't installed.
        try:
            from inference import get_model
        except ImportError as e:
            raise ImportError(
                "The `inference` package is required for court keypoint "
                "detection. Install with:\n"
                "    pip install inference inference-sdk\n"
            ) from e

        os.environ.setdefault("ROBOFLOW_API_KEY", api_key)
        self._model = get_model(model_id=model_id, api_key=api_key)

    # ── Inference ────────────────────────────────────────────────────────────
    def detect(
        self,
        frame:           np.ndarray,
        conf_threshold:  Optional[float] = None,
    ) -> List[Keypoint]:
        """
        Run keypoint detection on a single frame.

        Parameters
        ----------
        frame           : BGR image as np.ndarray (H, W, 3)
        conf_threshold  : Override the default per-call. None → use
                          self.conf_threshold.

        Returns
        -------
        list of (model_index, x, y, confidence) tuples for every
        keypoint above the threshold.

        Empty list if the model finds no court detection in the frame.
        """
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        result = self._model.infer(frame, confidence=thresh)

        # The inference SDK sometimes returns a list, sometimes a single
        # response. Normalise to a single response object.
        if isinstance(result, list):
            result = result[0] if result else None
        if result is None:
            return []

        return load_keypoints_from_inference_result(result, conf_threshold=thresh)

    # ── Identification ───────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"CourtKeypointDetector(model_id={self.model_id!r}, "
            f"conf_threshold={self.conf_threshold})"
        )


# ─────────────────────────────────────────────────────────────────────────────
def load_keypoints_from_inference_result(
    result,
    conf_threshold: float = 0.30,
) -> List[Keypoint]:
    """
    Extract keypoints from a Roboflow inference response.

    Handles both response shapes the SDK can return:
      - Pydantic response object (newer versions)
      - Plain dict (older versions, or when fed JSON)

    Parameters
    ----------
    result          : The object returned by `model.infer(frame)` (or one
                      element of it if the SDK returned a list).
    conf_threshold  : Minimum confidence to keep a keypoint.

    Returns
    -------
    list of (model_index, x, y, confidence) tuples.
    """
    # Normalise to a dict
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    else:
        data = result

    if not isinstance(data, dict):
        return []

    preds = data.get("predictions", [])
    if not preds:
        return []

    # Take the most confident "court" detection (typically only one)
    best = max(preds, key=lambda p: p.get("confidence", 0.0))
    raw_kps = best.get("keypoints", [])

    out: List[Keypoint] = []
    for i, kp in enumerate(raw_kps):
        x = float(kp.get("x", 0.0))
        y = float(kp.get("y", 0.0))
        c = float(kp.get("confidence", 0.0))
        # Some schemas use class_id; fall back to enumeration index
        idx = int(kp.get("class_id", i))

        # Drop low-confidence and zero-fill placeholder keypoints
        if c >= conf_threshold and not (x == 0.0 and y == 0.0):
            out.append((idx, x, y, c))

    return out
