"""
src/analytics/court_detection/overlay.py
─────────────────────────────────────────
Visualization utilities for court keypoints.

Kept separate from detector.py so that:
  • non-rendering callers (e.g. analytics scripts) don't pay for OpenCV
    imports they don't use,
  • the drawing style is consistent across the codebase (the team's
    visualizer.py + this module + main.py all draw the same way).
"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

# Type alias matching detector.py
Keypoint = Tuple[int, float, float, float]   # (idx, x, y, conf)


# ─────────────────────────────────────────────────────────────────────────────
def draw_keypoints(
    frame:        np.ndarray,
    keypoints:    Iterable[Keypoint],
    radius:       int = 8,
    fill_color:   tuple[int, int, int] = (0, 255, 0),       # green
    label_color:  tuple[int, int, int] = (0, 255, 255),     # yellow
    show_labels:  bool = True,
) -> np.ndarray:
    """
    Draw court keypoints onto a frame, in place.

    Each keypoint is rendered as a green filled circle with a black
    outer ring (readable on light AND dark courts) and an optional
    yellow index label with a black outline (readable on any background).

    Parameters
    ----------
    frame        : BGR image (modified in place — also returned for chaining)
    keypoints    : iterable of (idx, x, y, conf) tuples from
                   CourtKeypointDetector.detect()
    radius       : circle radius in pixels
    fill_color   : BGR fill color of the dot
    label_color  : BGR color of the index label
    show_labels  : if False, only the dots are drawn (no numbers)

    Returns
    -------
    The same `frame` array, mutated.
    """
    for idx, x, y, _conf in keypoints:
        center = (int(x), int(y))

        # Filled circle + black outline ring → visible on any court color
        cv2.circle(frame, center, radius,     fill_color, -1)
        cv2.circle(frame, center, radius + 2, (0, 0, 0),   2)

        if show_labels:
            text = str(idx)
            pos  = (center[0] + radius + 4, center[1] - 6)
            # Black "halo" first
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            # Yellow text on top
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, label_color, 2, cv2.LINE_AA)

    return frame
