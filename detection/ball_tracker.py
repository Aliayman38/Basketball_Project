import numpy as np
import cv2
from collections import deque
from detection.detector import Detection


class BallTracker:
    """
    Lightweight tracker specifically for the basketball.

    Responsibilities
    ----------------
    - Select the most-likely ball detection each frame (highest conf,
      plausible size, closest to last known position).
    - Keep a rolling position history for trail drawing.
    - Simple linear interpolation to fill short gaps when the ball
      is momentarily occluded.
    """

    def __init__(
        self,
        trail_length: int = 25,
        max_gap_frames: int = 8,
        max_jump_px: int = 300,
        min_area: int = 50,
        max_area: int = 8000,
    ):
        self.trail_length  = trail_length
        self.max_gap       = max_gap_frames
        self.max_jump      = max_jump_px
        self.min_area      = min_area
        self.max_area      = max_area

        # History: deque of (cx, cy) or None
        self._history: deque = deque(maxlen=trail_length)
        self._last_center: tuple | None = None
        self._missing_count: int = 0

        # Interpolated positions inserted during gaps
        self._interpolated: set[int] = set()

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, ball_detections: list[Detection]) -> Detection | None:
        """
        Feed all 'basketball' detections for this frame.
        Returns the chosen Detection or None if ball not found.
        """
        best = self._select_best(ball_detections)

        if best is not None:
            self._missing_count = 0
            self._last_center   = best.center
            self._history.append(best.center)
        else:
            self._missing_count += 1
            self._history.append(None)   # placeholder

        return best

    def get_trail(self) -> list[tuple[int, int]]:
        """Return list of (cx, cy) positions (Nones excluded)."""
        return [p for p in self._history if p is not None]

    def is_lost(self) -> bool:
        return self._missing_count > self.max_gap

    def reset(self):
        self._history.clear()
        self._last_center  = None
        self._missing_count = 0

    # ------------------------------------------------------------------ #
    #  Drawing                                                              #
    # ------------------------------------------------------------------ #

    def draw_trail(self, frame: np.ndarray) -> np.ndarray:
        """Draw a fading orange trail on the frame (in-place)."""
        trail = self.get_trail()
        for i in range(1, len(trail)):
            alpha  = int(255 * i / len(trail))
            radius = max(2, int(6 * i / len(trail)))
            color  = (0, int(165 * i / len(trail)), 255)
            cv2.line(frame, trail[i - 1], trail[i], color, radius)
            cv2.circle(frame, trail[i], radius, color, -1)
        return frame

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _select_best(self, candidates: list[Detection]) -> Detection | None:
        if not candidates:
            return None

        # Filter by size plausibility
        valid = [
            d for d in candidates
            if self.min_area <= d.area <= self.max_area
        ]
        if not valid:
            valid = candidates   # fall back to all if all fail size check

        # If we have a known last position, prefer the closest
        if self._last_center is not None:
            def score(d: Detection):
                dist = np.linalg.norm(
                    np.array(d.center) - np.array(self._last_center)
                )
                return dist - d.confidence * 50   # favour high confidence

            valid = [d for d in valid
                     if np.linalg.norm(
                         np.array(d.center) - np.array(self._last_center)
                     ) < self.max_jump]
            if not valid:
                return None
            return min(valid, key=score)

        # First detection — pick highest confidence
        return max(valid, key=lambda d: d.confidence)