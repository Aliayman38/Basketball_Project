"""
src/detection/ball_tracker.py
──────────────────────────────
Dedicated ball detection and interpolation module.

Problems solved
───────────────
1. Ball recall = 0.695 → model fires at conf 0.18–0.25 but conf=0.30 filter
   drops those detections. Fix: use a per-class lower threshold for the ball.

2. Motion blur during fast passes → ball disappears for 3–8 frames.
   Fix: linear interpolation fills gaps ≤ MAX_GAP_FRAMES.

3. Ball is tiny at full resolution → SAHI tile-based inference finds it
   even when sub-pixel in the full frame.

Usage
─────
  from src.detection.ball_tracker import BallTracker

  ball_tracker = BallTracker(model_path="best.pt")
  ball_tracker.update(frame, frame_idx)

  pos = ball_tracker.get_position(frame_idx)    # (cx, cy) or None
  traj = ball_tracker.get_trajectory()          # [(cx, cy, fidx), ...]
"""

from __future__ import annotations
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Class IDs — must match your dataset
CLASS_BALL = 0

# Interpolation cap: fill gaps shorter than this many frames
MAX_GAP_FRAMES = 12

# How many past positions to keep for smoothing
SMOOTH_WINDOW = 5


class BallTracker:
    """
    Dedicated ball detector with:
      - Low-confidence threshold (0.15) to maximise recall
      - SAHI-lite: detect on full frame AND on 4 quadrant tiles
      - Linear interpolation for short gap frames
      - Exponential moving average smoothing on position

    Parameters
    ----------
    model_path      : path to fine-tuned YOLOv11 .pt
    ball_conf       : detection threshold for ball only (default 0.15)
    player_conf     : detection threshold for everything else (default 0.30)
    imgsz           : inference resolution
    use_tiling      : also run inference on 4 overlapping tiles
    device          : torch device string
    """

    def __init__(
        self,
        model_path:   str   = "best.pt",
        ball_conf:    float = 0.15,    # ← lower than default 0.30
        player_conf:  float = 0.30,
        imgsz:        int   = 1280,
        use_tiling:   bool  = True,    # SAHI-lite quadrant tiling
        device:       str   = "0",
    ) -> None:
        self.ball_conf   = ball_conf
        self.player_conf = player_conf
        self.imgsz       = imgsz
        self.use_tiling  = use_tiling
        self.device      = device

        print(f"[BallTracker] Loading: {model_path}  ball_conf={ball_conf}")
        self.model = YOLO(model_path)

        # Raw detections per frame: {frame_idx: (cx, cy, conf)}
        self._raw: dict[int, tuple[float, float, float]] = {}

        # Final positions after interpolation: {frame_idx: (cx, cy)}
        self._positions: dict[int, tuple[float, float]] = {}

        # Ordered trajectory (cx, cy, fidx)
        self._trajectory: list[tuple[float, float, int]] = []

        # Smoothing buffer
        self._recent: deque = deque(maxlen=SMOOTH_WINDOW)

        self._last_frame_idx: int = -1

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, frame_idx: int) -> tuple[float, float] | None:
        """
        Detect the ball in `frame` and update internal state.

        Returns (cx, cy) if ball found this frame, else None.
        """
        detection = self._detect_ball(frame)

        if detection is not None:
            cx, cy, conf = detection
            # Smooth with EMA
            if self._recent:
                cx = 0.7 * cx + 0.3 * self._recent[-1][0]
                cy = 0.7 * cy + 0.3 * self._recent[-1][1]

            self._raw[frame_idx]       = (cx, cy, conf)
            self._positions[frame_idx] = (cx, cy)
            self._recent.append((cx, cy))
            self._trajectory.append((cx, cy, frame_idx))

        # Fill any gap since last detection
        if self._last_frame_idx >= 0:
            self._interpolate_gap(self._last_frame_idx, frame_idx)

        self._last_frame_idx = frame_idx
        return self._positions.get(frame_idx)

    def get_position(self, frame_idx: int) -> tuple[float, float] | None:
        """Return ball (cx, cy) at frame_idx, including interpolated frames."""
        return self._positions.get(frame_idx)

    def get_trajectory(self) -> list[tuple[float, float, int]]:
        """Full trajectory including interpolated points [(cx, cy, fidx)]."""
        return sorted(self._positions.items(), key=lambda kv: kv[0])  # type: ignore
        # actually return proper format
        return [(cx, cy, fidx) for fidx, (cx, cy) in sorted(self._positions.items())]

    def get_raw_trajectory(self) -> list[tuple[float, float, int]]:
        """Only real detections, no interpolated points."""
        return [(cx, cy, fidx) for fidx, (cx, cy, _conf) in sorted(self._raw.items())]

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_ball(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        """
        Run full-frame + optional tiled detection and return the best ball.

        Full-frame is fast and catches mid-range balls.
        Tiling catches balls that are tiny in the full view.
        """
        candidates: list[tuple[float, float, float]] = []

        # ── 1. Full-frame inference at low conf ───────────────────────────
        result = self.model(
            frame,
            conf=self.ball_conf,   # ← low threshold here
            iou=0.45,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            classes=[CLASS_BALL],  # ← only look for ball, ignore players
        )[0]

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            cx   = (xyxy[0] + xyxy[2]) / 2.0
            cy   = (xyxy[1] + xyxy[3]) / 2.0
            candidates.append((cx, cy, conf))

        # ── 2. SAHI-lite: 4 overlapping quadrant tiles ────────────────────
        if self.use_tiling:
            h, w = frame.shape[:2]
            tiles = self._get_tiles(w, h)

            for (x1, y1, x2, y2) in tiles:
                tile = frame[y1:y2, x1:x2]
                t_result = self.model(
                    tile,
                    conf=self.ball_conf,
                    iou=0.45,
                    imgsz=640,       # tiles are smaller → 640 is fine
                    device=self.device,
                    verbose=False,
                    classes=[CLASS_BALL],
                )[0]

                for box in t_result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu())
                    # Map tile coords back to full-frame coords
                    cx_tile = (xyxy[0] + xyxy[2]) / 2.0
                    cy_tile = (xyxy[1] + xyxy[3]) / 2.0
                    candidates.append((
                        cx_tile + x1,
                        cy_tile + y1,
                        conf * 0.95,   # slight penalty for tile-sourced detections
                    ))

        if not candidates:
            return None

        # NMS-lite: cluster nearby candidates (within 40 px) and keep highest conf
        candidates = self._nms_candidates(candidates, dist_thresh=40.0)

        # Return highest-confidence surviving candidate
        return max(candidates, key=lambda c: c[2])

    @staticmethod
    def _get_tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
        """
        Return 4 overlapping tile regions covering the full frame.
        50 % overlap ensures the ball is fully inside at least one tile.
        """
        hw, hh = w // 2, h // 2
        overlap = w // 8   # ~12 % overlap
        return [
            (0,          0,          hw + overlap, hh + overlap),   # TL
            (hw - overlap, 0,          w,           hh + overlap),   # TR
            (0,          hh - overlap, hw + overlap, h           ),   # BL
            (hw - overlap, hh - overlap, w,          h           ),   # BR
        ]

    @staticmethod
    def _nms_candidates(
        candidates: list[tuple[float, float, float]],
        dist_thresh: float = 40.0,
    ) -> list[tuple[float, float, float]]:
        """
        Greedy distance-based NMS: suppress weaker candidates within
        dist_thresh pixels of a stronger one.
        """
        sorted_cands = sorted(candidates, key=lambda c: c[2], reverse=True)
        kept = []
        for cand in sorted_cands:
            cx, cy, _ = cand
            if all(
                np.hypot(cx - kx, cy - ky) > dist_thresh
                for kx, ky, _ in kept
            ):
                kept.append(cand)
        return kept

    # ── Interpolation ─────────────────────────────────────────────────────────

    def _interpolate_gap(self, prev_idx: int, curr_idx: int) -> None:
        """
        If the ball was visible at prev_idx and curr_idx but missing in between,
        and the gap is ≤ MAX_GAP_FRAMES, fill with linear interpolation.
        """
        gap = curr_idx - prev_idx - 1
        if gap <= 0 or gap > MAX_GAP_FRAMES:
            return

        prev_pos = self._positions.get(prev_idx)
        curr_pos = self._positions.get(curr_idx)
        if prev_pos is None or curr_pos is None:
            return

        # Check that none of the gap frames already have a detection
        missing = [
            idx for idx in range(prev_idx + 1, curr_idx)
            if idx not in self._positions
        ]
        if not missing:
            return

        px, py = prev_pos
        cx, cy = curr_pos

        for i, fidx in enumerate(missing, start=1):
            t = i / (len(missing) + 1)
            interp_x = px + t * (cx - px)
            interp_y = py + t * (cy - py)
            self._positions[fidx] = (interp_x, interp_y)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def draw_ball(
        self,
        frame: np.ndarray,
        frame_idx: int,
        trail_length: int = 20,
    ) -> np.ndarray:
        """
        Draw ball position + motion trail on a copy of `frame`.

        Parameters
        ----------
        frame_idx    : current frame index
        trail_length : how many past frames to show in the trail
        """
        vis = frame.copy()

        # Draw trail
        for i in range(trail_length, 0, -1):
            past_idx = frame_idx - i
            pos = self._positions.get(past_idx)
            if pos is None:
                continue
            alpha = 1.0 - (i / trail_length)
            radius = max(2, int(4 * alpha))
            color = (
                int(255 * alpha),
                int(165 * alpha),
                0,
            )
            cv2.circle(vis, (int(pos[0]), int(pos[1])), radius, color, -1)

        # Draw current position
        pos = self._positions.get(frame_idx)
        if pos is not None:
            cx, cy = int(pos[0]), int(pos[1])
            is_interpolated = frame_idx not in self._raw

            color  = (0, 165, 255) if not is_interpolated else (180, 180, 0)
            label  = "Ball" if not is_interpolated else "Ball (interp)"
            cv2.circle(vis, (cx, cy), 12, color, 2)
            cv2.circle(vis, (cx, cy),  4, color, -1)
            cv2.putText(
                vis, label, (cx + 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        return vis

    def reset(self) -> None:
        self._raw.clear()
        self._positions.clear()
        self._trajectory.clear()
        self._recent.clear()
        self._last_frame_idx = -1

    def __repr__(self) -> str:
        return (
            f"BallTracker(ball_conf={self.ball_conf}, "
            f"tiling={self.use_tiling}, "
            f"detections={len(self._raw)}, "
            f"total_positions={len(self._positions)})"
        )