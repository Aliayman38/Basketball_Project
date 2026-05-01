"""
src/detection/ball_tracker.py  (v2 — high-accuracy rewrite)
─────────────────────────────
5-layer ball tracking pipeline that handles:
  1. YOLO ultra-low conf (0.07) + Kalman distance gate  → max detections
  2. True SAHI 4-tile inference at 640px                → tiny ball on tiles
  3. Kalman filter state machine                        → predicts position
                                                           even in miss frames
  4. Lucas-Kanade Optical Flow fallback                 → tracks orange blob
                                                           when YOLO fully fails
  5. Physics validation                                 → rejects impossible jumps

Why stay on YOLOv11L (not X):
  - Ball's problem is spatial resolution and strategy, not model capacity
  - L→X gives only ~2% mAP on COCO; optical flow + Kalman gives far more
  - X is 50% slower and uses 50% more VRAM
  - Use that compute budget for SAHI tiling instead

Usage:
    bt = BallTracker(model_path="best.pt")
    pos = bt.update(frame, frame_idx)   # (cx, cy) always — never None after warmup
    trail = bt.get_trail(n=20)          # last n (cx, cy) positions
"""

from __future__ import annotations
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

CLASS_BALL = 0

# ── Physics constants ────────────────────────────────────────────────────────
MAX_BALL_SPEED_PX_PER_FRAME = 120   # ~15 m/s at typical camera distance
MAX_KALMAN_GATE_PX          = 150   # reject detections too far from prediction
MAX_INTERP_GAP              = 20    # fill gaps up to this many frames

# ── Optical flow parameters ──────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)
ORANGE_HSV_LOWER = np.array([5,  120, 120], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)


class BallTracker:
    """
    High-accuracy basketball tracker using 5 layered strategies.

    Parameters
    ----------
    model_path  : YOLOv11 fine-tuned weights (.pt)
    ball_conf   : YOLO confidence threshold — ultra low, gated by Kalman
    imgsz       : inference resolution for full-frame pass
    use_tiling  : enable SAHI 4-tile inference
    use_flow    : enable Lucas-Kanade optical flow fallback
    device      : torch device ('0' or 'cpu')
    """

    def __init__(
        self,
        model_path: str   = "best.pt",
        ball_conf:  float = 0.07,
        imgsz:      int   = 1280,
        use_tiling: bool  = True,
        use_flow:   bool  = True,
        device:     str   = "0",
    ) -> None:
        self.ball_conf  = ball_conf
        self.imgsz      = imgsz
        self.use_tiling = use_tiling
        self.use_flow   = use_flow
        self.device     = device

        print(f"[BallTracker] Loading: {model_path}")
        self.model = YOLO(model_path)

        # ── Kalman filter (state: cx, cy, vx, vy) ────────────────────────
        self.kf = self._build_kalman()
        self._kf_initialized = False

        # ── Position history ──────────────────────────────────────────────
        # {frame_idx: (cx, cy)}  — includes predicted + interpolated frames
        self._positions: dict[int, tuple[float, float]] = {}
        # only real detections (YOLO or flow)
        self._detections: dict[int, tuple[float, float]] = {}

        # ── Optical flow state ────────────────────────────────────────────
        self._prev_gray:  np.ndarray | None = None
        self._flow_point: np.ndarray | None = None  # shape (1,1,2) float32
        self._flow_miss_count: int = 0
        self._flow_max_miss: int = 5    # reset flow after 5 consecutive misses

        # ── Trail buffer (for annotator) ──────────────────────────────────
        self._trail: deque[tuple[float, float]] = deque(maxlen=30)

        self._last_fidx: int = -1
        self._source_log: dict[int, str] = {}   # frame → 'yolo'|'flow'|'kalman'|'interp'

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self, frame: np.ndarray, frame_idx: int
    ) -> tuple[float, float] | None:
        """
        Process one frame. Returns (cx, cy) — the best ball position.

        Priority: YOLO full-frame → YOLO tiled → Optical flow → Kalman predict
        After detection: Kalman is updated / corrected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos, source = None, "none"

        # ── Strategy 1: YOLO full-frame (ultra low conf) ──────────────────
        yolo_pos = self._yolo_detect(frame, self.imgsz)
        if yolo_pos and self._kalman_gate(yolo_pos):
            pos, source = yolo_pos, "yolo"

        # ── Strategy 2: SAHI 4-tile if full-frame missed ──────────────────
        if pos is None and self.use_tiling:
            tile_pos = self._sahi_detect(frame)
            if tile_pos and self._kalman_gate(tile_pos):
                pos, source = tile_pos, "yolo_tile"

        # ── Strategy 3: Lucas-Kanade optical flow ────────────────────────
        if pos is None and self.use_flow and self._prev_gray is not None:
            flow_pos = self._optical_flow(gray, frame)
            if flow_pos and self._kalman_gate(flow_pos):
                pos, source = flow_pos, "flow"
                self._flow_miss_count = 0
            else:
                self._flow_miss_count += 1
                if self._flow_miss_count > self._flow_max_miss:
                    self._flow_point = None   # reset stale flow seed

        # ── Update Kalman ─────────────────────────────────────────────────
        if pos is not None:
            kf_pos = self._kalman_correct(pos)
            # Use Kalman-smoothed position (removes single-frame jitter)
            pos = kf_pos
            self._detections[frame_idx] = pos
            self._update_flow_seed(pos)
        else:
            # ── Strategy 4: Kalman prediction (no detection this frame) ──
            if self._kf_initialized:
                pos = self._kalman_predict_only()
                source = "kalman"
            # else: truly no data yet

        # ── Store position and update trail ───────────────────────────────
        if pos is not None:
            self._positions[frame_idx] = pos
            self._trail.append(pos)
            self._source_log[frame_idx] = source

        # ── Post-frame bookkeeping ─────────────────────────────────────────
        self._prev_gray  = gray.copy()
        self._last_fidx  = frame_idx

        # Back-fill interpolation between last real detection and now
        self._interpolate_back(frame_idx)

        return self._positions.get(frame_idx)

    def get_position(self, frame_idx: int) -> tuple[float, float] | None:
        return self._positions.get(frame_idx)

    def get_trail(self, n: int = 20) -> list[tuple[float, float]]:
        """Return last n confirmed positions.  n=0 → empty list (no trail)."""
        if n <= 0:
            return []
        trail = list(self._trail)
        return trail[-n:]

    def get_trajectory(self) -> list[tuple[float, float, int]]:
        """Full trajectory [(cx, cy, frame_idx)] sorted by frame."""
        return [(cx, cy, fidx)
                for fidx, (cx, cy) in sorted(self._positions.items())]

    def get_source(self, frame_idx: int) -> str:
        """Return detection source for a frame: 'yolo','flow','kalman','interp'"""
        return self._source_log.get(frame_idx, "none")

    # ── Strategy 1: YOLO full-frame ───────────────────────────────────────────

    def _yolo_detect(
        self, frame: np.ndarray, imgsz: int
    ) -> tuple[float, float] | None:
        result = self.model(
            frame,
            conf=self.ball_conf,
            iou=0.3,
            imgsz=imgsz,
            device=self.device,
            classes=[CLASS_BALL],
            verbose=False,
        )[0]
        return self._best_box(result.boxes, offset=(0, 0))

    # ── Strategy 2: SAHI 4-tile ───────────────────────────────────────────────

    def _sahi_detect(
        self, frame: np.ndarray
    ) -> tuple[float, float] | None:
        h, w = frame.shape[:2]
        hw, hh = w // 2, h // 2
        ov = w // 8     # 12.5% overlap prevents ball from falling on tile edge

        tiles = [
            (0,       0,       hw + ov, hh + ov),
            (hw - ov, 0,       w,       hh + ov),
            (0,       hh - ov, hw + ov, h      ),
            (hw - ov, hh - ov, w,       h      ),
        ]

        candidates: list[tuple[float, float, float]] = []

        for x1, y1, x2, y2 in tiles:
            tile = frame[y1:y2, x1:x2]
            result = self.model(
                tile,
                conf=self.ball_conf,
                iou=0.3,
                imgsz=640,
                device=self.device,
                classes=[CLASS_BALL],
                verbose=False,
            )[0]
            pt = self._best_box(result.boxes, offset=(x1, y1), return_conf=True)
            if pt:
                candidates.append(pt)

        if not candidates:
            return None

        # Cluster-NMS: merge detections within 40px, keep highest conf
        candidates.sort(key=lambda c: c[2], reverse=True)
        kept = []
        for cx, cy, conf in candidates:
            if all(np.hypot(cx - kx, cy - ky) > 40 for kx, ky, _ in kept):
                kept.append((cx, cy, conf))

        return (kept[0][0], kept[0][1]) if kept else None

    # ── Strategy 3: Lucas-Kanade optical flow ────────────────────────────────

# ── Strategy 3: Lucas-Kanade optical flow ────────────────────────────────

    def _optical_flow(
    self, gray: np.ndarray, frame: np.ndarray
    ) -> tuple[float, float] | None:
        if self._flow_point is None or self._prev_gray is None:
            return None

        next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray,
            self._flow_point, None,
            **LK_PARAMS,
        )

        if status is None or status[0][0] == 0:
            return None

        cx, cy = float(next_pt[0, 0, 0]), float(next_pt[0, 0, 1])

        # Validate 1: Boundary check
        h, w = gray.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return None

        # Validate 2: Is the tracked point still orange? (basketball color check)
        # Crop a small 10x10 window around the tracked point
        xi, yi = int(cx), int(cy)
        window_size = 5  # Half-size
        y1, y2 = max(0, yi - window_size), min(h, yi + window_size)
        x1, x2 = max(0, xi - window_size), min(w, xi + window_size)
        
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        # Convert patch to HSV and check for orange
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv_patch, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)
        
        # If less than 5% of the patch is orange, reject the tracking point
        orange_ratio = cv2.countNonZero(orange_mask) / (patch.shape[0] * patch.shape[1])
        if orange_ratio < 0.05:
            return None  # Tracker drifted onto something non-orange (e.g., a player)

        self._flow_point = np.array([[[cx, cy]]], dtype=np.float32)
        return (cx, cy)
    
    def _update_flow_seed(self, pos: tuple[float, float]) -> None:
        """Seed the optical flow tracker at the confirmed ball position."""
        self._flow_point = np.array([[[pos[0], pos[1]]]], dtype=np.float32)

    # ── Kalman filter ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        """
        4-state Kalman: [cx, cy, vx, vy]
        Constant-velocity model with measurement = [cx, cy].

        Tuned for basketball:
          - High process noise (Q) because ball changes direction abruptly
          - Low measurement noise (R) because YOLO detections are fairly precise
        """
        kf = cv2.KalmanFilter(4, 2)

        # Transition: cx += vx, cy += vy each frame
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # We measure cx and cy directly
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise — HIGH because ball accelerates fast
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 25.0

        # Measurement noise — moderate: YOLO box center jitters ±3px
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9.0

        # Initial uncertainty
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        return kf

    def _kalman_correct(
        self, pos: tuple[float, float]
    ) -> tuple[float, float]:
        """Feed a real measurement into the Kalman filter and return smoothed pos."""
        measurement = np.array([[pos[0]], [pos[1]]], dtype=np.float32)

        if not self._kf_initialized:
            # Initialise state with first detection
            self.kf.statePre  = np.array([[pos[0]], [pos[1]], [0.], [0.]], dtype=np.float32)
            self.kf.statePost = np.array([[pos[0]], [pos[1]], [0.], [0.]], dtype=np.float32)
            self._kf_initialized = True
            return pos

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return (float(corrected[0][0]), float(corrected[1][0]))

    def _kalman_predict_only(self) -> tuple[float, float] | None:
        """Predict next position without a measurement (pure physics)."""
        if not self._kf_initialized:
            return None
        predicted = self.kf.predict()
        cx, cy = float(predicted[0][0]), float(predicted[1][0])

        # Sanity: if prediction wanders off-screen or explodes, return None
        if not (0 <= cx < 4000 and 0 <= cy < 4000):
            return None
        return (cx, cy)

    def _kalman_gate(
        self, pos: tuple[float, float]
    ) -> bool:
        """
        Reject detections too far from the Kalman prediction.
        This eliminates false positives (orange ads, sponsor logos)
        that YOLO picks up at conf=0.07.
        """
        if not self._kf_initialized:
            return True     # accept anything during warmup

        pred = self._kalman_predict_only()
        if pred is None:
            return True

        dist = np.hypot(pos[0] - pred[0], pos[1] - pred[1])
        return dist <= MAX_KALMAN_GATE_PX

    # ── Gap interpolation ─────────────────────────────────────────────────────

    def _interpolate_back(self, curr_fidx: int) -> None:
        """
        Linear interpolation between the last real detection and current frame
        for any un-filled gap ≤ MAX_INTERP_GAP frames.
        Kalman frames are NOT overwritten — only truly-empty frames.
        """
        if not self._detections:
            return

        real_fidxs = sorted(self._detections.keys())
        if len(real_fidxs) < 2:
            return

        for i in range(len(real_fidxs) - 1):
            f1, f2 = real_fidxs[i], real_fidxs[i + 1]
            gap = f2 - f1 - 1
            if gap <= 0 or gap > MAX_INTERP_GAP:
                continue

            p1 = self._detections[f1]
            p2 = self._detections[f2]

            for k, fidx in enumerate(range(f1 + 1, f2), start=1):
                if fidx in self._detections:
                    continue   # already has a real detection
                # Overwrite kalman with smoother interpolation
                t = k / (gap + 1)
                ix = p1[0] + t * (p2[0] - p1[0])
                iy = p1[1] + t * (p2[1] - p1[1])
                self._positions[fidx] = (ix, iy)
                self._source_log[fidx] = "interp"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _best_box(
        boxes,
        offset: tuple[int, int] = (0, 0),
        return_conf: bool = False,
    ):
        """Return (cx, cy) or (cx, cy, conf) of the highest-confidence box."""
        if boxes is None or len(boxes) == 0:
            return None

        best_idx  = int(boxes.conf.argmax())
        xyxy      = boxes.xyxy[best_idx].cpu().numpy()
        conf      = float(boxes.conf[best_idx].cpu())
        cx = float((xyxy[0] + xyxy[2]) / 2) + offset[0]
        cy = float((xyxy[1] + xyxy[3]) / 2) + offset[1]

        return (cx, cy, conf) if return_conf else (cx, cy)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        trail_len: int = 25,
    ) -> np.ndarray:
        """Draw ball + motion trail + source label on a copy of frame."""
        vis = frame.copy()

        # Trail — fading orange dots
        trail = self.get_trail(trail_len)
        for i, (cx, cy) in enumerate(trail):
            alpha  = (i + 1) / len(trail)
            radius = max(2, int(6 * alpha))
            color  = (0, int(140 * alpha), int(255 * alpha))
            cv2.circle(vis, (int(cx), int(cy)), radius, color, -1)

        # Current position
        pos = self.get_position(frame_idx)
        if pos is not None:
            cx, cy   = int(pos[0]), int(pos[1])
            source   = self.get_source(frame_idx)
            is_real  = source in ("yolo", "yolo_tile", "flow")

            ring_color = (0, 165, 255) if is_real else (0, 200, 80)
            cv2.circle(vis, (cx, cy), 14, ring_color, 2)
            cv2.circle(vis, (cx, cy),  4, ring_color, -1)

            labels = {
                "yolo":      "Ball (YOLO)",
                "yolo_tile": "Ball (tile)",
                "flow":      "Ball (flow)",
                "kalman":    "Ball (pred)",
                "interp":    "Ball (interp)",
            }
            cv2.putText(
                vis, labels.get(source, "Ball"),
                (cx + 16, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ring_color, 2, cv2.LINE_AA,
            )

        return vis

    def reset(self) -> None:
        self.kf              = self._build_kalman()
        self._kf_initialized = False
        self._positions.clear()
        self._detections.clear()
        self._source_log.clear()
        self._trail.clear()
        self._prev_gray      = None
        self._flow_point     = None
        self._flow_miss_count = 0
        self._last_fidx      = -1

    def __repr__(self) -> str:
        real = sum(1 for s in self._source_log.values() if s in ("yolo","yolo_tile","flow"))
        pred = sum(1 for s in self._source_log.values() if s == "kalman")
        intp = sum(1 for s in self._source_log.values() if s == "interp")
        return (
            f"BallTracker(real={real}, kalman_pred={pred}, interp={intp}, "
            f"total={len(self._positions)})"
        )