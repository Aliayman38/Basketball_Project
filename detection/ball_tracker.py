"""
src/detection/ball_tracker.py  (v3 — TrackNet primary)
────────────────────────────────────────────────────────
Ball tracking: TrackNet primary → RT-DETR fallback → Optical flow → Kalman → Interpolation
"""
from __future__ import annotations
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import RTDETR
from detection.tracknet import TrackNet, heatmap_to_point

CLASS_BALL           = 0
TRACKNET_CONF_THR    = 0.50
RTDETR_BALL_CONF     = 0.07
KALMAN_GATE_PX       = 150
MAX_INTERP_GAP       = 20
TRACKNET_W           = 640
TRACKNET_H           = 360

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)
ORANGE_HSV_LOWER = np.array([5,  120, 120], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)


class BallTracker:
    """
    5-layer ball tracker:
    1. TrackNet  — spatiotemporal heatmap (3 frames stacked)
    2. RT-DETR  — single-frame fallback  (conf=0.07)
    3. Opt. flow — Lucas-Kanade + orange HSV validation
    4. Kalman    — constant-velocity prediction
    5. Interp    — linear gap fill ≤ 20 frames
    """

    def __init__(
        self,
        tracknet_path: str       = "weights/tracknet_best.pt",
        rtdetr_path:   str       = "models/RT-DETR/RT-DETR.pt",
        device:        str | int = "0",
        use_flow:      bool      = True,
    ) -> None:
        self.device_str = str(device)
        self.use_flow   = use_flow
        self.torch_device = torch.device(
            "cuda" if torch.cuda.is_available() and str(device) != "cpu" else "cpu"
        )

        # TrackNet
        self._tracknet:   TrackNet | None = None
        self._tracknet_w: int = TRACKNET_W
        self._tracknet_h: int = TRACKNET_H

        if tracknet_path and _exists(tracknet_path):
            print(f"[BallTracker] Loading TrackNet <- {tracknet_path}")
            self._tracknet = self._load_tracknet(tracknet_path)
            print(f"[BallTracker] TrackNet ready on {self.torch_device}")
        else:
            print(f"[BallTracker] WARNING: no TrackNet weights at '{tracknet_path}'")
            print(f"[BallTracker] Falling back to RT-DETR-only mode.")

        # RT-DETR fallback
        print(f"[BallTracker] Loading RT-DETR <- {rtdetr_path}")
        self._rtdetr = RTDETR(rtdetr_path)

        # Kalman
        self._kf             = _build_kalman()
        self._kf_initialized = False

        # 3-frame buffer for TrackNet
        self._frame_buf: deque[np.ndarray] = deque(maxlen=3)

        # Position stores
        self._positions:  dict[int, tuple] = {}
        self._detections: dict[int, tuple] = {}
        self._sources:    dict[int, str]   = {}

        # Trail
        self._trail: deque[tuple] = deque(maxlen=30)

        # Optical flow
        self._prev_gray:       np.ndarray | None = None
        self._flow_pt:         np.ndarray | None = None
        self._flow_miss_count: int               = 0

        self._last_fidx: int = -1

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, frame_idx: int) -> tuple | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_buf.append(frame.copy())

        pos, source = None, "none"

        # Layer 1 — TrackNet
        if self._tracknet is not None and len(self._frame_buf) == 3:
            tp = self._tracknet_detect()
            if tp and self._kalman_gate(tp):
                pos, source = tp, "tracknet"

        # Layer 2 — RT-DETR
        if pos is None:
            yp = self._rtdetr_detect(frame)
            if yp and self._kalman_gate(yp):
                pos, source = yp, "rtdetr"

        # Layer 3 — Optical flow
        if pos is None and self.use_flow and self._prev_gray is not None:
            fp = self._optical_flow(gray, frame)
            if fp and self._kalman_gate(fp):
                pos, source = fp, "flow"
                self._flow_miss_count = 0
            else:
                self._flow_miss_count += 1
                if self._flow_miss_count > 5:
                    self._flow_pt = None

        # Layer 4 — Kalman update / predict
        if pos is not None:
            pos = self._kalman_correct(pos)
            self._detections[frame_idx] = pos
            self._update_flow_seed(pos)
        elif self._kf_initialized:
            pos    = self._kalman_predict_only()
            source = "kalman"

        # Store
        if pos is not None:
            self._positions[frame_idx] = pos
            self._trail.append(pos)
            self._sources[frame_idx] = source

        self._prev_gray = gray.copy()
        self._last_fidx = frame_idx
        self._interpolate_back(frame_idx)
        return self._positions.get(frame_idx)

    def get_position(self, fidx: int):
        return self._positions.get(fidx)

    def get_trail(self, n: int = 25):
        return list(self._trail)[-n:]

    def get_source(self, fidx: int) -> str:
        return self._sources.get(fidx, "none")

    def get_trajectory(self):
        return [(cx, cy, fi) for fi, (cx, cy) in sorted(self._positions.items())]

    # ── TrackNet ──────────────────────────────────────────────────────────────

    def _load_tracknet(self, path: str) -> TrackNet:
        model = TrackNet(in_frames=3).to(self.torch_device)
        ckpt  = torch.load(path, map_location=self.torch_device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        model.eval()
        if "input_w" in ckpt:
            self._tracknet_w = ckpt["input_w"]
            self._tracknet_h = ckpt["input_h"]
        return model

    def _tracknet_detect(self):
        frames = list(self._frame_buf)
        orig_h, orig_w = frames[-1].shape[:2]
        tensors = []
        for f in frames:
            r = cv2.resize(f, (self._tracknet_w, self._tracknet_h))
            t = torch.from_numpy(r).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        x = torch.cat(tensors, dim=0).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            heatmap = self._tracknet(x)[0]
        pt = heatmap_to_point(heatmap, threshold=TRACKNET_CONF_THR)
        if pt is None:
            return None
        return (pt[0] / self._tracknet_w * orig_w,
                pt[1] / self._tracknet_h * orig_h)

    # ── RT-DETR ───────────────────────────────────────────────────────────────────

    def _rtdetr_detect(self, frame: np.ndarray):
        res = self._rtdetr(
            frame, conf=RTDETR_BALL_CONF, iou=0.3, imgsz=640,
            device=self.device_str, classes=[CLASS_BALL], verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        best = int(res.boxes.conf.argmax())
        b    = res.boxes.xyxy[best].cpu().numpy()
        return (float((b[0]+b[2])/2), float((b[1]+b[3])/2))

    # ── Optical flow ──────────────────────────────────────────────────────────

    def _optical_flow(self, gray: np.ndarray, frame: np.ndarray):
        if self._flow_pt is None or self._prev_gray is None:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._flow_pt, None, **LK_PARAMS)
        if status is None or status[0][0] == 0:
            return None
        cx, cy = float(nxt[0,0,0]), float(nxt[0,0,1])
        h, w   = gray.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        xi, yi = int(cx), int(cy)
        ws = 5
        patch = frame[max(0,yi-ws):min(h,yi+ws), max(0,xi-ws):min(w,xi+ws)]
        if patch.size == 0:
            return None
        hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)
        ratio = cv2.countNonZero(mask) / (patch.shape[0]*patch.shape[1])
        if ratio < 0.05:
            return None
        self._flow_pt = np.array([[[cx, cy]]], dtype=np.float32)
        return (cx, cy)

    def _update_flow_seed(self, pos):
        self._flow_pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)

    # ── Kalman ────────────────────────────────────────────────────────────────

    def _kalman_correct(self, pos):
        m = np.array([[pos[0]], [pos[1]]], dtype=np.float32)
        if not self._kf_initialized:
            self._kf.statePre  = np.array([[pos[0]],[pos[1]],[0.],[0.]], dtype=np.float32)
            self._kf.statePost = np.array([[pos[0]],[pos[1]],[0.],[0.]], dtype=np.float32)
            self._kf_initialized = True
            return pos
        self._kf.predict()
        c = self._kf.correct(m)
        return (float(c[0][0]), float(c[1][0]))

    def _kalman_predict_only(self):
        if not self._kf_initialized:
            return None
        p = self._kf.predict()
        cx, cy = float(p[0][0]), float(p[1][0])
        return (cx, cy) if (0 <= cx < 5000 and 0 <= cy < 5000) else None

    def _kalman_gate(self, pos) -> bool:
        if not self._kf_initialized:
            return True
        pred = self._kalman_predict_only()
        if pred is None:
            return True
        return np.hypot(pos[0]-pred[0], pos[1]-pred[1]) <= KALMAN_GATE_PX

    # ── Interpolation ─────────────────────────────────────────────────────────

    def _interpolate_back(self, curr_fidx: int) -> None:
        if len(self._detections) < 2:
            return
        fidxs = sorted(self._detections.keys())
        for i in range(len(fidxs)-1):
            f1, f2 = fidxs[i], fidxs[i+1]
            gap    = f2 - f1 - 1
            if gap <= 0 or gap > MAX_INTERP_GAP:
                continue
            p1, p2 = self._detections[f1], self._detections[f2]
            for k, fidx in enumerate(range(f1+1, f2), start=1):
                if fidx in self._detections:
                    continue
                t = k / (gap+1)
                self._positions[fidx] = (p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1]))
                self._sources[fidx]   = "interp"

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, frame_idx: int, trail_len: int = 25) -> np.ndarray:
        vis   = frame.copy()
        trail = self.get_trail(trail_len)
        for i, (cx, cy) in enumerate(trail):
            a = (i+1) / max(len(trail), 1)
            cv2.circle(vis, (int(cx), int(cy)), max(2, int(6*a)),
                       (0, int(140*a), int(255*a)), -1)
        pos = self.get_position(frame_idx)
        if pos is not None:
            cx, cy  = int(pos[0]), int(pos[1])
            src     = self.get_source(frame_idx)
            is_real = src in ("tracknet", "rtdetr", "flow")
            color   = (0, 165, 255) if is_real else (0, 200, 80)
            labels  = {"tracknet":"Ball (TrackNet)","rtdetr":"Ball (RT-DETR)",
                       "flow":"Ball (flow)","kalman":"Ball (pred)","interp":"Ball (interp)"}
            cv2.circle(vis, (cx, cy), 14, color, 2)
            cv2.circle(vis, (cx, cy),  4, color, -1)
            cv2.putText(vis, labels.get(src,"Ball"),
                        (cx+16, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis

    def reset(self) -> None:
        self._kf = _build_kalman(); self._kf_initialized = False
        self._frame_buf.clear(); self._positions.clear()
        self._detections.clear(); self._sources.clear()
        self._trail.clear(); self._prev_gray = None
        self._flow_pt = None; self._flow_miss_count = 0; self._last_fidx = -1

    def __repr__(self) -> str:
        r = sum(1 for s in self._sources.values() if s in ("tracknet","rtdetr","flow"))
        return (f"BallTracker(tracknet={'ok' if self._tracknet else 'MISSING'},"
                f" real={r}, total={len(self._positions)})")


# ── Module helpers ────────────────────────────────────────────────────────────

def _exists(path: str) -> bool:
    import os; return bool(path and os.path.exists(path))

def _build_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 25.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100.0
    return kf