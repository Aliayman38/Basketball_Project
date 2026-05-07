import cv2
import os
import time
import numpy as np
import torch
import scipy.linalg
from pathlib import Path

from detection.detector import BasketballDetector
from boxmot.trackers.botsort.botsort import BotSort

# ── Kalman stability patch ────────────────────────────────────────────────────
from boxmot.motion.kalman_filters.base import BaseKalmanFilter

def _stable_update_state(self, z, R=None, H=None):
    _H = self._resolve_matrix(H, self.H)
    _R = self._resolve_matrix(R, self.R)
    if np.isscalar(_R):
        _R = np.eye(self.dim_z) * float(_R)

    projected_mean, projected_cov = self.project_state(H=_H, R=_R)

    eps = 1e-6
    while True:
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False
            )
            break
        except np.linalg.LinAlgError:
            projected_cov += np.eye(projected_cov.shape[0]) * eps
            eps *= 10
            if eps > 1.0:
                return self.x, self.P

    self.K = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(self.P, _H.T).T, check_finite=False
    ).T
    self.y  = z.reshape(-1, 1)[:self.dim_z] - projected_mean
    self.S  = projected_cov
    self.SI = scipy.linalg.cho_solve(
        (chol_factor, lower), np.eye(self.dim_z), check_finite=False
    )
    self.x      = self.x + np.dot(self.K, self.y)
    self.P      = self.P - np.linalg.multi_dot((self.K, projected_cov, self.K.T))
    self.z      = z.reshape(-1, 1)[:self.dim_z].copy()
    self.x_post = self.x.copy()
    self.P_post = self.P.copy()
    return self.x, self.P

BaseKalmanFilter.update_state = _stable_update_state
# ── end patch ─────────────────────────────────────────────────────────────────


BALL_CLASS_ID = 0
CLASS_NAMES   = {0: 'basketball', 1: 'net', 2: 'player', 3: 'referee'}


# ── Custom ID manager (replaces ocsort_tracker.py entirely) ──────────────────

class IDManager:
    """
    Maps BotSort's internal track IDs → clean sequential IDs per class.
    player 1-10 | referee 1-4 | net 1-2
    Recycles IDs after a track has been absent for max_absent frames.
    """
    LIMITS = {'player': 10, 'referee': 4, 'net': 2}
    MAX_ABSENT = 60   # frames before a lost track's ID is recycled

    def __init__(self):
        # ori_id → custom_id
        self._map: dict[str, dict[int, int]] = {c: {} for c in self.LIMITS}
        # ori_id → frames absent
        self._age: dict[str, dict[int, int]] = {c: {} for c in self.LIMITS}

    def get_id(self, cls_name: str, ori_id: int) -> int | None:
        if cls_name not in self.LIMITS:
            return ori_id

        m = self._map[cls_name]

        if ori_id in m:
            self._age[cls_name][ori_id] = 0   # reset age
            return m[ori_id]

        free = self._next_free(cls_name)
        if free is None:
            return None

        m[ori_id] = free
        self._age[cls_name][ori_id] = 0
        return free

    def update_ages(self, active: dict[str, set[int]]):
        """Call once per frame with the set of ori_ids seen this frame."""
        for cls_name in self.LIMITS:
            seen = active.get(cls_name, set())
            for ori_id in list(self._age[cls_name]):
                if ori_id in seen:
                    self._age[cls_name][ori_id] = 0
                else:
                    self._age[cls_name][ori_id] += 1
                    if self._age[cls_name][ori_id] >= self.MAX_ABSENT:
                        del self._map[cls_name][ori_id]
                        del self._age[cls_name][ori_id]

    def _next_free(self, cls_name: str) -> int | None:
        used = set(self._map[cls_name].values())
        for i in range(1, self.LIMITS[cls_name] + 1):
            if i not in used:
                return i
        return None


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    video_path  = 'data/test_3.mp4'
    model_path  = 'models/weights/last.pt'
    output_path = 'runs/bot-sort tracking/tracking_botsort.mp4'
    reid_path   = 'osnet_x0_25_msmt17.pt'
    device      = torch.device('cuda:0')

    os.makedirs('runs', exist_ok=True)

    detector   = BasketballDetector(model_path)
    id_manager = IDManager()

    tracker = BotSort(
        reid_weights      = Path(reid_path),
        device            = device,
        half              = True,
        track_high_thresh = 0.30,
        track_low_thresh  = 0.10,
        new_track_thresh  = 0.40,
        track_buffer      = 120,
        match_thresh      = 0.80,
        proximity_thresh  = 0.50,
        appearance_thresh = 0.40,
        cmc_method        = 'ecc',
        frame_rate        = 30,
        with_reid         = True,
        min_hits          = 1,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )

    print('🚀 Starting BoT-SORT Tracking...')
    frame_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.model.predict(frame, conf=0.3, verbose=False)[0]

        ball_boxes   = []
        tracker_dets = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf   = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            if cls_id == BALL_CLASS_ID:
                ball_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
            else:
                tracker_dets.append([x1, y1, x2, y2, conf, float(cls_id)])

        dets   = (np.array(tracker_dets, dtype=float)
                  if tracker_dets else np.empty((0, 6)))
        tracks = tracker.update(dets, frame)

        # assign clean IDs and draw
        active: dict[str, set] = {}
        for track in tracks:
            x1, y1, x2, y2, ori_id, conf, cls_idx = track[:7]
            cls_name = CLASS_NAMES.get(int(cls_idx), 'unknown')
            ori_id   = int(ori_id)

            custom_id = id_manager.get_id(cls_name, ori_id)
            if custom_id is None:
                continue

            active.setdefault(cls_name, set()).add(ori_id)

            color = detector.colors.get(cls_name, (255, 255, 255))
            draw_box(frame,
                     int(x1), int(y1), int(x2), int(y2),
                     f'{cls_name} {custom_id}', color)

        id_manager.update_ages(active)

        # single best ball (highest confidence)
        if ball_boxes:
            x1, y1, x2, y2, _ = max(ball_boxes, key=lambda b: b[4])
            draw_box(frame, x1, y1, x2, y2,
                     'basketball', detector.colors['basketball'])

        writer.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Frame {frame_count}  |  {frame_count / (time.time() - t0):.1f} FPS')

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f'\n✅ Done — {frame_count} frames in {elapsed:.1f}s '
          f'({frame_count / elapsed:.1f} FPS avg)')
    print(f'   Saved → {output_path}')


if __name__ == '__main__':
    main()