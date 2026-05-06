import numpy as np


# ---------------------------------------------------------------------------
# Kalman Filter (1 per track)
# ---------------------------------------------------------------------------
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: list):
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.id                = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history    = []
        self.hits       = 0
        self.hit_streak = 0
        self.age        = 0

        # OC-SORT: store previous observations for velocity estimation
        self.observations: dict[int, np.ndarray] = {}
        self.last_observation = np.array(bbox)
        self.delta_t = 3

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([cx, cy, s, r]).reshape(4, 1)

    @staticmethod
    def _z_to_bbox(x, score=None):
        cx, cy, s, r = x[0], x[1], x[2], x[3]
        w = np.sqrt(abs(s * r))
        h = abs(s) / (w + 1e-6)
        bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        if score is None:
            return bbox
        return bbox + [score]

    # ------------------------------------------------------------------
    def update(self, bbox: list):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        # OC-SORT: store observation with current age
        self.observations[self.age] = np.array(bbox)
        self.last_observation = np.array(bbox)

        self.kf.update(self._bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred = self._z_to_bbox(self.kf.x)
        self.history.append(pred)
        return self.history[-1]

    def get_state(self):
        return self._z_to_bbox(self.kf.x)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------
def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[...,  2] - bb_gt[...,  0]) * (bb_gt[...,  3] - bb_gt[...,  1])

    return inter / (area_test + area_gt - inter + 1e-6)


def linear_assignment(cost_matrix: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.stack([row_ind, col_ind], axis=1)
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty(0, dtype=int)
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(int)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))

    unmatched_detections = [d for d in range(len(detections))
                            if d not in matched_indices[:, 0]]
    unmatched_trackers   = [t for t in range(len(trackers))
                            if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.concatenate(matches, axis=0) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ---------------------------------------------------------------------------
# OC-SORT Tracker
# ---------------------------------------------------------------------------
class OCSortTracker:
    """
    Simplified OC-SORT tracker.
    Accepts raw bbox detections and returns tracked objects with stable IDs.
    """

    def __init__(
        self,
        max_age: int        = 30,
        min_hits: int       = 3,
        iou_threshold: float = 0.3,
        det_thresh: float   = 0.3,
        delta_t: int        = 3,
    ):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.det_thresh    = det_thresh
        self.delta_t       = delta_t

        self.trackers:   list[KalmanBoxTracker] = []
        self.frame_count = 0

        KalmanBoxTracker.count = 0   # reset IDs on new tracker

    # ------------------------------------------------------------------
    def update(self, detections: list) -> np.ndarray:
        """
        detections : list of dicts with 'bbox' and 'confidence'
        Returns   : np.ndarray shape (N, 5) → [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Build detection array [x1,y1,x2,y2,conf]
        dets_array = np.zeros((0, 5))
        if detections:
            dets_array = np.array([
                d["bbox"] + [d["confidence"]]
                for d in detections
                if d["confidence"] >= self.det_thresh
            ])

        # Predict existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t_idx, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t_idx] = pos + [0]
            if np.any(np.isnan(pos)):
                to_del.append(t_idx)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t_idx in reversed(to_del):
            self.trackers.pop(t_idx)

        # Associate
        if dets_array.shape[0] > 0:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                dets_array[:, :4], trks[:, :4], self.iou_threshold
            )
        else:
            matched          = np.empty((0, 2), dtype=int)
            unmatched_dets   = np.empty(0,      dtype=int)
            unmatched_trks   = np.arange(len(self.trackers))

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets_array[m[0], :4].tolist())

        # Create new trackers for unmatched detections
        for idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets_array[idx, :4].tolist()))

        # Collect outputs
        results = []
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                results.append([*d, trk.id + 1])   # 1-indexed ID

        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return np.array(results) if results else np.empty((0, 5))

    def reset(self):
        self.trackers    = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0