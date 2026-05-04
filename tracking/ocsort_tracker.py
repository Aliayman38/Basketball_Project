"""
src/tracking/ocsort_tracker.py
────────────────────────────────
OC-SORT tracker with appearance-based Re-ID for basketball.

Why OC-SORT over SORT/ByteTrack
─────────────────────────────────
SORT uses Kalman-predicted velocity for matching. During occlusion the
Kalman velocity drifts, so when the player reappears the tracker either
assigns a wrong ID or creates a new one. OC-SORT fixes this with two ideas:

  OCM — Observation-Centric Momentum
    Velocity is estimated from the last delta_t REAL observations, not from
    the Kalman filter. Real observations are always accurate; Kalman velocity
    is a noisy estimate that drifts when there are no updates.

  ORU — Observation-Centric Re-Update
    When a lost track is recovered, linearly interpolate N observations
    between the last real detection and the current one, then feed each
    interpolated point through the Kalman correct() step. This gradually
    corrects the Kalman velocity without a discontinuous jump.

Re-ID Gallery
─────────────
Every confirmed track maintains an HSV appearance histogram (48-dim, EMA).
When a track exceeds max_age it is archived in the gallery rather than
simply deleted. When a new unmatched detection appears it is compared
against the gallery using:

    combined = 0.65 × appearance_cost + 0.35 × position_cost

    appearance_cost : Bhattacharyya distance on HSV histogram  (0=identical)
    position_cost   : normalised euclidean from last position  (0=same spot)

If combined < reid_threshold the old track_id is reused — the player gets
their original ID back transparently.

ID ranges
─────────
  Players  :   1 … 9 999
  Referees : REF_ID_OFFSET+1 … (10 001+)
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_BALL    = 0
CLASS_PLAYER  = 4
CLASS_REF     = 5
REF_ID_OFFSET = 10_000

TRAIL_LEN = 45   # frames stored per trail

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (235, 110,  40),   # Team A — vivid blue
    1:  ( 40, 200,  60),   # Team B — vivid green
    2:  ( 50,  50, 220),   # Referee — red
    -1: (160, 160, 160),   # Unknown — grey
}

# ── Kalman helpers ────────────────────────────────────────────────────────────

def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → column vector [cx, cy, w, h]."""
    w  = float(bbox[2] - bbox[0])
    h  = float(bbox[3] - bbox[1])
    cx = float(bbox[0]) + w * 0.5
    cy = float(bbox[1]) + h * 0.5
    return np.array([[cx], [cy], [w], [h]], dtype=np.float32)


def _z_to_bbox(state: np.ndarray) -> np.ndarray:
    """Kalman state [cx, cy, w, h, …] → [x1,y1,x2,y2]."""
    # cv2.KalmanFilter returns (8,1) shaped arrays — index with [i][0]
    cx = float(state[0][0]) if state.ndim == 2 else float(state[0])
    cy = float(state[1][0]) if state.ndim == 2 else float(state[1])
    w  = max(float(state[2][0]) if state.ndim == 2 else float(state[2]), 1.0)
    h  = max(float(state[3][0]) if state.ndim == 2 else float(state[3]), 1.0)
    return np.array([cx - w*0.5, cy - h*0.5, cx + w*0.5, cy + h*0.5],
                    dtype=np.float32)


def _build_kalman() -> cv2.KalmanFilter:
    """
    8-state constant-velocity Kalman filter.
    State:        [cx, cy, w, h,  vcx, vcy, vw, vh]
    Measurement:  [cx, cy, w, h]
    """
    kf = cv2.KalmanFilter(8, 4)

    # State transition — position += velocity per frame
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0,  1, 0, 0, 0],
        [0, 1, 0, 0,  0, 1, 0, 0],
        [0, 0, 1, 0,  0, 0, 1, 0],
        [0, 0, 0, 1,  0, 0, 0, 1],
        [0, 0, 0, 0,  1, 0, 0, 0],
        [0, 0, 0, 0,  0, 1, 0, 0],
        [0, 0, 0, 0,  0, 0, 1, 0],
        [0, 0, 0, 0,  0, 0, 0, 1],
    ], dtype=np.float32)

    # Measurement matrix — observe cx, cy, w, h directly
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
        kf.measurementMatrix[i, i] = 1.0

    # Process noise — how much each state element may change per frame
    #   Position (cx,cy) : allow moderate drift
    #   Size (w,h)       : players don't change size quickly
    #   Velocity (v*)    : allow moderate velocity change (basketball is fast)
    kf.processNoiseCov = np.diag(
        [2.0, 2.0, 1.0, 1.0,  10.0, 10.0, 1.0, 1.0]
    ).astype(np.float32)

    # Measurement noise — RT-DETR bbox noise
    kf.measurementNoiseCov = np.diag(
        [4.0, 4.0, 16.0, 16.0]
    ).astype(np.float32)

    # High initial uncertainty — especially on velocities
    P = np.diag([10.0, 10.0, 10.0, 10.0,
                 1e4,  1e4,  1e4,  1e4]).astype(np.float32)
    kf.errorCovPost = P.copy()
    kf.errorCovPre  = P.copy()

    return kf


# ── Vectorised IoU ────────────────────────────────────────────────────────────

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between all pairs.  a: (N,4), b: (M,4) → (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ── Appearance ────────────────────────────────────────────────────────────────

def _extract_histogram(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """
    48-dim HSV histogram from the jersey (torso) region.
    Layout:  [0..31] hue histogram (coloured pixels only)
             [32..47] saturation histogram (all pixels)

    Coloured-pixel mask (S>40, V>40) excludes shadows and white jerseys from
    the hue channel, so two white-jersey teams don't collide on hue.
    The saturation channel captures white vs coloured: white jerseys push
    saturation toward zero, giving a distinct fingerprint.
    """
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(frame.shape[1] - 1, int(bbox[2]))
    y2 = min(frame.shape[0] - 1, int(bbox[3]))

    if x2 - x1 < 16 or y2 - y1 < 32:
        return None

    crop  = frame[y1:y2, x1:x2]
    bh    = crop.shape[0]
    torso = crop[int(bh * 0.15): int(bh * 0.55), :]
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Hue histogram — coloured pixels only
    col_mask = (S > 40) & (V > 40)
    h_pix    = H[col_mask].reshape(-1, 1).astype(np.float32)
    if len(h_pix) >= 20:
        h_hist = cv2.calcHist([h_pix], [0], None, [32], [0, 180]).flatten()
        hs = h_hist.sum()
        if hs > 0:
            h_hist /= hs
    else:
        h_hist = np.zeros(32, dtype=np.float32)

    # Saturation histogram — all pixels
    s_pix  = S.reshape(-1, 1).astype(np.float32)
    s_hist = cv2.calcHist([s_pix], [0], None, [16], [0, 256]).flatten()
    ss = s_hist.sum()
    if ss > 0:
        s_hist /= ss

    return np.concatenate([h_hist, s_hist]).astype(np.float32)


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two normalised histograms.  [0,1]."""
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


# ── KalmanBoxTracker ──────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Single-object tracker — one Kalman filter per person.

    Maintains:
      kf             : Kalman filter for smooth bbox prediction
      obs_history    : ring-buffer of real observations for OCM velocity
      histogram      : EMA of 48-dim HSV features for Re-ID
      trail          : recent centre points for visualisation
    """

    def __init__(
        self,
        bbox:      np.ndarray,
        class_id:  int,
        conf:      float,
        track_id:  int,
        frame_idx: int,
        delta_t:   int = 3,
    ) -> None:
        self.track_id  = track_id
        self.class_id  = class_id
        self.conf      = conf

        # Kalman filter — initialise state from first detection
        self.kf = _build_kalman()
        z = _bbox_to_z(bbox)  # shape (4,1)
        state = np.zeros((8, 1), dtype=np.float32)
        state[0, 0] = z[0, 0]   # cx
        state[1, 0] = z[1, 0]   # cy
        state[2, 0] = z[2, 0]   # w
        state[3, 0] = z[3, 0]   # h
        # velocities stay 0 — we don't know initial velocity
        self.kf.statePost = state.copy()
        self.kf.statePre  = state.copy()

        # Lifecycle counters
        self.age               = 0   # total frames since creation
        self.hits              = 1   # number of real detections
        self.hit_streak        = 1   # consecutive detections
        self.time_since_update = 0   # frames since last real detection

        # OCM: ring buffer of (frame_idx, bbox) for real observations
        self.delta_t   = delta_t
        self.obs_hist: deque[tuple[int, np.ndarray]] = deque(maxlen=delta_t + 1)
        self.obs_hist.append((frame_idx, bbox.astype(np.float32)))
        self.last_obs_bbox  = bbox.astype(np.float32)
        self.last_obs_frame = frame_idx

        # Appearance — EMA of 48-dim HSV histograms
        self.histogram: np.ndarray | None = None

        # Trail of (cx, cy) for drawing
        self.trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_LEN)
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        self.trail.append((cx, cy))

    # ── Kalman lifecycle ───────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """
        Kalman predict step.
        Called EVERY frame regardless of whether a detection is available.
        Returns the predicted [x1,y1,x2,y2].
        """
        self.kf.predict()
        self.age               += 1
        self.time_since_update += 1
        if self.time_since_update > 1:
            self.hit_streak = 0
        return self.get_state()

    def update(self, bbox: np.ndarray, conf: float, frame_idx: int) -> None:
        """
        Kalman correct step with a real detection.
        Stores observation for OCM and updates appearance trail.
        """
        self.conf               = conf
        self.hits              += 1
        self.hit_streak        += 1
        self.time_since_update  = 0
        self.last_obs_bbox      = bbox.astype(np.float32)
        self.last_obs_frame     = frame_idx
        self.obs_hist.append((frame_idx, bbox.astype(np.float32)))

        self.kf.correct(_bbox_to_z(bbox))

        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        self.trail.append((cx, cy))

    def update_oru(self, bbox: np.ndarray, conf: float, frame_idx: int) -> None:
        """
        OC-SORT Observation-Centric Re-Update (ORU).

        When a track is recovered after N missing frames, linearly interpolate
        N intermediate observations between the last real detection and the
        current one, feeding each through the Kalman filter.  This gradually
        corrects the velocity without a sharp discontinuity.

        Without ORU: Kalman correction produces a velocity shock → wrong
        predictions in the next chunk → ID switch on the next frame.

        With ORU: velocity is smoothly re-aligned, tracking stays stable.
        """
        n_miss = self.time_since_update   # e.g. 5 if absent for 5 frames
        if n_miss > 1:
            old = self.last_obs_bbox
            for step in range(1, n_miss):
                alpha  = step / n_miss
                interp = (old * (1.0 - alpha) + bbox * alpha).astype(np.float32)
                self.kf.predict()
                self.kf.correct(_bbox_to_z(interp))

        # Final update with the actual detection
        self.update(bbox, conf, frame_idx)

    # ── OCM prediction ─────────────────────────────────────────────────────────

    def get_ocm_bbox(self, frame_idx: int) -> np.ndarray:
        """
        OC-SORT Observation-Centric Momentum (OCM).

        Returns a predicted bbox using velocity computed from the last
        delta_t REAL observations rather than Kalman predicted velocity.
        Real-observation velocity is accurate even during long occlusions
        because it is derived from detector outputs, not drifting predictions.

        If fewer than 2 real observations are available, falls back to the
        Kalman prediction.
        """
        if len(self.obs_hist) < 2:
            return self.get_state()

        fi_old, bb_old = self.obs_hist[0]
        fi_new, bb_new = self.obs_hist[-1]
        dt = max(1.0, float(fi_new - fi_old))

        vel = (bb_new - bb_old) / dt                  # pixels per frame
        frames_ahead = frame_idx - self.last_obs_frame
        pred = self.last_obs_bbox + vel * frames_ahead

        # Sanity check — if size has collapsed, use Kalman instead
        if (pred[2] - pred[0]) < 4 or (pred[3] - pred[1]) < 4:
            return self.get_state()

        return pred.astype(np.float32)

    # ── State accessors ────────────────────────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """Current Kalman posterior as [x1,y1,x2,y2]."""
        return _z_to_bbox(self.kf.statePost)

    @property
    def center(self) -> tuple[float, float]:
        bbox = self.get_state()
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    def update_histogram(self, hist: np.ndarray) -> None:
        """EMA update of appearance histogram.  α=0.10 → slow adaptation."""
        if self.histogram is None:
            self.histogram = hist.copy()
        else:
            self.histogram = 0.90 * self.histogram + 0.10 * hist


# ── Re-ID gallery ─────────────────────────────────────────────────────────────

@dataclass
class _GalleryEntry:
    track_id:   int
    histogram:  np.ndarray       # 48-dim HSV
    last_bbox:  np.ndarray       # [x1,y1,x2,y2] when track was lost
    last_frame: int
    class_id:   int
    hits:       int              # total confirmed frames — quality proxy


class ReIDGallery:
    """
    Archive of lost tracks for long-term re-identification.

    Matching cost:
        combined = w_app × Bhattacharyya(hist_new, hist_stored)
                 + w_pos × min(1, dist / max_pos_dist)

    A match is accepted when combined < reid_threshold.
    Gallery entries expire after reid_max_age frames.
    """

    def __init__(
        self,
        reid_max_age:    int   = 150,   # ~5 s at 30 fps
        reid_threshold:  float = 0.40,
        w_appearance:    float = 0.65,
        w_position:      float = 0.35,
        max_pos_dist:    float = 500.0, # pixels — anything further → pos_cost=1
        min_hits:        int   = 5,     # minimum hits to enter gallery
    ) -> None:
        self.max_age        = reid_max_age
        self.threshold      = reid_threshold
        self.w_app          = w_appearance
        self.w_pos          = w_position
        self.max_pos_dist   = max_pos_dist
        self.min_hits       = min_hits
        self._store: dict[int, _GalleryEntry] = {}

    def add(self, track: KalmanBoxTracker, frame_idx: int) -> None:
        """Archive a lost track.  Only confirmed tracks enter the gallery."""
        if track.histogram is None:
            return
        if track.hits < self.min_hits:
            return
        self._store[track.track_id] = _GalleryEntry(
            track_id   = track.track_id,
            histogram  = track.histogram.copy(),
            last_bbox  = track.get_state().copy(),
            last_frame = frame_idx,
            class_id   = track.class_id,
            hits       = track.hits,
        )

    def match(
        self,
        histogram: np.ndarray | None,
        bbox:      np.ndarray,
        class_id:  int,
        frame_idx: int,
    ) -> tuple[int | None, float]:
        """
        Find the best gallery match for a new detection.
        Returns (track_id, cost) where cost < threshold means a match.
        Returns (None, inf) if gallery is empty or no match found.
        """
        if not self._store or histogram is None:
            return None, float("inf")

        cx_new = (bbox[0] + bbox[2]) * 0.5
        cy_new = (bbox[1] + bbox[3]) * 0.5

        best_id, best_cost = None, float("inf")

        for entry in self._store.values():
            if entry.class_id != class_id:
                continue

            # Appearance cost [0, 1]
            app_cost = _hist_distance(histogram, entry.histogram)

            # Position cost [0, 1]
            cx_old = (entry.last_bbox[0] + entry.last_bbox[2]) * 0.5
            cy_old = (entry.last_bbox[1] + entry.last_bbox[3]) * 0.5
            pos_d  = float(np.hypot(cx_new - cx_old, cy_new - cy_old))
            pos_cost = min(1.0, pos_d / max(self.max_pos_dist, 1.0))

            cost = self.w_app * app_cost + self.w_pos * pos_cost
            if cost < best_cost:
                best_cost = cost
                best_id   = entry.track_id

        if best_cost < self.threshold:
            return best_id, best_cost
        return None, best_cost

    def remove(self, track_id: int) -> None:
        self._store.pop(track_id, None)

    def cleanup(self, frame_idx: int) -> None:
        """Remove entries that have been in the gallery too long."""
        expired = [
            tid for tid, e in self._store.items()
            if frame_idx - e.last_frame > self.max_age
        ]
        for tid in expired:
            del self._store[tid]

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"ReIDGallery(entries={len(self._store)}, threshold={self.threshold})"


# ── OCSortTracker ─────────────────────────────────────────────────────────────

class OCSortTracker:
    """
    OC-SORT tracker with appearance-based Re-ID.

    Per-frame pipeline
    ──────────────────
    1. Predict all active tracks (Kalman + OCM for cost matrix).
    2. First association: Hungarian on IoU(OCM_pred, high_conf_dets).
    3. Second association: Hungarian on IoU(Kalman_pred, low_conf_dets)
       for still-unmatched tracks.
    4. Unmatched tracks: age them; archive in Re-ID gallery when max_age hit.
    5. Unmatched high-conf detections:
         a. Compute appearance histogram.
         b. Query Re-ID gallery — if match found, reuse old track_id (ORU).
         c. Otherwise create new track.
    6. Update histogram EMA for all matched tracks.
    7. Collect and return output dicts.

    Parameters
    ──────────
    max_age          : Frames a track can be absent before gallery archival.
    min_hits         : Detections before a track appears in output.
    iou_threshold    : Primary association IoU threshold.
    iou_threshold_2  : Secondary association IoU threshold (relaxed).
    delta_t          : Real-observation lookback for OCM velocity.
    reid_max_age     : Frames to keep gallery entries.
    reid_threshold   : Combined cost threshold for Re-ID acceptance.
    min_hits_gallery : Minimum hits before a track enters the gallery.
    """

    def __init__(
        self,
        max_age:          int   = 90,
        min_hits:         int   = 3,
        iou_threshold:    float = 0.30,
        iou_threshold_2:  float = 0.10,
        delta_t:          int   = 3,
        reid_max_age:     int   = 300,
        reid_threshold:   float = 0.55,
        min_hits_gallery: int   = 3,
        max_players:      int   = 10,
        max_refs:         int   = 3,
    ) -> None:
        self.max_age         = max_age
        self.min_hits        = min_hits
        self.iou_thr         = iou_threshold
        self.iou_thr_2       = iou_threshold_2
        self.delta_t         = delta_t
        self.max_players     = max_players
        self.max_refs        = max_refs

        # Active tracks keyed by track_id
        self._tracks: dict[int, KalmanBoxTracker] = {}

        # Re-ID gallery
        self._gallery = ReIDGallery(
            reid_max_age   = reid_max_age,
            reid_threshold = reid_threshold,
            min_hits       = min_hits_gallery,
        )

        # ID counters — separate ranges for players and refs
        self._next_player_id = 1
        self._next_ref_id    = REF_ID_OFFSET + 1

        # Ref display-number (1, 2, 3 for annotations)
        self._ref_display: dict[int, int] = {}
        self._next_ref_disp = 1

        # Statistics
        self.total_player_tracks = 0
        self.total_ref_tracks    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        frame:      np.ndarray,
        detections: list[dict],
        frame_idx:  int,
    ) -> list[dict]:
        """
        Main per-frame entry point.

        Parameters
        ----------
        frame      : BGR video frame
        detections : list of dicts from RT-DETR detector
                     each dict must have: bbox (4,), conf, class_id
        frame_idx  : global frame index (0-based)

        Returns
        -------
        List of tracking result dicts:
            track_id, bbox, center, class_id, conf, team_id, source
        """
        self._gallery.cleanup(frame_idx)

        output: list[dict] = []
        for class_id in (CLASS_PLAYER, CLASS_REF):
            class_dets   = [d for d in detections if d["class_id"] == class_id]
            class_tracks = [t for t in self._tracks.values()
                            if t.class_id == class_id]
            output.extend(
                self._update_class(frame, class_dets, class_tracks,
                                   class_id, frame_idx)
            )
        return output

    # ── Per-class update ──────────────────────────────────────────────────────

    def _update_class(
        self,
        frame:       np.ndarray,
        dets:        list[dict],
        tracks:      list[KalmanBoxTracker],
        class_id:    int,
        frame_idx:   int,
    ) -> list[dict]:

        # ── Step 1: Predict all tracks ─────────────────────────────────────
        pred_kalman = []   # Kalman predictions (for second association)
        pred_ocm    = []   # OCM  predictions  (for first  association)
        for t in tracks:
            pred_kalman.append(t.predict())          # advances Kalman state
            pred_ocm.append(t.get_ocm_bbox(frame_idx))  # OCM velocity estimate

        # ── Step 2: Split detections into high and low confidence ──────────
        high_dets = [d for d in dets if d.get("conf", 1.0) >= 0.45]
        low_dets  = [d for d in dets if d.get("conf", 1.0) <  0.45]

        high_bboxes = np.array([d["bbox"][:4] for d in high_dets],
                               dtype=np.float32) if high_dets else np.empty((0, 4))
        low_bboxes  = np.array([d["bbox"][:4] for d in low_dets],
                               dtype=np.float32) if low_dets  else np.empty((0, 4))

        # ── Step 3: First association — OCM preds ↔ high-conf dets ────────
        matches1, unmatched_t, unmatched_d_high = self._associate(
            [np.array(p) for p in pred_ocm], high_bboxes, self.iou_thr
        )

        matched_track_set: set[int] = set()
        for t_idx, d_idx in matches1:
            track = tracks[t_idx]
            det   = high_dets[d_idx]
            bbox  = det["bbox"][:4].astype(np.float32)

            # ORU if track was absent for multiple frames
            if track.time_since_update > 1:
                track.update_oru(bbox, det.get("conf", 1.0), frame_idx)
            else:
                track.update(bbox, det.get("conf", 1.0), frame_idx)

            # Appearance update
            hist = _extract_histogram(frame, bbox)
            if hist is not None:
                track.update_histogram(hist)

            matched_track_set.add(t_idx)

        # ── Step 4: Second association — Kalman preds ↔ low-conf dets ─────
        rem_tracks  = [tracks[i] for i in unmatched_t]
        rem_kalman  = [pred_kalman[i] for i in unmatched_t]

        if rem_tracks and len(low_dets) > 0:
            matches2, unmatched_t2, _ = self._associate(
                [np.array(p) for p in rem_kalman], low_bboxes, self.iou_thr_2
            )
            for t_idx2, d_idx2 in matches2:
                track = rem_tracks[t_idx2]
                det   = low_dets[d_idx2]
                bbox  = det["bbox"][:4].astype(np.float32)
                track.update(bbox, det.get("conf", 1.0), frame_idx)
                hist = _extract_histogram(frame, bbox)
                if hist is not None:
                    track.update_histogram(hist)
                # Mark the original index in `tracks` as matched
                matched_track_set.add(unmatched_t[t_idx2])

            final_unmatched_t = [unmatched_t[i] for i in unmatched_t2]
        else:
            final_unmatched_t = list(unmatched_t)

        # ── Step 5: Handle unmatched tracks ────────────────────────────────
        to_delete: list[int] = []
        for t_idx in final_unmatched_t:
            track = tracks[t_idx]
            if track.time_since_update > self.max_age:
                self._gallery.add(track, frame_idx)
                to_delete.append(track.track_id)
        for tid in to_delete:
            self._tracks.pop(tid, None)

        # ── Step 6: Handle unmatched high-conf detections ──────────────────
        #
        # Basketball has a known max number of players (10) and refs (3).
        # When we are AT the cap, every unmatched detection MUST be a
        # returning player — force Re-ID even if the gallery cost exceeds the
        # normal threshold, taking the best available match.
        # This is the single most important change to stop runaway track counts.

        used_gallery: set[int] = set()

        cap          = self.max_players if class_id == CLASS_PLAYER else self.max_refs
        n_active     = sum(1 for t in self._tracks.values()
                           if t.class_id == class_id)
        n_gallery    = sum(1 for e in self._gallery._store.values()
                           if e.class_id == class_id)
        at_cap       = (n_active + n_gallery) >= cap

        for d_idx in unmatched_d_high:
            det  = high_dets[d_idx]
            bbox = det["bbox"][:4].astype(np.float32)
            hist = _extract_histogram(frame, bbox)

            # Query Re-ID gallery
            old_id, cost = self._gallery.match(hist, bbox, class_id, frame_idx)

            # If at cap: force Re-ID — accept best match regardless of cost
            if at_cap and old_id is None:
                # Find best gallery match ignoring the threshold
                best_id, best_cost = None, float("inf")
                cx = (bbox[0] + bbox[2]) * 0.5
                cy = (bbox[1] + bbox[3]) * 0.5
                for entry in self._gallery._store.values():
                    if entry.class_id != class_id:
                        continue
                    if entry.track_id in used_gallery:
                        continue
                    app_c = (_hist_distance(hist, entry.histogram)
                             if hist is not None else 0.5)
                    ecx = (entry.last_bbox[0] + entry.last_bbox[2]) * 0.5
                    ecy = (entry.last_bbox[1] + entry.last_bbox[3]) * 0.5
                    pos_c = min(1.0, float(np.hypot(cx-ecx, cy-ecy)) / 500.0)
                    c = 0.65 * app_c + 0.35 * pos_c
                    if c < best_cost:
                        best_cost = c
                        best_id   = entry.track_id
                if best_id is not None:
                    old_id, cost = best_id, best_cost

            if old_id is not None and old_id not in used_gallery:
                # ── Player has returned — reuse their original ID ──────────
                track = self._create_track(
                    bbox, class_id, det.get("conf", 1.0),
                    frame_idx, track_id=old_id
                )
                if hist is not None:
                    track.update_histogram(hist)
                self._gallery.remove(old_id)
                used_gallery.add(old_id)
                disp = old_id - REF_ID_OFFSET if class_id == CLASS_REF else old_id
                forced = " [forced]" if at_cap else ""
                print(f"[OCSortTracker] Re-ID  ← track {disp} "
                      f"re-acquired at frame {frame_idx}  "
                      f"(cost={cost:.3f}{forced})")

            elif not at_cap:
                # ── Under cap: genuinely new player/ref ───────────────────
                track = self._create_track(
                    bbox, class_id, det.get("conf", 1.0),
                    frame_idx, track_id=None
                )
                if hist is not None:
                    track.update_histogram(hist)
                if class_id == CLASS_PLAYER:
                    self.total_player_tracks += 1
                else:
                    self.total_ref_tracks    += 1
            # else: at cap and no gallery entry → ignore spurious detection

        # ── Step 7: Collect output ─────────────────────────────────────────
        return self._collect_output(class_id)

    # ── Assignment ────────────────────────────────────────────────────────────

    def _associate(
        self,
        pred_bboxes: list[np.ndarray],
        det_bboxes:  np.ndarray,
        threshold:   float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Hungarian assignment between predicted bboxes and detections.

        Returns
        -------
        matches         : list of (track_index, det_index) pairs
        unmatched_tracks: track indices with no valid detection
        unmatched_dets  : detection indices with no valid track
        """
        n_t = len(pred_bboxes)
        n_d = len(det_bboxes)

        if n_t == 0:
            return [], [], list(range(n_d))
        if n_d == 0:
            return [], list(range(n_t)), []

        pred_arr = np.array(pred_bboxes, dtype=np.float32)
        iou_mat  = _iou_batch(pred_arr, det_bboxes)   # (n_t, n_d)
        cost_mat = 1.0 - iou_mat

        if _SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
        else:
            # Greedy fallback — sort all pairs by cost, pick greedily
            all_pairs = sorted(
                [(r, c) for r in range(n_t) for c in range(n_d)],
                key=lambda rc: cost_mat[rc[0], rc[1]]
            )
            used_r, used_c = set(), set()
            row_ind, col_ind = [], []
            for r, c in all_pairs:
                if r not in used_r and c not in used_c:
                    row_ind.append(r)
                    col_ind.append(c)
                    used_r.add(r)
                    used_c.add(c)

        matches:       list[tuple[int, int]] = []
        matched_t:     set[int]              = set()
        matched_d:     set[int]              = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= threshold:
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)

        unmatched_t = [r for r in range(n_t) if r not in matched_t]
        unmatched_d = [c for c in range(n_d) if c not in matched_d]
        return matches, unmatched_t, unmatched_d

    # ── Track creation ────────────────────────────────────────────────────────

    def _create_track(
        self,
        bbox:      np.ndarray,
        class_id:  int,
        conf:      float,
        frame_idx: int,
        track_id:  int | None,
    ) -> KalmanBoxTracker:
        """Allocate a KalmanBoxTracker and register it."""
        if track_id is None:
            if class_id == CLASS_PLAYER:
                track_id = self._next_player_id
                self._next_player_id += 1
            else:
                track_id = self._next_ref_id
                self._next_ref_id += 1

        if class_id == CLASS_REF and track_id not in self._ref_display:
            self._ref_display[track_id] = self._next_ref_disp
            self._next_ref_disp += 1

        track = KalmanBoxTracker(
            bbox=bbox, class_id=class_id, conf=conf,
            track_id=track_id, frame_idx=frame_idx, delta_t=self.delta_t,
        )
        self._tracks[track_id] = track
        return track

    # ── Output collection ─────────────────────────────────────────────────────

    def _collect_output(self, class_id: int) -> list[dict]:
        """
        Assemble output dicts for confirmed tracks that were updated this frame.

        A track is included when:
          - It was detected this frame (time_since_update == 0)
          - It has reached min_hits confirmations (or is brand-new)
        """
        output = []
        for tid, track in self._tracks.items():
            if track.class_id   != class_id:
                continue
            if track.time_since_update > 0:
                continue   # no detection this frame — skip
            if track.hits < self.min_hits and track.age > self.min_hits:
                continue   # not yet confirmed

            bbox = track.get_state()
            cx   = (bbox[0] + bbox[2]) * 0.5
            cy   = (bbox[1] + bbox[3]) * 0.5

            output.append({
                "track_id":  tid,
                "bbox":      bbox,
                "center":    (cx, cy),
                "class_id":  class_id,
                "conf":      track.conf,
                "team_id":   -1,      # filled by TeamClusterer
                "source":    "ocsort",
            })
        return output

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame:       np.ndarray,
        frame_dets:  list[dict],
        show_trails: bool = True,
        show_ids:    bool = True,
        show_teams:  bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Annotate a frame with bounding boxes, IDs, team labels, and trails."""
        vis = frame.copy()

        for det in frame_dets:
            tid     = det["track_id"]
            cid     = det["class_id"]
            team_id = det.get("team_id", -1)
            bbox    = det["bbox"]
            x1, y1  = int(bbox[0]), int(bbox[1])
            x2, y2  = int(bbox[2]), int(bbox[3])

            # Color and label
            if cid == CLASS_REF:
                color = TEAM_COLORS[2]
                disp  = self._ref_display.get(tid, tid - REF_ID_OFFSET)
                label = f"Ref#{disp}  Referee"
            else:
                color = TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
                label = f"P#{tid}"
                if show_teams and team_id in (0, 1):
                    label += f"  Team {'A' if team_id == 0 else 'B'}"

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label background + text
            if show_ids:
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              color, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            1, cv2.LINE_AA)

            # Motion trail
            if show_trails:
                track = self._tracks.get(tid)
                if track is not None:
                    trail = list(track.trail)
                    n     = len(trail)
                    for i in range(1, n):
                        alpha = i / n
                        if alpha < 0.25:
                            continue
                        p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                        p2 = (int(trail[i][0]),   int(trail[i][1]))
                        # Skip physically impossible jumps (Re-ID warp)
                        if np.hypot(p2[0]-p1[0], p2[1]-p1[1]) > 80:
                            continue
                        c = tuple(int(v * alpha) for v in color)
                        cv2.line(vis, p1, p2, c,
                                 max(1, int(3 * alpha)), cv2.LINE_AA)

        return vis

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_active(self) -> int:
        return len(self._tracks)

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    def __repr__(self) -> str:
        return (
            f"OCSortTracker("
            f"active={self.n_active}, "
            f"gallery={self.gallery_size}, "
            f"players={self.total_player_tracks}, "
            f"refs={self.total_ref_tracks})"
        )