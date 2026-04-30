"""
src/tracking/tracker.py
────────────────────────
Multi-Object Tracking (MOT) for basketball players and referees.

Wraps ByteTrack (via the `supervision` library) with a clean interface
that accepts detection dicts from BasketballDetector.parse() and returns
the same dicts enriched with `track_id` and `team_id`.

Output dict schema  (superset of BasketballDetector.parse() output)
────────────────────────────────────────────────────────────────────
  bbox        : np.ndarray  shape (4,)  [x1, y1, x2, y2]
  center      : tuple       (cx, cy)
  conf        : float
  class_id    : int
  class_name  : str
  track_id    : int         ← NEW: persistent ID across frames
  team_id     : int         ← NEW: TEAM_UNKNOWN(-1) until TeamClusterer fills it

Design decisions
────────────────
• Two separate ByteTrack instances — one for CLASS_PLAYER, one for CLASS_REF.
  Refs and players have very different speeds and densities, so independent
  motion models give cleaner results.

• REF_ID_OFFSET = 10_000 is added to all referee track IDs.
  This guarantees player track #3 and referee track #3 never share an ID,
  which is critical for TeamClusterer's {track_id: team_id} dict.

• Ball (CLASS_BALL) is intentionally excluded — BallTracker in
  src/detection/ball_tracker.py handles it with SAHI + interpolation.
  Ball dets are passed through with track_id = -1.

• team_id is seeded as TEAM_UNKNOWN here and populated by TeamClusterer
  in the main pipeline loop.  This module is tracking-only.

• IoU-based bbox matching maps supervision's output back to the original
  detection dicts so every key from BasketballDetector.parse() is preserved.

Pipeline integration
────────────────────
  detector  = BasketballDetector(...)
  tracker   = PlayerTracker()
  clusterer = TeamClusterer()

  for frame_idx, frame in enumerate(video):
      raw_dets     = detector.parse(detector.detect(frame))
      tracked_dets = tracker.update(raw_dets)          # adds track_id
      clusterer.update(frame, tracked_dets)            # fills team_id in-place
      for det in tracked_dets:
          det["team_id"] = clusterer.get_team(det["track_id"])

Requires
────────
  pip install supervision>=0.20.0
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict

try:
    import supervision as sv
except ImportError as exc:
    raise ImportError(
        "supervision is required for tracking.\n"
        "Install with:  pip install supervision>=0.20.0"
    ) from exc

from src.team_clustering.clusterer import (
    CLASS_PLAYER,
    CLASS_REF,
    TEAM_UNKNOWN,
    TEAM_COLORS,
    TEAM_NAMES,
)
from src.detection.detector import CLASS_ID_TO_NAME, _DEFAULT_CLASS_COLORS


# ── Constants ─────────────────────────────────────────────────────────────────

# Referee track IDs are offset by this value to prevent collisions with
# player track IDs inside TeamClusterer's {track_id: team_id} mapping.
REF_ID_OFFSET = 10_000

# How many past (cx, cy) positions to keep per track for trail drawing.
TRAIL_LENGTH = 30


# ── Version-safe ByteTrack factory ───────────────────────────────────────────
# supervision renamed ByteTrack's constructor parameters in v0.19+:
#   Old (≤0.18):  track_thresh, track_buffer, match_thresh
#   New (≥0.19):  track_activation_threshold, lost_track_buffer,
#                 minimum_matching_threshold
# We try the new names first and fall back to the old ones automatically.

def _make_bytetrack(
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    frame_rate:   int,
) -> sv.ByteTrack:
    try:
        return sv.ByteTrack(
            track_activation_threshold = track_thresh,
            lost_track_buffer          = track_buffer,
            minimum_matching_threshold = match_thresh,
            frame_rate                 = frame_rate,
        )
    except TypeError:
        # Older supervision (≤ 0.18) uses the original parameter names
        return sv.ByteTrack(
            track_thresh = track_thresh,
            track_buffer = track_buffer,
            match_thresh = match_thresh,
            frame_rate   = frame_rate,
        )


# ─────────────────────────────────────────────────────────────────────────────
class PlayerTracker:
    """
    Multi-object tracker for basketball players and referees.

    Wraps two ByteTrack instances so player and referee tracks share no IDs
    and can be tuned independently.

    Parameters
    ----------
    track_thresh : float
        Minimum detection confidence to initialise a new track.
        Lower values increase recall at the cost of more false tracks.
    track_buffer : int
        Frames a track can survive without a matching detection before
        being removed.  30 frames ≈ 1 s at 30 fps.
    match_thresh : float
        IoU threshold for the Hungarian assignment step.
        Higher = stricter matching, fewer ID switches on crowded plays.
    frame_rate   : int
        Source video FPS.  ByteTrack uses this to scale its Kalman filter.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int   = 30,
        match_thresh: float = 0.80,
        frame_rate:   int   = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate   = frame_rate

        # ── Two ByteTrack instances ───────────────────────────────────────────
        self._player_tracker = _make_bytetrack(
            track_thresh, track_buffer, match_thresh, frame_rate
        )
        # Refs move slower and are fewer → tighter buffer avoids ghost tracks
        self._ref_tracker = _make_bytetrack(
            track_thresh, max(10, track_buffer // 2), match_thresh, frame_rate
        )

        # Position history per track_id → used for trail drawing and analytics
        # {track_id: [(cx, cy), ...]}
        self._history: dict[int, list[tuple[float, float]]] = defaultdict(list)

        # Cumulative sets of unique IDs seen (survives a reset-free clip)
        self._all_player_ids: set[int] = set()
        self._all_ref_ids:    set[int] = set()

        self._frame_idx: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Run tracking on one frame's worth of detections.

        Parameters
        ----------
        detections : list[dict]
            Direct output of BasketballDetector.parse().
            Must contain keys: bbox, center, conf, class_id, class_name.

        Returns
        -------
        list[dict]
            Same dicts with two new keys:
              track_id : int  Persistent ID across frames.
                              Players:   1 – 9 999
                              Referees:  10 001 – 19 999  (REF_ID_OFFSET applied)
                              Ball/other: -1  (untracked)
              team_id  : int  TEAM_UNKNOWN (-1) — filled by TeamClusterer later.

        Notes
        -----
        The returned list is ordered as:  tracked players, tracked refs, other.
        Ball detections are appended at the end with track_id = -1.
        """
        # ── 1. Partition by class ─────────────────────────────────────────────
        player_dets = [d for d in detections if d["class_id"] == CLASS_PLAYER]
        ref_dets    = [d for d in detections if d["class_id"] == CLASS_REF]
        other_dets  = [
            d for d in detections
            if d["class_id"] not in (CLASS_PLAYER, CLASS_REF)
        ]

        # ── 2. Track each class independently ────────────────────────────────
        tracked_players = self._run_tracker(
            self._player_tracker, player_dets, id_offset=0
        )
        tracked_refs = self._run_tracker(
            self._ref_tracker, ref_dets, id_offset=REF_ID_OFFSET
        )

        # ── 3. Pass-through for ball & other classes (no tracking) ────────────
        passthrough = [
            {**d, "track_id": -1, "team_id": TEAM_UNKNOWN}
            for d in other_dets
        ]

        # ── 4. Update position history and ID registry ────────────────────────
        for det in tracked_players:
            tid = det["track_id"]
            if tid != -1:
                self._append_history(tid, det["center"])
                self._all_player_ids.add(tid)

        for det in tracked_refs:
            tid = det["track_id"]
            if tid != -1:
                self._append_history(tid, det["center"])
                self._all_ref_ids.add(tid)

        self._frame_idx += 1
        return tracked_players + tracked_refs + passthrough

    def get_tracked_players(self, tracked_dets: list[dict]) -> list[dict]:
        """Return only CLASS_PLAYER dets that have an active track_id."""
        return [
            d for d in tracked_dets
            if d["class_id"] == CLASS_PLAYER and d.get("track_id", -1) != -1
        ]

    def get_tracked_refs(self, tracked_dets: list[dict]) -> list[dict]:
        """Return only CLASS_REF dets that have an active track_id."""
        return [
            d for d in tracked_dets
            if d["class_id"] == CLASS_REF and d.get("track_id", -1) != -1
        ]

    def get_track_history(self, track_id: int) -> list[tuple[float, float]]:
        """
        Return the stored position history [(cx, cy), ...] for a track.

        Useful for the analytics module (distance, speed calculation).
        """
        return self._history.get(track_id, [])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_player_tracks(self) -> int:
        """Cumulative unique player IDs seen since last reset()."""
        return len(self._all_player_ids)

    @property
    def total_ref_tracks(self) -> int:
        """Cumulative unique referee IDs seen since last reset()."""
        return len(self._all_ref_ids)

    @property
    def frame_count(self) -> int:
        """Number of frames processed since last reset()."""
        return self._frame_idx

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame:        np.ndarray,
        tracked_dets: list[dict],
        show_trails:  bool = True,
        show_ids:     bool = True,
        show_teams:   bool = True,
        show_conf:    bool = False,
    ) -> np.ndarray:
        """
        Annotate a copy of `frame` with bounding boxes, IDs, team labels,
        and optional motion trails.

        Parameters
        ----------
        frame        : BGR frame (not modified in-place).
        tracked_dets : Output of update().
        show_trails  : Draw fading motion trail behind each track.
        show_ids     : Overlay the track ID on each box.
        show_teams   : Colour boxes by team assignment (requires clusterer
                       to have populated team_id in the dicts).
        show_conf    : Append detection confidence to the label.

        Returns
        -------
        Annotated BGR frame.
        """
        vis = frame.copy()

        for det in tracked_dets:
            tid     = det.get("track_id", -1)
            team_id = det.get("team_id", TEAM_UNKNOWN)
            cid     = det["class_id"]

            # Ball / untracked → skip (BallTracker handles ball drawing)
            if tid == -1:
                continue

            # ── Colour: team colour if assigned, else class default ────────
            color = (
                TEAM_COLORS.get(team_id)
                if show_teams and team_id != TEAM_UNKNOWN
                else _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180))
            )

            # ── Motion trail ──────────────────────────────────────────────
            if show_trails:
                history = self._history.get(tid, [])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    c     = tuple(int(v * alpha) for v in color)
                    thick = max(1, int(3 * alpha))
                    pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                    pt2 = (int(history[i][0]),     int(history[i][1]))
                    cv2.line(vis, pt1, pt2, c, thick, cv2.LINE_AA)

            # ── Bounding box ──────────────────────────────────────────────
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # ── Label ─────────────────────────────────────────────────────
            if show_ids or show_teams:
                # Show user-friendly ID (remove REF_ID_OFFSET for display)
                display_id   = tid - REF_ID_OFFSET if tid >= REF_ID_OFFSET else tid
                class_prefix = "Ref" if cid == CLASS_REF else "P"
                parts        = [f"{class_prefix}#{display_id}"]

                if show_teams and team_id != TEAM_UNKNOWN:
                    parts.append(TEAM_NAMES.get(team_id, ""))

                if show_conf:
                    parts.append(f"{det['conf']:.2f}")

                label = "  ".join(p for p in parts if p)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )

        return vis

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset both trackers, history, and counters.

        Call between clips or game quarters to avoid stale tracks.
        """
        self._player_tracker.reset()
        self._ref_tracker.reset()
        self._history.clear()
        self._all_player_ids.clear()
        self._all_ref_ids.clear()
        self._frame_idx = 0
        print("[PlayerTracker] Reset.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_tracker(
        self,
        tracker:   sv.ByteTrack,
        dets:      list[dict],
        id_offset: int = 0,
    ) -> list[dict]:
        """
        Feed one class's detections into a ByteTrack instance and return
        enriched dicts with track_id and team_id filled.

        Parameters
        ----------
        tracker   : The ByteTrack instance to update.
        dets      : Detection dicts for a single class (all same class_id).
        id_offset : Added to all raw tracker IDs (use REF_ID_OFFSET for refs).

        Returns
        -------
        list[dict] with track_id and team_id added.
        Unmatched input dets (dropped by ByteTrack during warm-up) are
        included with track_id = -1 to avoid silently losing detections.
        """
        if not dets:
            return []

        # ── Build sv.Detections ───────────────────────────────────────────────
        bboxes    = np.array([d["bbox"]     for d in dets], dtype=np.float32)
        confs     = np.array([d["conf"]     for d in dets], dtype=np.float32)
        class_ids = np.array([d["class_id"] for d in dets], dtype=int)

        sv_dets = sv.Detections(
            xyxy       = bboxes,
            confidence = confs,
            class_id   = class_ids,
        )

        # ── Run ByteTrack ─────────────────────────────────────────────────────
        tracked_sv = tracker.update_with_detections(sv_dets)

        # No tracks active yet (e.g. very first frame) → return all as untracked
        if len(tracked_sv) == 0 or tracked_sv.tracker_id is None:
            return [{**d, "track_id": -1, "team_id": TEAM_UNKNOWN} for d in dets]

        # ── Map tracked detections back to original dicts via IoU ─────────────
        tracked_bboxes = tracked_sv.xyxy          # shape (M, 4)
        tracker_ids    = tracked_sv.tracker_id    # shape (M,)

        # Track which original indices have been matched to avoid duplicates
        used_orig_indices: set[int] = set()
        result: list[dict] = []

        for t_idx in range(len(tracked_sv)):
            orig_idx = self._match_bbox(
                tracked_bboxes[t_idx], bboxes, used_orig_indices
            )

            if orig_idx == -1:
                # ByteTrack predicted a track position with no matching det
                # (carry-over from previous frame) — create a minimal entry
                x1, y1, x2, y2 = tracked_bboxes[t_idx]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                raw_tid  = int(tracker_ids[t_idx])
                track_id = raw_tid + id_offset
                result.append({
                    "bbox":       tracked_bboxes[t_idx],
                    "center":     (cx, cy),
                    "conf":       0.0,      # no real detection backing this
                    "class_id":   dets[0]["class_id"] if dets else -1,
                    "class_name": dets[0]["class_name"] if dets else "unknown",
                    "track_id":   track_id,
                    "team_id":    TEAM_UNKNOWN,
                })
            else:
                used_orig_indices.add(orig_idx)
                raw_tid  = int(tracker_ids[t_idx])
                track_id = raw_tid + id_offset
                result.append({
                    **dets[orig_idx],
                    "track_id": track_id,
                    "team_id":  TEAM_UNKNOWN,
                })

        # Append any input dets that ByteTrack discarded (tentative / low-conf)
        for i, det in enumerate(dets):
            if i not in used_orig_indices:
                result.append({**det, "track_id": -1, "team_id": TEAM_UNKNOWN})

        return result

    @staticmethod
    def _match_bbox(
        target:        np.ndarray,
        candidates:    np.ndarray,
        used_indices:  set[int],
        iou_thresh:    float = 0.30,
    ) -> int:
        """
        Return the index of the candidate bbox with the highest IoU to
        `target`, skipping already-used indices.

        Returns -1 if no candidate clears iou_thresh.
        """
        if len(candidates) == 0:
            return -1

        # Vectorised IoU over all candidates in one pass
        ix1 = np.maximum(candidates[:, 0], target[0])
        iy1 = np.maximum(candidates[:, 1], target[1])
        ix2 = np.minimum(candidates[:, 2], target[2])
        iy2 = np.minimum(candidates[:, 3], target[3])

        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

        area_t = (target[2] - target[0]) * (target[3] - target[1])
        area_c = (
            (candidates[:, 2] - candidates[:, 0])
            * (candidates[:, 3] - candidates[:, 1])
        )
        union = area_t + area_c - inter

        iou = np.where(union > 0.0, inter / union, 0.0)

        # Mask out already-matched candidates
        for idx in used_indices:
            iou[idx] = 0.0

        best = int(np.argmax(iou))
        return best if iou[best] >= iou_thresh else -1

    def _append_history(self, track_id: int, center: tuple[float, float]) -> None:
        """Append `center` to the history buffer and enforce TRAIL_LENGTH cap."""
        buf = self._history[track_id]
        buf.append(center)
        if len(buf) > TRAIL_LENGTH:
            # Remove oldest entry (no deque overhead — list is fine at len 30)
            self._history[track_id] = buf[-TRAIL_LENGTH:]

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PlayerTracker("
            f"frame={self._frame_idx}, "
            f"players_seen={self.total_player_tracks}, "
            f"refs_seen={self.total_ref_tracks}, "
            f"track_thresh={self.track_thresh}, "
            f"track_buffer={self.track_buffer})"
        )