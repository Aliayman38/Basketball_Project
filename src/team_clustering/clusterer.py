"""
src/team_clustering/clusterer.py
─────────────────────────────────
Unsupervised team assignment via jersey-colour clustering.

Dataset class map  (Roboflow basketball-players v11)
─────────────────────────────────────────────────────
  0 → Ball
  1 → Clock
  2 → Hoop
  3 → Overlay
  4 → Player   ← clustered here
  5 → Ref      ← detected separately; auto-assigned to TEAM_REF

Design decision
────────────────
Referees are their own YOLO class (id=5), so they are never fed into
KMeans.  Only CLASS_PLAYER detections enter the colour pipeline.
We therefore need k=2 (Team A vs Team B) — not k=3.
Ref tracks receive the fixed label TEAM_REF=2 automatically.

Algorithm
──────────
1.  Each frame: for every detection with class_id == CLASS_PLAYER(4),
    crop the bbox, take the torso strip (12 %–52 % of box height:
    jersey region — avoids face and shorts).
2.  Convert torso to HSV.  Discard low-saturation and very-dark pixels
    (white court floor, shadows, specular highlights).
3.  Buffer the per-player median-HSV vector per frame.
4.  After warm_up_frames frames: fit KMeans(k=2) on the per-player
    median-colour representations.
5.  Predict label for every new track that appears after the fit.
6.  Optional refine() re-fits on all data (call at end-of-video).

Why HSV over RGB?
  Hue is invariant to brightness/exposure changes.  RGB clusters shift
  drastically under different arena lighting sections; HSV stays stable.

Importable names
─────────────────
  TeamClusterer                 — main class
  CLASS_PLAYER, CLASS_REF       — int  (4, 5)
  TEAM_A, TEAM_B, TEAM_REF      — int  (0, 1, 2)
  TEAM_UNKNOWN                  — int  (-1)
  TEAM_COLORS                   — dict[int, tuple[int,int,int]]  BGR
  TEAM_NAMES                    — dict[int, str]

Usage in detector.py
─────────────────────
  from src.team_clustering.clusterer import (
      TeamClusterer,
      CLASS_PLAYER, CLASS_REF,
      TEAM_A, TEAM_B, TEAM_REF, TEAM_UNKNOWN,
      TEAM_COLORS, TEAM_NAMES,
  )
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ── Dataset class IDs  (must match data/basketball.yaml `names`) ──────────────
CLASS_BALL    = 0
CLASS_CLOCK   = 1
CLASS_HOOP    = 2
CLASS_OVERLAY = 3
CLASS_PLAYER  = 4   # ← players to cluster
CLASS_REF     = 5   # ← referees detected separately

# ── Team label constants ──────────────────────────────────────────────────────
TEAM_A       = 0   # KMeans cluster 0
TEAM_B       = 1   # KMeans cluster 1
TEAM_REF     = 2   # Always assigned to CLASS_REF detections
TEAM_UNKNOWN = -1  # Not yet assigned

# ── Display colours per team  (BGR for OpenCV) ────────────────────────────────
TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    TEAM_A:       (235, 110,  40),   # vivid blue  — Team A
    TEAM_B:       ( 40, 200,  60),   # vivid green — Team B
    TEAM_REF:     ( 50,  50, 220),   # vivid red   — Referees
    TEAM_UNKNOWN: (160, 160, 160),   # grey        — not yet assigned
}

# ── Human-readable names ──────────────────────────────────────────────────────
TEAM_NAMES: dict[int, str] = {
    TEAM_A:       "Team A",
    TEAM_B:       "Team B",
    TEAM_REF:     "Referee",
    TEAM_UNKNOWN: "Unknown",
}

# ── Default torso slice  (fraction of bounding-box height) ───────────────────
_TORSO_TOP  = 0.12   # skip face / head
_TORSO_BOT  = 0.52   # stop before shorts / legs


# ─────────────────────────────────────────────────────────────────────────────
class TeamClusterer:
    """
    Assigns basketball player tracks to teams via jersey-colour clustering.

    Parameters
    ----------
    warm_up_frames : int
        Frames to collect before the first KMeans fit.  60 frames ≈ 2 s at
        30 fps — enough for stable per-player colour estimates.
    torso_ratio : tuple[float, float]
        (top_frac, bottom_frac) vertical crop window inside the bbox.
    method : {'kmeans', 'gmm'}
        Clustering backend.  GMM handles dark-blue vs black jerseys better.
    min_color_obs : int
        Minimum per-player frame observations before including in KMeans fit.
    """

    def __init__(
        self,
        warm_up_frames: int               = 60,
        torso_ratio:    tuple[float, float] = (_TORSO_TOP, _TORSO_BOT),
        method:         str               = "kmeans",
        min_color_obs:  int               = 5,
    ) -> None:
        self.warm_up_frames = warm_up_frames
        self.torso_ratio    = torso_ratio
        self.method         = method
        self.min_color_obs  = min_color_obs

        # {track_id: [median_hsv_vec, ...]}  — raw per-frame observations
        self._color_buffer: dict[int, list[np.ndarray]] = defaultdict(list)

        # {track_id: team_id}  — final assignments
        self._team_labels:  dict[int, int] = {}

        # Fitted sklearn model (KMeans or GMM)
        self._model: KMeans | GaussianMixture | None = None

        self.is_fitted:  bool = False
        self._frame_idx: int  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, tracked_dets: list[dict]) -> None:
        """
        Call once per frame with the current BGR frame and all tracked dets.

        Each detection dict must have:
            track_id  : int
            bbox      : np.ndarray  [x1, y1, x2, y2]
            class_id  : int

        Behaviour:
          • CLASS_REF     → immediately labelled TEAM_REF, never clustered.
          • CLASS_PLAYER  → colour buffered; clustered after warm-up.
          • All others    → ignored.
        """
        for det in tracked_dets:
            cid = int(det["class_id"])
            tid = int(det["track_id"])

            if cid == CLASS_REF:
                # Referees: fixed label, no colour processing needed
                self._team_labels[tid] = TEAM_REF
                continue

            if cid != CLASS_PLAYER:
                continue

            colour = self._extract_jersey_hsv(frame, det["bbox"])
            if colour is not None:
                self._color_buffer[tid].append(colour)

        self._frame_idx += 1

        # First-time fit at warm-up boundary
        if self._frame_idx == self.warm_up_frames and not self.is_fitted:
            self._fit()

        # Assign any player whose buffer just crossed min_color_obs
        if self.is_fitted:
            self._assign_pending()

    def get_team(self, track_id: int) -> int:
        """Return TEAM_A(0), TEAM_B(1), TEAM_REF(2), or TEAM_UNKNOWN(-1)."""
        return self._team_labels.get(track_id, TEAM_UNKNOWN)

    def get_team_name(self, track_id: int) -> str:
        """Human-readable team label string."""
        return TEAM_NAMES[self.get_team(track_id)]

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        """BGR tuple for drawing bounding boxes / overlay text."""
        return TEAM_COLORS[self.get_team(track_id)]

    def get_team_rosters(self) -> dict[int, list[int]]:
        """
        {team_id: [track_id, ...]} for every team.

        Used by the analytics engine to aggregate per-team statistics
        (total distance, avg speed, etc.).
        """
        rosters: dict[int, list[int]] = {TEAM_A: [], TEAM_B: [], TEAM_REF: []}
        for tid, team in self._team_labels.items():
            rosters.setdefault(team, []).append(tid)
        return rosters

    def refine(self) -> None:
        """
        Re-fit the model on ALL buffered data collected so far.

        Recommended usage: call once after the video loop completes to
        produce the cleanest team assignments for the final analytics report.
        """
        self._fit(label="REFINE")

    def print_roster(self) -> None:
        """Pretty-print the team rosters to stdout."""
        rosters = self.get_team_rosters()
        print("\n" + "─" * 48)
        print("  TEAM ROSTERS")
        print("─" * 48)
        for team_id in (TEAM_A, TEAM_B, TEAM_REF):
            tids = sorted(rosters.get(team_id, []))
            name = TEAM_NAMES[team_id]
            print(f"  {name:<10s}  ({len(tids):2d} tracks)  IDs: {tids}")
        print("─" * 48 + "\n")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_jersey_hsv(
        self,
        frame: np.ndarray,
        bbox:  np.ndarray,
    ) -> np.ndarray | None:
        """
        Crop the torso region and return its median HSV as a (3,) array.

        Steps
        ─────
        1. Clamp bbox to frame bounds.
        2. Slice vertically by torso_ratio.
        3. Convert to HSV.
        4. Discard unsaturated (S ≤ 40) and very-dark (V ≤ 30) pixels.
        5. Return median of remaining pixels; fall back to all-pixel median
           if fewer than 10 saturated pixels remain (handles dark jerseys).
        """
        x1 = max(0, int(bbox[0]));  y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1] - 1, int(bbox[2]))
        y2 = min(frame.shape[0] - 1, int(bbox[3]))

        if (x2 - x1) < 8 or (y2 - y1) < 16:
            return None   # bbox too small — skip

        crop   = frame[y1:y2, x1:x2]
        h_box  = crop.shape[0]
        t_top  = int(h_box * self.torso_ratio[0])
        t_bot  = int(h_box * self.torso_ratio[1])
        torso  = crop[t_top:t_bot, :]

        if torso.size == 0:
            return None

        hsv    = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Saturation > 40  AND  Value > 30  → colourful, non-shadow pixels
        mask   = (pixels[:, 1] > 40) & (pixels[:, 2] > 30)
        valid  = pixels[mask]

        return np.median(valid if len(valid) >= 10 else pixels, axis=0)

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Aggregate per-player colour buffers → (N, 3) feature matrix.

        Only players with ≥ min_color_obs observations are included.
        Each player is represented by the median of their observation
        vectors (robust to per-frame noise).
        """
        feats: list[np.ndarray] = []
        tids:  list[int]        = []

        for tid, obs in self._color_buffer.items():
            if len(obs) >= self.min_color_obs:
                feats.append(np.median(obs, axis=0))
                tids.append(tid)

        if not feats:
            return np.empty((0, 3), dtype=np.float32), []

        return np.array(feats, dtype=np.float32), tids

    def _fit(self, label: str = "FIT") -> None:
        """Fit the clustering model and assign labels to all buffered players."""
        X, tids = self._build_feature_matrix()

        if len(X) < 2:
            print(
                f"[TeamClusterer] {label} — need ≥ 2 players with sufficient "
                f"observations, got {len(X)}.  Waiting for more frames…"
            )
            return

        if self.method == "gmm":
            self._model = GaussianMixture(
                n_components=2, random_state=42, n_init=5, covariance_type="full"
            )
            labels = self._model.fit_predict(X)
        else:
            self._model = KMeans(
                n_clusters=2, random_state=42, n_init=10
            )
            labels = self._model.fit_predict(X)

        for tid, lab in zip(tids, labels):
            self._team_labels[tid] = int(lab)

        self.is_fitted = True
        n_a = int((labels == 0).sum())
        n_b = int((labels == 1).sum())
        print(
            f"[TeamClusterer] {label} — {len(X)} players clustered.  "
            f"Team A: {n_a}  Team B: {n_b}"
        )

    def _assign_pending(self) -> None:
        """
        Predict team label for any player track whose buffer recently crossed
        min_color_obs but hasn't been assigned yet (appeared after warm-up).
        """
        for tid, obs in self._color_buffer.items():
            if tid in self._team_labels:
                continue
            if len(obs) < self.min_color_obs:
                continue
            feat  = np.median(obs, axis=0).reshape(1, -1).astype(np.float32)
            label = int(self._model.predict(feat)[0])
            self._team_labels[tid] = label

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TeamClusterer("
            f"method={self.method!r}, "
            f"fitted={self.is_fitted}, "
            f"assigned={len(self._team_labels)}, "
            f"warm_up={self.warm_up_frames})"
        )
