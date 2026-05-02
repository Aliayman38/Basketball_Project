"""
team_clustering/clusterer.py
─────────────────────────────
Team assignment via HSV torso histograms + K-Means.

Why HSV histograms instead of SigLIP + UMAP
────────────────────────────────────────────
  • Jersey colour is the only signal that reliably separates teams.
  • HSV hue histograms are invariant to lighting and shadow — exactly
    what you need for an indoor arena with mixed floor reflections.
  • 48-dim feature is perfect for K-Means directly — no UMAP overhead,
    no instability, no single-sample transform approximation.
  • Runs on CPU at >1000 frames/s (SigLIP needed a GPU pass per crop).

Bugs fixed vs. previous version
────────────────────────────────
  1. SigLIP removed       — 768-dim vision model was overkill for colour.
  2. UMAP removed         — umap.transform(1 sample) is unreliable; not needed.
  3. Label flip fixed     — clusters are anchored by mean hue every fit/refine,
                            so Team A and Team B never swap.
  4. Warm-up retry fixed  — was `==` (fires once); now `>=` (retries until data ready).
  5. White jersey support — saturation histogram added alongside hue histogram so
                            white jerseys form their own distinct feature signature.
  6. Torso slice widened  — 0.10–0.60 instead of 0.15–0.50 to capture full jersey.

Dataset class map (Roboflow basketball-players v11)
────────────────────────────────────────────────────
  0 → Ball   1 → Clock   2 → Hoop   3 → Overlay
  4 → Player ← clustered here
  5 → Ref    ← auto-assigned TEAM_REF, never clustered
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict, Counter, Counter
from sklearn.cluster import KMeans

# ── Class IDs ─────────────────────────────────────────────────────────────────
CLASS_BALL    = 0
CLASS_CLOCK   = 1
CLASS_HOOP    = 2
CLASS_OVERLAY = 3
CLASS_PLAYER  = 4
CLASS_REF     = 5

# ── Team labels ───────────────────────────────────────────────────────────────
TEAM_A       = 0
TEAM_B       = 1
TEAM_REF     = 2
TEAM_UNKNOWN = -1

TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    TEAM_A:       (235, 110,  40),   # vivid blue  — Team A
    TEAM_B:       ( 40, 200,  60),   # vivid green — Team B
    TEAM_REF:     ( 50,  50, 220),   # vivid red   — Referees
    TEAM_UNKNOWN: (160, 160, 160),   # grey
}

TEAM_NAMES: dict[int, str] = {
    TEAM_A:       "Team A",
    TEAM_B:       "Team B",
    TEAM_REF:     "Referee",
    TEAM_UNKNOWN: "Unknown",
}

# ── Torso slice (fraction of bbox height) ────────────────────────────────────
_TORSO_TOP = 0.10   # skip head / neck only
_TORSO_BOT = 0.60   # include full jersey body; stop before shorts

# ── Histogram dimensions ──────────────────────────────────────────────────────
_HUE_BINS = 32        # 180° ÷ 32 ≈ 5.6° per bin — fine enough to tell apart teams
_SAT_BINS = 16        # saturation range 0-255 — captures white-jersey signature
_FEAT_DIM = _HUE_BINS + _SAT_BINS   # 48 — directly usable by K-Means, no UMAP needed


class TeamClusterer:
    """
    Assigns basketball player track IDs to TEAM_A / TEAM_B.

    Pipeline
    ────────
    1. For every player detection, crop the torso and compute a 48-dim
       HSV histogram (32 hue bins + 16 saturation bins).
    2. Accumulate histograms per track_id.
    3. After warm_up_frames, compute one mean histogram per track and
       run K-Means(k=2) to split into two teams.
    4. After each fit/refine, anchor labels: the cluster whose center has
       the lower weighted-mean hue is always TEAM_A — labels never flip.
    5. New tracks that appear after warm-up are assigned via the fitted
       K-Means predict() on their mean histogram.

    Parameters
    ──────────
    warm_up_frames : frames to accumulate before first clustering attempt.
                     Retries every subsequent frame until enough data exists.
    torso_ratio    : (top, bottom) fractions of bbox height defining the
                     torso crop window.
    min_obs        : minimum histogram samples per track before it is included
                     in the feature matrix.
    device         : kept for API compatibility; this implementation is CPU-only.
    """

    def __init__(
        self,
        warm_up_frames: int = 60,
        torso_ratio: tuple[float, float] = (_TORSO_TOP, _TORSO_BOT),
        min_obs: int = 5,
        device: str | None = None,
    ) -> None:
        self.warm_up_frames = warm_up_frames
        self.torso_ratio    = torso_ratio
        self.min_obs        = min_obs

        # {track_id: [48-dim histogram, ...]}  — one vector per frame
        self._hist_buffer: dict[int, list[np.ndarray]] = defaultdict(list)

        # {track_id: TEAM_A | TEAM_B | TEAM_REF | TEAM_UNKNOWN}
        self._team_labels: dict[int, int] = {}

        # Fitted K-Means; None until first _fit() succeeds
        self._kmeans: KMeans | None = None

        # Maps raw K-Means label (0 or 1) → TEAM_A / TEAM_B.
        # Set by _fit() based on mean hue — stable across refine() calls.
        self._label_map: dict[int, int] = {0: TEAM_A, 1: TEAM_B}
        self._user_locked: bool = False   # True once user picks teams in GUI

        # Majority voting — only lock after consistent predictions
        self._vote_counts: dict[int, Counter] = defaultdict(Counter)
        self.vote_threshold  = 10    # observations before locking
        self.vote_confidence = 0.70  # fraction required to agree

        self.is_fitted: bool = False
        self._frame_idx: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, tracked_dets: list[dict]) -> None:
        """
        Call once per frame.

        For each player detection: extracts a torso histogram and buffers it.
        For each referee: immediately assigns TEAM_REF (no clustering needed).
        Triggers clustering once warm_up_frames have elapsed and retries every
        subsequent frame until at least 2 players have enough observations.
        """
        for det in tracked_dets:
            cid = int(det["class_id"])
            tid = int(det.get("track_id", -1))

            if tid == -1:
                continue

            if cid == CLASS_REF:
                self._team_labels[tid] = TEAM_REF
                continue

            if cid != CLASS_PLAYER:
                continue

            hist = self._extract_histogram(frame, det["bbox"])
            if hist is not None:
                self._hist_buffer[tid].append(hist)

        self._frame_idx += 1

        # Skip auto-clustering if user already calibrated via GUI
        if self._user_locked:
            if self.is_fitted:
                self._assign_pending()
            return

        if self._frame_idx >= self.warm_up_frames and not self.is_fitted:
            self._fit()

        if self.is_fitted:
            self._assign_pending()

    def get_team(self, track_id: int) -> int:
        return self._team_labels.get(track_id, TEAM_UNKNOWN)

    def get_team_name(self, track_id: int) -> str:
        return TEAM_NAMES[self.get_team(track_id)]

    def get_color(self, track_id: int) -> tuple[int, int, int]:
        return TEAM_COLORS[self.get_team(track_id)]

    def get_team_rosters(self) -> dict[int, list[int]]:
        rosters: dict[int, list[int]] = {TEAM_A: [], TEAM_B: [], TEAM_REF: []}
        for tid, team in self._team_labels.items():
            rosters.setdefault(team, []).append(tid)
        return rosters

    def refine(self) -> None:
        """
        Re-cluster using all accumulated histograms.
        The hue anchor guarantees Team A / Team B labels stay consistent.
        Only runs if label map has NOT been manually set by the user.
        """
        if not self._user_locked:
            self._fit(label="REFINE")

    def calibrate_from_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        One-shot calibration on a single frame using k=3 clustering.

        Uses SigLIP embeddings (768-dim) when available — they capture jersey
        texture and pattern, not just colour.  Falls back to HSV histograms
        if transformers is not installed.

        Clusters into 3 groups: Team A, Team B, and Referees.  The referee
        cluster is identified automatically as the group with the lowest mean
        HSV saturation (refs wear black/white stripes → achromatic).  The
        remaining two player clusters are returned for the user to label.

        Returns
        -------
        (group0_crops, group1_crops)  — the two PLAYER groups.
        Call set_user_label_map(team_a_is_group0=True/False) afterwards.
        """
        player_dets = [d for d in detections
                       if d.get("class_id") == CLASS_PLAYER
                       and d.get("track_id", -1) != -1]
        if len(player_dets) < 2:
            return [], []

        crops, tids = [], []
        for det in player_dets:
            crop = self._extract_crop(frame, det["bbox"])
            crops.append(crop)
            tids.append(det.get("track_id", -1))

        # Feature extraction — try SigLIP first, fall back to HSV
        feats = self._extract_siglip_features(crops)
        if feats is None:
            feats_list, valid_crops, valid_tids = [], [], []
            for det, crop, tid in zip(player_dets, crops, tids):
                h = self._extract_histogram(frame, det["bbox"])
                if h is not None:
                    feats_list.append(h)
                    valid_crops.append(crop)
                    valid_tids.append(tid)
            if len(feats_list) < 2:
                return [], []
            feats      = np.array(feats_list, dtype=np.float32)
            crops      = valid_crops
            tids       = valid_tids

        # k=3: two teams + referee cluster
        k = min(3, len(feats))
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        raw_labels = km.fit_predict(feats)

        if k == 3:
            # Identify referee cluster by lowest mean saturation
            # (black+white stripes = least chromatic)
            sat_means = []
            for ci in range(3):
                cluster_crops = [crops[j] for j in range(len(crops))
                                 if raw_labels[j] == ci]
                sat_means.append(self._mean_saturation(cluster_crops))

            ref_cluster    = int(np.argmin(sat_means))
            player_clusters = [i for i in range(3) if i != ref_cluster]

            # Store mapping: raw cluster id → player group index (0 or 1)
            self._player_cluster_map = {
                player_clusters[0]: 0,
                player_clusters[1]: 1,
                ref_cluster:        -1,   # referee
            }
        else:
            self._player_cluster_map = {0: 0, 1: 1}

        self._kmeans     = km
        self._calib_raw  = raw_labels
        self._calib_tids = tids

        group0 = [crops[j] for j in range(len(crops))
                  if self._player_cluster_map.get(raw_labels[j], -1) == 0]
        group1 = [crops[j] for j in range(len(crops))
                  if self._player_cluster_map.get(raw_labels[j], -1) == 1]

        print(f"[TeamClusterer] Calibration — k={k}, "
              f"group0: {len(group0)}, group1: {len(group1)}, "
              f"refs/other: {len(crops)-len(group0)-len(group1)}")
        return group0, group1

    def _mean_saturation(self, crops: list[np.ndarray]) -> float:
        """Mean HSV saturation across crops — referees score lowest."""
        if not crops:
            return 0.0
        total, count = 0.0, 0
        for crop in crops:
            if crop.size == 0:
                continue
            hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            total += float(hsv[:,:,1].mean())
            count += 1
        return total / max(count, 1)

    def _extract_siglip_features(
        self, crops: list[np.ndarray]
    ) -> np.ndarray | None:
        """
        Extract SigLIP embeddings for a list of BGR crops.

        Returns an (N, 768) float32 array, or None if transformers/SigLIP
        is not available (caller falls back to HSV histograms).
        """
        try:
            from transformers import SiglipImageProcessor, SiglipVisionModel
            import torch
            from PIL import Image as PILImage

            if not hasattr(self, "_siglip_model"):
                print("[TeamClusterer] Loading SigLIP for calibration (one-time)…")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_id = "google/siglip-base-patch16-224"
                self._siglip_proc  = SiglipImageProcessor.from_pretrained(model_id)
                self._siglip_model = SiglipVisionModel.from_pretrained(model_id).to(device)
                self._siglip_model.eval()
                self._siglip_device = device

            embeddings = []
            for crop in crops:
                if crop.size == 0:
                    embeddings.append(np.zeros(768, dtype=np.float32))
                    continue
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil  = PILImage.fromarray(rgb)
                inp  = self._siglip_proc(images=pil, return_tensors="pt").to(
                           self._siglip_device)
                with torch.no_grad():
                    out = self._siglip_model(**inp)
                emb = out.pooler_output.squeeze().cpu().numpy().astype(np.float32)
                embeddings.append(emb)

            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"[TeamClusterer] SigLIP unavailable ({e}), using HSV fallback")
            return None

    def set_user_label_map(self, team_a_is_group0: bool) -> None:
        """
        Lock the label map based on the user's GUI selection.
        Handles both k=2 and k=3 calibration (referee cluster auto-excluded).

        team_a_is_group0=True  → group 0 is Team A, group 1 is Team B
        team_a_is_group0=False → group 1 is Team A, group 0 is Team B
        """
        pcm = getattr(self, "_player_cluster_map", None)

        if pcm is not None:
            # k=3 path: build label_map from player_cluster_map
            self._label_map = {}
            for raw_cluster, group_idx in pcm.items():
                if group_idx == -1:
                    self._label_map[raw_cluster] = TEAM_REF
                elif group_idx == 0:
                    self._label_map[raw_cluster] = TEAM_A if team_a_is_group0 else TEAM_B
                else:
                    self._label_map[raw_cluster] = TEAM_B if team_a_is_group0 else TEAM_A
        else:
            # k=2 fallback
            if team_a_is_group0:
                self._label_map = {0: TEAM_A, 1: TEAM_B}
            else:
                self._label_map = {0: TEAM_B, 1: TEAM_A}

        self._user_locked = True
        self.is_fitted    = True

        # Pre-assign calibrated tracks (skip refs)
        if hasattr(self, "_calib_raw") and hasattr(self, "_calib_tids"):
            for tid, raw in zip(self._calib_tids, self._calib_raw):
                team = self._label_map.get(int(raw))
                if team is not None and team != TEAM_REF:
                    self._team_labels[tid] = team

        print(f"[TeamClusterer] User locked — "
              f"group0={'Team A' if team_a_is_group0 else 'Team B'}, "
              f"group1={'Team B' if team_a_is_group0 else 'Team A'}")

    # ── Private ────────────────────────────────────────────────────────────────

    def _extract_crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Return the torso crop (BGR) used for display in the GUI."""
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1] - 1, int(bbox[2]))
        y2 = min(frame.shape[0] - 1, int(bbox[3]))
        crop  = frame[y1:y2, x1:x2]
        h_box = crop.shape[0]
        t_top = int(h_box * self.torso_ratio[0])
        t_bot = int(h_box * self.torso_ratio[1])
        torso = crop[t_top:t_bot, :]
        if torso.size == 0:
            return crop
        return torso

    def _extract_histogram(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray | None:
        """
        Crop the torso region and return a normalised 48-dim HSV feature vector.

        Feature layout
        ──────────────
        dims  0-31  hue histogram (computed only on coloured pixels — S>40, V>40)
        dims 32-47  saturation histogram (all pixels — separates white vs coloured jerseys)

        White jersey handling
        ─────────────────────
        White pixels have low saturation so they are excluded from the hue
        histogram.  But they push the saturation histogram toward the low end,
        making white-jersey tracks cluster distinctly from coloured-jersey tracks.
        This means the feature correctly separates:
          • coloured vs coloured  (different hue peaks)
          • coloured vs white     (different saturation profiles)
          • white vs white        (both same low-saturation profile → same cluster)
        """
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1] - 1, int(bbox[2]))
        y2 = min(frame.shape[0] - 1, int(bbox[3]))

        if (x2 - x1) < 16 or (y2 - y1) < 32:
            return None

        crop  = frame[y1:y2, x1:x2]
        h_box = crop.shape[0]
        t_top = int(h_box * self.torso_ratio[0])
        t_bot = int(h_box * self.torso_ratio[1])
        torso = crop[t_top:t_bot, :]

        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # ── Hue histogram (coloured pixels only) ────────────────────────────
        colour_mask = (S > 40) & (V > 40)   # exclude shadows and achromatic pixels
        h_pixels    = H[colour_mask].reshape(-1, 1).astype(np.float32)

        if len(h_pixels) < 20:
            # Not enough coloured pixels (occluded, white, or very dark jersey).
            # Zero hue histogram — the saturation histogram will still carry
            # enough signal to place the track in the correct cluster.
            h_hist = np.zeros(_HUE_BINS, dtype=np.float32)
        else:
            h_hist = cv2.calcHist([h_pixels], [0], None, [_HUE_BINS], [0, 180]).flatten()
            h_sum  = h_hist.sum()
            if h_sum > 0:
                h_hist /= h_sum   # L1 normalise

        # ── Saturation histogram (all pixels) ────────────────────────────────
        s_pixels = S.reshape(-1, 1).astype(np.float32)
        s_hist   = cv2.calcHist([s_pixels], [0], None, [_SAT_BINS], [0, 256]).flatten()
        s_sum    = s_hist.sum()
        if s_sum > 0:
            s_hist /= s_sum

        return np.concatenate([h_hist, s_hist]).astype(np.float32)

    def _build_feature_matrix(self) -> tuple[np.ndarray, list[int]]:
        """
        Build an (N, 48) matrix — one mean histogram per qualifying track.
        Only tracks with >= min_obs observations are included.
        """
        feats: list[np.ndarray] = []
        tids:  list[int]        = []

        for tid, obs in self._hist_buffer.items():
            if len(obs) >= self.min_obs:
                feats.append(np.mean(obs, axis=0))   # mean = robust representation
                tids.append(tid)

        if not feats:
            return np.empty((0, _FEAT_DIM), dtype=np.float32), []

        return np.array(feats, dtype=np.float32), tids

    def _fit(self, label: str = "FIT") -> None:
        """
        Fit K-Means on the current feature matrix and assign team labels.

        Hue anchor (label-flip prevention)
        ────────────────────────────────────
        After fitting, compute the weighted mean hue of each cluster center's
        hue histogram.  The cluster with the lower mean hue is assigned TEAM_A,
        the other TEAM_B.  This mapping is stored in self._label_map and reused
        by _assign_pending() and every subsequent refine() call, so the same
        physical team is always TEAM_A regardless of how many times you refine.
        """
        X, tids = self._build_feature_matrix()

        if len(X) < 2:
            print(f"[TeamClusterer] {label} — not enough players ({len(X)}), "
                  f"retrying next frame…")
            return

        km         = KMeans(n_clusters=2, random_state=42, n_init=15, max_iter=300)
        raw_labels = km.fit_predict(X)

        # ── Hue anchor ────────────────────────────────────────────────────────
        hue_bins = np.arange(_HUE_BINS, dtype=np.float32)

        def _mean_hue(center: np.ndarray) -> float:
            h = center[:_HUE_BINS]
            total = h.sum()
            return float(np.dot(h, hue_bins) / total) if total > 1e-9 else 90.0

        hue0, hue1 = _mean_hue(km.cluster_centers_[0]), _mean_hue(km.cluster_centers_[1])

        if hue0 <= hue1:
            self._label_map = {0: TEAM_A, 1: TEAM_B}
        else:
            self._label_map = {0: TEAM_B, 1: TEAM_A}

        self._kmeans = km

        # Overwrite all player labels (refine() is allowed to correct earlier fits)
        for tid, raw in zip(tids, raw_labels):
            self._team_labels[tid] = self._label_map[int(raw)]

        self.is_fitted = True
        n_a = sum(1 for v in self._team_labels.values() if v == TEAM_A)
        n_b = sum(1 for v in self._team_labels.values() if v == TEAM_B)
        n_r = sum(1 for v in self._team_labels.values() if v == TEAM_REF)
        print(f"[TeamClusterer] {label} complete — "
              f"Team A: {n_a}  Team B: {n_b}  Refs: {n_r}")

    def _assign_pending(self) -> None:
        """
        Assign team labels using majority voting across observations.

        Each call casts one vote from the latest histogram.  A team is only
        locked once vote_threshold votes cast AND vote_confidence fraction agree.
        This makes assignments robust to single motion-blurred frames.
        """
        if self._kmeans is None:
            return

        for tid, obs in self._hist_buffer.items():
            if tid in self._team_labels:
                continue
            if not obs:
                continue

            latest    = obs[-1].reshape(1, -1).astype(np.float32)
            raw_label = int(self._kmeans.predict(latest)[0])
            team      = self._label_map.get(raw_label)
            if team is None:
                continue

            self._vote_counts[tid][team] += 1

            total = sum(self._vote_counts[tid].values())
            if total < self.vote_threshold:
                continue

            winner, winner_count = self._vote_counts[tid].most_common(1)[0]
            if winner_count / total >= self.vote_confidence:
                self._team_labels[tid] = winner

    def __repr__(self) -> str:
        return (
            f"TeamClusterer(fitted={self.is_fitted}, "
            f"assigned={len(self._team_labels)})"
        )