"""
team_clustering.py
──────────────────
Offline two-pass team clustering for basketball tracking.
Modified: Fixed Cython Buffer dtype mismatch (Forced float64 everywhere).
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jersey_crop(bgr_crop: np.ndarray) -> np.ndarray:
    """Return only the torso region to focus on jersey colour."""
    h, w = bgr_crop.shape[:2]
    y0 = int(h * 0.20)
    y1 = int(h * 0.60)
    x0 = int(w * 0.20)
    x1 = int(w * 0.80)
    return bgr_crop[y0:y1, x0:x1]


def _colour_histogram(bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """L*a*b* histogram. Returns a flat, L2-normalised feature vector."""
    if bgr is None or bgr.size == 0:
        return np.zeros(bins * 2, dtype=np.float64) # Changed to float64

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    feats = []
    for ch in (1, 2):          
        hist = cv2.calcHist([lab], [ch], None, [bins], [0, 256])
        feats.append(hist.flatten())

    # Changed to float64 to prevent Cython buffer mismatch
    vec = np.concatenate(feats).astype(np.float64) 
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TeamClusterer:
    TEAM_COLORS = {
        0: (0,   180, 255),    # Team A  — amber/orange
        1: (180,  60, 255),    # Team B  — violet
    }
    TEAM_NAMES = {0: 'A', 1: 'B'}

    def __init__(
        self,
        n_teams: int = 2,
        max_crops_per_track: int = 60,
        bins: int = 32,
        min_crops: int = 5,
    ):
        self.n_teams             = n_teams
        self.max_crops_per_track = max_crops_per_track
        self.bins                = bins
        self.min_crops           = min_crops

        self._features: dict[int, list[np.ndarray]] = defaultdict(list)
        self._team_map: dict[int, int] = {}
        self._is_fitted = False
        self.km = None 

    # ── Pass 1 ───────────────────────────────────────────────────────────────
    def collect(self, ori_id: int, crop: np.ndarray) -> None:
        if len(self._features[ori_id]) >= self.max_crops_per_track:
            return
        torso = _jersey_crop(crop)
        if torso.size == 0:
            return
        feat = _colour_histogram(torso, bins=self.bins)
        self._features[ori_id].append(feat)

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self) -> None:
        valid_ids: list[int] = []
        rep_vecs:  list[np.ndarray] = []

        for ori_id, feats in self._features.items():
            if len(feats) < self.min_crops:
                continue
            # Changed to float64
            rep = np.median(np.stack(feats), axis=0).astype(np.float64)
            norm = np.linalg.norm(rep)
            if norm > 0:
                rep /= norm
            valid_ids.append(ori_id)
            rep_vecs.append(rep)

        if len(valid_ids) < self.n_teams:
            print(f"[TeamClusterer] ⚠ Not enough valid tracks. All players 'unknown'.")
            self._is_fitted = True
            return

        X = np.stack(rep_vecs)
        self.km = KMeans(n_clusters=self.n_teams, n_init=20, random_state=42)
        labels = self.km.fit_predict(X)

        for ori_id, label in zip(valid_ids, labels):
            self._team_map[ori_id] = int(label)

        self._is_fitted = True
        print("[TeamClusterer] ✅ Clustering done.")

    # ── THE NEW PREDICTION METHOD ────────────────────────────────────────────
    def predict(self, crop: np.ndarray) -> int | None:
        """Predicts the team directly from the image crop (ignores faulty IDs)."""
        if not self._is_fitted or self.km is None:
            return None
            
        torso = _jersey_crop(crop)
        if torso.size == 0:
            return None
            
        feat = _colour_histogram(torso, bins=self.bins)
        
        # Now everything is strictly float64!
        team_idx = int(self.km.predict([feat])[0])
        return team_idx

    def team_summary(self) -> dict[str, list[int]]:
        summary: dict[str, list[int]] = defaultdict(list)
        for ori_id, idx in self._team_map.items():
            summary[self.TEAM_NAMES.get(idx, 'unknown')].append(ori_id)
        return dict(summary)