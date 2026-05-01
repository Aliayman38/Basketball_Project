"""
src/analytics/heatmap.py
─────────────────────────
Builds occupancy heatmaps from player trajectories on the top-down court.

Core idea
─────────
Accumulate (court_x, court_y) positions over many frames into a 2D array,
blur it to convert sparse points into a smooth density field, then color
it with a heat colormap and blend onto the court canvas.

Two modes
─────────
- Per-player heatmap   → individual coverage zones
- Team heatmap         → sum across all players in a team

Public API
──────────
    builder = HeatmapBuilder(transformer)
    for frame_dets in tracking_results.values():
        builder.add_frame(frame_dets)

    # Visualisations
    img = builder.build_player_heatmap(track_id=5)
    img = builder.build_team_heatmap(team_id=0)

    # Stats
    builder.get_dominant_zone(5)        →  (cx, cy)
    builder.get_court_coverage(5)       →  0.34   (i.e. covered 34% of court)
"""

from __future__ import annotations

import numpy as np
import cv2
from collections import defaultdict

from .homography import HomographyTransformer


# ── Defaults ─────────────────────────────────────────────────────────────────
# Blur radius (canvas pixels). On a 1060x560 court (~28m × 15m), 25 px ≈ 0.66m
# of "personal space" per player.
DEFAULT_BLUR_SIGMA = 25

# INFERNO is cleaner and more readable than JET for occupancy data
DEFAULT_COLORMAP = cv2.COLORMAP_INFERNO


# ─────────────────────────────────────────────────────────────────────────────
class HeatmapBuilder:
    """
    Accumulates player positions over time and produces heatmap images.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        transformer: HomographyTransformer,
        blur_sigma:  int = DEFAULT_BLUR_SIGMA,
        colormap:    int = DEFAULT_COLORMAP,
    ) -> None:
        self.transformer = transformer
        self.blur_sigma  = blur_sigma
        self.colormap    = colormap

        W = transformer.court_width_px
        H = transformer.court_height_px

        # One accumulator per track_id and per team_id.
        self._player_acc: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((H, W), dtype=np.float32)
        )
        self._team_acc: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((H, W), dtype=np.float32)
        )

        self._frames_seen = 0

    # ── Population ────────────────────────────────────────────────────────────

    def add_frame(self, frame_dets: list[dict]) -> None:
        """
        Process one frame's detections. Each dict should have:
            - bbox      : (x1, y1, x2, y2)
            - class_id  : int
            - track_id  : int (≥ 0)
            - team_id   : int (optional)
        """
        # Lazy import to avoid circular dependency at module-load time.
        # Falls back to project constants if team_clustering can't import
        # its heavy deps (torch, transformers) — keeps heatmap usable in
        # lightweight scripts that don't need clustering.
        try:
            from team_clustering.clusterer import CLASS_PLAYER, TEAM_UNKNOWN
        except ImportError:
            CLASS_PLAYER = 4
            TEAM_UNKNOWN = -1

        W = self.transformer.court_width_px
        H = self.transformer.court_height_px

        for det in frame_dets:
            if det.get("class_id") != CLASS_PLAYER:
                continue   # heatmaps are players-only

            tid = det.get("track_id", -1)
            if tid == -1:
                continue

            court_x, court_y = self.transformer.transform_bbox_foot(det["bbox"])

            if not self.transformer.is_on_court((court_x, court_y), margin=10):
                continue

            cx = max(0, min(W - 1, int(court_x)))
            cy = max(0, min(H - 1, int(court_y)))

            self._player_acc[tid][cy, cx] += 1.0

            team_id = det.get("team_id", TEAM_UNKNOWN)
            if team_id != TEAM_UNKNOWN:
                self._team_acc[team_id][cy, cx] += 1.0

        self._frames_seen += 1

    # ── Rendering ─────────────────────────────────────────────────────────────

    def build_player_heatmap(
        self,
        track_id: int,
        overlay:  bool  = True,
        alpha:    float = 0.55,
    ) -> np.ndarray | None:
        """Render heatmap for one player. None if no data."""
        acc = self._player_acc.get(track_id)
        if acc is None or acc.sum() == 0:
            return None
        return self._render(acc, overlay, alpha)

    def build_team_heatmap(
        self,
        team_id: int,
        overlay: bool  = True,
        alpha:   float = 0.55,
    ) -> np.ndarray | None:
        """Render heatmap for one team. None if no data."""
        acc = self._team_acc.get(team_id)
        if acc is None or acc.sum() == 0:
            return None
        return self._render(acc, overlay, alpha)

    def _render(
        self,
        acc:     np.ndarray,
        overlay: bool,
        alpha:   float,
    ) -> np.ndarray:
        """
        Accumulator → colored heatmap image.

        Pipeline:
            1. Gaussian blur (sparse points → smooth density)
            2. Normalize to 0..255
            3. Apply colormap
            4. Optionally blend onto court canvas with per-pixel alpha
        """
        # 1. Blur
        ksize = int(self.blur_sigma * 4) | 1   # odd kernel size
        blurred = cv2.GaussianBlur(
            acc, (ksize, ksize),
            sigmaX=self.blur_sigma, sigmaY=self.blur_sigma,
        )

        # 2. Normalize
        max_v = blurred.max()
        if max_v < 1e-6:
            return self.transformer.make_court_canvas()
        norm = (blurred / max_v * 255).astype(np.uint8)

        # 3. Colormap
        heat = cv2.applyColorMap(norm, self.colormap)

        if not overlay:
            return heat

        # 4. Per-pixel alpha blending — cold areas stay clear, hot areas opaque
        canvas = self.transformer.make_court_canvas()
        alpha_mask = (norm.astype(np.float32) / 255.0 * alpha)[..., None]
        blended = (heat.astype(np.float32) * alpha_mask +
                   canvas.astype(np.float32) * (1.0 - alpha_mask))
        return blended.astype(np.uint8)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_dominant_zone(
        self, track_id: int
    ) -> tuple[int, int] | None:
        """
        (court_x, court_y) of the player's most-occupied position.
        """
        acc = self._player_acc.get(track_id)
        if acc is None or acc.sum() == 0:
            return None
        cy, cx = np.unravel_index(acc.argmax(), acc.shape)
        return (int(cx), int(cy))

    def get_court_coverage(
        self,
        track_id:  int,
        threshold: float = 0.05,
    ) -> float:
        """
        Fraction of the court (0..1) where the player spent meaningful time.
        threshold = minimum occupancy (relative to peak) to count as "covered".
        """
        acc = self._player_acc.get(track_id)
        if acc is None or acc.sum() == 0:
            return 0.0
        ksize = int(self.blur_sigma * 4) | 1
        blurred = cv2.GaussianBlur(acc, (ksize, ksize), self.blur_sigma)
        peak = blurred.max()
        if peak < 1e-6:
            return 0.0
        active = (blurred >= peak * threshold).sum()
        return float(active) / blurred.size

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_player_heatmap(self, track_id: int, path: str) -> bool:
        img = self.build_player_heatmap(track_id, overlay=True)
        if img is None:
            return False
        cv2.imwrite(path, img)
        return True

    def save_team_heatmap(self, team_id: int, path: str) -> bool:
        img = self.build_team_heatmap(team_id, overlay=True)
        if img is None:
            return False
        cv2.imwrite(path, img)
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_players_tracked(self) -> int:
        return len(self._player_acc)

    @property
    def frames_processed(self) -> int:
        return self._frames_seen

    def __repr__(self) -> str:
        return (
            f"HeatmapBuilder("
            f"frames={self._frames_seen}, "
            f"players={self.n_players_tracked}, "
            f"teams={len(self._team_acc)})"
        )
