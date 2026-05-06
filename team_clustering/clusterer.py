import cv2
import numpy as np
from sklearn.cluster import KMeans


class Clusterer:
    """
    Extracts the dominant jersey color from a player crop and
    clusters all players into 2 teams using KMeans.

    Strategy:
    - Crop only the TOP HALF of the player bounding box (jersey, not shorts)
    - Remove the background using a secondary 2-cluster KMeans on the crop
    - Use the resulting foreground color as the player's representative color
    - Fit a global 2-cluster KMeans on the first frame to define Team 0 / Team 1
    """

    def __init__(self):
        self.team_kmeans: KMeans | None = None
        self.team_colors: dict = {}   # {team_id: [R, G, B]}

    # ------------------------------------------------------------------
    def _get_jersey_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Returns a single representative BGR color for the player's jersey."""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Clamp to frame boundaries
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, 0])

        crop = frame[y1:y2, x1:x2]

        # Use only top 50 % (jersey area)
        top_half = crop[: crop.shape[0] // 2, :]

        if top_half.size == 0:
            return np.array([0, 0, 0])

        # Resize for speed
        small = cv2.resize(top_half, (30, 30))
        pixels = small.reshape(-1, 3).astype(float)

        # Remove background: cluster the crop into 2 (bg vs jersey)
        km = KMeans(n_clusters=2, random_state=42, n_init=3)
        km.fit(pixels)

        # Corner pixels are typically background
        corners = [pixels[0], pixels[29], pixels[870], pixels[899]]
        corner_labels = km.predict(corners)
        bg_label = int(np.bincount(corner_labels).argmax())
        jersey_label = 1 - bg_label

        return km.cluster_centers_[jersey_label]   # BGR float

    # ------------------------------------------------------------------
    def fit(self, frame: np.ndarray, player_bboxes: list):
        """
        Fit the 2-cluster KMeans on the FIRST usable frame.
        Call this ONCE, then use assign_team() for all subsequent frames.
        """
        colors = []
        for bbox in player_bboxes:
            color = self._get_jersey_color(frame, bbox)
            colors.append(color)

        if len(colors) < 2:
            return   # not enough players to determine teams

        self.team_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.team_kmeans.fit(colors)

        # Store representative colors for visualization
        self.team_colors[0] = self.team_kmeans.cluster_centers_[0].astype(int).tolist()
        self.team_colors[1] = self.team_kmeans.cluster_centers_[1].astype(int).tolist()

    # ------------------------------------------------------------------
    def assign_team(self, frame: np.ndarray, bbox: list) -> int:
        """
        Returns team label (0 or 1) for a single player.
        Requires fit() to have been called first.
        """
        if self.team_kmeans is None:
            return -1

        color = self._get_jersey_color(frame, bbox)
        label = int(self.team_kmeans.predict([color])[0])
        return label

    # ------------------------------------------------------------------
    def get_team_color_bgr(self, team_id: int) -> tuple:
        """Return the BGR tuple for a team's representative color."""
        color = self.team_colors.get(team_id, [128, 128, 128])
        return tuple(int(c) for c in color)

    def is_fitted(self) -> bool:
        return self.team_kmeans is not None