import numpy as np
from team_clustering.clusterer import Clusterer


class TeamAssigner:
    """
    Maintains a persistent mapping of track_id → team_id across frames.
    Uses majority-vote over recent frames to avoid flickering assignments.
    """

    def __init__(self, vote_window: int = 10):
        self.clusterer   = Clusterer()
        self.assignments: dict[int, int]       = {}   # track_id → team_id
        self.history:     dict[int, list[int]] = {}   # track_id → recent votes
        self.vote_window = vote_window
        self._fitted     = False

    # ------------------------------------------------------------------
    def fit_on_frame(self, frame: np.ndarray, player_tracks: np.ndarray):
        """
        Call on the first frame (or any frame with many players visible).
        player_tracks : (N, 5) array → [x1, y1, x2, y2, track_id]
        """
        if len(player_tracks) < 2:
            return
        bboxes = player_tracks[:, :4].tolist()
        self.clusterer.fit(frame, bboxes)
        self._fitted = True

    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray, player_tracks: np.ndarray) -> dict:
        """
        Assigns a team to every tracked player in this frame.
        Returns dict: {track_id: team_id}
        """
        if not self._fitted or len(player_tracks) == 0:
            return {}

        frame_assignments = {}

        for track in player_tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            bbox = [x1, y1, x2, y2]

            # Get raw cluster label for this frame
            raw_label = self.clusterer.assign_team(frame, bbox)

            # Accumulate votes
            if track_id not in self.history:
                self.history[track_id] = []
            self.history[track_id].append(raw_label)

            # Limit vote window
            if len(self.history[track_id]) > self.vote_window:
                self.history[track_id].pop(0)

            # Majority vote → stable assignment
            votes = self.history[track_id]
            team_id = int(np.bincount(votes).argmax())
            self.assignments[track_id] = team_id
            frame_assignments[track_id] = team_id

        return frame_assignments

    # ------------------------------------------------------------------
    def get_team(self, track_id: int) -> int:
        """Returns the current team assignment for a track_id, or -1."""
        return self.assignments.get(track_id, -1)

    def get_team_color(self, team_id: int) -> tuple:
        return self.clusterer.get_team_color_bgr(team_id)

    def is_ready(self) -> bool:
        return self._fitted