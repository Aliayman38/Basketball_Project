"""
src/team_clustering/team_assigner.py
──────────────────────────────────────
CLIP-based team assignment — classifies players by jersey appearance
using the patrickjohncyh/fashion-clip vision-language model.

Original class is unchanged.
read_stub / save_stub are inlined here to avoid a naming conflict with
the 'utils' package that may be installed in the active virtual environment.
"""

from __future__ import annotations

import os
import pickle

from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel


# ── Inlined stub helpers (originally from utils.py) ──────────────────────────

def read_stub(read_from_stub: bool, stub_path: str | None):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            return pickle.load(f)
    return None


def save_stub(stub_path: str | None, data) -> None:
    if not stub_path:
        return
    parent = os.path.dirname(stub_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(data, f)


# ─────────────────────────────────────────────────────────────────────────────

class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self,
                 team_1_class_name="white shirt",
                 team_2_class_name="dark blue shirt",
                 ):
        self.team_colors = {}
        self.player_team_dict = {}

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        """Loads the pre-trained vision model for jersey color classification."""
        self.model     = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Returns:
            str: The classified jersey color/description.
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image     = pil_image

        classes = [self.team_1_class_name, self.team_2_class_name]

        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        outputs          = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs            = logits_per_image.softmax(dim=1)

        return classes[probs.argmax(dim=1)[0]]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Gets the team assignment for a player, using cached results if available.

        Returns:
            int: Team ID (1 or 2) assigned to the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_across_frames(self, video_frames, player_tracks,
                                       read_from_stub=False, stub_path=None):
        """
        Processes all video frames to assign teams to players, with optional caching.

        Returns:
            list: List of dicts mapping player IDs to team assignments for each frame.
        """
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model()

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):
            player_assignment.append({})

            if frame_num % 50 == 0:
                self.player_team_dict = {}

            for player_id, track in player_track.items():
                team = self.get_player_team(
                    video_frames[frame_num],
                    track['bbox'],
                    player_id,
                )
                player_assignment[frame_num][player_id] = team

        save_stub(stub_path, player_assignment)
        return player_assignment
