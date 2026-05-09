"""
src/analytics/possession_overlay.py
Draw possession highlights on video frames: glowing bbox + label for ball carrier.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


# ── Color Palette ────────────────────────────────────────────────────────────
# BGR format for OpenCV
POSSESSION_GLOW_COLOR = (0, 255, 255)      # Cyan/Yellow glow — highly visible
POSSESSION_LABEL_BG   = (0, 200, 200)      # Slightly darker for label background
POSSESSION_TEXT_COLOR = (0, 0, 0)          # Black text

TEAM_COLORS = {
    "Team 0": (0, 165, 255),    # Orange
    "Team 1": (255, 100, 0),    # Blue-ish
}


def draw_glowing_bbox(
    frame: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int] = POSSESSION_GLOW_COLOR,
    thickness: int = 4,
    glow_layers: int = 3,
    glow_alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw a bounding box with a glowing halo effect.

    Args:
        frame: Input image (modified in-place)
        bbox: [x1, y1, x2, y2]
        color: BGR tuple for the glow
        thickness: Main box line thickness
        glow_layers: Number of outer glow rings
        glow_alpha: Transparency of glow layers

    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Create a glow layer
    glow = np.zeros_like(frame)

    # Draw expanding rectangles for glow effect
    for i in range(glow_layers, 0, -1):
        offset = i * 6
        alpha = glow_alpha * (1 - i / (glow_layers + 1))
        glow_color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)

        gx1 = max(0, x1 - offset)
        gy1 = max(0, y1 - offset)
        gx2 = min(w, x2 + offset)
        gy2 = min(h, y2 + offset)

        cv2.rectangle(glow, (gx1, gy1), (gx2, gy2), glow_color, thickness + i)

    # Blend glow onto frame
    frame = cv2.addWeighted(frame, 1.0, glow, 0.6, 0)

    # Draw main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    return frame


def draw_possession_label(
    frame: np.ndarray,
    bbox: List[float],
    player_id: str,
    team: Optional[str] = None,
    label_height: int = 28,
) -> np.ndarray:
    """
    Draw a label above the bbox showing "HAS BALL" + player info.

    Args:
        frame: Input image (modified in-place)
        bbox: [x1, y1, x2, y2]
        player_id: Track ID of the player
        team: Team name (optional)
        label_height: Height of the label bar in pixels

    Returns:
        Modified frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    label_text = f"🏀 HAS BALL — P{player_id}"
    if team:
        label_text += f" [{team}]"

    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Label background position (above the bbox)
    label_y1 = max(0, y1 - label_height - 4)
    label_y2 = max(label_height, y1 - 2)
    label_x1 = x1
    label_x2 = min(w, x1 + text_w + 16)

    # Draw label background
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), POSSESSION_LABEL_BG, -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), POSSESSION_GLOW_COLOR, 2)

    # Draw text
    text_x = label_x1 + 8
    text_y = label_y1 + label_height - 6
    cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, POSSESSION_TEXT_COLOR, thickness)

    return frame


def draw_possession_indicator(
    frame: np.ndarray,
    bbox: List[float],
    player_id: str,
    team: Optional[str] = None,
) -> np.ndarray:
    """
    Full possession visualization: glowing bbox + label.

    Args:
        frame: Input image
        bbox: [x1, y1, x2, y2]
        player_id: Track ID
        team: Team name

    Returns:
        Modified frame with possession highlight
    """
    frame = draw_glowing_bbox(frame, bbox, color=POSSESSION_GLOW_COLOR)
    frame = draw_possession_label(frame, bbox, player_id, team)
    return frame


def render_possession_video(
    input_video_path: str,
    output_video_path: str,
    trajectories: dict,
    possession_by_frame: Dict[int, Optional[str]],
    fps: Optional[float] = None,
) -> None:
    """
    Render a new video with possession highlights overlaid.

    Args:
        input_video_path: Path to the tracked video (with existing bboxes)
        output_video_path: Where to save the highlighted video
        trajectories: Full trajectories dict (for player bbox lookup)
        possession_by_frame: {frame: player_id} from get_possession_by_frame()
        fps: Override FPS (auto-detected if None)
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video_path}")

    _fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, _fps, (w, h))

    # Build player bbox lookup by frame
    player_bbox_by_frame: Dict[int, Dict[str, dict]] = {}
    for pid, records in trajectories.get("players", {}).items():
        for rec in records:
            frame_num = rec["frame"]
            if frame_num not in player_bbox_by_frame:
                player_bbox_by_frame[frame_num] = {}
            player_bbox_by_frame[frame_num][pid] = {
                "bbox": rec.get("bbox", [0, 0, 0, 0]),
                "team": rec.get("team"),
            }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        possessor_id = possession_by_frame.get(frame_idx)

        if possessor_id and frame_idx in player_bbox_by_frame:
            player_data = player_bbox_by_frame[frame_idx].get(possessor_id)
            if player_data:
                bbox = player_data["bbox"]
                team = player_data.get("team")
                frame = draw_possession_indicator(frame, bbox, possessor_id, team)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"   Possession video → {output_video_path}")
