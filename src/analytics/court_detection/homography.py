"""
src/analytics/court_detection/homography.py
─────────────────────────────────────────────
Per-frame homography using roboflow/sports ViewTransformer.
"""
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    from sports import ViewTransformer
except ImportError:
    ViewTransformer = None

from .court_template import VERTICES

Keypoint = Tuple[int, float, float, float]


def build_transformer(
    keypoints: List[Keypoint],
    vertices: list = VERTICES,
    conf_threshold: float = 0.5,
    min_correspondences: int = 4,
) -> Optional[object]:
    filtered = [
        (idx, x, y, c)
        for idx, x, y, c in keypoints
        if c >= conf_threshold and 0 <= idx < len(vertices)
    ]
    if len(filtered) < min_correspondences:
        return None

    frame_landmarks = np.array(
        [[x, y] for (_, x, y, _) in filtered], dtype=np.float32
    )
    court_landmarks = np.array(
        [vertices[idx] for (idx, _, _, _) in filtered], dtype=np.float32
    )

    if ViewTransformer is not None:
        try:
            return ViewTransformer(source=frame_landmarks, target=court_landmarks)
        except Exception:
            return None
    else:
        H, mask = cv2.findHomography(
            frame_landmarks, court_landmarks, cv2.RANSAC, 10.0,
            maxIters=2000, confidence=0.995
        )
        if H is None:
            return None
        if mask is not None and int(mask.sum()) < min_correspondences:
            return None
        return H


def project_points(points: np.ndarray, transformer) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    if ViewTransformer is not None and isinstance(transformer, ViewTransformer):
        return transformer.transform_points(points=pts)
    else:
        pts_h = pts.reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts_h, transformer)
        return out.reshape(-1, 2)


def project_point(point: Tuple[float, float], transformer) -> Tuple[float, float]:
    result = project_points(np.array([point], dtype=np.float32), transformer)
    return float(result[0, 0]), float(result[0, 1])


class HomographyTracker:
    def __init__(self, vertices=VERTICES, conf_threshold=0.5,
                 min_correspondences=4, max_stale_frames=60):
        self.vertices = vertices
        self.conf_threshold = conf_threshold
        self.min_correspondences = min_correspondences
        self.max_stale_frames = max_stale_frames
        self.last_transformer = None
        self.frames_since_fresh = 0
        self.stats = {"frames_total": 0, "frames_fresh": 0,
                      "frames_reused": 0, "frames_no_h": 0}

    def update(self, keypoints):
        self.stats["frames_total"] += 1
        t = build_transformer(keypoints, self.vertices,
                              self.conf_threshold, self.min_correspondences)
        if t is not None:
            self.last_transformer = t
            self.frames_since_fresh = 0
            self.stats["frames_fresh"] += 1
            return t
        self.frames_since_fresh += 1
        if self.last_transformer and self.frames_since_fresh <= self.max_stale_frames:
            self.stats["frames_reused"] += 1
            return self.last_transformer
        if self.frames_since_fresh > self.max_stale_frames:
            self.last_transformer = None
        self.stats["frames_no_h"] += 1
        return None

    def summary(self):
        return dict(self.stats)
