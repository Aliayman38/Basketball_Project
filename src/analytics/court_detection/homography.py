"""
src/analytics/court_detection/homography.py
─────────────────────────────────────────────
Per-frame homography solver for basketball broadcast video.

Takes per-frame keypoint detections (from the trained YOLO26-pose
court model) and computes a 3×3 homography matrix H that maps:
    pixel coordinates (image)  →  world coordinates (court meters)

Robust to:
  • Outlier keypoints (RANSAC inside cv2.findHomography)
  • Frames with too few keypoints (falls back to last good H)
  • Camera cuts (resets the fallback when needed)

Public API
──────────
    compute_h(keypoints, kp_to_world, ...) -> H or None
    project_point(point, H) -> (x_m, y_m)
    project_points(points, H) -> np.ndarray of meters
    HomographyTracker — class that maintains last-good-H across frames
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2


Keypoint = Tuple[int, float, float, float]   # (idx, x_pix, y_pix, conf)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-frame solver
# ─────────────────────────────────────────────────────────────────────────────
def compute_h(
    keypoints: List[Keypoint],
    kp_to_world: dict,
    min_correspondences: int = 4,
    ransac_threshold_px: float = 8.0,
) -> Optional[np.ndarray]:
    """
    Compute the homography matrix H from keypoint detections.

    Parameters
    ----------
    keypoints           : list of (idx, x_px, y_px, conf) — output of
                          detect_landmarks() in landmarks_overlay.py
    kp_to_world         : dict mapping keypoint index → (x_m, y_m).
                          Use court_template.KP_TO_WORLD.
    min_correspondences : minimum number of (image, world) pairs needed.
                          4 is the mathematical minimum; 6+ gives RANSAC
                          enough leeway to discard outliers.
    ransac_threshold_px : maximum reprojection error (in pixels) for a
                          correspondence to be considered an inlier.
                          Larger = more permissive.

    Returns
    -------
    3×3 numpy array H if a valid homography was found, else None.
    None means: not enough keypoints, or RANSAC couldn't find a
    consistent transform — caller should fall back to last good H.
    """
    if not keypoints or len(keypoints) < min_correspondences:
        return None

    # Build matched pairs: (image_pt, world_pt)
    image_pts = []
    world_pts = []
    for idx, x_px, y_px, conf in keypoints:
        world = kp_to_world.get(idx)
        if world is None:
            continue   # this index not in the mapping; skip
        image_pts.append([x_px, y_px])
        world_pts.append([world[0], world[1]])

    if len(image_pts) < min_correspondences:
        return None

    image_pts = np.asarray(image_pts, dtype=np.float32)
    world_pts = np.asarray(world_pts, dtype=np.float32)

    # cv2.findHomography returns (H, mask). RANSAC method automatically
    # filters outliers — robust against keypoints that landed on the
    # crowd, on the wrong court line, etc.
    H, mask = cv2.findHomography(
        srcPoints       = image_pts,
        dstPoints       = world_pts,
        method          = cv2.RANSAC,
        ransacReprojThreshold = ransac_threshold_px,
        maxIters        = 2000,
        confidence      = 0.995,
    )

    if H is None:
        return None

    # Quality check: how many points did RANSAC accept as inliers?
    # If less than 4, the homography is likely garbage.
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_correspondences:
        return None

    return H


# ─────────────────────────────────────────────────────────────────────────────
#  Projection utilities
# ─────────────────────────────────────────────────────────────────────────────
def project_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Project a single pixel coordinate to world coordinates using H.

    Parameters
    ----------
    point : (x_pixel, y_pixel)
    H     : 3×3 homography matrix

    Returns
    -------
    (x_meters, y_meters) on the court.
    """
    pt = np.asarray([[point]], dtype=np.float32)   # shape (1, 1, 2) for cv2
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def project_points(points: Iterable[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    """
    Project many pixels to world coordinates in one call.

    Parameters
    ----------
    points : iterable of (x, y) pixel coordinates
    H      : 3×3 homography matrix

    Returns
    -------
    np.ndarray of shape (N, 2) with (x_meters, y_meters) rows.
    """
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Stateful tracker — handles the camera-cut / few-keypoints problem
# ─────────────────────────────────────────────────────────────────────────────
class HomographyTracker:
    """
    Maintains a running estimate of H across frames.

    On every frame:
        - If there are enough good keypoints → compute fresh H, store it
        - If not enough keypoints (close-up, replay, cut) → reuse last H

    This handles broadcast video naturally: gameplay frames produce
    fresh H per frame; close-up cutaways reuse the last gameplay H,
    which keeps any visualization (top-down minimap, projected paths)
    sensible even during the cutaway.

    Use:
        tracker = HomographyTracker(KP_TO_WORLD)
        for frame_kps in per_frame_keypoints:
            H = tracker.update(frame_kps)   # H may be None on early frames
            if H is not None:
                player_world = project_point(player_pixel, H)
    """

    def __init__(
        self,
        kp_to_world: dict,
        min_correspondences:    int   = 6,    # higher than 4 = stricter
        ransac_threshold_px:    float = 8.0,
        max_stale_frames:       int   = 60,   # ~2 seconds at 30 fps
    ):
        """
        Parameters
        ----------
        kp_to_world          : the index→meters mapping
        min_correspondences  : require this many keypoints before computing H
        ransac_threshold_px  : RANSAC pixel threshold for inlier acceptance
        max_stale_frames     : after this many frames without fresh H,
                               clear the cached H (we've drifted too far,
                               better no H than wrong H)
        """
        self.kp_to_world         = kp_to_world
        self.min_correspondences = min_correspondences
        self.ransac_threshold_px = ransac_threshold_px
        self.max_stale_frames    = max_stale_frames

        self.last_H:        Optional[np.ndarray] = None
        self.frames_since_fresh: int             = 0

        # Stats for end-of-run reporting
        self.stats = {
            "frames_total":   0,
            "frames_fresh":   0,   # produced new H from this frame's kps
            "frames_reused":  0,   # used cached last_H
            "frames_no_h":    0,   # no H at all (too stale or never had one)
        }

    def update(self, keypoints: List[Keypoint]) -> Optional[np.ndarray]:
        """
        Process one frame's keypoints. Returns the H to use for this
        frame (could be fresh or reused from a previous frame).
        Returns None only when no H is available at all (e.g. start of
        video and never seen enough keypoints yet).
        """
        self.stats["frames_total"] += 1

        H_new = compute_h(
            keypoints           = keypoints,
            kp_to_world         = self.kp_to_world,
            min_correspondences = self.min_correspondences,
            ransac_threshold_px = self.ransac_threshold_px,
        )

        if H_new is not None:
            # Fresh H this frame
            self.last_H = H_new
            self.frames_since_fresh = 0
            self.stats["frames_fresh"] += 1
            return H_new

        # No fresh H. Try to reuse cached, unless it's too stale.
        self.frames_since_fresh += 1
        if self.last_H is not None and self.frames_since_fresh <= self.max_stale_frames:
            self.stats["frames_reused"] += 1
            return self.last_H

        # No H available
        if self.frames_since_fresh > self.max_stale_frames:
            # Cached H is too old — clear it
            self.last_H = None
        self.stats["frames_no_h"] += 1
        return None

    def reset(self) -> None:
        """Manually clear the cached H. Useful at known scene cuts."""
        self.last_H = None
        self.frames_since_fresh = 0

    def summary(self) -> dict:
        """Return a copy of the stats dict for end-of-run reporting."""
        return dict(self.stats)
