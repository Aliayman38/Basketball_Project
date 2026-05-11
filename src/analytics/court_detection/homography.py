"""
src/analytics/court_detection/homography.py
─────────────────────────────────────────────
Per-frame homography solver for basketball broadcast video.

Inspired by abdullahtarek/basketball_analysis tactical_view_converter.

Pipeline:
  1. Take per-frame keypoint detections from the trained YOLO-pose model
  2. Filter outliers via validate_keypoints (Tarek's approach)
  3. Run cv2.findHomography with RANSAC (extra safety against outliers)
  4. Cache last good H to handle frames with too few keypoints

Public API
──────────
    compute_h(keypoints, kp_to_world, ...) -> H or None
    project_point(point, H) -> (x_px, y_px) on canvas
    project_points(points, H) -> array of canvas pixels
    validate_keypoints(keypoints, kp_to_world, threshold_px) -> filtered keypoints
    HomographyTracker — maintains H state across frames
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Iterable

import numpy as np
import cv2


Keypoint = Tuple[int, float, float, float]   # (idx, x_pix, y_pix, conf)


# ─────────────────────────────────────────────────────────────────────────────
#  Keypoint validation (Tarek's approach)
# ─────────────────────────────────────────────────────────────────────────────
def validate_keypoints(
    keypoints: List[Keypoint],
    kp_to_world: dict,
    threshold_px: float = 50.0,
    min_correspondences: int = 4,
) -> List[Keypoint]:
    """
    Filter keypoints that don't fit a consistent homography with the rest.

    Approach (inspired by Tarek's tactical_view_converter.validate_keypoints):
      1. Compute an initial H from ALL keypoints with mapped world points
      2. For each keypoint, project it through H to a predicted world point
      3. Compare against its claimed world point (from kp_to_world)
      4. If the reprojection error > threshold_px, discard the keypoint
      5. Return the surviving keypoints

    This filters out keypoints where the MODEL got the pixel location
    wrong (e.g. detected on the crowd or on the wrong court feature).
    Even if RANSAC inside findHomography handles some outliers, doing
    this BEFORE solving makes the final H much more accurate.

    Returns the filtered list of keypoints. Returns the original list
    if there aren't enough to even compute an initial H.
    """
    mapped = [(idx, x, y, c) for (idx, x, y, c) in keypoints if idx in kp_to_world]
    if len(mapped) < min_correspondences:
        return keypoints

    # Build initial H from all mapped keypoints
    image_pts = np.asarray([[x, y] for (_, x, y, _) in mapped], dtype=np.float32)
    world_pts = np.asarray([kp_to_world[idx] for (idx, *_) in mapped], dtype=np.float32)

    H_init, _ = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 10.0)
    if H_init is None:
        return keypoints   # can't validate, return as-is

    # For each keypoint, check reprojection error
    image_arr = image_pts.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(image_arr, H_init).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_pts, axis=1)

    valid = []
    for (kp, err) in zip(mapped, errors):
        if err <= threshold_px:
            valid.append(kp)

    # Also keep keypoints with no world mapping (they'll be filtered later
    # in compute_h but might be useful for diagnostics)
    unmapped = [(idx, x, y, c) for (idx, x, y, c) in keypoints if idx not in kp_to_world]
    return valid + unmapped


# ─────────────────────────────────────────────────────────────────────────────
#  Single-frame H solver
# ─────────────────────────────────────────────────────────────────────────────
def compute_h(
    keypoints: List[Keypoint],
    kp_to_world: dict,
    min_correspondences: int = 4,
    ransac_threshold_px: float = 10.0,
    validate: bool = True,
    validation_threshold_px: float = 60.0,
) -> Optional[np.ndarray]:
    """
    Compute the homography matrix H from keypoint detections.

    Parameters
    ----------
    keypoints           : list of (idx, x_px, y_px, conf) from detect_landmarks
    kp_to_world         : dict {idx → (x_canvas, y_canvas)} from court_template
    min_correspondences : minimum (image, world) pairs needed (default 4)
    ransac_threshold_px : RANSAC reprojection threshold IN CANVAS PIXELS
                          (because world points are canvas pixels now)
    validate            : if True, run validate_keypoints first
    validation_threshold_px : max reprojection error during validation

    Returns
    -------
    3×3 numpy array H if successful, else None.
    """
    if validate:
        keypoints = validate_keypoints(
            keypoints, kp_to_world,
            threshold_px=validation_threshold_px,
            min_correspondences=min_correspondences,
        )

    if not keypoints or len(keypoints) < min_correspondences:
        return None

    image_pts = []
    world_pts = []
    for idx, x_px, y_px, conf in keypoints:
        world = kp_to_world.get(idx)
        if world is None:
            continue
        image_pts.append([x_px, y_px])
        world_pts.append([world[0], world[1]])

    if len(image_pts) < min_correspondences:
        return None

    image_pts = np.asarray(image_pts, dtype=np.float32)
    world_pts = np.asarray(world_pts, dtype=np.float32)

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

    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_correspondences:
        return None

    return H


# ─────────────────────────────────────────────────────────────────────────────
#  Projection utilities
# ─────────────────────────────────────────────────────────────────────────────
def project_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Project a single pixel coordinate to canvas pixel using H."""
    pt = np.asarray([[point]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def project_points(points: Iterable[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    """Project many pixel coords to canvas pixels in one call."""
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Stateful tracker — handles camera cuts / few-keypoints frames
# ─────────────────────────────────────────────────────────────────────────────
class HomographyTracker:
    """
    Maintains a running H across frames.

      - Each frame: try to compute fresh H from this frame's keypoints
      - If too few keypoints / RANSAC fails: reuse last good H
      - After too many stale frames: clear cache

    Use:
        tracker = HomographyTracker(KP_TO_WORLD)
        for frame_kps in per_frame_keypoints:
            H = tracker.update(frame_kps)
            if H is not None:
                canvas_xy = project_point(player_pixel, H)
    """

    def __init__(
        self,
        kp_to_world: dict,
        min_correspondences: int   = 4,
        ransac_threshold_px: float = 10.0,
        max_stale_frames:    int   = 60,
        validate:            bool  = True,
    ):
        self.kp_to_world         = kp_to_world
        self.min_correspondences = min_correspondences
        self.ransac_threshold_px = ransac_threshold_px
        self.max_stale_frames    = max_stale_frames
        self.validate            = validate

        self.last_H:        Optional[np.ndarray] = None
        self.frames_since_fresh: int             = 0

        self.stats = {
            "frames_total":   0,
            "frames_fresh":   0,
            "frames_reused":  0,
            "frames_no_h":    0,
        }

    def update(self, keypoints: List[Keypoint]) -> Optional[np.ndarray]:
        self.stats["frames_total"] += 1

        H_new = compute_h(
            keypoints           = keypoints,
            kp_to_world         = self.kp_to_world,
            min_correspondences = self.min_correspondences,
            ransac_threshold_px = self.ransac_threshold_px,
            validate            = self.validate,
        )

        if H_new is not None:
            self.last_H = H_new
            self.frames_since_fresh = 0
            self.stats["frames_fresh"] += 1
            return H_new

        self.frames_since_fresh += 1
        if self.last_H is not None and self.frames_since_fresh <= self.max_stale_frames:
            self.stats["frames_reused"] += 1
            return self.last_H

        if self.frames_since_fresh > self.max_stale_frames:
            self.last_H = None
        self.stats["frames_no_h"] += 1
        return None

    def reset(self) -> None:
        self.last_H = None
        self.frames_since_fresh = 0

    def summary(self) -> dict:
        return dict(self.stats)
