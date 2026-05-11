"""
src/analytics/court_detection/landmarks_overlay.py
Court landmark detection module using a custom-trained YOLO-pose model.
"""
from __future__ import annotations

import os
import time
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import cv2
import numpy as np


Keypoint = Tuple[int, float, float, float]   # (idx, x, y, conf)


def load_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("pip install ultralytics") from e
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(weights_path)


def detect_landmarks(frame, model, conf_threshold=0.30):
    res = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.xy) == 0:
        return []
    xy   = res.keypoints.xy[0].cpu().numpy()
    conf = res.keypoints.conf[0].cpu().numpy()
    out = []
    for i, ((x, y), c) in enumerate(zip(xy, conf)):
        if c >= conf_threshold and not (x == 0.0 and y == 0.0):
            out.append((i, float(x), float(y), float(c)))
    return out


def draw_landmarks(frame, keypoints, radius=8, show_indices=True):
    for idx, x, y, conf in keypoints:
        center = (int(x), int(y))
        if   conf >= 0.50: fill = (0, 255,   0)
        elif conf >= 0.30: fill = (255, 255, 0)
        else:              fill = (0, 165, 255)
        cv2.circle(frame, center, radius,     fill,         -1)
        cv2.circle(frame, center, radius + 2, (0, 0, 0),     2)
        if show_indices:
            text = str(idx)
            pos  = (center[0] + radius + 4, center[1] - 6)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0),     4, cv2.LINE_AA)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_landmarks_video(input_video_path, output_video_path,
                           weights_path="models/weights/court_kp.pt",
                           conf_threshold=0.30, log_every=30):
    """
    Standalone renderer (kept for backward compatibility).
    Use run_landmarks() below for pipeline integration.
    """
    print(f"[Landmarks] Loading model: {weights_path}")
    model = load_model(weights_path)
    print(f"[Landmarks] Model ready.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video_path}")
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"[Landmarks] {W}x{H} @ {fps:.1f} fps, {n_tot} frames")

    per_frame_keypoints = []
    n_with_kp = 0
    total_kp  = 0
    fi = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            keypoints = detect_landmarks(frame, model, conf_threshold)
            per_frame_keypoints.append(keypoints)
            if keypoints:
                n_with_kp += 1
                total_kp += len(keypoints)
                draw_landmarks(frame, keypoints)
            writer.write(frame)
            fi += 1
            if fi % log_every == 0:
                elapsed = time.time() - t0
                rate = fi / max(elapsed, 1e-3)
                print(f"   frame {fi}/{n_tot}  ({rate:.1f} fps)  "
                      f"kp_rate={100*n_with_kp/fi:.0f}%")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    print(f"\n[Landmarks] ✓ {fi} frames in {elapsed:.1f}s")
    print(f"[Landmarks]   kp rate: {n_with_kp}/{fi} ({100*n_with_kp/max(1,fi):.1f}%)")

    return {
        "total_frames":     fi,
        "frames_with_kp":   n_with_kp,
        "avg_kp_per_frame": total_kp / max(1, fi),
        "elapsed_seconds":  elapsed,
        "per_frame":        per_frame_keypoints,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Pipeline Integration (NEW)
# ═════════════════════════════════════════════════════════════════════════════

def run_landmarks(
    input_video_path: str,
    output_video_path: str,
    analytics_dir: Path,
    weights_path: str = "models/weights/court_kp.pt",
    conf_threshold: float = 0.30,
    log_every: int = 30,
) -> Dict:
    """
    Court landmark detection — pipeline-friendly wrapper.

    Args:
        input_video_path:  Path to the video to process.
        output_video_path: Where to save the video with landmarks drawn.
        analytics_dir:     Directory for saving landmark data JSON.
        weights_path:      Path to YOLO-pose weights.
        conf_threshold:    Confidence threshold for keypoints.
        log_every:         Log progress every N frames.

    Returns:
        Dict with stats and per-frame keypoint data.
    """
    print("\n🏀 Running Court Landmark Detection...")

    result = render_landmarks_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        log_every=log_every,
    )

    # Save per-frame keypoints to analytics dir
    landmarks_json_path = analytics_dir / "landmarks.json"
    landmarks_json_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = {
        "total_frames": result["total_frames"],
        "frames_with_kp": result["frames_with_kp"],
        "avg_kp_per_frame": result["avg_kp_per_frame"],
        "elapsed_seconds": result["elapsed_seconds"],
        "per_frame": [
            [{"idx": idx, "x": x, "y": y, "conf": conf} for idx, x, y, conf in frame_kps]
            for frame_kps in result["per_frame"]
        ],
    }

    with open(landmarks_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)

    print(f"   Landmarks JSON → {landmarks_json_path}")
    print(f"   Landmarks video → {output_video_path}")

    return result
