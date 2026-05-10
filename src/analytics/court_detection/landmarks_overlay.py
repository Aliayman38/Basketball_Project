"""
src/analytics/court_detection/landmarks_overlay.py
Court landmark detection module using a custom-trained YOLO-pose model.
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

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