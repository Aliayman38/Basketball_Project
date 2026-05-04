"""
src/tracking/tracker.py
────────────────────────
Multi-Object Tracking using YOLOv8 for detection and 
DeepOCSORT (via BoxMOT) for advanced non-linear tracking.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# BULLETPROOF BOXMOT IMPORT
# Safely hunts down DeepOCSORT regardless of BoxMOT version
# ==========================================
try:
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort as DeepOCSORT
except ImportError:
    from boxmot import DeepOCSORT

from team_clustering.clusterer import (
    CLASS_PLAYER,
    CLASS_REF,
    CLASS_HOOP,
    CLASS_OVERLAY,
    TEAM_UNKNOWN,
    TEAM_COLORS,
    TEAM_NAMES,
)
from detection.detector import CLASS_ID_TO_NAME, _DEFAULT_CLASS_COLORS

# ── Track ID namespace ────────────────────────────────────────────────────────
REF_ID_OFFSET     = 10_000
HOOP_ID_OFFSET    = 20_000
OVERLAY_ID_OFFSET = 30_000

TRAIL_LENGTH = 30
TRACKED_CLASSES = [CLASS_PLAYER, CLASS_REF, CLASS_HOOP, CLASS_OVERLAY]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _draw_label(frame, label, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ─────────────────────────────────────────────────────────────────────────────
class PlayerTracker:
    def __init__(
        self,
        model_path:  str,
        conf:        float = 0.25,
        iou:         float = 0.45,
        imgsz:       int   = 1280,
        device:      str   = "0",
        tracker_cfg: str   = None, # No longer needed for DeepOCSORT
    ) -> None:
        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz
        self.device      = device

        print(f"[PlayerTracker] Loading YOLO: {model_path}")
        self.model = YOLO(model_path)

        print("[PlayerTracker] Initializing DeepOCSORT...")
        # ==========================================
        # EXTREME BASKETBALL TUNING INITIALIZATION
        # ==========================================
        self.tracker = DeepOCSORT(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device=f"cuda:{self.device}" if self.device != "cpu" else "cpu",
            half=True,
            
            # --- Basketball Specific Parameters ---
            max_age=300,           # Keep IDs alive in memory for 10 FULL SECONDS (at 30fps)
            iou_threshold=0.05,    # Be incredibly forgiving about spatial teleportation
            w_association_emb=0.8, # Force the tracker to lean 80% on visual Re-ID
            min_hits=3,            # Drop this slightly so returning players are recognized faster
            det_thresh=0.50        # Filter out background noise/crowds
        )

        self._history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self._all_player_ids: set[int] = set()
        self._all_ref_ids:    set[int] = set()
        self._frame_idx: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[dict]:
        # 1. Run YOLO Detection (No built-in tracking here)
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            classes=TRACKED_CLASSES,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # 2. Format for BoxMOT: numpy array of [x1, y1, x2, y2, conf, cls]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().reshape(-1, 1)
        clss = boxes.cls.cpu().numpy().reshape(-1, 1)
        raw_dets = np.hstack((xyxy, confs, clss))

        # 3. Apply Custom Cross-Class NMS (Kill players overlapping with refs)
        clean_dets = self._apply_cross_class_nms(raw_dets)

        if len(clean_dets) == 0:
            return []

        # 4. Feed clean detections AND the image frame into DeepOCSORT
        # BoxMOT returns: [x1, y1, x2, y2, track_id, conf, cls, ind]

        # ==========================================
        # BULLETPROOF ANTI-CRASH FILTER & CLAMP
        # ==========================================
        if len(clean_dets) > 0:
            # 1. Force float32 to prevent internal type errors
            clean_dets = np.array(clean_dets, dtype=np.float32)
            
            # 2. Strip impossible math values (NaNs and Infinities)
            clean_dets = clean_dets[~np.isnan(clean_dets).any(axis=1)]
            clean_dets = clean_dets[~np.isinf(clean_dets).any(axis=1)]
            
            if len(clean_dets) > 0:
                # 3. THE CLAMP: Force all coordinates to stay strictly inside the video frame
                img_h, img_w = frame.shape[:2]
                clean_dets[:, 0] = np.clip(clean_dets[:, 0], 0, img_w) # x1
                clean_dets[:, 1] = np.clip(clean_dets[:, 1], 0, img_h) # y1
                clean_dets[:, 2] = np.clip(clean_dets[:, 2], 0, img_w) # x2
                clean_dets[:, 3] = np.clip(clean_dets[:, 3], 0, img_h) # y2

                # 4. Calculate width and height of the clamped boxes
                widths = clean_dets[:, 2] - clean_dets[:, 0]
                heights = clean_dets[:, 3] - clean_dets[:, 1]
                
                # 5. Destroy anything smaller than 5 pixels
                valid_mask = (widths > 5) & (heights > 5)
                
                bad_boxes = len(valid_mask) - valid_mask.sum()
                if bad_boxes > 0:
                    print(f"\n[Tracker] ⚠️ Caught and destroyed {bad_boxes} corrupted bounding boxes!")
                    
                clean_dets = clean_dets[valid_mask]

        # ==========================================
        # KALMAN FILTER SHOCK ABSORBER
        # ==========================================
        try:
            # Attempt to update the tracker normally
            tracked_output = self.tracker.update(clean_dets, frame)
        except Exception as e:
            # If the internal math explodes, catch it, print a warning, and skip the frame
            print(f"\n[Tracker] 💥 Math crash caught at current frame! Skipping to save pipeline...")
            tracked_output = np.empty((0, 8)) # Return empty tracks for this single millisecond

        # 5. Parse back into our project's dictionary format
        tracked_dets = self._parse_result(tracked_output)

        # 6. Update History Trails
        for det in tracked_dets:
            tid = det["track_id"]
            if tid == -1: continue
            self._append_history(tid, det["center"])
            if det["class_id"] == CLASS_PLAYER:
                self._all_player_ids.add(tid)
            elif det["class_id"] == CLASS_REF:
                self._all_ref_ids.add(tid)

        self._frame_idx += 1
        return tracked_dets

    # ── Filtering & Parsing ───────────────────────────────────────────────────

    def _apply_cross_class_nms(self, dets: np.ndarray) -> np.ndarray:
        """Removes Player detections that heavily overlap with Non-Players."""
        players = dets[dets[:, 5] == CLASS_PLAYER]
        non_players = dets[dets[:, 5] != CLASS_PLAYER]
        
        surviving_players = []
        for p in players:
            is_overlapping = False
            for np_det in non_players:
                if _compute_iou(p[:4], np_det[:4]) > 0.60:
                    is_overlapping = True
                    break
            if not is_overlapping:
                surviving_players.append(p)
                
        if len(surviving_players) > 0:
            return np.vstack((non_players, np.array(surviving_players)))
        return non_players

    def _parse_result(self, tracked_output: np.ndarray) -> list[dict]:
        """Converts DeepOCSORT array output into our pipeline's dict format."""
        tracked = []
        if tracked_output is None or len(tracked_output) == 0:
            return tracked

        for row in tracked_output:
            x1, y1, x2, y2, raw_id, conf, cls_id, _ = row
            
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls_id == CLASS_REF:
                track_id = int(raw_id) + REF_ID_OFFSET
            elif cls_id == CLASS_HOOP:
                track_id = int(raw_id) + HOOP_ID_OFFSET
            elif cls_id == CLASS_OVERLAY:
                track_id = int(raw_id) + OVERLAY_ID_OFFSET
            else:
                track_id = int(raw_id)

            tracked.append({
                "bbox":       np.array([x1, y1, x2, y2]),
                "center":     (float(cx), float(cy)),
                "conf":       float(conf),
                "class_id":   int(cls_id),
                "class_name": CLASS_ID_TO_NAME.get(int(cls_id), "unknown"),
                "track_id":   track_id,
                "team_id":    TEAM_UNKNOWN,
            })
        return tracked

    # ── Boilerplate getters and drawing logic ─────────────────────────────────

    def get_tracked_players(self, tracked_dets: list[dict]) -> list[dict]:
        return [d for d in tracked_dets if d["class_id"] == CLASS_PLAYER and d.get("track_id", -1) != -1]

    def get_tracked_refs(self, tracked_dets: list[dict]) -> list[dict]:
        return [d for d in tracked_dets if d["class_id"] == CLASS_REF and d.get("track_id", -1) != -1]

    @property
    def total_player_tracks(self) -> int: return len(self._all_player_ids)
    
    @property
    def total_ref_tracks(self) -> int: return len(self._all_ref_ids)

    def draw_tracks(self, frame, tracked_dets, show_trails=True, show_ids=True, show_teams=True, show_conf=False):
        vis = frame.copy()
        for det in tracked_dets:
            tid, team_id, cid = det.get("track_id", -1), det.get("team_id", TEAM_UNKNOWN), det["class_id"]
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cx, cy = int(det["center"][0]), int(det["center"][1])

            color = TEAM_COLORS.get(team_id, _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180))) if (show_teams and team_id != TEAM_UNKNOWN) else _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180))

            if cid == CLASS_HOOP:
                r = max(14, (x2 - x1) // 2)
                cv2.circle(vis, (cx, cy), r, color, 2, cv2.LINE_AA)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                if show_ids: _draw_label(vis, f"Hoop #{tid - HOOP_ID_OFFSET}", x1, y1, color)
                continue

            if cid == CLASS_OVERLAY:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                continue

            if show_trails:
                history = self._history.get(tid, [])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(vis, (int(history[i-1][0]), int(history[i-1][1])), (int(history[i][0]), int(history[i][1])), c, max(1, int(3 * alpha)), cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if show_ids or show_teams:
                display_id = tid - REF_ID_OFFSET if tid >= REF_ID_OFFSET else tid
                prefix = "Ref" if cid == CLASS_REF else "P"
                parts = [f"{prefix}#{display_id}"]
                if show_teams and team_id != TEAM_UNKNOWN: parts.append(TEAM_NAMES.get(team_id, ""))
                _draw_label(vis, "  ".join(parts), x1, y1, color)

        return vis

    def _append_history(self, track_id: int, center: tuple[float, float]) -> None:
        buf = self._history[track_id]
        buf.append(center)
        if len(buf) > TRAIL_LENGTH: self._history[track_id] = buf[-TRAIL_LENGTH:]