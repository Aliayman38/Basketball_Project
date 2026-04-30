"""
src/tracking/tracker.py
────────────────────────
Multi-Object Tracking using ultralytics ByteTrack (model.track persist=True).

Track ID namespace
──────────────────
  Players  :     1 –  9 999
  Referees : 10 001 – 19 999   (REF_ID_OFFSET)
  Hoops    : 20 001 – 20 010   (HOOP_ID_OFFSET)
  Overlays : 30 001 – 30 020   (OVERLAY_ID_OFFSET)
  Ball     : -1
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

from src.team_clustering.clusterer import (
    CLASS_PLAYER, CLASS_REF, CLASS_HOOP, CLASS_OVERLAY,
    TEAM_UNKNOWN, TEAM_COLORS, TEAM_NAMES,
)
from src.detection.detector import CLASS_ID_TO_NAME, _DEFAULT_CLASS_COLORS

REF_ID_OFFSET     = 10_000
HOOP_ID_OFFSET    = 20_000
OVERLAY_ID_OFFSET = 30_000
TRAIL_LENGTH      = 30
TRACKED_CLASSES   = [CLASS_PLAYER, CLASS_REF, CLASS_HOOP, CLASS_OVERLAY]

_DEFAULT_CFG = str(Path(__file__).resolve().parents[2] / "config" / "bytetrack.yaml")


def _draw_label(frame, label, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


class PlayerTracker:

    def __init__(
        self,
        model_path:  str   = "models/weights/best.pt",
        conf:        float = 0.25,
        iou:         float = 0.45,
        imgsz:       int   = 1280,
        device:      str   = "0",
        tracker_cfg: str   = _DEFAULT_CFG,
    ) -> None:
        self.conf        = conf
        self.iou         = iou
        self.imgsz       = imgsz
        self.device      = device
        self.tracker_cfg = tracker_cfg if os.path.exists(tracker_cfg) else "bytetrack.yaml"

        print(f"[PlayerTracker] Loading  : {model_path}")
        print(f"[PlayerTracker] Config   : {self.tracker_cfg}")
        self.model = YOLO(model_path)

        self._history: dict[int, list[tuple]] = defaultdict(list)
        self._all_player_ids: set[int] = set()
        self._all_ref_ids:    set[int] = set()
        self._frame_idx: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> list[dict]:
        results = self.model.track(
            frame,
            persist  = True,
            conf     = self.conf,
            iou      = self.iou,
            imgsz    = self.imgsz,
            device   = self.device,
            tracker  = self.tracker_cfg,
            classes  = TRACKED_CLASSES,
            verbose  = False,
        )

        tracked = self._parse_result(results[0])

        # Cross-class NMS: remove player boxes overlapping hoop/ref
        priority = [d for d in tracked if d["class_id"] != CLASS_PLAYER]
        players  = [d for d in tracked if d["class_id"] == CLASS_PLAYER]
        players  = [p for p in players
                    if not any(_iou(p["bbox"], h["bbox"]) > 0.60 for h in priority)]
        tracked  = priority + players

        for det in tracked:
            tid = det["track_id"]
            if tid == -1:
                continue
            self._append_history(tid, det["center"])
            if det["class_id"] == CLASS_PLAYER:
                self._all_player_ids.add(tid)
            elif det["class_id"] == CLASS_REF:
                self._all_ref_ids.add(tid)

        self._frame_idx += 1
        return tracked

    def get_tracked_players(self, t): return [d for d in t if d["class_id"]==CLASS_PLAYER and d.get("track_id",-1)!=-1]
    def get_tracked_refs(self, t):    return [d for d in t if d["class_id"]==CLASS_REF    and d.get("track_id",-1)!=-1]
    def get_track_history(self, tid): return self._history.get(tid, [])

    @property
    def total_player_tracks(self): return len(self._all_player_ids)
    @property
    def total_ref_tracks(self):    return len(self._all_ref_ids)
    @property
    def frame_count(self):         return self._frame_idx

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_tracks(self, frame, tracked_dets,
                    show_trails=True, show_ids=True,
                    show_teams=True, show_conf=False):
        vis = frame.copy()
        for det in tracked_dets:
            tid     = det.get("track_id", -1)
            team_id = det.get("team_id", TEAM_UNKNOWN)
            cid     = det["class_id"]

            x1, y1, x2, y2 = det["bbox"].astype(int)
            cx, cy          = int(det["center"][0]), int(det["center"][1])

            color = (
                TEAM_COLORS.get(team_id, _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180)))
                if (show_teams and team_id != TEAM_UNKNOWN)
                else _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180))
            )

            if tid == -1:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                continue

            if cid == CLASS_HOOP:
                r = max(14, (x2 - x1) // 2)
                cv2.circle(vis, (cx, cy), r,      color, 2, cv2.LINE_AA)
                cv2.circle(vis, (cx, cy), r // 2, color, 2, cv2.LINE_AA)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                if show_ids:
                    _draw_label(vis, f"Hoop #{tid - HOOP_ID_OFFSET}", x1, y1, color)
                continue

            if cid == CLASS_OVERLAY:
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                continue

            # Player / Ref
            if show_trails:
                history = self._history.get(tid, [])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    c     = tuple(int(v * alpha) for v in color)
                    cv2.line(vis,
                             (int(history[i-1][0]), int(history[i-1][1])),
                             (int(history[i][0]),   int(history[i][1])),
                             c, max(1, int(3*alpha)), cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if show_ids or show_teams:
                display_id = tid - REF_ID_OFFSET if tid >= REF_ID_OFFSET else tid
                prefix     = "Ref" if cid == CLASS_REF else "P"
                parts      = [f"{prefix}#{display_id}"]
                if show_teams and team_id != TEAM_UNKNOWN:
                    parts.append(TEAM_NAMES.get(team_id, ""))
                if show_conf:
                    parts.append(f"{det['conf']:.2f}")
                _draw_label(vis, "  ".join(p for p in parts if p), x1, y1, color)

        return vis

    def reset(self):
        self._history.clear()
        self._all_player_ids.clear()
        self._all_ref_ids.clear()
        self._frame_idx = 0

    def _parse_result(self, result) -> list[dict]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        bboxes  = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.int().cpu().numpy()
        confs   = boxes.conf.float().cpu().numpy()
        has_ids = boxes.id is not None
        raw_ids = boxes.id.int().cpu().numpy() if has_ids else [-1] * len(boxes)

        out = []
        for xyxy, raw_id, cls_id, conf in zip(bboxes, raw_ids, cls_ids, confs):
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0

            if raw_id == -1:              track_id = -1
            elif cls_id == CLASS_REF:     track_id = int(raw_id) + REF_ID_OFFSET
            elif cls_id == CLASS_HOOP:    track_id = int(raw_id) + HOOP_ID_OFFSET
            elif cls_id == CLASS_OVERLAY: track_id = int(raw_id) + OVERLAY_ID_OFFSET
            else:                         track_id = int(raw_id)

            out.append({
                "bbox":       xyxy,
                "center":     (float(cx), float(cy)),
                "conf":       float(conf),
                "class_id":   int(cls_id),
                "class_name": CLASS_ID_TO_NAME.get(int(cls_id), "unknown"),
                "track_id":   track_id,
                "team_id":    TEAM_UNKNOWN,
            })
        return out

    def _append_history(self, tid, center):
        buf = self._history[tid]
        buf.append(center)
        if len(buf) > TRAIL_LENGTH:
            self._history[tid] = buf[-TRAIL_LENGTH:]

    def __repr__(self):
        return (f"PlayerTracker(frame={self._frame_idx}, "
                f"players={self.total_player_tracks}, refs={self.total_ref_tracks})")