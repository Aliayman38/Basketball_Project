"""
src/detection/detector.py
──────────────────────────
RT-DETR detection wrapper — aligned to the Roboflow basketball-players v11
dataset with 6 classes.

Dataset class map  (data/basketball.yaml)
──────────────────────────────────────────
  0 → Ball
  1 → Clock
  2 → Hoop
  3 → Overlay
  4 → Player
  5 → Ref
"""

from __future__ import annotations

import numpy as np
from ultralytics import RTDETR

# ── Single source of truth for class IDs ─────────────────────────────────────
from team_clustering.clusterer import (
    CLASS_BALL,
    CLASS_CLOCK,
    CLASS_HOOP,
    CLASS_OVERLAY,
    CLASS_PLAYER,
    CLASS_REF,
    TEAM_COLORS,
    TEAM_NAMES,
)

# ── Class-id → name  (mirrors data/basketball.yaml `names` list) ─────────────
CLASS_ID_TO_NAME: dict[int, str] = {
    CLASS_BALL:    "Ball",
    CLASS_CLOCK:   "Clock",
    CLASS_HOOP:    "Hoop",
    CLASS_OVERLAY: "Overlay",
    CLASS_PLAYER:  "Player",
    CLASS_REF:     "Ref",
}

CLASS_NAME_TO_ID: dict[str, int] = {v: k for k, v in CLASS_ID_TO_NAME.items()}

# Classes we actually care about at inference time
CLASSES_OF_INTEREST = {CLASS_BALL, CLASS_PLAYER, CLASS_REF, CLASS_HOOP, CLASS_OVERLAY}


# ─────────────────────────────────────────────────────────────────────────────
class BasketballDetector:
    """
    Thin wrapper around an RT-DETR model for basketball detection.

    Parameters
    ----------
    model_path : str   Path to RT-DETR .pt weights.
    conf       : float Detection confidence threshold.
    iou        : float NMS IoU threshold.
    imgsz      : int   Inference resolution (RT-DETR native = 640).
    device     : str   '0' for first GPU, 'cpu' for CPU-only.
    """

    def __init__(
        self,
        model_path: str   = "models/RT-DETR/RT-DETR.pt",
        conf:       float = 0.30,
        iou:        float = 0.45,
        imgsz:      int   = 640,
        device:     str   = "0",
    ) -> None:
        self.model_path = model_path
        self.conf       = conf
        self.iou        = iou
        self.imgsz      = imgsz
        self.device     = device

        print(f"[Detector] Loading RT-DETR weights: {model_path}")
        self.model = RTDETR(model_path)
        print(f"[Detector] Ready — conf={conf}  iou={iou}  imgsz={imgsz}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        """
        Run inference on a single BGR frame.

        Returns
        -------
        ultralytics Results object  (result.boxes contains raw detections)
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return results[0]

    def parse(self, result) -> list[dict]:
        """
        Parse a YOLO Results object into a list of detection dicts.

        Each dict contains:
            bbox       : np.ndarray  shape (4,)  [x1, y1, x2, y2]  float
            center     : tuple       (cx, cy)    float
            conf       : float
            class_id   : int
            class_name : str
        """
        detections: list[dict] = []
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cid   = int(box.cls[0].cpu())

            # Skip classes we don't use (Clock, Hoop, Overlay)
            if cid not in CLASSES_OF_INTEREST:
                continue

            xyxy  = box.xyxy[0].cpu().numpy().astype(float)
            conf  = float(box.conf[0].cpu())
            cx    = (xyxy[0] + xyxy[2]) / 2.0
            cy    = (xyxy[1] + xyxy[3]) / 2.0

            detections.append(
                {
                    "bbox":       xyxy,
                    "center":     (cx, cy),
                    "conf":       conf,
                    "class_id":   cid,
                    "class_name": CLASS_ID_TO_NAME.get(cid, "unknown"),
                }
            )

        return detections

    # ── Convenience filters ───────────────────────────────────────────────────

    def get_players(self, detections: list[dict]) -> list[dict]:
        """Return only CLASS_PLAYER detections."""
        return [d for d in detections if d["class_id"] == CLASS_PLAYER]

    def get_referees(self, detections: list[dict]) -> list[dict]:
        """Return only CLASS_REF detections."""
        return [d for d in detections if d["class_id"] == CLASS_REF]

    def get_ball(self, detections: list[dict]) -> dict | None:
        """
        Return the single highest-confidence CLASS_BALL detection, or None.
        (There is only one ball on court — take the top-confidence pick.)
        """
        balls = [d for d in detections if d["class_id"] == CLASS_BALL]
        return max(balls, key=lambda d: d["conf"]) if balls else None

    # ── Visualisation helper ──────────────────────────────────────────────────

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        team_labels: dict[int, int] | None = None,
        track_ids:   dict[int, int] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of `frame`.

        Parameters
        ----------
        frame        : BGR frame (not modified in-place)
        detections   : list of detection dicts from parse()
        team_labels  : optional {det_index: team_id} to colour by team
        track_ids    : optional {det_index: track_id} to show IDs

        Returns
        -------
        Annotated BGR frame.
        """
        import cv2
        vis = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"].astype(int)
            cid  = det["class_id"]
            name = det["class_name"]
            conf = det["conf"]

            # Choose colour: team colour if assigned, else class default
            team = (team_labels or {}).get(i, -1)
            color = TEAM_COLORS.get(team, _DEFAULT_CLASS_COLORS.get(cid, (200, 200, 200)))

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label text
            tid_str  = f" #{track_ids[i]}" if track_ids and i in track_ids else ""
            team_str = f" {TEAM_NAMES.get(team, '')}" if team != -1 else ""
            label    = f"{name}{tid_str}{team_str} {conf:.2f}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return vis

    # ── Misc ──────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Dummy forward pass so the first real frame isn't slow."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[Detector] Warmup complete.")

    def __repr__(self) -> str:
        return (
            f"BasketballDetector("
            f"weights={self.model_path!r}, "
            f"conf={self.conf}, iou={self.iou}, imgsz={self.imgsz})"
        )


# ── Default per-class box colours (used before team assignment) ───────────────
_DEFAULT_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    CLASS_BALL:    (  0, 165, 255),   # orange  — ball
    CLASS_CLOCK:   (200, 200,   0),   # yellow  — clock
    CLASS_HOOP:    (  0, 255, 255),   # cyan    — hoop
    CLASS_OVERLAY: (180,   0, 180),   # purple  — overlay
    CLASS_PLAYER:  (160, 160, 160),   # grey    — player (before team cluster)
    CLASS_REF:     ( 50,  50, 220),   # red     — referee
}