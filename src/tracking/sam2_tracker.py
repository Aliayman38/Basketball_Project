"""
src/tracking/sam2_tracker.py
─────────────────────────────
SAM2-based player tracker with temporal memory bank and multi-frame re-prompting.

Why players disappear and don't come back
──────────────────────────────────────────
SAM2 is initialised with bbox prompts on frame 0. If a player goes off-screen,
SAM2 has no new signal and the track fades from its memory bank. When the player
re-enters, SAM2 may fail to re-identify them.

Fix: multi-frame anchor prompting
───────────────────────────────────
SAM2 accepts prompts at ANY frame index, not just frame 0.
Every REPROMPT_INTERVAL frames, we run YOLO detection and match each detection
to the nearest known track by centroid distance. We then call
predictor.add_new_points_or_box() for that frame, giving SAM2 a fresh anchor
point to recover from. This dramatically improves re-identification after
off-screen exits.

Requires
────────
  pip install sam2
  mkdir -p models/sam2
  wget -P models/sam2 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
"""

from __future__ import annotations

import os
import cv2
import shutil
import tempfile
import numpy as np
import torch
from collections import defaultdict

try:
    from sam2.build_sam import build_sam2_video_predictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False

from src.team_clustering.clusterer import (
    CLASS_PLAYER, CLASS_REF,
    TEAM_UNKNOWN, TEAM_COLORS, TEAM_NAMES,
)
from src.detection.detector import CLASS_ID_TO_NAME, _DEFAULT_CLASS_COLORS


REF_ID_OFFSET    = 10_000
TRAIL_LENGTH     = 30
REPROMPT_INTERVAL = 30   # add anchor prompts every N frames

_MODEL_CFGS = {
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cleanup_mask(mask: np.ndarray, max_dist_ratio: float = 0.3) -> np.ndarray:
    """Remove disconnected segments — keeps only the main player body."""
    mask_u8  = mask.astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if n <= 2:
        return mask
    sizes    = stats[1:, cv2.CC_STAT_AREA]
    main_idx = int(np.argmax(sizes)) + 1
    mx, my   = centroids[main_idx]
    diag     = np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
    clean    = np.zeros_like(mask)
    for i in range(1, n):
        cx, cy = centroids[i]
        if i == main_idx or np.hypot(cx - mx, cy - my) <= max_dist_ratio * diag:
            clean[labels == i] = True
    return clean


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _draw_label(frame, label, x1, y1, color):
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
class SAM2Tracker:
    """
    SAM2-based tracker with multi-frame re-prompting for stable IDs.

    Parameters
    ----------
    checkpoint        : Path to SAM2 .pt checkpoint.
    model_size        : 'small' | 'base' | 'large'
    device            : 'cuda' or 'cpu'
    reprompt_interval : Add YOLO anchor prompts every N frames (default 30).
                        Lower = more stable but slower initialisation.
    max_match_dist    : Max centroid distance (px) to match a YOLO detection
                        to an existing track during re-prompting.
    """

    def __init__(
        self,
        checkpoint:        str   = "models/sam2/sam2.1_hiera_small.pt",
        model_size:        str   = "small",
        device:            str   = "cuda",
        reprompt_interval: int   = 30,
        max_match_dist:    float = 150.0,
    ) -> None:
        if not _SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is required.\n"
                "Install: pip install sam2\n"
                "Weights: wget -P models/sam2 "
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
            )

        self.device            = device
        self.model_size        = model_size
        self.reprompt_interval = reprompt_interval
        self.max_match_dist    = max_match_dist

        model_cfg = _MODEL_CFGS[model_size]
        print(f"[SAM2Tracker] Loading SAM2 ({model_size}) from {checkpoint}")
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )
        print(f"[SAM2Tracker] Ready.")

        # {sam2_obj_id: stable_track_id}
        self._obj_to_track:  dict[int, int] = {}
        # {stable_track_id: sam2_obj_id}
        self._track_to_obj:  dict[int, int] = {}
        # {stable_track_id: class_id}
        self._track_class:   dict[int, int] = {}
        # {stable_track_id: last known (cx, cy)}
        self._last_pos:      dict[int, tuple[float, float]] = {}

        self._history:       dict[int, list[tuple]] = defaultdict(list)
        self._inference_state = None
        self._temp_dir:      str | None = None

        # {ref_track_id: display_number} — independent 1,2,3 for refs
        self._ref_display_id: dict[int, int] = {}

        self._all_player_ids: set[int] = set()
        self._all_ref_ids:    set[int] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(
        self,
        frames:      list[np.ndarray],
        frame0_dets: list[dict],
        detector=None,           # BasketballDetector — used for re-prompting
    ) -> None:
        """
        Write frames to temp dir, init SAM2, add prompts on frame 0,
        then add anchor prompts every reprompt_interval frames.

        Parameters
        ----------
        frames       : All video frames.
        frame0_dets  : YOLO detections on frame 0.
        detector     : BasketballDetector instance for periodic re-prompting.
                       Pass None to skip re-prompting (frame 0 only).
        """
        # Write frames as numbered JPEGs
        self._temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        print(f"[SAM2Tracker] Writing {len(frames)} frames to temp dir…")
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(self._temp_dir, f"{i:05d}.jpg"), frame)

        self._inference_state = self.predictor.init_state(
            video_path           = self._temp_dir,
            offload_video_to_cpu = True,
            offload_state_to_cpu = True,
        )
        self.predictor.reset_state(self._inference_state)

        # ── Prompt on frame 0 — separate counters for players and refs ───────────
        p_dets = sorted(
            [d for d in frame0_dets if d["class_id"] == CLASS_PLAYER],
            key=lambda d: d["center"][0],
        )
        r_dets = sorted(
            [d for d in frame0_dets if d["class_id"] == CLASS_REF],
            key=lambda d: d["center"][0],
        )

        # Players: obj_id 1, 2, 3 … → track_id = obj_id (display: P#1, P#2 …)
        obj_id = 1
        for det in p_dets:
            track_id = obj_id
            self._obj_to_track[obj_id]   = track_id
            self._track_to_obj[track_id] = obj_id
            self._track_class[track_id]  = CLASS_PLAYER
            self._last_pos[track_id]     = det["center"]
            self.predictor.add_new_points_or_box(
                inference_state = self._inference_state,
                frame_idx = 0, obj_id = obj_id,
                box = np.array(det["bbox"][:4], dtype=np.float32),
            )
            obj_id += 1

        # Refs: obj_id continues from where players left off (no collision),
        # but display ID restarts at 1 independently (Ref#1, Ref#2 …)
        ref_display = 1
        for det in r_dets:
            track_id = obj_id + REF_ID_OFFSET   # internal unique key
            self._obj_to_track[obj_id]   = track_id
            self._track_to_obj[track_id] = obj_id
            self._track_class[track_id]  = CLASS_REF
            self._last_pos[track_id]     = det["center"]
            self._ref_display_id[track_id] = ref_display
            self.predictor.add_new_points_or_box(
                inference_state = self._inference_state,
                frame_idx = 0, obj_id = obj_id,
                box = np.array(det["bbox"][:4], dtype=np.float32),
            )
            obj_id      += 1
            ref_display += 1

        print(f"[SAM2Tracker] Frame 0: prompted {len(p_dets)} players, {len(r_dets)} refs")

        # ── Add anchor prompts at regular intervals ────────────────────────────
        if detector is not None and self.reprompt_interval > 0:
            self._add_anchor_prompts(frames, detector)

    def propagate(self, frames: list[np.ndarray]) -> dict[int, list[dict]]:
        """Propagate SAM2 through all frames. Returns {frame_idx: [det_dict]}."""
        if self._inference_state is None:
            raise RuntimeError("Call initialize() before propagate().")

        all_results: dict[int, list[dict]] = defaultdict(list)

        print("[SAM2Tracker] Propagating…")
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                self._inference_state
            ):
                frame_dets = []
                for i, obj_id in enumerate(obj_ids):
                    track_id = self._obj_to_track.get(int(obj_id))
                    if track_id is None:
                        continue

                    mask = (mask_logits[i, 0] > 0.0).cpu().numpy()
                    mask = cleanup_mask(mask)

                    bbox = mask_to_bbox(mask)
                    if bbox is None:
                        continue

                    cx  = (bbox[0] + bbox[2]) / 2.0
                    cy  = (bbox[1] + bbox[3]) / 2.0
                    cid = self._track_class.get(track_id, CLASS_PLAYER)

                    self._last_pos[track_id] = (cx, cy)

                    frame_dets.append({
                        "bbox":       bbox,
                        "center":     (float(cx), float(cy)),
                        "conf":       1.0,
                        "class_id":   cid,
                        "class_name": CLASS_ID_TO_NAME.get(cid, "unknown"),
                        "track_id":   track_id,
                        "team_id":    TEAM_UNKNOWN,
                        "mask":       mask,
                    })

                    self._append_history(track_id, (float(cx), float(cy)))
                    if cid == CLASS_PLAYER:
                        self._all_player_ids.add(track_id)
                    elif cid == CLASS_REF:
                        self._all_ref_ids.add(track_id)

                all_results[frame_idx] = frame_dets

                if frame_idx % 30 == 0:
                    pct = 100.0 * frame_idx / max(len(frames) - 1, 1)
                    print(f"\r[SAM2Tracker] {frame_idx}/{len(frames)} ({pct:.0f}%)",
                          end="", flush=True)

        print(f"\n[SAM2Tracker] Propagation complete.")

        # Clean up temp dir
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        return dict(all_results)

    # ── Re-prompting ──────────────────────────────────────────────────────────

    def _add_anchor_prompts(self, frames: list[np.ndarray], detector) -> None:
        """
        Every reprompt_interval frames, run YOLO and add bbox prompts for
        each known track found in the detection results.

        Matching strategy: nearest centroid within max_match_dist pixels,
        restricted to same class namespace (player vs ref).
        """
        anchor_frames = list(range(
            self.reprompt_interval,
            len(frames),
            self.reprompt_interval,
        ))
        print(f"[SAM2Tracker] Adding anchor prompts at {len(anchor_frames)} frames "
              f"(every {self.reprompt_interval} frames)…")

        for frame_idx in anchor_frames:
            frame = frames[frame_idx]
            dets  = detector.parse(detector.detect(frame))
            dets  = [d for d in dets if d["class_id"] in (CLASS_PLAYER, CLASS_REF)]

            if not dets:
                continue

            # For each known track, find the nearest matching detection
            matched_dets: set[int] = set()   # det indices already matched

            for track_id, (last_cx, last_cy) in self._last_pos.items():
                cid      = self._track_class.get(track_id, CLASS_PLAYER)
                obj_id   = self._track_to_obj.get(track_id)
                if obj_id is None:
                    continue

                best_idx  = -1
                best_dist = float("inf")

                for det_idx, det in enumerate(dets):
                    if det_idx in matched_dets:
                        continue
                    if det["class_id"] != cid:
                        continue
                    dcx, dcy = det["center"]
                    dist = np.hypot(dcx - last_cx, dcy - last_cy)
                    if dist < best_dist and dist <= self.max_match_dist:
                        best_dist = dist
                        best_idx  = det_idx

                if best_idx != -1:
                    matched_dets.add(best_idx)
                    box = np.array(dets[best_idx]["bbox"][:4], dtype=np.float32)
                    self.predictor.add_new_points_or_box(
                        inference_state = self._inference_state,
                        frame_idx       = frame_idx,
                        obj_id          = obj_id,
                        box             = box,
                    )
                    # Update last known position for next anchor interval
                    self._last_pos[track_id] = dets[best_idx]["center"]

        print(f"[SAM2Tracker] Anchor prompts added.")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame:       np.ndarray,
        frame_dets:  list[dict],
        show_trails: bool  = True,
        show_ids:    bool  = True,
        show_teams:  bool  = True,
        show_masks:  bool  = True,
        mask_alpha:  float = 0.25,
    ) -> np.ndarray:
        vis = frame.copy()

        for det in frame_dets:
            tid     = det.get("track_id", -1)
            team_id = det.get("team_id", TEAM_UNKNOWN)
            cid     = det["class_id"]

            x1, y1, x2, y2 = det["bbox"].astype(int)

            color = (
                TEAM_COLORS.get(team_id, _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180)))
                if (show_teams and team_id != TEAM_UNKNOWN)
                else _DEFAULT_CLASS_COLORS.get(cid, (180, 180, 180))
            )

            # Mask overlay
            if show_masks and "mask" in det:
                overlay = vis.copy()
                overlay[det["mask"]] = color
                vis = cv2.addWeighted(overlay, mask_alpha, vis, 1 - mask_alpha, 0)

            # Motion trail
            if show_trails:
                history = self._history.get(tid, [])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    c     = tuple(int(v * alpha) for v in color)
                    cv2.line(vis,
                             (int(history[i-1][0]), int(history[i-1][1])),
                             (int(history[i][0]),   int(history[i][1])),
                             c, max(1, int(3 * alpha)), cv2.LINE_AA)

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label
            if show_ids:
                if cid == CLASS_REF:
                    display_id = self._ref_display_id.get(tid, tid - REF_ID_OFFSET)
                    id_str = f"Ref#{display_id}"
                else:
                    id_str = f"P#{tid}"
                parts = [id_str]
                if show_teams and team_id != TEAM_UNKNOWN:
                    parts.append(TEAM_NAMES.get(team_id, ""))
                _draw_label(vis, "  ".join(p for p in parts if p), x1, y1, color)

        return vis

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_player_tracks(self): return len(self._all_player_ids)

    @property
    def total_ref_tracks(self): return len(self._all_ref_ids)

    def get_track_history(self, tid): return self._history.get(tid, [])

    def _append_history(self, tid, center):
        buf = self._history[tid]
        buf.append(center)
        if len(buf) > TRAIL_LENGTH:
            self._history[tid] = buf[-TRAIL_LENGTH:]

    def __repr__(self):
        return (f"SAM2Tracker(size={self.model_size}, "
                f"players={self.total_player_tracks}, "
                f"refs={self.total_ref_tracks})")