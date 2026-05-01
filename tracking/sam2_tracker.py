"""
tracking/sam2_tracker.py
─────────────────────────
SAM2 tracker with chunked processing to prevent RAM/VRAM crashes.

Memory strategy
───────────────
Instead of loading all frames at once (~1.3 GB RAM), the video is split
into overlapping chunks of CHUNK_SIZE frames. Each chunk:
  1. Loads N frames into a temp JPEG directory
  2. Initialises SAM2 state for that chunk
  3. Propagates tracking
  4. Frees all memory before next chunk

Overlap between chunks ensures track continuity across boundaries.

Requires
────────
  pip install sam2
  mkdir -p models/sam2
  wget -P models/sam2 \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
"""

from __future__ import annotations

import os
import gc
import cv2
import shutil
import tempfile
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

try:
    from sam2.build_sam import build_sam2_video_predictor
    _SAM2_AVAILABLE = True
except ImportError:
    _SAM2_AVAILABLE = False

from team_clustering.clusterer import (
    CLASS_PLAYER, CLASS_REF,
    TEAM_UNKNOWN, TEAM_COLORS, TEAM_NAMES,
)
from detection.detector import CLASS_ID_TO_NAME, _DEFAULT_CLASS_COLORS


REF_ID_OFFSET = 10_000
TRAIL_LENGTH  = 30

_MODEL_CFGS = {
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cleanup_mask(mask: np.ndarray, max_dist_ratio: float = 0.3) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
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
    SAM2 video tracker with chunked processing.

    Parameters
    ----------
    checkpoint        : Path to SAM2 .pt checkpoint.
    model_size        : 'small' | 'base' | 'large'
    device            : 'cuda' or 'cpu'
    chunk_size        : Frames per chunk. Lower = less RAM.
                        60  frames ≈ 300 MB  (safe for most laptops)
                        30  frames ≈ 150 MB  (if still crashing)
    chunk_overlap     : Frames shared between chunks for continuity.
    reprompt_interval : Add YOLO anchor prompts every N frames.
    max_match_dist    : Max centroid distance (px) for anchor matching.
    gpu_scale         : Resize frames for SAM2. 0.5 = 4× less VRAM.
    """

    def __init__(
        self,
        checkpoint:        str   = "models/sam2/sam2.1_hiera_small.pt",
        model_size:        str   = "small",
        device:            str   = "cuda",
        chunk_size:        int   = 60,
        chunk_overlap:     int   = 10,
        reprompt_interval: int   = 30,
        max_match_dist:    float = 150.0,
        gpu_scale:         float = 0.5,
    ) -> None:
        if not _SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is required.\n"
                "Install: pip install sam2\n"
                "Weights: wget -P models/sam2 "
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
                "sam2.1_hiera_small.pt"
            )

        self.device            = device
        self.model_size        = model_size
        self.chunk_size        = chunk_size
        self.chunk_overlap     = chunk_overlap
        self.reprompt_interval = reprompt_interval
        self.max_match_dist    = max_match_dist
        self.gpu_scale         = gpu_scale

        model_cfg = _MODEL_CFGS[model_size]
        print(f"[SAM2Tracker] Loading SAM2 ({model_size}) from {checkpoint}")
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )
        print(f"[SAM2Tracker] Ready — chunk={chunk_size} overlap={chunk_overlap} "
              f"scale={gpu_scale}")

        # ID tracking state
        self._obj_to_track:    dict[int, int]              = {}
        self._track_to_obj:    dict[int, int]              = {}
        self._track_class:     dict[int, int]              = {}
        self._last_pos:        dict[int, tuple[float,float]] = {}
        self._ref_display_id:  dict[int, int]              = {}
        self._history:         dict[int, list[tuple]]      = defaultdict(list)
        self._scale:           float                       = gpu_scale

        self._all_player_ids:  set[int] = set()
        self._all_ref_ids:     set[int] = set()

        # Counters persist across chunks
        self._next_player_obj: int = 1
        self._next_ref_obj:    int = 5001
        self._next_ref_display:int = 1

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(
        self,
        frames:      list[np.ndarray],
        frame0_dets: list[dict],
        anchor_dets: dict[int, list[dict]] | None = None,
    ) -> dict[int, list[dict]]:
        """
        Process the full video in memory-safe chunks.

        Parameters
        ----------
        frames       : All video frames.
        frame0_dets  : YOLO detections on frame 0.
        anchor_dets  : Pre-computed {frame_idx: dets} for anchor frames.

        Returns
        -------
        {frame_idx: [det_dict, ...]}
        """
        total        = len(frames)
        all_results: dict[int, list[dict]] = {}

        self._assign_initial_ids(frame0_dets)

        chunks = self._make_chunks(total)
        print(f"[SAM2Tracker] {total} frames → {len(chunks)} chunks "
              f"of ~{self.chunk_size} frames")

        for chunk_idx, (start, end) in enumerate(chunks):
            print(f"\n[SAM2Tracker] Chunk {chunk_idx+1}/{len(chunks)}  "
                  f"frames {start}–{end-1}")

            chunk_frames = frames[start:end]

            # Collect anchor prompts that fall in this chunk
            chunk_anchors: dict[int, list[dict]] = {}
            if anchor_dets:
                for fi, dets in anchor_dets.items():
                    if start < fi < end:   # skip frame 0 (handled separately)
                        chunk_anchors[fi - start] = dets

            chunk_results = self._process_chunk(
                chunk_frames  = chunk_frames,
                global_offset = start,
                chunk_anchors = chunk_anchors,
                is_first      = (chunk_idx == 0),
                frame0_dets   = frame0_dets if chunk_idx == 0 else None,
            )

            # Skip overlap frames from non-first chunks to avoid duplicates
            skip = self.chunk_overlap if chunk_idx > 0 else 0
            for local_fi, dets in chunk_results.items():
                if local_fi >= skip:
                    all_results[start + local_fi] = dets

            del chunk_frames
            free_memory()

            pct = 100.0 * end / total
            print(f"[SAM2Tracker] Chunk {chunk_idx+1} done ({pct:.0f}% complete)")

        print(f"\n[SAM2Tracker] Complete — "
              f"players={self.total_player_tracks}  refs={self.total_ref_tracks}")
        return all_results

    # ── Chunk processing ──────────────────────────────────────────────────────

    def _process_chunk(
        self,
        chunk_frames:  list[np.ndarray],
        global_offset: int,
        chunk_anchors: dict[int, list[dict]],
        is_first:      bool,
        frame0_dets:   list[dict] | None,
    ) -> dict[int, list[dict]]:
        temp_dir = tempfile.mkdtemp(prefix="sam2_chunk_")
        h0, w0   = chunk_frames[0].shape[:2]

        if self.gpu_scale < 1.0:
            new_w = int(w0 * self.gpu_scale)
            new_h = int(h0 * self.gpu_scale)
        else:
            new_w, new_h = w0, h0

        try:
            # Write chunk frames as JPEGs
            for i, frame in enumerate(chunk_frames):
                f = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA) \
                    if self.gpu_scale < 1.0 else frame
                cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), f)

            inference_state = self.predictor.init_state(
                video_path           = temp_dir,
                offload_video_to_cpu = True,
                offload_state_to_cpu = True,
            )
            self.predictor.reset_state(inference_state)

            # Prompt frame 0 of this chunk
            if is_first and frame0_dets is not None:
                self._prompt_with_dets(inference_state, 0, frame0_dets)
            else:
                self._prompt_from_last_pos(inference_state, 0)

            # Add anchor prompts inside this chunk
            for local_fi, dets in sorted(chunk_anchors.items()):
                if local_fi > 0:
                    self._prompt_with_dets(inference_state, local_fi, dets)

            # Propagate
            chunk_results: dict[int, list[dict]] = {}
            with torch.inference_mode(), \
                 torch.autocast(self.device, dtype=torch.bfloat16):
                for fi, obj_ids, mask_logits in \
                        self.predictor.propagate_in_video(inference_state):
                    dets = []
                    for i, obj_id in enumerate(obj_ids):
                        track_id = self._obj_to_track.get(int(obj_id))
                        if track_id is None:
                            continue

                        mask = (mask_logits[i, 0] > 0.0).cpu().numpy()
                        mask = cleanup_mask(mask)
                        bbox = mask_to_bbox(mask)
                        if bbox is None:
                            continue

                        if self.gpu_scale < 1.0:
                            bbox = bbox / self.gpu_scale

                        cx  = (bbox[0] + bbox[2]) / 2.0
                        cy  = (bbox[1] + bbox[3]) / 2.0
                        cid = self._track_class.get(track_id, CLASS_PLAYER)

                        self._last_pos[track_id] = (cx, cy)
                        self._append_history(track_id, (cx, cy))

                        if cid == CLASS_PLAYER:
                            self._all_player_ids.add(track_id)
                        elif cid == CLASS_REF:
                            self._all_ref_ids.add(track_id)

                        dets.append({
                            "bbox":       bbox,
                            "center":     (float(cx), float(cy)),
                            "conf":       1.0,
                            "class_id":   cid,
                            "class_name": CLASS_ID_TO_NAME.get(cid, "unknown"),
                            "track_id":   track_id,
                            "team_id":    TEAM_UNKNOWN,
                            "mask":       mask,
                        })

                    chunk_results[fi] = dets

            self.predictor.reset_state(inference_state)
            del inference_state

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return chunk_results

    # ── ID assignment ─────────────────────────────────────────────────────────

    def _assign_initial_ids(self, frame0_dets: list[dict]) -> None:
        p_dets = sorted(
            [d for d in frame0_dets if d["class_id"] == CLASS_PLAYER],
            key=lambda d: d["center"][0],
        )
        r_dets = sorted(
            [d for d in frame0_dets if d["class_id"] == CLASS_REF],
            key=lambda d: d["center"][0],
        )
        for det in p_dets:
            obj_id   = self._next_player_obj
            track_id = obj_id
            self._obj_to_track[obj_id]   = track_id
            self._track_to_obj[track_id] = obj_id
            self._track_class[track_id]  = CLASS_PLAYER
            self._last_pos[track_id]     = det["center"]
            self._next_player_obj       += 1

        for det in r_dets:
            obj_id   = self._next_ref_obj
            track_id = obj_id + REF_ID_OFFSET
            self._obj_to_track[obj_id]       = track_id
            self._track_to_obj[track_id]     = obj_id
            self._track_class[track_id]      = CLASS_REF
            self._last_pos[track_id]         = det["center"]
            self._ref_display_id[track_id]   = self._next_ref_display
            self._next_ref_obj              += 1
            self._next_ref_display          += 1

        print(f"[SAM2Tracker] IDs assigned: {len(p_dets)} players, {len(r_dets)} refs")

    def _prompt_with_dets(
        self, state, local_fi: int, dets: list[dict]
    ) -> None:
        matched: set[int] = set()
        for track_id, (lx, ly) in self._last_pos.items():
            obj_id = self._track_to_obj.get(track_id)
            if obj_id is None:
                continue
            cid = self._track_class.get(track_id, CLASS_PLAYER)
            best_idx, best_dist = -1, float("inf")
            for di, det in enumerate(dets):
                if di in matched or det["class_id"] != cid:
                    continue
                dx, dy = det["center"]
                d = np.hypot(dx - lx, dy - ly)
                if d < best_dist:
                    best_dist, best_idx = d, di
            if best_idx != -1 and best_dist <= self.max_match_dist:
                matched.add(best_idx)
                box = np.array(
                    dets[best_idx]["bbox"][:4], dtype=np.float32
                ) * self.gpu_scale
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=local_fi,
                    obj_id=obj_id, box=box,
                )

    def _prompt_from_last_pos(self, state, local_fi: int) -> None:
        for track_id, (lx, ly) in self._last_pos.items():
            obj_id = self._track_to_obj.get(track_id)
            if obj_id is None:
                continue
            point  = np.array([[lx * self.gpu_scale, ly * self.gpu_scale]],
                               dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=state, frame_idx=local_fi,
                obj_id=obj_id, points=point, labels=labels,
            )

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame:       np.ndarray,
        frame_dets:  list[dict],
        show_trails: bool  = True,
        show_ids:    bool  = True,
        show_teams:  bool  = True,
        show_masks:  bool  = True,
        mask_alpha:  float = 0.55,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = frame.shape[:2]

        for det in frame_dets:
            tid     = det.get("track_id", -1)
            team_id = det.get("team_id", TEAM_UNKNOWN)
            cid     = det["class_id"]
            x1, y1, x2, y2 = det["bbox"].astype(int)

            color = (
                TEAM_COLORS.get(team_id, _DEFAULT_CLASS_COLORS.get(cid, (180,180,180)))
                if (show_teams and team_id != TEAM_UNKNOWN)
                else _DEFAULT_CLASS_COLORS.get(cid, (180,180,180))
            )

            # Mask + contour for players and refs only
            if show_masks and "mask" in det and det["mask"] is not None:
                mask = det["mask"]
                if mask.shape != (h, w):
                    mask = cv2.resize(
                        mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                overlay = vis.copy()
                overlay[mask] = color
                vis = cv2.addWeighted(overlay, mask_alpha, vis, 1-mask_alpha, 0)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(vis, contours, -1, color, 1, cv2.LINE_AA)
            else:
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)

            # Motion trail
            if show_trails:
                history = self._history.get(tid, [])
                for i in range(1, len(history)):
                    alpha = i / len(history)
                    c     = tuple(int(v * alpha) for v in color)
                    cv2.line(vis,
                             (int(history[i-1][0]), int(history[i-1][1])),
                             (int(history[i][0]),   int(history[i][1])),
                             c, max(1, int(3*alpha)), cv2.LINE_AA)

            # Label — no bounding box
            if show_ids:
                if cid == CLASS_REF:
                    disp   = self._ref_display_id.get(tid, tid - REF_ID_OFFSET)
                    id_str = f"Ref#{disp}"
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
    def total_ref_tracks(self):    return len(self._all_ref_ids)

    def get_track_history(self, tid): return self._history.get(tid, [])

    def _make_chunks(self, total: int) -> list[tuple[int, int]]:
        chunks, start = [], 0
        while start < total:
            end = min(start + self.chunk_size, total)
            chunks.append((start, end))
            if end == total:
                break
            start = end - self.chunk_overlap
        return chunks

    def _append_history(self, tid, center):
        buf = self._history[tid]
        buf.append(center)
        if len(buf) > TRAIL_LENGTH:
            self._history[tid] = buf[-TRAIL_LENGTH:]

    def __repr__(self):
        return (f"SAM2Tracker(chunk={self.chunk_size}, "
                f"scale={self.gpu_scale}, "
                f"players={self.total_player_tracks}, "
                f"refs={self.total_ref_tracks})")