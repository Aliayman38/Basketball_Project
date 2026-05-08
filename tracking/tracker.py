"""
src/tracker.py
Basketball tracking pipeline: detection, tracking, team classification, trajectory recording.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Any

from detection.detector import BasketballDetector
from team_clustering.clusterer import CLIPTeamClusterer
from boxmot.trackers.botsort.botsort import BotSort

# ── Kalman stability patch ────────────────────────────────────────────────────
from boxmot.motion.kalman_filters.base import BaseKalmanFilter
import scipy.linalg

def _stable_update_state(self, z, R=None, H=None):
    _H = self._resolve_matrix(H, self.H)
    _R = self._resolve_matrix(R, self.R)
    if np.isscalar(_R):
        _R = np.eye(self.dim_z) * float(_R)
    projected_mean, projected_cov = self.project_state(H=_H, R=_R)
    eps = 1e-6
    while True:
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            projected_cov += np.eye(projected_cov.shape[0]) * eps
            eps *= 10
            if eps > 1.0:
                return self.x, self.P
    self.K = scipy.linalg.cho_solve(
        (chol_factor, lower), np.dot(self.P, _H.T).T, check_finite=False).T
    self.y  = z.reshape(-1, 1)[:self.dim_z] - projected_mean
    self.S  = projected_cov
    self.SI = scipy.linalg.cho_solve(
        (chol_factor, lower), np.eye(self.dim_z), check_finite=False)
    self.x      = self.x + np.dot(self.K, self.y)
    self.P      = self.P - np.linalg.multi_dot((self.K, projected_cov, self.K.T))
    self.z      = z.reshape(-1, 1)[:self.dim_z].copy()
    self.x_post = self.x.copy()
    self.P_post = self.P.copy()
    return self.x, self.P

BaseKalmanFilter.update_state = _stable_update_state
# ─────────────────────────────────────────────────────────────────────────────

BALL_CLASS_ID = 0
CLASS_NAMES   = {0: 'basketball', 1: 'net', 2: 'player', 3: 'referee'}

TEAM_0_DESC = "a basketball player wearing a yellow jersey"
TEAM_1_DESC = "a basketball player wearing a dark blue jersey"

CLIP_REFRESH_EVERY = 30

TEAM_BOX_COLORS = {
    0: (255, 255, 255),
    1: (0,   0,   255),
}

TEAM_NAMES_SHORT = {0: "WHITE", 1: "BLUE"}


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


# ── Trajectory helpers ────────────────────────────────────────────────────────

def make_trajectory_record(frame_idx, x1, y1, x2, y2, extra=None):
    record = {
        "frame":  frame_idx,
        "bbox":   [x1, y1, x2, y2],
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
    }
    if extra:
        record.update(extra)
    return record


# ── ID Manager ────────────────────────────────────────────────────────────────

class IDManager:
    LIMITS     = {'player': 10, 'referee': 4, 'net': 2}
    MAX_ABSENT = 60

    def __init__(self):
        self._map = {c: {} for c in self.LIMITS}
        self._age = {c: {} for c in self.LIMITS}

    def get_id(self, cls_name, ori_id):
        if cls_name not in self.LIMITS:
            return ori_id
        m = self._map[cls_name]
        if ori_id in m:
            self._age[cls_name][ori_id] = 0
            return m[ori_id]
        free = self._next_free(cls_name)
        if free is None:
            return None
        m[ori_id] = free
        self._age[cls_name][ori_id] = 0
        return free

    def update_ages(self, active):
        for cls_name in self.LIMITS:
            seen = active.get(cls_name, set())
            for ori_id in list(self._age[cls_name]):
                if ori_id in seen:
                    self._age[cls_name][ori_id] = 0
                else:
                    self._age[cls_name][ori_id] += 1
                    if self._age[cls_name][ori_id] >= self.MAX_ABSENT:
                        del self._map[cls_name][ori_id]
                        del self._age[cls_name][ori_id]

    def _next_free(self, cls_name):
        used = set(self._map[cls_name].values())
        for i in range(1, self.LIMITS[cls_name] + 1):
            if i not in used:
                return i
        return None


# ── Tracking Pipeline ─────────────────────────────────────────────────────────

class BasketballTracker:
    """Encapsulates the full detection → tracking → team classification pipeline."""

    def __init__(
        self,
        model_path: str,
        reid_path: str,
        device: torch.device,
        team_0_desc: str = TEAM_0_DESC,
        team_1_desc: str = TEAM_1_DESC,
        clip_refresh: int = CLIP_REFRESH_EVERY,
    ):
        self.detector   = BasketballDetector(model_path)
        self.id_manager = IDManager()
        self.clip       = CLIPTeamClusterer(team_0_desc=team_0_desc, team_1_desc=team_1_desc)
        self.clip_refresh = clip_refresh

        self.tracker = BotSort(
            reid_weights      = Path(reid_path),
            device            = device,
            half              = True,
            track_high_thresh = 0.30,
            track_low_thresh  = 0.10,
            new_track_thresh  = 0.40,
            track_buffer      = 120,
            match_thresh      = 0.80,
            proximity_thresh  = 0.50,
            appearance_thresh = 0.40,
            cmc_method        = 'ecc',
            frame_rate        = 30,
            with_reid         = True,
            min_hits          = 1,
        )

        # Team assignment state
        self.team_cache: dict[int, tuple[int, int]] = {}       # custom_id -> (team_idx, frame)
        self.persistent_teams: dict[int, int] = {}              # custom_id -> team_idx

        # Trajectory accumulators
        self.trajectories = {
            "players":  defaultdict(list),
            "referees": defaultdict(list),
            "net":      defaultdict(list),
            "ball":     [],
        }

        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect, track, classify teams, draw, record."""

        # 1. Detect
        results = self.detector.model.predict(frame, conf=0.3, verbose=False)[0]

        ball_boxes   = []
        tracker_dets = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf   = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id == BALL_CLASS_ID:
                ball_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
            else:
                tracker_dets.append([x1, y1, x2, y2, conf, float(cls_id)])

        dets = (np.array(tracker_dets, dtype=float)
                if tracker_dets else np.empty((0, 6)))

        # 2. Track
        tracks = self.tracker.update(dets, frame)

        # 3. IDs + CLIP + draw + record
        active: dict[str, set] = {}

        for track in tracks:
            x1, y1, x2, y2, ori_id, conf, cls_idx = track[:7]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_name = CLASS_NAMES.get(int(cls_idx), 'unknown')
            ori_id   = int(ori_id)

            custom_id = self.id_manager.get_id(cls_name, ori_id)
            if custom_id is None:
                continue

            active.setdefault(cls_name, set()).add(ori_id)

            color = self.detector.colors.get(cls_name, (255, 255, 255))
            label = f'{cls_name} {custom_id}'
            team_name = None
            team_idx = None

            if cls_name == 'player':
                team_idx = self._get_team(frame, x1, y1, x2, y2, custom_id)
                if team_idx is not None:
                    color     = TEAM_BOX_COLORS[team_idx]
                    team_name = self.clip.TEAM_NAMES[team_idx]
                    label     = f'{team_name} {custom_id}'

                record = make_trajectory_record(
                    self.frame_count, x1, y1, x2, y2,
                    extra={"team": team_name} if team_name else None
                )
                self.trajectories["players"][str(custom_id)].append(record)

            elif cls_name == 'referee':
                self.trajectories["referees"][str(custom_id)].append(
                    make_trajectory_record(self.frame_count, x1, y1, x2, y2)
                )

            elif cls_name == 'net':
                self.trajectories["net"][str(custom_id)].append(
                    make_trajectory_record(self.frame_count, x1, y1, x2, y2)
                )

            draw_box(frame, x1, y1, x2, y2, label, color)

        self.id_manager.update_ages(active)

        # 4. Ball
        if ball_boxes:
            x1, y1, x2, y2, _ = max(ball_boxes, key=lambda b: b[4])
            draw_box(frame, x1, y1, x2, y2,
                     'basketball', self.detector.colors['basketball'])
            self.trajectories["ball"].append(
                make_trajectory_record(self.frame_count, x1, y1, x2, y2)
            )

        self.frame_count += 1
        return frame

    def _get_team(self, frame, x1, y1, x2, y2, custom_id: int) -> int | None:
        """Get team assignment for a player, using cache and CLIP as needed."""
        if custom_id in self.persistent_teams:
            team_idx = self.persistent_teams[custom_id]
            self.team_cache[custom_id] = (team_idx, self.frame_count)
            return team_idx

        cached = self.team_cache.get(custom_id)
        needs_update = (cached is None or
                        (self.frame_count - cached[1]) >= self.clip_refresh)

        if needs_update:
            try:
                team_idx = self.clip.predict(frame, x1, y1, x2, y2)
                self.team_cache[custom_id] = (team_idx, self.frame_count)
                self.persistent_teams[custom_id] = team_idx
            except Exception as e:
                print(f"[WARNING] CLIP error for player {custom_id}: {e}")
                team_idx = cached[0] if cached else None
        else:
            team_idx = cached[0]

        return team_idx

    def get_trajectories(self) -> dict:
        """Return trajectories as plain dicts (not defaultdicts)."""
        return {
            "players":  dict(self.trajectories["players"]),
            "referees": dict(self.trajectories["referees"]),
            "net":      dict(self.trajectories["net"]),
            "ball":     list(self.trajectories["ball"]),
        }

    def reset(self):
        """Reset all state for a new video."""
        self.trajectories = {
            "players":  defaultdict(list),
            "referees": defaultdict(list),
            "net":      defaultdict(list),
            "ball":     [],
        }
        self.frame_count = 0
        self.team_cache.clear()
        self.persistent_teams.clear()
        self.id_manager = IDManager()
