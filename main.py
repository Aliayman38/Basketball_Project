import cv2
import os
import json
import time
import numpy as np
import torch
import scipy.linalg
from pathlib import Path
from collections import defaultdict
import csv

from detection.detector              import BasketballDetector
from team_clustering.clusterer       import CLIPTeamClusterer
from boxmot.trackers.botsort.botsort import BotSort

# ── Analytics modules ───────────────────────────────────────────────────────
import sys
analytics_dir = Path(__file__).parent / "analytics"
if str(analytics_dir) not in sys.path:
    sys.path.insert(0, str(analytics_dir.parent))

from src.analytics.distance import build_report as build_distance_report, export_csv as export_distance_csv
from src.analytics.speed  import build_speed_report, export_csv as export_speed_csv

# ── Kalman stability patch ────────────────────────────────────────────────────
from boxmot.motion.kalman_filters.base import BaseKalmanFilter

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

TEAM_0_DESC = "a basketball player wearing a white jersey"
TEAM_1_DESC = "a basketball player wearing a dark blue jersey"

CLIP_REFRESH_EVERY = 15

TEAM_BOX_COLORS = {
    0: (255, 255, 255),
    1: (0,   0,   255),
}


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


def save_trajectories(trajectories: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f'   Trajectories → {path}')


# ── Analytics ───────────────────────────────────────────────────────────────────

def run_analytics(trajectories: dict, fps: float, analytics_dir: Path,
                  meters_per_pixel: float | None = None,
                  include_referees: bool = False):
    """Compute and export distance & speed reports from trajectories."""
    
    # ── Distance report ─────────────────────────────────────────────────────
    flat_trajectories: dict[str, list] = {}
    for pid, pts in trajectories.get("players", {}).items():
        flat_trajectories[pid] = pts
    if include_referees:
        for rid, pts in trajectories.get("referees", {}).items():
            flat_trajectories[f"ref_{rid}"] = pts
    
    distance_rows = build_distance_report(flat_trajectories, meters_per_pixel)
    distance_path = analytics_dir / "distance_report.csv"
    export_distance_csv(distance_rows, distance_path)
    
    # ── Speed report ─────────────────────────────────────────────────────────
    speed_input: dict[str, list[list[float]]] = {}
    for pid, records in trajectories.get("players", {}).items():
        speed_input[pid] = [
            [r["center"][0], r["center"][1], float(r["frame"])]
            for r in records
        ]
    
    speed_rows = build_speed_report(speed_input, fps=fps, meters_per_pixel=meters_per_pixel)
    speed_path = analytics_dir / "speed_report.csv"
    export_speed_csv(speed_rows, speed_path)
    
    print(f"\n📊 Analytics saved to {analytics_dir}/")


# ── Visualization ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> dict[str, dict[str, str]]:
    """Parse a CSV into {player_id: {column: value}} dict."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("player_id", "").strip()
            if pid:
                result[pid] = {k.strip(): v.strip() for k, v in row.items()}
    return result


def load_frame_index(path: str, gap_threshold: int = 30) -> dict[int, dict[str, tuple[int, int]]]:
    """Build frame-indexed lookup with linear interpolation across tracking gaps."""
    with open(path, encoding="utf-8") as f:
        players: dict[str, list] = json.load(f)["players"]

    frames: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)

    for pid, points in players.items():
        detections: list[tuple[int, int, int]] = []
        for point in points:
            try:
                frame = int(point["frame"])
                x, y = point["center"]
                detections.append((frame, int(x), int(y)))
            except (TypeError, IndexError, ValueError):
                continue

        if not detections:
            continue

        detections.sort()

        for frame, x, y in detections:
            frames[frame][pid] = (x, y)

        for (f0, x0, y0), (f1, x1, y1) in zip(detections, detections[1:]):
            gap = f1 - f0
            if gap <= 1 or gap > gap_threshold:
                continue
            for step in range(1, gap):
                t = step / gap
                xi = round(x0 + t * (x1 - x0))
                yi = round(y0 + t * (y1 - y0))
                frames[f0 + step].setdefault(pid, (xi, yi))

    return frames


_UNIT_MAP: dict[str, str] = {
    "total_distance_m":  "m",
    "total_distance_px": "px",
    "avg_speed_m_s":     "m/s",
    "avg_speed_px_s":    "px/s",
}


def pick(row: dict[str, str], metric_key: str, pixel_key: str) -> tuple[str, str]:
    """Return (value, display_unit) preferring metric, falling back to pixels."""
    if metric_key in row and row[metric_key]:
        return row[metric_key], _UNIT_MAP.get(metric_key, "m")
    if pixel_key in row and row[pixel_key]:
        return row[pixel_key], _UNIT_MAP.get(pixel_key, "px")
    return "0", "px"


def run_visualization(
    video_path: str,
    trajectories_path: str,
    distance_csv: str,
    speed_csv: str,
    output_path: str,
) -> None:
    """Overlay tracked player statistics (distance & speed) onto the tracked video."""
    
    frames_data = load_frame_index(trajectories_path)
    dist_data   = load_csv(distance_csv)
    speed_data  = load_csv(speed_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for pid, (x, y) in frames_data.get(frame_idx, {}).items():
            d_row = dist_data.get(pid, {})
            s_row = speed_data.get(pid, {})

            dist,  d_unit = pick(d_row, "total_distance_m",  "total_distance_px")
            speed, s_unit = pick(s_row, "avg_speed_m_s",     "avg_speed_px_s")

            text = f"D: {float(dist):.1f}{d_unit} | S: {float(speed):.2f}{s_unit}"
            label_x = max(5, min(x - 120, frame.shape[1] - 350))
            label_y = max(30, y - 60)

            cv2.putText(
                frame,
                text,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n🎬 Final video with stats → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Configuration ───────────────────────────────────────────────────────
    video_path        = 'data/video_2.mp4'
    model_path        = 'models/weights/last.pt'
    output_path       = 'runs/bot-sort tracking/tracking_botsort.mp4'
    trajectories_path = 'runs/bot-sort tracking/analytics/trajectories.json'
    reid_path         = 'osnet_x0_25_msmt17.pt'
    device            = torch.device('cuda:0')
    
    # Analytics config
    meters_per_pixel = 0.0264
    include_referees = False

    os.makedirs('runs', exist_ok=True)

    detector   = BasketballDetector(model_path)
    id_manager = IDManager()
    clip       = CLIPTeamClusterer(team_0_desc=TEAM_0_DESC, team_1_desc=TEAM_1_DESC)

    tracker = BotSort(
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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )

    print('🚀 Starting BoT-SORT + CLIP Team Classification...')

    # ── trajectory accumulators ───────────────────────────────────────────────
    trajectories = {
        "players":  defaultdict(list),
        "referees": defaultdict(list),
        "net":      defaultdict(list),
        "ball":     [],
    }

    frame_count = 0
    team_cache: dict[int, tuple[int, int]] = {}
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Detect ──────────────────────────────────────────────────────
        results = detector.model.predict(frame, conf=0.3, verbose=False)[0]

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

        dets   = (np.array(tracker_dets, dtype=float)
                  if tracker_dets else np.empty((0, 6)))

        # ── 2. Track ───────────────────────────────────────────────────────
        tracks = tracker.update(dets, frame)

        # ── 3. IDs + CLIP + draw + record trajectories ─────────────────────
        active: dict[str, set] = {}

        for track in tracks:
            x1, y1, x2, y2, ori_id, conf, cls_idx = track[:7]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_name = CLASS_NAMES.get(int(cls_idx), 'unknown')
            ori_id   = int(ori_id)

            custom_id = id_manager.get_id(cls_name, ori_id)
            if custom_id is None:
                continue

            active.setdefault(cls_name, set()).add(ori_id)

            color = detector.colors.get(cls_name, (255, 255, 255))
            label = f'{cls_name} {custom_id}'
            team_name = None

            if cls_name == 'player':
                cached       = team_cache.get(ori_id)
                needs_update = (cached is None or
                                (frame_count - cached[1]) >= CLIP_REFRESH_EVERY)
                if needs_update:
                    try:
                        team_idx = clip.predict(frame, x1, y1, x2, y2)
                        team_cache[ori_id] = (team_idx, frame_count)
                    except Exception:
                        team_idx = cached[0] if cached else None
                else:
                    team_idx = cached[0]

                if team_idx is not None:
                    color     = TEAM_BOX_COLORS[team_idx]
                    team_name = clip.TEAM_NAMES[team_idx]
                    label     = f'{team_name} {custom_id}'

                record = make_trajectory_record(
                    frame_count, x1, y1, x2, y2,
                    extra={"team": team_name} if team_name else None
                )
                trajectories["players"][str(custom_id)].append(record)

            elif cls_name == 'referee':
                trajectories["referees"][str(custom_id)].append(
                    make_trajectory_record(frame_count, x1, y1, x2, y2)
                )

            elif cls_name == 'net':
                trajectories["net"][str(custom_id)].append(
                    make_trajectory_record(frame_count, x1, y1, x2, y2)
                )

            draw_box(frame, x1, y1, x2, y2, label, color)

        id_manager.update_ages(active)

        # ── 4. Ball ────────────────────────────────────────────────────────
        if ball_boxes:
            x1, y1, x2, y2, _ = max(ball_boxes, key=lambda b: b[4])
            draw_box(frame, x1, y1, x2, y2,
                     'basketball', detector.colors['basketball'])
            trajectories["ball"].append(
                make_trajectory_record(frame_count, x1, y1, x2, y2)
            )

        writer.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f'Frame {frame_count}  |  '
                  f'{frame_count / (time.time() - t0):.1f} FPS')

    # ── Save ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    trajectories["players"]  = dict(trajectories["players"])
    trajectories["referees"] = dict(trajectories["referees"])
    trajectories["net"]      = dict(trajectories["net"])

    save_trajectories(trajectories, trajectories_path)

    # ── Analytics ──────────────────────────────────────────────────────────
    analytics_dir = Path(trajectories_path).parent
    run_analytics(
        trajectories=trajectories,
        fps=fps,
        analytics_dir=analytics_dir,
        meters_per_pixel=meters_per_pixel,
        include_referees=include_referees,
    )

    # ── Visualization ────────────────────────────────────────────────────────
    final_output = str(Path(output_path).parent / "final_output.mp4")
    run_visualization(
        video_path=output_path,
        trajectories_path=trajectories_path,
        distance_csv=str(analytics_dir / "distance_report.csv"),
        speed_csv=str(analytics_dir / "speed_report.csv"),
        output_path=final_output,
    )

    elapsed = time.time() - t0
    print(f'\n✅ Done — {frame_count} frames in {elapsed:.1f}s '
          f'({frame_count / elapsed:.1f} FPS avg)')
    print(f'   Tracked video  → {output_path}')
    print(f'   Final video    → {final_output}')


if __name__ == '__main__':
    main()