"""
dashboard.py
Player statistics dashboard + score banner + shot flash overlay.
"""

from __future__ import annotations

import cv2
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any

from src.analytics.shot_detector import ShotResult


# ═════════════════════════════════════════════════════════════════════════════
#  Config & Helpers
# ═════════════════════════════════════════════════════════════════════════════

_UNIT = {
    "total_distance_m": "m", "total_distance_px": "px",
    "avg_speed_m_s": "m/s", "avg_speed_px_s": "px/s",
}


def _pick(row: dict, mk: str, pk: str) -> tuple[str, str]:
    """Return (value, unit) preferring metric, falling back to pixels."""
    if row.get(mk):
        return row[mk], _UNIT.get(mk, "m")
    if row.get(pk):
        return row[pk], _UNIT.get(pk, "px")
    return "0", "px"


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            r.get("player_id", "").strip(): dict(r)
            for r in csv.DictReader(f)
            if r.get("player_id", "").strip()
        }


def _load_frame_index(path: str, gap: int = 30) -> dict[int, dict[str, tuple[int, int]]]:
    with open(path, encoding="utf-8") as f:
        players = json.load(f)["players"]
    frames: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    for pid, points in players.items():
        dets = [(int(p["frame"]), int(p["center"][0]), int(p["center"][1])) for p in points]
        dets.sort()
        for fr, x, y in dets:
            frames[fr][pid] = (x, y)
        for (f0, x0, y0), (f1, x1, y1) in zip(dets, dets[1:]):
            g = f1 - f0
            if 1 < g <= gap:
                for step in range(1, g):
                    t = step / g
                    frames[f0 + step].setdefault(pid, (round(x0 + t*(x1-x0)), round(y0 + t*(y1-y0))))
    return frames


def _load_player_teams(traj_path: str) -> dict[str, str]:
    """Most common team per player from trajectories."""
    with open(traj_path, encoding="utf-8") as f:
        data = json.load(f)
    teams: dict[str, str] = {}
    for pid, recs in data.get("players", {}).items():
        tlist = [r.get("team") for r in recs if r.get("team")]
        if tlist:
            teams[pid] = Counter(tlist).most_common(1)[0][0]
    return teams


# ═════════════════════════════════════════════════════════════════════════════
#  Score Banner
# ═════════════════════════════════════════════════════════════════════════════

def build_frame_scores(events: list[tuple[int, int]], total: int) -> dict[int, dict[int, int]]:
    """Map frame → {team: cumulative_score}."""
    events = sorted(events, key=lambda x: x[0])
    scores: dict[int, dict[int, int]] = {}
    cur = {0: 0, 1: 0}
    idx = 0
    for f in range(total):
        while idx < len(events) and events[idx][0] <= f:
            cur[events[idx][1]] += 1
            idx += 1
        scores[f] = {0: cur[0], 1: cur[1]}
    return scores


def draw_score_banner(frame, w: int, scores: dict[int, int]):
    h = 50
    cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (w, h), (100, 100, 100), 2)
    text = f"WHITE: {scores[0]}  |  BLUE: {scores[1]}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(frame, text, (w//2 - tw//2, h//2 + th//2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  Shot Flash
# ═════════════════════════════════════════════════════════════════════════════

def draw_shot_flash(frame, width: int, shot_num: int):
    text = f"🏀 SHOT MADE #{shot_num}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cx = width // 2
    cv2.rectangle(frame, (cx - tw//2 - 20, 60), (cx + tw//2 + 20, 60 + th + 20),
                  (0, 165, 255), -1)
    cv2.putText(frame, text, (cx - tw//2, 60 + th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
#  Player Dashboard
# ═════════════════════════════════════════════════════════════════════════════

class PlayerDashboard:
    """Fixed corner dashboard showing all visible player stats."""

    TEAM_COLORS = {
        "T1": (255, 255, 255),      # White
        "T2": (0, 100, 255),        # Blue (BGR)
        "WHITE": (255, 255, 255),
        "BLUE": (0, 100, 255),
        "—": (150, 150, 150),
    }

    def __init__(self, width: int, height: int, corner: str = "top-right"):
        self.row_h = 30
        self.header_h = 36
        self.col_w = [45, 115, 115, 75]          # ID | DISTANCE | SPEED | TEAM
        self.pad = 10
        self.panel_w = sum(self.col_w) + self.pad * 2
        self.panel_h = self.header_h + self.row_h * 10 + self.pad * 2

        if corner == "top-right":
            self.x = width - self.panel_w - 15
            self.y = 65
        elif corner == "top-left":
            self.x = 15
            self.y = 65
        elif corner == "bottom-right":
            self.x = width - self.panel_w - 15
            self.y = height - self.panel_h - 15
        else:
            self.x = 15
            self.y = height - self.panel_h - 15

    def _team_color(self, team: str) -> tuple:
        return self.TEAM_COLORS.get(team, (200, 200, 200))

    def draw(self, frame: Any, player_stats: list[dict]):
        n = min(len(player_stats), 10)
        actual_h = self.header_h + self.row_h * n + self.pad * 2

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y),
                      (self.x + self.panel_w, self.y + actual_h), (25, 25, 35), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Border
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.panel_w, self.y + actual_h), (80, 80, 120), 2)
        cv2.rectangle(frame, (self.x + 2, self.y + 2),
                      (self.x + self.panel_w - 2, self.y + actual_h - 2), (60, 60, 80), 1)

        # Title bar
        tb = self.header_h - 4
        cv2.rectangle(frame, (self.x + 3, self.y + 3),
                      (self.x + self.panel_w - 3, self.y + tb), (50, 50, 70), -1)
        title = "📊 PLAYER STATISTICS"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, title, (self.x + (self.panel_w - tw)//2, self.y + self.pad + th - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 240), 2, cv2.LINE_AA)

        # Headers
        headers = ["ID", "DISTANCE", "SPEED", "TEAM"]
        hx = self.x + self.pad
        hy = self.y + self.header_h + 2
        for i, h in enumerate(headers):
            cv2.putText(frame, h, (hx + sum(self.col_w[:i]) + 8, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 200), 1, cv2.LINE_AA)
        cv2.line(frame, (self.x + self.pad, hy + 6),
                 (self.x + self.panel_w - self.pad, hy + 6), (80, 80, 100), 1)

        # Rows
        yo = hy + self.row_h + 2
        for i, st in enumerate(player_stats[:10]):
            team = st.get("team", "—")
            color = self._team_color(team)
            bg = (35, 35, 45) if i % 2 == 0 else (30, 30, 40)
            cv2.rectangle(frame, (self.x + 4, yo - self.row_h + 8),
                          (self.x + self.panel_w - 4, yo + 2), bg, -1)
            cv2.rectangle(frame, (self.x + 4, yo - self.row_h + 8),
                          (self.x + 8, yo + 2), color, -1)

            # ID (centered)
            (iw, _), _ = cv2.getTextSize(str(st["id"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, str(st["id"]), (hx + (self.col_w[0] - iw)//2, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # Distance
            cv2.putText(frame, st.get("distance", "0.0"), (hx + self.col_w[0] + 8, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            # Speed
            cv2.putText(frame, st.get("speed", "0.0"), (hx + self.col_w[0] + self.col_w[1] + 8, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 100), 1, cv2.LINE_AA)
            # Team
            cv2.putText(frame, team, (hx + self.col_w[0] + self.col_w[1] + self.col_w[2] + 8, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            yo += self.row_h


# ═════════════════════════════════════════════════════════════════════════════
#  Main Visualization Orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def render_video(
    video_path: str,
    traj_path: str,
    dist_csv: Path,
    speed_csv: Path,
    out_path: str,
    shots: list[ShotResult] | None = None,
    shot_events: list[tuple[int, int]] | None = None,
):
    """Render final video with dashboard, score banner, and shot flashes."""

    frames_data = _load_frame_index(traj_path)
    dist_data   = _load_csv(dist_csv)
    speed_data  = _load_csv(speed_csv)
    player_teams = _load_player_teams(traj_path)

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    dashboard = PlayerDashboard(width, height, corner="top-right")
    frame_scores = build_frame_scores(shot_events, total) if shot_events else {}

    shot_frames: dict[int, list[int]] = defaultdict(list)
    if shots:
        for i, s in enumerate(shots, 1):
            for f in range(s.arc_start_frame, min(s.arc_end_frame + 1, total)):
                shot_frames[f].append(i)

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build visible player stats
        visible = []
        for pid, (x, y) in frames_data.get(fidx, {}).items():
            d, du = _pick(dist_data.get(pid, {}), "total_distance_m", "total_distance_px")
            s, su = _pick(speed_data.get(pid, {}), "avg_speed_m_s", "avg_speed_px_s")
            visible.append({
                "id": pid,
                "distance": f"{float(d):.1f}{du}",
                "speed": f"{float(s):.2f}{su}",
                "team": player_teams.get(pid, "—"),
            })
        visible.sort(key=lambda x: int(x["id"]))

        # Draw overlays
        dashboard.draw(frame, visible)

        if fidx in shot_frames:
            for sn in shot_frames[fidx]:
                draw_shot_flash(frame, width, sn)

        if frame_scores:
            draw_score_banner(frame, width, frame_scores.get(fidx, {0: 0, 1: 0}))

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    print(f"\n🎬 Final video with dashboard → {out_path}")
