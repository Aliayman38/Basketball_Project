"""
src/analytics/possession.py
Ball possession analytics: track which player/team has the ball per frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class PossessionFrame:
    """Possession state for a single frame."""
    frame: int
    player_id: Optional[str] = None
    team: Optional[str] = None
    distance: Optional[float] = None  # pixels


@dataclass
class PossessionReport:
    """Aggregated possession report across the full video."""
    frames: List[PossessionFrame] = field(default_factory=list)

    # Per-player stats: {player_id: total_frames}
    player_possession_frames: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Per-team stats: {team_name: total_frames}
    team_possession_frames: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Possession changes: list of (frame, from_team, to_team)
    changes: List[Tuple[int, Optional[str], Optional[str]]] = field(default_factory=list)

    total_frames: int = 0

    @property
    def player_percentages(self) -> Dict[str, float]:
        """Possession percentage per player."""
        if self.total_frames == 0:
            return {}
        return {
            pid: (count / self.total_frames) * 100
            for pid, count in self.player_possession_frames.items()
        }

    @property
    def team_percentages(self) -> Dict[str, float]:
        """Possession percentage per team."""
        if self.total_frames == 0:
            return {}
        return {
            team: (count / self.total_frames) * 100
            for team, count in self.team_possession_frames.items()
        }


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point from bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def build_possession_report(
    trajectories: dict,
    ball_threshold: float = 80.0,
    min_consecutive_frames: int = 3,
    team_names: Tuple[str, str] = ("Team 0", "Team 1"),
) -> PossessionReport:
    """
    Build a possession report from trajectories.

    Args:
        trajectories: Dict with "players" and "ball" keys from tracker output.
        ball_threshold: Max pixel distance between ball and player center to count as possession.
        min_consecutive_frames: Minimum consecutive frames to confirm a possession change (debounce).
        team_names: Tuple of (team_0_name, team_1_name) for mapping.

    Returns:
        PossessionReport with per-frame and aggregated stats.
    """
    report = PossessionReport()

    players = trajectories.get("players", {})
    ball_records = trajectories.get("ball", [])

    if not ball_records:
        print("⚠️  No ball trajectory data found in trajectories.")
        return report

    # Build frame-indexed lookups
    # Players: {frame: {player_id: {"center": (x,y), "team": str, "bbox": [...]}}}
    player_by_frame: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for pid, records in players.items():
        for rec in records:
            frame = rec["frame"]
            player_by_frame[frame][pid] = {
                "center": tuple(rec["center"]),
                "team": rec.get("team"),
                "bbox": rec.get("bbox", [0, 0, 0, 0]),
            }

    # Ball: {frame: {"center": (x,y), "bbox": [...]}}
    ball_by_frame: Dict[int, dict] = {}
    for rec in ball_records:
        frame = rec["frame"]
        center = tuple(rec["center"]) if "center" in rec else get_bbox_center(rec.get("bbox", [0,0,0,0]))
        ball_by_frame[frame] = {
            "center": center,
            "bbox": rec.get("bbox", [0, 0, 0, 0]),
        }

    all_frames = sorted(set(ball_by_frame.keys()) | set(player_by_frame.keys()))
    report.total_frames = len(all_frames)

    last_team: Optional[str] = None
    last_player: Optional[str] = None
    consecutive_count = 0
    confirmed_team: Optional[str] = None
    confirmed_player: Optional[str] = None

    for frame in all_frames:
        ball_data = ball_by_frame.get(frame)
        frame_players = player_by_frame.get(frame, {})

        best_player: Optional[str] = None
        best_team: Optional[str] = None
        best_dist: float = float('inf')

        if ball_data and frame_players:
            ball_center = ball_data["center"]

            for pid, pdata in frame_players.items():
                player_center = pdata["center"]
                dist = euclidean_distance(ball_center, player_center)

                if dist < ball_threshold and dist < best_dist:
                    best_dist = dist
                    best_player = pid
                    best_team = pdata["team"]

        # Debounce possession changes
        if best_player == last_player and best_team == last_team:
            consecutive_count += 1
        else:
            consecutive_count = 1
            last_player = best_player
            last_team = best_team

        # Confirm possession only after min_consecutive_frames
        if consecutive_count >= min_consecutive_frames:
            if confirmed_player != best_player or confirmed_team != best_team:
                report.changes.append((frame, confirmed_team, best_team))
                confirmed_player = best_player
                confirmed_team = best_team

        # Use confirmed possession for stats (or current if not enough frames yet)
        effective_player = confirmed_player if consecutive_count >= min_consecutive_frames else confirmed_player
        effective_team = confirmed_team if consecutive_count >= min_consecutive_frames else confirmed_team

        report.frames.append(PossessionFrame(
            frame=frame,
            player_id=effective_player,
            team=effective_team,
            distance=best_dist if best_player else None
        ))

        if effective_player:
            report.player_possession_frames[effective_player] += 1
        if effective_team:
            report.team_possession_frames[effective_team] += 1

    return report


def get_possession_by_frame(report: PossessionReport) -> Dict[int, Optional[str]]:
    """
    Build a fast lookup: {frame_number: player_id} for visualization.
    Returns None for frames with no possession.
    """
    return {fr.frame: fr.player_id for fr in report.frames}


def export_possession_csv(report: PossessionReport, path: Path) -> None:
    """Export possession report to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        f.write("player_id,team,possession_frames,possession_pct\n")
        for pid, frames in sorted(report.player_possession_frames.items(), key=lambda x: -x[1]):
            pct = report.player_percentages.get(pid, 0.0)
            team = next((fr.team for fr in report.frames if fr.player_id == pid), "unknown")
            f.write(f"{pid},{team},{frames},{pct:.2f}\n")

        f.write("\nteam,possession_frames,possession_pct\n")
        for team, frames in sorted(report.team_possession_frames.items(), key=lambda x: -x[1]):
            pct = report.team_percentages.get(team, 0.0)
            f.write(f"{team},{frames},{pct:.2f}\n")

    print(f"   Possession CSV → {path}")


def export_possession_json(report: PossessionReport, path: Path) -> None:
    """Export possession report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "total_frames": report.total_frames,
            "team_possession": {
                team: {"frames": frames, "percentage": round(pct, 2)}
                for team, (frames, pct) in {
                    t: (c, report.team_percentages.get(t, 0.0))
                    for t, c in report.team_possession_frames.items()
                }.items()
            },
            "player_possession": {
                pid: {"frames": frames, "percentage": round(pct, 2), "team": next(
                    (fr.team for fr in report.frames if fr.player_id == pid), None
                )}
                for pid, (frames, pct) in {
                    p: (c, report.player_percentages.get(p, 0.0))
                    for p, c in report.player_possession_frames.items()
                }.items()
            },
            "possession_changes": len(report.changes),
        },
        "changes": [
            {"frame": frame, "from_team": f, "to_team": t}
            for frame, f, t in report.changes
        ],
        "per_frame": [
            {
                "frame": fr.frame,
                "player_id": fr.player_id,
                "team": fr.team,
                "distance": round(fr.distance, 2) if fr.distance else None,
            }
            for fr in report.frames
        ],
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"   Possession JSON → {path}")


def print_possession_summary(report: PossessionReport, fps: float = 30.0) -> None:
    """Print a formatted possession summary to console."""
    print(f"\n🏀 Possession Analysis ({report.total_frames} frames, {report.total_frames/fps:.1f}s)")
    print("-" * 50)

    print("\n📊 Team Possession:")
    for team, frames in sorted(report.team_possession_frames.items(), key=lambda x: -x[1]):
        pct = report.team_percentages[team]
        seconds = frames / fps
        print(f"   {team}: {frames} frames ({pct:.1f}%) — {seconds:.1f}s")

    print("\n👤 Player Possession:")
    for pid, frames in sorted(report.player_possession_frames.items(), key=lambda x: -x[1]):
        pct = report.player_percentages[pid]
        seconds = frames / fps
        team = next((fr.team for fr in report.frames if fr.player_id == pid), "?")
        print(f"   Player {pid} ({team}): {frames} frames ({pct:.1f}%) — {seconds:.1f}s")

    print(f"\n🔄 Possession Changes: {len(report.changes)}")
    for frame, from_t, to_t in report.changes[:10]:  # Show first 10
        print(f"   Frame {frame}: {from_t or 'None'} → {to_t or 'None'}")
    if len(report.changes) > 10:
        print(f"   ... and {len(report.changes) - 10} more")
