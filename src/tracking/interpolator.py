"""
src/tracking/interpolator.py
──────────────────────────────
Trajectory gap-filling for players who disappear during jumps or occlusions.

Why players vanish during jumps
────────────────────────────────
1. Aspect ratio change: standing player bbox is tall/narrow; mid-jump is
   shorter/wider. ByteTrack's Kalman filter uses constant-velocity motion —
   vertical acceleration during a jump violates this assumption.
2. Confidence drops: jumping players are partially motion-blurred and may
   overlap teammates, lowering detection confidence below the threshold.
3. Short occlusion: player passes behind another player at the apex.

Two strategies are combined here
──────────────────────────────────
A. Linear interpolation for gaps ≤ MAX_GAP_FRAMES:
   If a track is present at frame t₁ and frame t₂, fill t₁+1..t₂-1 with
   lerp(pos_t1, pos_t2). Works well for normal movement.

B. Parabolic interpolation for jump-shaped gaps:
   Detects jump signature (large upward Δy followed by gap followed by
   downward Δy) and fits a parabola through the known endpoints.
   This is optional — enable with use_parabolic=True.

Usage
─────
  from src.tracking.interpolator import TrajectoryInterpolator

  interp = TrajectoryInterpolator()
  filled = interp.fill_all(trajectories)   # call AFTER the full video loop
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict


# Gaps shorter than this are filled; longer gaps stay as-is
MAX_GAP_FRAMES = 15

# A gap where the player was moving upward before and downward after
# → treat as a jump and fit a parabola instead of a straight line
JUMP_VERTICAL_PX = 30    # pre-gap upward velocity (px/frame)


class TrajectoryInterpolator:
    """
    Post-processing step: fills short gaps in player trajectories.

    Call after the video loop completes, before running analytics.

    Parameters
    ----------
    max_gap       : int   Fill gaps ≤ this many frames.
    use_parabolic : bool  Use parabola for jump-shaped gaps (optional).
    """

    def __init__(
        self,
        max_gap:       int  = MAX_GAP_FRAMES,
        use_parabolic: bool = True,
    ) -> None:
        self.max_gap       = max_gap
        self.use_parabolic = use_parabolic

    # ── Public API ────────────────────────────────────────────────────────────

    def fill_all(
        self,
        trajectories: dict[int, list[tuple]],
    ) -> dict[int, list[tuple]]:
        """
        Fill gaps in all tracks.

        Parameters
        ----------
        trajectories : {track_id: [(cx, cy, frame_idx), ...]}
            As returned by PlayerTracker.get_all_trajectories().

        Returns
        -------
        {track_id: [(cx, cy, frame_idx), ...]}
            Same structure but with interpolated points inserted.
        """
        filled: dict[int, list[tuple]] = {}
        stats = {"tracks": 0, "gaps_filled": 0, "frames_added": 0}

        for tid, traj in trajectories.items():
            if len(traj) < 2:
                filled[tid] = traj
                continue

            new_traj, gaps, frames = self._fill_track(traj)
            filled[tid] = new_traj
            stats["tracks"] += 1
            stats["gaps_filled"] += gaps
            stats["frames_added"] += frames

        print(
            f"[Interpolator] Processed {stats['tracks']} tracks. "
            f"Filled {stats['gaps_filled']} gaps, "
            f"added {stats['frames_added']} interpolated frames."
        )
        return filled

    def fill_single(
        self, trajectory: list[tuple]
    ) -> list[tuple]:
        """Fill gaps in a single track trajectory."""
        result, _, _ = self._fill_track(trajectory)
        return result

    # ── Core logic ────────────────────────────────────────────────────────────

    def _fill_track(
        self, traj: list[tuple]
    ) -> tuple[list[tuple], int, int]:
        """
        Returns (filled_trajectory, num_gaps_filled, num_frames_added).
        """
        # Sort by frame index
        sorted_traj = sorted(traj, key=lambda p: p[2])
        result      = list(sorted_traj)

        gaps_filled   = 0
        frames_added  = 0
        inserted_pts: list[tuple] = []

        for i in range(len(sorted_traj) - 1):
            p1 = sorted_traj[i]
            p2 = sorted_traj[i + 1]

            x1, y1, f1 = p1[0], p1[1], p1[2]
            x2, y2, f2 = p2[0], p2[1], p2[2]

            gap = f2 - f1 - 1
            if gap <= 0 or gap > self.max_gap:
                continue

            # Detect jump signature
            is_jump = self._is_jump_gap(sorted_traj, i)

            if is_jump and self.use_parabolic:
                pts = self._parabolic_fill(x1, y1, f1, x2, y2, f2)
            else:
                pts = self._linear_fill(x1, y1, f1, x2, y2, f2)

            inserted_pts.extend(pts)
            gaps_filled  += 1
            frames_added += len(pts)

        result.extend(inserted_pts)
        result = sorted(result, key=lambda p: p[2])
        return result, gaps_filled, frames_added

    @staticmethod
    def _linear_fill(
        x1: float, y1: float, f1: int,
        x2: float, y2: float, f2: int,
    ) -> list[tuple]:
        """Linear interpolation between two trajectory points."""
        pts = []
        n = f2 - f1 - 1
        for k in range(1, n + 1):
            t = k / (n + 1)
            pts.append((
                x1 + t * (x2 - x1),
                y1 + t * (y2 - y1),
                f1 + k,
            ))
        return pts

    @staticmethod
    def _parabolic_fill(
        x1: float, y1: float, f1: int,
        x2: float, y2: float, f2: int,
    ) -> list[tuple]:
        """
        Parabolic arc for jump gaps.

        Fits a parabola y = a*t^2 + b*t + c through the endpoints.
        The apex is estimated as the midpoint with added vertical
        displacement (player goes up then comes down).
        """
        n     = f2 - f1 - 1
        pts   = []
        dy    = y2 - y1   # positive = player moved down in image space
        apex_y = y1 + dy * 0.5 - abs(dy) * 0.4  # rise to apex, then fall

        for k in range(1, n + 1):
            t = k / (n + 1)           # 0..1
            # Lerp X linearly
            ix = x1 + t * (x2 - x1)
            # Parabola: 0 at t=0, apex at t=0.5, 0 at t=1 → 4t(1-t) profile
            parabola = 4 * t * (1 - t)
            # Blend: linear base + parabolic vertical component
            iy = y1 + t * (y2 - y1) + parabola * (apex_y - (y1 + 0.5 * (y2 - y1)))
            pts.append((ix, iy, f1 + k))

        return pts

    @staticmethod
    def _is_jump_gap(
        traj: list[tuple], gap_start_idx: int
    ) -> bool:
        """
        Heuristic: is the gap before/after this index likely caused by a jump?
        A jump signature = player was moving upward (decreasing y) before the gap.
        In image space: smaller y = higher on screen = physically higher.
        """
        i = gap_start_idx
        if i < 1:
            return False

        # Vertical velocity before gap (negative = moving up in image = jumping)
        _, y_prev, f_prev = traj[i - 1]
        _, y_curr, f_curr = traj[i]
        dt = max(f_curr - f_prev, 1)
        vy = (y_curr - y_prev) / dt

        # If player was moving upward significantly before the gap → jump
        return vy < -JUMP_VERTICAL_PX

    # ── Reporting ─────────────────────────────────────────────────────────────
    def stitch_out_of_bounds(
        self, 
        trajectories: dict[int, list[tuple]], 
        frame_width: int, 
        edge_margin: int = 80
    ) -> dict[int, list[tuple]]:
        """
        Post-processing: If an ID dies near the edge of the screen, and a NEW ID 
        spawns near that same edge later, merge them into a single continuous ID.
        """
        stitched = dict(trajectories)
        
        # 1. Gather the first and last known positions for every track
        track_info = {}
        for tid, traj in stitched.items():
            if len(traj) < 2: 
                continue
            sorted_traj = sorted(traj, key=lambda p: p[2])
            start_pt, end_pt = sorted_traj[0], sorted_traj[-1]
            track_info[tid] = {
                'start_f': start_pt[2], 'start_x': start_pt[0],
                'end_f': end_pt[2],   'end_x': end_pt[0]
            }
            
        exits = []   
        entries = [] 
        
        # 2. Classify tracks as Edge Exits or Edge Entries
        for tid, info in track_info.items():
            if info['end_x'] < edge_margin:
                exits.append({'tid': tid, 'frame': info['end_f'], 'edge': 'left'})
            elif info['end_x'] > frame_width - edge_margin:
                exits.append({'tid': tid, 'frame': info['end_f'], 'edge': 'right'})
                
            if info['start_f'] > 10: 
                if info['start_x'] < edge_margin:
                    entries.append({'tid': tid, 'frame': info['start_f'], 'edge': 'left'})
                elif info['start_x'] > frame_width - edge_margin:
                    entries.append({'tid': tid, 'frame': info['start_f'], 'edge': 'right'})
        
        exits.sort(key=lambda x: x['frame'])
        used_entries = set()
        
        # --- FIX 1: Handle Chained Merges (Player A -> B -> C) ---
        alias_map = {}
        def get_root(t):
            while t in alias_map:
                t = alias_map[t]
            return t
        
        # 3. Match Exits to Entries
        for exit_data in exits:
            original_exit_tid = exit_data['tid']
            root_exit_tid = get_root(original_exit_tid)
            
            # --- FIX 2: Stop players merging with hoops/referees ---
            # Player IDs are < 10000. Referees are 10000+. Hoops are 20000+.
            # Integer division by 10000 groups them into matching "buckets".
            namespace_bucket = root_exit_tid // 10000
            
            candidates = [
                e for e in entries 
                if e['edge'] == exit_data['edge'] 
                and e['frame'] > exit_data['frame']
                and e['tid'] not in used_entries
                and (e['tid'] // 10000) == namespace_bucket  # Must be the same object type!
            ]
            
            if not candidates:
                continue
                
            candidates.sort(key=lambda x: x['frame'])
            best_match = candidates[0]
            entry_tid = best_match['tid']
            
            print(f"[Interpolator] Edge Stitch: Merging Track #{entry_tid} back into Root Track #{root_exit_tid} ({exit_data['edge']} edge)")
            stitched[root_exit_tid].extend(stitched[entry_tid])
            
            del stitched[entry_tid]
            used_entries.add(entry_tid)
            alias_map[entry_tid] = root_exit_tid # Remember this merge for chains
            
        return stitched


    def get_gap_report(
        self, trajectories: dict[int, list[tuple]]
    ) -> dict[int, list[dict]]:
        """
        Return {track_id: [{start, end, length, type}, ...]} for all gaps.
        Useful for debugging specific players.
        """
        report = {}
        for tid, traj in trajectories.items():
            sorted_traj = sorted(traj, key=lambda p: p[2])
            gaps = []
            for i in range(len(sorted_traj) - 1):
                f1 = sorted_traj[i][2]
                f2 = sorted_traj[i + 1][2]
                gap = f2 - f1 - 1
                if gap > 0:
                    is_jump = self._is_jump_gap(sorted_traj, i)
                    gaps.append({
                        "start":  f1,
                        "end":    f2,
                        "length": gap,
                        "type":   "jump" if is_jump else "occlusion",
                        "filled": gap <= self.max_gap,
                    })
            if gaps:
                report[tid] = gaps
        return report

    def __repr__(self) -> str:
        return (
            f"TrajectoryInterpolator("
            f"max_gap={self.max_gap}, "
            f"parabolic={self.use_parabolic})"
        )