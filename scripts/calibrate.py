"""
scripts/calibrate.py
─────────────────────
Interactive tool to compute the homography matrix by clicking
the 4 court corners on the first frame of your video.

Run:
    python scripts/calibrate.py --video path/to/game.mp4

Controls:
    Left click  → place a corner point (order: TL → TR → BR → BL)
    R           → reset all points
    S           → save and exit (requires all 4 points)
    Q / ESC     → quit without saving

Output:
    config/homography.npy   ← the 3×3 H matrix
"""

from __future__ import annotations
import argparse
import sys
import os

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from src.analytics.homography import HomographyTransformer


CORNER_LABELS = ["TL (top-left)", "TR (top-right)", "BR (bottom-right)", "BL (bottom-left)"]
CORNER_COLORS = [
    (0,   255,   0),    # green  – TL
    (255, 165,   0),    # orange – TR
    (0,   0,   255),    # red    – BR
    (255, 0,   255),    # purple – BL
]
WINDOW_NAME = "Calibrate — click 4 court corners"


class CalibrationTool:
    def __init__(self, frame: np.ndarray, out_path: str, court_w: int, court_h: int) -> None:
        self.original  = frame.copy()
        self.out_path  = out_path
        self.court_w   = court_w
        self.court_h   = court_h
        self.points: list[list[int]] = []

    def run(self) -> bool:
        """Main loop. Returns True if homography was saved."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        cv2.setMouseCallback(WINDOW_NAME, self._on_click)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):          # Q or ESC → quit
                cv2.destroyAllWindows()
                print("[Calibrate] Cancelled — no file saved.")
                return False

            if key == ord("r"):                # R → reset
                self.points.clear()
                self._redraw()

            if key == ord("s"):                # S → save
                if len(self.points) < 4:
                    print(f"[Calibrate] Need 4 points, have {len(self.points)}. Keep clicking.")
                    continue
                self._save()
                cv2.destroyAllWindows()
                return True

        cv2.destroyAllWindows()
        return False

    def _on_click(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 4:
                print("[Calibrate] Already have 4 points. Press R to reset or S to save.")
                return
            self.points.append([x, y])
            idx = len(self.points) - 1
            print(f"[Calibrate] Point {idx + 1}/4 — {CORNER_LABELS[idx]}: ({x}, {y})")
            self._redraw()

    def _redraw(self) -> None:
        vis = self.original.copy()

        # Instructions
        instructions = [
            "Click the 4 court corners in order:",
            "1) Top-Left   2) Top-Right   3) Bottom-Right   4) Bottom-Left",
            "R = reset   S = save (needs 4 pts)   Q = quit",
        ]
        for i, line in enumerate(instructions):
            y = 32 + i * 26
            cv2.rectangle(vis, (0, y - 20), (vis.shape[1], y + 8), (20, 20, 20), -1)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        # Next point prompt
        if len(self.points) < 4:
            nxt = CORNER_LABELS[len(self.points)]
            cv2.putText(vis, f"Next: {nxt}", (12, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, CORNER_COLORS[len(self.points)], 2, cv2.LINE_AA)

        # Draw placed points + polygon
        pts_np = np.array(self.points, dtype=np.int32)
        for i, (px, py) in enumerate(self.points):
            c = CORNER_COLORS[i]
            cv2.circle(vis, (px, py), 10, c, -1)
            cv2.circle(vis, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(vis, str(i + 1), (px + 15, py + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2, cv2.LINE_AA)

        if len(self.points) >= 2:
            cv2.polylines(vis, [pts_np], isClosed=(len(self.points) == 4),
                          color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, vis)

    def _save(self) -> None:
        transformer = HomographyTransformer(
            src_points=self.points,
            court_width_px=self.court_w,
            court_height_px=self.court_h,
        )
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        transformer.save(self.out_path)
        print(f"\n[Calibrate] ✓ Homography saved → {self.out_path}")
        print(f"[Calibrate] H matrix:\n{transformer.H}\n")
        print("[Calibrate] Now run:  python main.py --video <your_video>")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball court homography calibration tool")
    parser.add_argument("--video",    required=True,               help="Path to game video")
    parser.add_argument("--output",   default="config/homography.npy", help="Output .npy path")
    parser.add_argument("--frame",    type=int, default=0,         help="Frame index to use for calibration")
    parser.add_argument("--court_w",  type=int, default=1060,      help="Court canvas width in pixels")
    parser.add_argument("--court_h",  type=int, default=560,       help="Court canvas height in pixels")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[Calibrate] ERROR: Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"[Calibrate] ERROR: Could not read frame {args.frame} from {args.video}")
        sys.exit(1)

    print(f"[Calibrate] Video loaded: {args.video}  ({frame.shape[1]}×{frame.shape[0]})")
    print("[Calibrate] Opening calibration window — follow the on-screen instructions.\n")

    tool = CalibrationTool(frame, args.output, args.court_w, args.court_h)
    tool.run()


if __name__ == "__main__":
    main()