"""
src/analytics/homography.py
────────────────────────────
Maps 2D pixel coordinates from the camera image to a flat top-down
basketball court canvas using projective homography.

Workflow
────────
1. Click the 4 court corners on a video frame (scripts/calibrate.py).
2. We compute a 3x3 H matrix that maps camera pixels → court canvas.
3. Every detected player's FOOT position is transformed through H to
   get its real position on the flat court.
4. These court coordinates feed:
     - heatmap.py      (occupancy density)
     - distance.py     (real distance traveled)
     - speed.py        (real m/s)
     - visualization   (mini-map mosaic on the output video)

Key concept
───────────
A homography is a 3x3 matrix in homogeneous coordinates. It can model
any plane-to-plane projective transform — exactly what we need to undo
the camera's tilted view of the (flat) court.

Public API
──────────
    transformer = HomographyTransformer(
        src_points     = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],
        court_width_px = 1060,
        court_height_px= 560,
    )
    transformer.transform_point((950, 720))       →  (530, 280)
    transformer.transform_bbox_foot(bbox)         →  (court_x, court_y)
    transformer.save("config/homography.npy")
    HomographyTransformer.load("config/homography.npy")
"""

from __future__ import annotations

import os
import numpy as np
import cv2


# ── Defaults ─────────────────────────────────────────────────────────────────
# A real NBA court is 94 ft × 50 ft (1.88 aspect ratio).
# 1060 × 560 ≈ 1.89 — visually faithful while easy to draw on.
DEFAULT_COURT_WIDTH  = 1060
DEFAULT_COURT_HEIGHT = 560

# Corner order MUST be: Top-Left → Top-Right → Bottom-Right → Bottom-Left.
# This matches scripts/calibrate.py and gives a non-mirrored top-down view.
CORNER_ORDER = ("TL", "TR", "BR", "BL")


# ─────────────────────────────────────────────────────────────────────────────
class HomographyTransformer:
    """
    Plane-to-plane projective transformer for basketball court mapping.

    Two construction modes:
      A) Compute from 4 clicked corners       → pass src_points
      B) Load a saved H matrix from disk      → use HomographyTransformer.load()
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        src_points:      list[list[int]] | np.ndarray | None = None,
        court_width_px:  int = DEFAULT_COURT_WIDTH,
        court_height_px: int = DEFAULT_COURT_HEIGHT,
        H:               np.ndarray | None = None,
    ) -> None:
        self.court_width_px  = court_width_px
        self.court_height_px = court_height_px

        # Destination corners — same TL→TR→BR→BL order as src_points
        self.dst_points = np.array([
            [0,                 0],
            [court_width_px,    0],
            [court_width_px,    court_height_px],
            [0,                 court_height_px],
        ], dtype=np.float32)

        if H is not None:
            # Path B — H supplied directly (loaded from disk)
            self.H          = np.asarray(H, dtype=np.float64)
            self.src_points = None
        elif src_points is not None:
            # Path A — compute H from clicked corners
            self.src_points = np.array(src_points, dtype=np.float32)
            if self.src_points.shape != (4, 2):
                raise ValueError(
                    f"src_points must be shape (4,2), got {self.src_points.shape}"
                )
            self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        else:
            raise ValueError(
                "Must provide either src_points (4 court corners) or "
                "H (a 3x3 matrix)."
            )

        # Pre-compute inverse for top-down → camera mapping
        self.H_inv = np.linalg.inv(self.H)

    # ── Camera → Court (forward) ──────────────────────────────────────────────

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        """
        Transform one pixel coordinate from camera space → court space.

        Returns
        -------
        (court_x, court_y) on the top-down canvas. The result may fall
        outside the canvas if the input pixel isn't on the court.
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return (float(out[0, 0, 0]), float(out[0, 0, 1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorised forward transform for many points at once.

        Parameters
        ----------
        points : array-like of shape (N, 2)

        Returns
        -------
        np.ndarray of shape (N, 2)
        """
        pts = np.asarray(points, dtype=np.float32)
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def transform_bbox_foot(self, bbox: np.ndarray) -> tuple[float, float]:
        """
        Convert a player's bounding box to their court position.

        Uses the BOTTOM-CENTER of the bbox (the player's feet) rather
        than the bbox center, because the camera tilt makes the chest
        appear at a different ground location than the feet.
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        return self.transform_point((foot_x, foot_y))

    # ── Court → Camera (reverse) ──────────────────────────────────────────────

    def inverse_transform_point(
        self, point: tuple[float, float]
    ) -> tuple[float, float]:
        """
        Map a top-down court coordinate back to a camera-image pixel.
        Useful for drawing court zones/lines onto the original frame.
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H_inv)
        return (float(out[0, 0, 0]), float(out[0, 0, 1]))

    # ── Validity check ────────────────────────────────────────────────────────

    def is_on_court(
        self,
        court_point: tuple[float, float],
        margin:      int = 20,
    ) -> bool:
        """
        True if a transformed point lies within the court canvas
        (with a tolerance margin for sideline players).
        """
        cx, cy = court_point
        return (
            -margin <= cx <= self.court_width_px  + margin and
            -margin <= cy <= self.court_height_px + margin
        )

    # ── Court canvas drawing ──────────────────────────────────────────────────

    def make_court_canvas(self) -> np.ndarray:
        """
        Generate a clean top-down basketball court image (BGR).

        Background    : warm wood tone
        Court lines   : white
        Hoops         : orange dots
        """
        W, H = self.court_width_px, self.court_height_px
        canvas = np.full((H, W, 3), (45, 90, 140), dtype=np.uint8)

        line_color = (255, 255, 255)
        line_thick = 2

        # Outer boundary
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), line_color, line_thick)

        # Half-court line
        cv2.line(canvas, (W // 2, 0), (W // 2, H - 1), line_color, line_thick)

        # Center circle
        cv2.circle(canvas, (W // 2, H // 2), int(0.07 * H), line_color, line_thick)

        # The two key/paint areas (rectangles near each end)
        paint_w = int(0.16 * W)
        paint_h = int(0.36 * H)
        for hoop_x in (0, W - paint_w):
            y_top = (H - paint_h) // 2
            cv2.rectangle(
                canvas,
                (hoop_x,           y_top),
                (hoop_x + paint_w, y_top + paint_h),
                line_color, line_thick,
            )

        # Hoop markers
        hoop_inset = int(0.04 * W)
        cv2.circle(canvas, (hoop_inset,     H // 2), 6, (0, 165, 255), -1)
        cv2.circle(canvas, (W - hoop_inset, H // 2), 6, (0, 165, 255), -1)

        return canvas

    def draw_players_on_canvas(
        self,
        canvas:      np.ndarray,
        players:     list[dict],
        team_colors: dict[int, tuple[int, int, int]] | None = None,
        radius:      int = 8,
    ) -> np.ndarray:
        """
        Draw player dots on a court canvas.

        Each player dict needs:
            court_pos : (cx, cy)   — already transformed coords
            team_id   : int        — optional, for color lookup
            track_id  : int        — optional, for label
        """
        out = canvas.copy()
        team_colors = team_colors or {}
        DEFAULT_COLOR = (180, 180, 180)

        for p in players:
            cx, cy = p["court_pos"]
            color  = team_colors.get(p.get("team_id", -1), DEFAULT_COLOR)

            cv2.circle(out, (int(cx), int(cy)), radius,     color,            -1)
            cv2.circle(out, (int(cx), int(cy)), radius + 1, (255, 255, 255),   1)

            if "track_id" in p:
                cv2.putText(
                    out, str(p["track_id"]),
                    (int(cx) - 6, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
                )

        return out

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save H + canvas dims (+ src_points if known) as a .npz bundle."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            H               = self.H,
            court_width_px  = self.court_width_px,
            court_height_px = self.court_height_px,
            src_points      = (self.src_points if self.src_points is not None
                               else np.zeros((4, 2), dtype=np.float32)),
        )
        print(f"[Homography] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "HomographyTransformer":
        """Reconstruct a HomographyTransformer from a saved bundle."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Homography file not found: {path}")

        data = np.load(path)
        transformer = cls(
            H               = data["H"],
            court_width_px  = int(data["court_width_px"]),
            court_height_px = int(data["court_height_px"]),
        )
        if "src_points" in data and not np.allclose(data["src_points"], 0):
            transformer.src_points = data["src_points"].astype(np.float32)
        print(f"[Homography] Loaded ← {path}")
        return transformer

    # ── Misc ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HomographyTransformer("
            f"canvas={self.court_width_px}×{self.court_height_px}, "
            f"calibrated={self.src_points is not None})"
        )
