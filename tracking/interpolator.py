import numpy as np
import pandas as pd


class Interpolator:
    """
    Fills in missing ball detections using linear interpolation.
    Smooths the trajectory so the ball path is continuous even
    when the detector misses it due to motion blur or occlusion.
    """

    @staticmethod
    def interpolate(positions: list, smooth: bool = True, window: int = 3) -> list:
        """
        positions : list of [x1,y1,x2,y2] or None  (one entry per frame)
        Returns   : same-length list with None gaps filled in
        """
        if not positions:
            return positions

        df = pd.DataFrame(
            positions,
            columns=["x1", "y1", "x2", "y2"]
        )

        # Mark missing rows
        # Each position is either a list of 4 ints or None
        # pd.DataFrame with None rows becomes NaN automatically when
        # built from a list containing None entries.

        # Interpolate linearly
        df = df.interpolate(method="linear", limit_direction="both")

        # Forward / backward fill any remaining edge NaNs
        df = df.ffill().bfill()

        if smooth and len(df) >= window:
            df = df.rolling(window=window, min_periods=1, center=True).mean()

        df = df.round().astype(int)
        return df.values.tolist()

    @staticmethod
    def build_position_list(raw_positions: list) -> list:
        """
        Converts raw_positions (list of bbox or None) to a form
        compatible with interpolate(). Ensures None entries become
        rows of NaN in the DataFrame.
        """
        result = []
        for pos in raw_positions:
            if pos is None:
                result.append([np.nan, np.nan, np.nan, np.nan])
            else:
                result.append(pos)
        return result

    @staticmethod
    def interpolate_ball(raw_positions: list, smooth: bool = True) -> list:
        """
        High-level convenience method.
        raw_positions : list of [x1,y1,x2,y2] or None
        Returns       : list of [x1,y1,x2,y2] with gaps filled
        """
        prepared = Interpolator.build_position_list(raw_positions)
        return Interpolator.interpolate(prepared, smooth=smooth)

    @staticmethod
    def get_center(bbox: list) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)