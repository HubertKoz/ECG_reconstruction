import os
import pandas as pd
import numpy as np
from scipy import signal

from config import TARGET_FS


class BaseLoader:
    def __init__(self, base_data_dir: str = "./data"):
        self.base_data_dir = base_data_dir

    def resample(self, df: pd.DataFrame, original_fs: int, target_fs: int = TARGET_FS) -> pd.DataFrame:
        """Resampluje DataFrame z original_fs do target_fs (domyślnie TARGET_FS z config)."""
        if original_fs == target_fs:
            return df.copy()

        gcd = np.gcd(target_fs, original_fs)
        up   = target_fs  // gcd
        down = original_fs // gcd

        resampled = {
            col: signal.resample_poly(df[col].values, up, down)
            for col in df.columns
        }
        return pd.DataFrame(resampled)
