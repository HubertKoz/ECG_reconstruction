import os
import pandas as pd
import numpy as np
from scipy import signal

class BaseLoader:
    def __init__(self, base_data_dir: str = "./data"):
        self.base_data_dir = base_data_dir

    def resample(self, df_ieee, original_fs, target_fs=256):
        # Obliczamy up/down (skrócony ułamek 8/25)
        gcd = np.gcd(target_fs, original_fs)
        up = target_fs // gcd
        down = original_fs // gcd
        
        resampled_dict = {}
        for col in df_ieee.columns:
            # Używamy resample_poly dla zachowania wierności sygnału
            resampled_dict[col] = signal.resample_poly(df_ieee[col].values, up, down)
        
        return pd.DataFrame(resampled_dict)
