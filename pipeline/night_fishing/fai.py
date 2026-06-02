# fai.py — Fishing Activity Index computation (v2: cross-night comparable)

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .config import W_MOTION, W_BSTD, W_ENTROPY, SG_WINDOW, SG_POLY


def compute_fai(df: pd.DataFrame, global_stats: dict = None) -> pd.DataFrame:
    """Compute Fishing Activity Index for a single night.

    Uses raw feature values (NOT per-night min-max) so that FAI is comparable
    across nights.  If global_stats is provided (p50/p90 of each component across
    all nights), the raw values are divided by the p90 for rough [0,~1] scaling.
    Otherwise falls back to dividing by a fixed reference.

    FAI = w_m * motion/p90 + w_b * bstd/p90 + w_e * |Δentropy|/p90
    """
    df = df.copy()
    motion  = df["motion_intensity"].values.astype(float)
    bstd    = df["brightness_std"].values.astype(float)
    entropy = df["entropy"].values.astype(float)
    entropy_diff = np.abs(np.diff(entropy, prepend=entropy[0]))

    # Reference scales (p90 across all data, or reasonable defaults)
    if global_stats:
        ref_m = global_stats.get("motion_p90", 20.0)
        ref_b = global_stats.get("bstd_p90", 60.0)
        ref_e = global_stats.get("entropy_diff_p90", 0.3)
    else:
        ref_m = 20.0
        ref_b = 60.0
        ref_e = 0.3

    df["fai_motion"]  = motion / max(ref_m, 1e-6)
    df["fai_bstd"]    = bstd / max(ref_b, 1e-6)
    df["fai_entropy"] = entropy_diff / max(ref_e, 1e-6)

    df["fai_raw"] = (
        W_MOTION  * df["fai_motion"]
        + W_BSTD  * df["fai_bstd"]
        + W_ENTROPY * df["fai_entropy"]
    )

    # Savitzky-Golay smoothing
    n = len(df)
    win = min(SG_WINDOW, n - (1 - n % 2))
    if win >= 5:
        try:
            df["fai_smooth"] = savgol_filter(df["fai_raw"].values, win, SG_POLY, mode="interp")
        except Exception:
            df["fai_smooth"] = df["fai_raw"].rolling(window=max(3, win), center=True, min_periods=1).mean()
    else:
        df["fai_smooth"] = df["fai_raw"].rolling(window=3, center=True, min_periods=1).mean()

    return df


def compute_global_stats(df_global: pd.DataFrame) -> dict:
    """Compute p50/p90 reference values across all nights for FAI scaling."""
    motion = df_global["motion_intensity"].values.astype(float)
    bstd   = df_global["brightness_std"].values.astype(float)
    entropy = df_global["entropy"].values.astype(float)
    entropy_diff = np.abs(np.diff(entropy, prepend=entropy[0]))

    return {
        "motion_p50": float(np.percentile(motion, 50)),
        "motion_p90": float(np.percentile(motion, 90)),
        "bstd_p50": float(np.percentile(bstd, 50)),
        "bstd_p90": float(np.percentile(bstd, 90)),
        "entropy_diff_p50": float(np.percentile(entropy_diff, 50)),
        "entropy_diff_p90": float(np.percentile(entropy_diff, 90)),
    }
