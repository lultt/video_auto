"""
fft_analysis.py — autocorrelation + FFT periodicity analysis for the motion signal.

Used to validate that the recovered cycles correspond to a real periodic rhythm
in the activity signal (typically ~45-65 min per net cycle).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def autocorrelation(motion: np.ndarray, max_lag: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    Naive autocorrelation up to `max_lag` samples.
    Returns (lags, autocorr_values).
    """
    motion = motion - np.nanmean(motion)
    max_lag = min(max_lag, len(motion) - 2)
    lags = np.arange(1, max_lag + 1)
    ac = []
    for lag in lags:
        a, b = motion[:-lag], motion[lag:]
        denom = np.nanstd(a) * np.nanstd(b)
        ac.append(0.0 if denom < 1e-9 else float(np.corrcoef(a, b)[0, 1]))
    return lags, np.array(ac)


def fft_periodicity(
    motion: np.ndarray,
    sample_period_sec: float = 60.0,
    period_min_low: float = 20.0,
    period_min_high: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute FFT power spectrum, restricted to the physically plausible
    cycle-period band [period_min_low, period_min_high] (in minutes).

    Returns (periods_minutes, powers, best_period_minutes).
    """
    motion = motion - np.nanmean(motion)
    freqs = rfftfreq(len(motion), d=sample_period_sec)
    power = np.abs(rfft(motion)) ** 2
    period_min = np.where(freqs > 0, 1 / freqs / 60.0, np.inf)
    valid = (period_min >= period_min_low) & (period_min <= period_min_high)
    if not np.any(valid):
        return period_min, power, float("nan")
    best_period = float(period_min[valid][np.argmax(power[valid])])
    return period_min[valid], power[valid], best_period