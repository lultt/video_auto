"""
change_point_detection.py — generic peak / change-point helpers.

Thin wrappers around scipy.signal.find_peaks so that detection params
live in one place and are easy to swap out later (e.g. for ruptures /
PELT, Bayesian changepoint, etc.).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def find_signal_peaks(
    signal: np.ndarray,
    prominence: float = 0.008,
    min_distance: int = 10,
    width: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Wrapper around find_peaks. Returns (indices, properties)."""
    kwargs = {"prominence": prominence, "distance": min_distance}
    if width is not None:
        kwargs["width"] = width
    return find_peaks(signal, **kwargs)


def find_rise_index(
    signal: np.ndarray,
    peak_idx: int,
    low_quantile: float = 0.30,
    rise_frac: float = 0.70,
    raw_signal: np.ndarray | None = None,
) -> int:
    """
    From a known peak, walk backward to find the last index below
    `low_quantile + rise_frac * (peak - low_quantile)` (i.e., the rise onset).
    """
    base = raw_signal if raw_signal is not None else signal
    peak_val = signal[peak_idx]
    thresh = np.quantile(base, low_quantile) + rise_frac * (peak_val - np.quantile(base, low_quantile))
    rising = np.where(signal[:peak_idx] < thresh)[0]
    return int(rising[-1]) if len(rising) > 0 else max(0, peak_idx - 15)


def find_fall_index(
    signal: np.ndarray,
    peak_idx: int,
    low_quantile: float = 0.40,
    rise_frac: float = 0.30,
    raw_signal: np.ndarray | None = None,
) -> int:
    """
    From a known peak, walk forward to find the first index below threshold
    (i.e., the fall offset).
    """
    base = raw_signal if raw_signal is not None else signal
    peak_val = signal[peak_idx]
    thresh = np.quantile(base, low_quantile) + rise_frac * (peak_val - np.quantile(base, low_quantile))
    falling = np.where(signal[peak_idx:] < thresh)[0]
    return int(peak_idx + falling[0]) if len(falling) > 0 else min(len(signal) - 1, peak_idx + 25)