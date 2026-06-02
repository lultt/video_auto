"""
valley_detection.py — valley-to-valley cycle boundary detection on motion signal.

Core assumption: net operations create motion peaks; the quiet interludes
between them (valleys) are the natural cycle boundaries.

Algorithm:
  1. Smooth motion signal (Savitzky-Golay).
  2. Invert → find_peaks on inverted signal  = valleys.
  3. Filter to only "deep" valleys (below a motion percentile).
  4. Each valley → valley segment is one cycle.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def smooth_signal(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    return savgol_filter(values, window, polyorder, mode="interp")


def detect_valleys(
    motion: np.ndarray,
    min_distance: int = 25,
    prominence: float = 1.2,
    depth_percentile: float = 35.0,
) -> np.ndarray:
    """
    Find valleys (local minima) in a motion signal.

    Returns array of *indices* into `motion` for deep valley positions.
    """
    valleys, _ = find_peaks(-motion, distance=min_distance, prominence=prominence)

    # Keep only "deep" valleys — where motion drops below the given percentile.
    threshold = np.percentile(motion, depth_percentile)
    valleys = np.array([v for v in valleys if motion[v] < threshold])
    return valleys


def detect_peaks(
    motion: np.ndarray,
    min_distance: int = 30,
    prominence: float = 3.5,
    min_width: int = 5,
):
    """Find motion peaks → (indices, properties dict)."""
    return find_peaks(motion, distance=min_distance, prominence=prominence, width=min_width)


def segment_cycles(valleys: np.ndarray, peaks: np.ndarray, peak_props: dict, motion: np.ndarray):
    """
    Build cycle entries from valley→valley segments, picking the strongest
    peak inside each segment as the activity anchor.
    """
    cycles = []
    for i in range(len(valleys) - 1):
        v_start, v_end = valleys[i], valleys[i + 1]
        seg_peaks = peaks[(peaks >= v_start) & (peaks <= v_end)]

        if len(seg_peaks) == 0:
            peak_idx = v_start + (v_end - v_start) // 2
            peak_prom = 0.0
        else:
            best = seg_peaks[np.argmax(motion[seg_peaks])]
            peak_idx = int(best)
            idx_in_pp = list(peaks).index(best)
            peak_prom = float(peak_props["prominences"][idx_in_pp])

        cycles.append({
            "start_idx": int(v_start),
            "peak_idx": peak_idx,
            "end_idx": int(v_end),
            "peak_prom": round(peak_prom, 2),
        })
    return cycles