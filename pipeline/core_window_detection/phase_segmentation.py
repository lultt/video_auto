"""
phase_segmentation.py — segment each cycle into phases (net-rising, net-on-deck,
fish-transfer, cleanup) using ROI-level feature signals.

This is the core logic of Layer 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


def smooth_features(seg_df: pd.DataFrame, roi_order=None, window=7, polyorder=2):
    """Add smoothed versions of edge_density and motion_intensity for each ROI."""
    if roi_order is None:
        roi_order = ["left_deck", "center_deck", "right_deck"]
    for roi in roi_order:
        for feat in ["edge_density", "motion_intensity"]:
            col = f"{roi}_{feat}"
            if col in seg_df.columns:
                seg_df[f"{roi}_{feat.split('_')[0]}_smooth" if feat == "edge_density" else f"{roi}_motion_smooth"] = \
                    savgol_filter(seg_df[col].to_numpy(), window, polyorder, mode="interp")
    return seg_df


def detect_net_on_deck(
    left_edge_smooth: np.ndarray,
    left_edge_raw: np.ndarray,
    prominence: float = 0.008,
    min_distance: int = 10,
    search_first_pct: float = 0.70,
    rise_low_quantile: float = 0.30,
    rise_frac: float = 0.70,
) -> tuple[int, int]:
    """
    Find net-on-deck and net-start indices from left-edge signal.

    Returns (net_start_idx, net_on_deck_idx).
    """
    n = len(left_edge_smooth)
    search_end = max(1, int(n * search_first_pct))

    peaks, props = find_peaks(left_edge_smooth[:search_end], prominence=prominence, distance=min_distance)

    if len(peaks) == 0:
        net_on_deck_idx = int(np.argmax(left_edge_smooth[:search_end]))
    else:
        candidates = [p for p in peaks if p < search_end]
        net_on_deck_idx = int(candidates[np.argmax(left_edge_smooth[candidates])]) if candidates else int(peaks[0])

    # Walk backward from the peak to find when the rising began
    peak_val = left_edge_smooth[net_on_deck_idx]
    rise_thresh = np.quantile(left_edge_raw, rise_low_quantile) + rise_frac * (peak_val - np.quantile(left_edge_raw, rise_low_quantile))
    rising = np.where(left_edge_smooth[:net_on_deck_idx] < rise_thresh)[0]
    net_start_idx = int(rising[-1]) if len(rising) > 0 else max(0, net_on_deck_idx - 15)

    return net_start_idx, net_on_deck_idx


def detect_fish_transfer(
    right_edge_smooth: np.ndarray,
    right_edge_raw: np.ndarray,
    net_on_deck_idx: int,
    prominence: float = 0.008,
    min_distance: int = 15,
    fall_low_quantile: float = 0.40,
    fall_rise_frac: float = 0.30,
) -> tuple[int, int]:
    """
    Find fish-transfer start and end indices from right-edge signal,
    constrained to begin AFTER net_on_deck_idx.

    Returns (fish_start_idx, fish_end_idx).
    """
    post_net = right_edge_smooth[net_on_deck_idx:]
    peaks, props = find_peaks(post_net, prominence=prominence, distance=min_distance)

    if len(peaks) == 0:
        right_q = np.quantile(post_net, 0.7)
        high = np.where(post_net >= right_q)[0]
        fish_start_rel = int(high[0]) if len(high) > 0 else 0
        fish_end_rel = int(high[-1]) if len(high) > 0 else len(post_net) - 1
    else:
        main_peak_rel = int(peaks[np.argmax(props["prominences"])])
        peak_val = post_net[main_peak_rel]
        fall_thresh = np.quantile(right_edge_raw, fall_low_quantile) + fall_rise_frac * (peak_val - np.quantile(right_edge_raw, fall_low_quantile))

        rising_r = np.where(post_net[:main_peak_rel + 1] < fall_thresh)[0]
        fish_start_rel = int(rising_r[-1]) if len(rising_r) > 0 else max(0, main_peak_rel - 20)

        falling_r = np.where(post_net[main_peak_rel:] < fall_thresh)[0]
        fish_end_rel = int(main_peak_rel + falling_r[0]) if len(falling_r) > 0 else min(len(post_net) - 1, main_peak_rel + 25)

    fish_start_idx = net_on_deck_idx + fish_start_rel
    fish_end_idx = net_on_deck_idx + fish_end_rel

    # Safety clamping
    fish_start_idx = max(net_on_deck_idx, fish_start_idx)
    fish_end_idx = min(len(right_edge_smooth) - 1, fish_end_idx)

    return int(fish_start_idx), int(fish_end_idx)