"""
detect_cycles.py — end-to-end cycle detection pipeline.

Loads coarse Parquet from a night folder, builds 1-min feature series,
detects valley-to-valley cycles, validates periodicity via FFT, and
saves detected_cycles.csv + cycle_statistics.csv.

This is the main entry point for Layer 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fft_analysis import autocorrelation, fft_periodicity
from .valley_detection import detect_peaks, detect_valleys, segment_cycles, smooth_signal
from ..shared.io import build_minute_series, load_parquet_folder, save_table
from ..shared.utils import build_night_mask


ROI_ORDER = ["left_deck", "center_deck", "right_deck"]


def run_cycle_detection(
    parquet_root: Path,
    out_dir: Path,
    night_start: str = "20:30",
    night_end: str = "08:30",
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    valley_distance: int = 25,
    valley_prominence: float = 1.2,
    valley_depth_pct: float = 35.0,
    peak_distance: int = 30,
    peak_prominence: float = 3.5,
    peak_width: int = 5,
    fft_low: float = 20.0,
    fft_high: float = 180.0,
    max_autocorr_lag: int = 600,
) -> dict:
    """
    Main cycle-detection routine. Returns (cycles_df, stats_df).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & preprocess
    df = load_parquet_folder(Path(parquet_root))
    wide = build_minute_series(df, roi_order=ROI_ORDER)

    # Derived aggregates
    wide["motion_mean"] = wide[[f"{r}_motion_intensity" for r in ROI_ORDER]].mean(axis=1)
    wide["edge_mean"] = wide[[f"{r}_edge_density" for r in ROI_ORDER]].mean(axis=1)
    wide["brightness_mean"] = wide[[f"{r}_mean_brightness" for r in ROI_ORDER]].mean(axis=1)
    wide["entropy_mean"] = wide[[f"{r}_entropy" for r in ROI_ORDER]].mean(axis=1)

    wide["motion_smooth"] = smooth_signal(
        wide["motion_mean"].to_numpy(), window=savgol_window, polyorder=savgol_polyorder
    )

    # Night mask
    night_mask = build_night_mask(wide.index, night_start, night_end)
    wide["night"] = night_mask

    night = wide[wide["night"]].copy()
    times = night.index
    motion = night["motion_smooth"].to_numpy()

    # 2. Detect valleys → cycles
    valleys = detect_valleys(
        motion,
        min_distance=valley_distance,
        prominence=valley_prominence,
        depth_percentile=valley_depth_pct,
    )
    peaks, peak_props = detect_peaks(
        motion,
        min_distance=peak_distance,
        prominence=peak_prominence,
        min_width=peak_width,
    )
    if len(valleys) < 2:
        raise RuntimeError(f"Not enough valleys found (got {len(valleys)}). Relax detection params.")

    cycles = segment_cycles(valleys, peaks, peak_props, motion)

    # 3. Build output tables
    rows = []
    for idx, c in enumerate(cycles, 1):
        seg = night.iloc[c["start_idx"]:c["end_idx"] + 1]
        rows.append({
            "cycle_id": idx,
            "start_time": times[c["start_idx"]].strftime("%Y-%m-%d %H:%M:%S"),
            "peak_time": times[c["peak_idx"]].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": times[c["end_idx"]].strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": round(
                (times[c["end_idx"]] - times[c["start_idx"]]).total_seconds() / 60, 1
            ),
            "peak_prom": c["peak_prom"],
            "mean_motion": round(float(seg["motion_mean"].mean()), 3),
            "max_motion": round(float(seg["motion_mean"].max()), 3),
            "mean_edge": round(float(seg["edge_mean"].mean()), 4),
            "max_edge": round(float(seg["edge_mean"].max()), 4),
            "mean_brightness": round(float(seg["brightness_mean"].mean()), 3),
            "min_brightness": round(float(seg["brightness_mean"].min()), 3),
            "mean_entropy": round(float(seg["entropy_mean"].mean()), 3),
        })

    cycles_df = pd.DataFrame(rows)
    save_table(cycles_df, out_dir / "detected_cycles.csv")

    stat_cols = [
        "duration_min", "mean_motion", "max_motion", "mean_edge",
        "max_edge", "mean_brightness", "min_brightness", "mean_entropy",
    ]
    stats = cycles_df[stat_cols].agg(["count", "mean", "std", "min", "max"]).round(3)
    save_table(stats, out_dir / "cycle_statistics.csv")

    # 4. Periodicity check (informational)
    _, _ = autocorrelation(motion, max_lag=max_autocorr_lag)
    _, _, best_period = fft_periodicity(motion, period_min_low=fft_low, period_min_high=fft_high)
    print(f"  FFT best period: {best_period:.1f} min" if not np.isnan(best_period) else "  FFT: no clear period found")

    # Save enriched minute series
    save_table(wide, out_dir / "minute_features_with_cycle_score.csv")

    return {
        "cycles_df": cycles_df,
        "stats_df": stats,
        "wide": wide,
        "night": night,
        "valleys": valleys,
        "peaks": peaks,
        "cycles_meta": cycles,
    }
