"""
detect_core_windows.py — Layer 3 orchestrator.

Given coarse Parquet data + detected cycles, extract the "core catch window"
(net-on-deck → fish-transfer-end) for each cycle.

Output: core_catch_windows.csv, core_window_statistics.csv, per-cycle stage plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .phase_segmentation import detect_fish_transfer, detect_net_on_deck, smooth_features
from ..shared.io import build_minute_series, load_parquet_folder, save_table


def run_core_window_detection(
    parquet_root: Path,
    cycles_csv: Path,
    out_dir: Path,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
    left_prominence: float = 0.008,
    left_min_dist: int = 10,
    left_search_pct: float = 0.70,
    left_rise_q: float = 0.30,
    left_rise_frac: float = 0.70,
    right_prominence: float = 0.008,
    right_min_dist: int = 15,
    right_fall_q: float = 0.40,
    right_fall_frac: float = 0.30,
    plot_stages: bool = True,
) -> pd.DataFrame:
    """
    Load cycles, extract core windows, save results.

    Returns results_df with one row per cycle.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_parquet_folder(Path(parquet_root))
    wide = build_minute_series(df, features=["edge_density", "motion_intensity", "entropy", "mean_brightness"])
    wide = smooth_features(wide, window=savgol_window, polyorder=savgol_polyorder)

    cycles_df = pd.read_csv(cycles_csv)
    results = []

    for _, row in cycles_df.iterrows():
        cycle_id = int(row["cycle_id"])
        t_start = pd.Timestamp(row["start_time"])
        t_end = pd.Timestamp(row["end_time"])

        mask = (wide.index >= t_start) & (wide.index <= t_end)
        seg = wide[mask].copy()
        if len(seg) < 10:
            print(f"  Cycle {cycle_id}: too short ({len(seg)} min), skipping")
            continue

        times = seg.index
        left_edge = seg["left_deck_edge_smooth"].to_numpy()
        right_edge = seg["right_deck_edge_smooth"].to_numpy()
        left_edge_raw = seg["left_deck_edge_density"].to_numpy()

        net_start_idx, net_on_deck_idx = detect_net_on_deck(
            left_edge, left_edge_raw,
            prominence=left_prominence, min_distance=left_min_dist,
            search_first_pct=left_search_pct,
            rise_low_quantile=left_rise_q, rise_frac=left_rise_frac,
        )
        fish_start_idx, fish_end_idx = detect_fish_transfer(
            right_edge, seg["right_deck_edge_density"].to_numpy(),
            net_on_deck_idx,
            prominence=right_prominence, min_distance=right_min_dist,
            fall_low_quantile=right_fall_q, fall_rise_frac=right_fall_frac,
        )

        core_start_idx = net_on_deck_idx
        core_end_idx = fish_end_idx
        core = seg.iloc[core_start_idx:core_end_idx + 1]
        pre_core = seg.iloc[:core_start_idx]

        right_edge_increase = round(
            float(core["right_deck_edge_density"].mean() / (pre_core["right_deck_edge_density"].mean() + 1e-6)), 2
        )

        result = {
            "cycle_id": cycle_id,
            "cycle_start": row["start_time"],
            "cycle_end": row["end_time"],
            "net_start_time": times[net_start_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "net_on_deck_time": times[net_on_deck_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "fish_start_time": times[fish_start_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "fish_end_time": times[fish_end_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "core_start_time": times[core_start_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "core_end_time": times[core_end_idx].strftime("%Y-%m-%d %H:%M:%S"),
            "duration_net_rising_min": round((times[net_on_deck_idx] - times[net_start_idx]).total_seconds() / 60, 1),
            "duration_fish_transfer_min": round((times[fish_end_idx] - times[fish_start_idx]).total_seconds() / 60, 1),
            "duration_core_window_min": round((times[core_end_idx] - times[core_start_idx]).total_seconds() / 60, 1),
            "duration_total_cycle_min": round((times[-1] - times[0]).total_seconds() / 60, 1),
            "left_edge_peak": round(float(left_edge[net_on_deck_idx]), 5),
            "right_edge_peak": round(float(right_edge[fish_start_idx:fish_end_idx + 1].max()), 5),
            "right_edge_increase_vs_pre": right_edge_increase,
        }
        results.append(result)

        if plot_stages:
            _plot_cycle_stages(seg, result, cycle_id, out_dir)
        print(f"  Cycle {cycle_id:2d}: core {result['core_start_time'][11:16]} → {result['core_end_time'][11:16]} "
              f"({result['duration_core_window_min']} min)")

    results_df = pd.DataFrame(results)
    save_table(results_df, out_dir / "core_catch_windows.csv")

    stats = results_df[[
        "duration_net_rising_min", "duration_fish_transfer_min",
        "duration_core_window_min", "duration_total_cycle_min",
        "right_edge_increase_vs_pre",
    ]].agg(["mean", "std", "min", "max"]).round(2)
    save_table(stats, out_dir / "core_window_statistics.csv")

    print(f"\nDone. {len(results_df)} core windows saved to {out_dir}")
    return results_df


def _plot_cycle_stages(seg_df, result, cycle_id, out_dir):
    """Generate per-cycle stage-annotation plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    times = seg_df.index
    fig, axes = plt.subplots(3, 1, figsize=(20, 9), sharex=True)

    # Left deck edge
    ax = axes[0]
    ax.plot(times, seg_df["left_deck_edge_density"], color="steelblue", alpha=0.4, linewidth=0.8, label="raw")
    ax.plot(times, seg_df["left_deck_edge_smooth"], color="darkblue", linewidth=1.5, label="smoothed")
    ax.axvline(pd.Timestamp(result["net_start_time"]), color="orange", linewidth=1.5, linestyle="--", label="net rising")
    ax.axvline(pd.Timestamp(result["net_on_deck_time"]), color="red", linewidth=2, label="net on deck")
    ax.axvspan(pd.Timestamp(result["core_start_time"]), pd.Timestamp(result["core_end_time"]), color="red", alpha=0.08, label="core window")
    ax.set_ylabel("Left Deck Edge Density")
    ax.set_title(f"Cycle {cycle_id}: Net Stage (Left Side)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Right deck edge
    ax = axes[1]
    ax.plot(times, seg_df["right_deck_edge_density"], color="forestgreen", alpha=0.4, linewidth=0.8, label="raw")
    ax.plot(times, seg_df["right_deck_edge_smooth"], color="darkgreen", linewidth=1.5, label="smoothed")
    ax.axvline(pd.Timestamp(result["fish_start_time"]), color="purple", linewidth=1.5, linestyle="--", label="fish start")
    ax.axvline(pd.Timestamp(result["fish_end_time"]), color="darkred", linewidth=2, label="fish end")
    ax.axvspan(pd.Timestamp(result["core_start_time"]), pd.Timestamp(result["core_end_time"]), color="red", alpha=0.08)
    ax.set_ylabel("Right Deck Edge Density")
    ax.set_title(f"Cycle {cycle_id}: Fish Transfer (Right Side)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Motion comparison
    ax = axes[2]
    ax.plot(times, seg_df.get("left_deck_motion_smooth", seg_df["left_deck_motion_intensity"]), color="darkblue", linewidth=1.2, label="left motion")
    ax.plot(times, seg_df.get("right_deck_motion_smooth", seg_df["right_deck_motion_intensity"]), color="darkgreen", linewidth=1.2, label="right motion")
    ax.axvspan(pd.Timestamp(result["core_start_time"]), pd.Timestamp(result["core_end_time"]), color="red", alpha=0.08)
    ax.axvline(pd.Timestamp(result["net_on_deck_time"]), color="red", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.axvline(pd.Timestamp(result["fish_end_time"]), color="darkred", linewidth=1.5, linestyle="-", alpha=0.7)
    ax.set_ylabel("Motion Intensity")
    ax.set_title("Left vs Right Motion")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / f"cycle_{cycle_id:02d}_stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig)