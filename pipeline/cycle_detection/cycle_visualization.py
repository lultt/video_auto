"""
cycle_visualization.py — diagnostic plots for Layer 2 cycle detection.

Generates:
  - cycle_boundaries.png   (4-panel: motion+valleys/peaks, edge, brightness, entropy)
  - peak_detection.png     (single-panel with cycle shading)
  - periodicity_analysis.png (autocorrelation + FFT power spectrum)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fft_analysis import autocorrelation, fft_periodicity

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_cycle_boundaries(
    wide: pd.DataFrame,
    night: pd.DataFrame,
    valleys: np.ndarray,
    peaks: np.ndarray,
    cycles_df: pd.DataFrame,
    out_path: Path,
):
    times = night.index
    fig, axes = plt.subplots(4, 1, figsize=(24, 12), sharex=True)

    ax = axes[0]
    ax.plot(times, night["motion_smooth"], color="#1f77b4", linewidth=1.0, label="smoothed motion")
    ax.scatter(times[valleys], night["motion_smooth"].iloc[valleys],
               color="green", s=40, zorder=5, marker="v", label="valley boundary")
    ax.scatter(times[peaks], night["motion_smooth"].iloc[peaks],
               color="red", s=50, zorder=5, marker="^", label="peak")
    ax.set_ylabel("Motion Intensity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    series = [
        ("edge_mean", "Edge Density", "#ff7f0e"),
        ("brightness_mean", "Brightness", "#2ca02c"),
        ("entropy_mean", "Entropy", "#9467bd"),
    ]
    for ax_i, (col, label, color) in zip(axes[1:], series):
        ax_i.plot(times, night[col], color=color, linewidth=1.0, label=label)
        ax_i.set_ylabel(label)
        ax_i.grid(True, alpha=0.25)
        ax_i.legend(loc="upper right")

    for _, row in cycles_df.iterrows():
        st, et = pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"])
        pt = pd.Timestamp(row["peak_time"])
        for ax_i in axes:
            ax_i.axvspan(st, et, color="red", alpha=0.06)
            ax_i.axvline(st, color="red", linewidth=0.8, alpha=0.4)
            ax_i.axvline(et, color="red", linewidth=0.8, alpha=0.4)
            ax_i.axvline(pt, color="black", linewidth=0.6, alpha=0.3, linestyle="--")
    for _, row in cycles_df.iterrows():
        y_top = axes[0].get_ylim()[1]
        axes[0].text(pd.Timestamp(row["peak_time"]), y_top, str(int(row["cycle_id"])),
                     ha="center", va="top", fontsize=10, color="red", fontweight="bold")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.suptitle("Night Net Cycle Detection — Valley-to-Valley Boundaries", fontsize=15)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_peak_detection(
    night: pd.DataFrame,
    valleys: np.ndarray,
    peaks: np.ndarray,
    cycles_df: pd.DataFrame,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(24, 6))
    times = night.index
    ax.plot(times, night["motion_smooth"], color="#1f77b4", linewidth=1.2, label="smoothed motion")
    ax.scatter(times[valleys], night["motion_smooth"].iloc[valleys],
               color="green", s=50, zorder=5, marker="v", label="valley boundary")
    ax.scatter(times[peaks], night["motion_smooth"].iloc[peaks],
               color="red", s=60, zorder=5, marker="^", label="motion peak")
    for _, row in cycles_df.iterrows():
        ax.axvspan(pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"]),
                   color="red", alpha=0.06)
        ax.text(pd.Timestamp(row["peak_time"]), row["max_motion"],
                str(int(row["cycle_id"])), ha="center", va="bottom",
                fontsize=11, color="red", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.set_title("Motion Peak + Valley Detection on Smoothed Motion Signal")
    ax.set_ylabel("Motion Intensity")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_periodicity(night: pd.DataFrame, out_path: Path):
    motion = night["motion_smooth"].to_numpy()
    lags, ac = autocorrelation(motion, max_lag=600)
    periods, power, best_period = fft_periodicity(motion)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].plot(lags, ac, color="black")
    axes[0].set_xlim(0, 300)
    axes[0].set_xlabel("Lag (minutes)")
    axes[0].set_ylabel("Autocorrelation")
    axes[0].set_title("Autocorrelation of Smoothed Motion (Night)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(periods, power, color="#1f77b4")
    axes[1].axvline(best_period, color="red", linestyle="--",
                    label=f"Best period ~{best_period:.1f} min" if not np.isnan(best_period) else "")
    axes[1].set_xlabel("Period (minutes)")
    axes[1].set_ylabel("FFT Power")
    axes[1].set_title("FFT Periodicity Analysis")
    axes[1].grid(True, alpha=0.3)
    if not np.isnan(best_period):
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(night, wide, valleys, peaks, cycles_df, out_dir: Path):
    """Generate the three diagnostic plots in one call."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_cycle_boundaries(wide, night, valleys, peaks, cycles_df, out_dir / "cycle_boundaries.png")
    plot_peak_detection(night, valleys, peaks, cycles_df, out_dir / "peak_detection.png")
    plot_periodicity(night, out_dir / "periodicity_analysis.png")