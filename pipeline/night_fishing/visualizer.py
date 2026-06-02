# visualizer.py — per-night diagnostic plots with valley-to-valley cycle regions

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import PLOT_DIR, PLOT_DPI


# Light palette for alternating cycle regions
CYCLE_COLORS = ["#e8f4fd", "#fef6e4", "#e8fde8", "#fde8f4",
                "#f4f4e8", "#e8f0fd", "#f0e8f4", "#e4fef6"]


def plot_night(df: pd.DataFrame, night_label: str, is_fishing: bool,
               net_count: int, peak_indices: list, is_complete: bool = True,
               warnings: list = None, segments: list = None,
               minute_info: dict = None) -> Optional[Path]:
    """Generate a night diagnostic figure with valley-to-valley cycle shading."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if len(df) < 3:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(15, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    (ax_fai, ax_motion, ax_bright) = axes

    # ---- Panel 1: FAI + cycle regions ----
    x = df["abs_time"].values
    t0 = x[0]
    t_hours = (x - t0).astype("timedelta64[s]").astype(float) / 3600.0

    ax_fai.plot(t_hours, df["fai_raw"], color="#cccccc", lw=0.6, alpha=0.4, label="FAI raw")
    ax_fai.plot(t_hours, df["fai_smooth"], color="#2E86AB", lw=1.8, label="FAI smoothed")
    ax_fai.axhline(y=0.30, color="#888888", ls=":", lw=0.8, alpha=0.6, label="activity threshold")

    # Shade valley-to-valley cycle regions
    if is_fishing and segments:
        for seg in segments:
            s_t = (seg["start_time"] - t0).total_seconds() / 3600.0
            e_t = (seg["end_time"] - t0).total_seconds() / 3600.0
            color = CYCLE_COLORS[seg["net_index"] % len(CYCLE_COLORS)]
            ax_fai.axvspan(s_t, e_t, alpha=0.35, color=color, zorder=0)
            mid = (s_t + e_t) / 2
            ax_fai.text(mid, ax_fai.get_ylim()[1] * 0.92,
                        f"#{seg['net_index']}", ha="center", fontsize=6.5,
                        color="#555555", fontweight="bold")

    # Mark valley positions on the FAI panel
    if minute_info is not None and minute_info.get("valleys") is not None:
        m_times = minute_info.get("times")
        valleys = minute_info.get("valleys")
        if m_times is not None and valleys is not None and len(valleys) > 0:
            v_hours = (m_times[valleys] - t0).total_seconds() / 3600.0
            for vh in v_hours:
                ax_fai.axvline(x=vh, color="#d62728", ls="--", lw=0.6, alpha=0.5, zorder=1)

    ax_fai.set_ylabel("FAI")
    ax_fai.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_fai.grid(True, alpha=0.3)
    ax_fai.margins(x=0.005)

    # Title
    status = "COMPLETE" if is_complete else "INCOMPLETE"
    fft_str = ""
    if minute_info is not None and minute_info.get("fft_period_min") is not None:
        period = minute_info["fft_period_min"]
        if not np.isnan(period):
            fft_str = f"  |  FFT period: {period:.0f} min"
    title_str = f"{night_label}   |   {status}"
    if is_complete:
        title_str += f"   |   Fishing: {is_fishing}   |   Net cycles: {net_count}{fft_str}"
    if warnings:
        title_str += "   |   " + " / ".join(warnings)
    ax_fai.set_title(title_str, loc="left", fontsize=11, fontweight="bold")

    # ---- Panel 2: 1-minute motion with valley markers + cycle shading ----
    if minute_info is not None:
        m_times = minute_info["times"]
        m_hours = (m_times - t0).total_seconds() / 3600.0
        ax_motion.plot(m_hours, minute_info["motion_raw"], color="#cccccc", lw=0.5, alpha=0.4)
        ax_motion.plot(m_hours, minute_info["motion_smooth"], color="#E63946", lw=1.3,
                       label="motion (1min, SG smoothed)")

        # Shade cycle regions on motion panel too
        if is_fishing and segments:
            for seg in segments:
                s_t = (seg["start_time"] - t0).total_seconds() / 3600.0
                e_t = (seg["end_time"] - t0).total_seconds() / 3600.0
                color = CYCLE_COLORS[seg["net_index"] % len(CYCLE_COLORS)]
                ax_motion.axvspan(s_t, e_t, alpha=0.25, color=color, zorder=0)

        # Valley markers
        valleys = minute_info.get("valleys")
        if valleys is not None and len(valleys) > 0:
            vh = m_hours[valleys]
            ax_motion.scatter(vh, minute_info["motion_smooth"][valleys],
                              color="#d62728", s=25, zorder=5, marker="v",
                              edgecolors="white", linewidths=0.5, label="valley")
        ax_motion.legend(loc="upper right", fontsize=8, framealpha=0.85)
    else:
        t_sec = (x - t0).astype("timedelta64[s]").astype(float)
        ax_motion.plot(t_sec / 3600, df["motion_intensity"], color="#E63946", lw=0.9, alpha=0.8)

    ax_motion.set_ylabel("Motion Intensity")
    ax_motion.grid(True, alpha=0.3)
    ax_motion.margins(x=0.005)

    # ---- Panel 3: Brightness ----
    ax_bright.plot(t_hours, df["mean_brightness"], color="#ff7f0e", lw=1.0, alpha=0.8, label="mean brightness")
    ax_bright.set_ylabel("Brightness")
    ax_bright.set_xlabel("Time within night (hours)")
    ax_bright.grid(True, alpha=0.3)
    ax_bright.legend(loc="upper right", fontsize=8)
    ax_bright.margins(x=0.005)
    span_h = t_hours[-1] if len(t_hours) > 0 else 12
    ax_bright.set_xlim(0, max(span_h, 1))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.12, top=0.94)
    out = PLOT_DIR / f"{night_label}.png"
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    return out
