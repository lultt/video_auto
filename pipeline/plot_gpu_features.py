"""
plot_gpu_features.py — Time-series visualization of LEFT/RIGHT ROI coarse features.

Reads the GPU-pipeline CSVs produced by cpp_pipeline.exe, then for each video:
  - one overview dashboard (6 features stacked, shared time axis)
  - one stand-alone high-DPI PNG per feature
And finally a corpus-wide summary figure.

Style: scientific / industrial dashboard — Inter / DejaVu Sans, sparse grid,
per-axis y-range, raw trace as faint background + Savitzky-Golay smoothed line
in foreground, real wall-clock x-axis (from filename timestamp), left/right
colour-coded with consistent legend.
"""

from __future__ import annotations

import re
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
CSV_DIR  = Path(r"J:\video_auto\outputs\feature_plots_gpu\gpu_features")
OUT_ROOT = Path(r"J:\video_auto\outputs\feature_plots_gpu")
PER_VIDEO_DIR = OUT_ROOT / "per_video"
SINGLE_FEAT_DIR = OUT_ROOT / "single_feature"
SUMMARY_DIR = OUT_ROOT / "summary"
for d in (PER_VIDEO_DIR, SINGLE_FEAT_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)

FEATURES = [
    ("mean_brightness",    "Mean Brightness",       "intensity (0–255)"),
    ("brightness_std",     "Brightness Std",        "std (0–128)"),
    ("mean_b",             "Mean B (Blue)",         "channel (0–255)"),
    ("mean_g",             "Mean G (Green)",        "channel (0–255)"),
    ("mean_r",             "Mean R (Red)",          "channel (0–255)"),
    ("motion_intensity",   "Motion Intensity",      "mean |Δ| frame-to-frame"),
    ("entropy",            "Entropy",               "bits (0–8)"),
    ("edge_density",       "Edge Density",          "fraction of edge pixels"),
    ("laplacian_variance", "Laplacian Variance",    "focus / texture proxy"),
]

# Colours: left = cool, right = warm; muted but high-contrast.
COLOR_LEFT  = "#2E86AB"   # steel blue
COLOR_RIGHT = "#E63946"   # crimson
RAW_ALPHA   = 0.18
SMOOTH_LW   = 1.65
RAW_LW      = 0.8
FIG_DPI     = 160

# ---------------------------------------------------------------------------
# Scientific style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#fafafa",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.labelcolor":   "#222222",
    "axes.titlecolor":   "#111111",
    "axes.titleweight":  "semibold",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        "#cccccc",
    "grid.linestyle":    ":",
    "grid.linewidth":    0.5,
    "grid.alpha":        0.7,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#dddddd",
    "legend.fontsize":   9,
    "font.family":       ["Inter", "DejaVu Sans", "Arial"],
    "font.size":         10,
    "savefig.facecolor": "#fafafa",
    "savefig.dpi":       FIG_DPI,
    "savefig.bbox":      "tight",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FNAME_RE = re.compile(r"(ch\d+)_(\d{8})_(\d{6})_(\d{6})")

def parse_video_start(video_id: str) -> datetime | None:
    """Extract wall-clock start time from ch01_YYYYMMDD_HHMMSS_HHMMSS pattern."""
    m = _FNAME_RE.search(video_id)
    if not m:
        return None
    return datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")

def smooth(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay if there's enough samples, else light rolling mean fallback."""
    n = len(y)
    if n < 7:
        return y.copy()
    win = min(31, n - (1 - n % 2))   # odd, ≤31, ≤n
    if win < 5:
        return y.copy()
    poly = 3 if win > 5 else 2
    try:
        return savgol_filter(y, win, poly, mode="interp")
    except Exception:
        return pd.Series(y).rolling(win, center=True, min_periods=1).mean().to_numpy()

def fmt_time_axis(ax, start: datetime | None, span_sec: float):
    if start is None:
        ax.set_xlabel("time within video (mm:ss)")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda s, _: f"{int(s)//60:02d}:{int(s)%60:02d}")
        )
        return
    ax.set_xlabel("wall-clock time")
    if span_sec <= 90 * 60:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int(span_sec / 60 / 8))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)

def time_axis_values(t_sec: np.ndarray, start: datetime | None):
    if start is None:
        return t_sec
    return np.array([start + timedelta(seconds=float(s)) for s in t_sec])

# ---------------------------------------------------------------------------
# Per-video plotting
# ---------------------------------------------------------------------------
def plot_one_video(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    video_id = df["video_id"].iloc[0]
    start = parse_video_start(video_id)

    df_l = df[df["roi_name"] == "left"].sort_values("timestamp_sec").reset_index(drop=True)
    df_r = df[df["roi_name"] == "right"].sort_values("timestamp_sec").reset_index(drop=True)
    if len(df_l) < 3 or len(df_r) < 3:
        return None

    span_sec = float(max(df_l["timestamp_sec"].max(), df_r["timestamp_sec"].max()))
    x_l = time_axis_values(df_l["timestamp_sec"].to_numpy(), start)
    x_r = time_axis_values(df_r["timestamp_sec"].to_numpy(), start)

    # ------------- overview dashboard (6 rows, 1 col) -------------
    n = len(FEATURES)
    fig, axes = plt.subplots(n, 1, figsize=(13.5, 2.0 * n + 1.2),
                             sharex=True, constrained_layout=False)

    for ax, (col, title, unit) in zip(axes, FEATURES):
        yl = df_l[col].to_numpy()
        yr = df_r[col].to_numpy()
        yl_s = smooth(yl)
        yr_s = smooth(yr)

        ax.plot(x_l, yl, color=COLOR_LEFT,  lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(x_r, yr, color=COLOR_RIGHT, lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(x_l, yl_s, color=COLOR_LEFT,  lw=SMOOTH_LW, label="left")
        ax.plot(x_r, yr_s, color=COLOR_RIGHT, lw=SMOOTH_LW, label="right")

        # Auto y-range with light padding; clip to non-negative for non-negative features.
        ymin = float(min(yl.min(), yr.min()))
        ymax = float(max(yl.max(), yr.max()))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        lo = max(ymin - pad, 0.0)
        ax.set_ylim(lo, ymax + pad)

        ax.set_title(f"{title}   ·   {unit}", loc="left", pad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.grid(True, axis="both", which="major")
        ax.margins(x=0.005)

    # only top subplot needs legend (shared semantics)
    axes[0].legend(loc="upper right", ncol=2, handlelength=2.2, borderpad=0.45)
    fmt_time_axis(axes[-1], start, span_sec)

    # Title
    start_str = start.strftime("%Y-%m-%d  %H:%M:%S") if start else "unknown start"
    dur_min = span_sec / 60
    fig.suptitle(
        f"ROI Feature Time Series — {video_id}",
        fontsize=14, fontweight="bold", y=0.995, x=0.02, ha="left",
    )
    fig.text(0.02, 0.972,
             f"start {start_str}   ·   span {dur_min:.1f} min   ·   "
             f"{len(df_l)} keyframes / ROI   ·   GPU NVDEC pipeline",
             ha="left", fontsize=9.5, color="#555555")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.945, bottom=0.055, hspace=0.45)

    out = PER_VIDEO_DIR / f"{video_id}__dashboard.png"
    fig.savefig(out)
    plt.close(fig)

    # ------------- stand-alone single-feature PNGs -------------
    for col, title, unit in FEATURES:
        yl = df_l[col].to_numpy()
        yr = df_r[col].to_numpy()
        yl_s = smooth(yl)
        yr_s = smooth(yr)

        fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=False)
        ax.plot(x_l, yl, color=COLOR_LEFT,  lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(x_r, yr, color=COLOR_RIGHT, lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(x_l, yl_s, color=COLOR_LEFT,  lw=SMOOTH_LW, label="left  (smoothed)")
        ax.plot(x_r, yr_s, color=COLOR_RIGHT, lw=SMOOTH_LW, label="right (smoothed)")

        ymin = float(min(yl.min(), yr.min()))
        ymax = float(max(yl.max(), yr.max()))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        lo = max(ymin - pad, 0.0)
        ax.set_ylim(lo, ymax + pad)

        ax.set_title(f"{title}  —  {video_id}", loc="left", pad=6, fontsize=12)
        ax.set_ylabel(unit)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax.grid(True)
        ax.margins(x=0.005)
        ax.legend(loc="upper right", ncol=2, handlelength=2.5)
        fmt_time_axis(ax, start, span_sec)

        feat_dir = SINGLE_FEAT_DIR / col
        feat_dir.mkdir(exist_ok=True)
        out = feat_dir / f"{video_id}.png"
        fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.16)
        fig.savefig(out)
        plt.close(fig)

    return {
        "video_id": video_id,
        "start": start,
        "span_sec": span_sec,
        "n_keyframes": len(df_l),
    }

# ---------------------------------------------------------------------------
# Corpus-wide summary (one row per feature × ROI, stacked over time)
# ---------------------------------------------------------------------------
def plot_corpus_summary(all_csvs: list[Path]):
    """Concatenate every video into one wall-clock time series and plot smoothed traces."""
    parts = []
    for csv in all_csvs:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        video_id = df["video_id"].iloc[0]
        start = parse_video_start(video_id)
        if start is None:
            continue
        df = df.copy()
        df["abs_time"] = [start + timedelta(seconds=float(s)) for s in df["timestamp_sec"]]
        parts.append(df)

    if not parts:
        print("  [warn] no usable CSVs for corpus summary")
        return

    big = pd.concat(parts, ignore_index=True)
    big = big.sort_values("abs_time")

    n = len(FEATURES)
    fig, axes = plt.subplots(n, 1, figsize=(15.5, 1.9 * n + 1.4),
                             sharex=True, constrained_layout=False)

    span_sec = (big["abs_time"].max() - big["abs_time"].min()).total_seconds()

    for ax, (col, title, unit) in zip(axes, FEATURES):
        for roi, color, label in [("left", COLOR_LEFT, "left"),
                                  ("right", COLOR_RIGHT, "right")]:
            sub = big[big["roi_name"] == roi].sort_values("abs_time")
            x = sub["abs_time"].to_numpy()
            y = sub[col].to_numpy()
            ys = smooth(y)
            ax.plot(x, y,  color=color, lw=0.6, alpha=0.12)
            ax.plot(x, ys, color=color, lw=1.3, label=label)

        ymin = float(big[col].min()); ymax = float(big[col].max())
        pad = max((ymax - ymin) * 0.07, 1e-3)
        ax.set_ylim(max(ymin - pad, 0.0), ymax + pad)
        ax.set_title(f"{title}   ·   {unit}", loc="left", pad=4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.grid(True)
        ax.margins(x=0.003)

    axes[0].legend(loc="upper right", ncol=2, handlelength=2.2)
    fmt_time_axis(axes[-1], big["abs_time"].iloc[0].to_pydatetime(), span_sec)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_sec / 3600 / 12))))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    first_ts = big["abs_time"].iloc[0]
    last_ts  = big["abs_time"].iloc[-1]
    fig.suptitle("Corpus-wide ROI Feature Time Series",
                 fontsize=15, fontweight="bold", y=0.995, x=0.02, ha="left")
    fig.text(0.02, 0.972,
             f"{first_ts:%Y-%m-%d %H:%M}  →  {last_ts:%Y-%m-%d %H:%M}   ·   "
             f"{len(all_csvs)} videos   ·   {len(big)//2} keyframes / ROI   ·   "
             f"GPU NVDEC pipeline",
             ha="left", fontsize=10, color="#555555")

    fig.subplots_adjust(left=0.05, right=0.99, top=0.945, bottom=0.06, hspace=0.45)
    out = SUMMARY_DIR / "corpus_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")

    # Per-feature corpus PNGs as well
    for col, title, unit in FEATURES:
        fig, ax = plt.subplots(figsize=(15, 4.3))
        for roi, color, label in [("left", COLOR_LEFT, "left"),
                                  ("right", COLOR_RIGHT, "right")]:
            sub = big[big["roi_name"] == roi].sort_values("abs_time")
            x = sub["abs_time"].to_numpy()
            y = sub[col].to_numpy()
            ys = smooth(y)
            ax.plot(x, y,  color=color, lw=0.5, alpha=0.13)
            ax.plot(x, ys, color=color, lw=1.4, label=label)
        ymin = float(big[col].min()); ymax = float(big[col].max())
        pad = max((ymax - ymin) * 0.07, 1e-3)
        ax.set_ylim(max(ymin - pad, 0.0), ymax + pad)
        ax.set_title(f"{title}  —  corpus  ·  {first_ts:%Y-%m-%d %H:%M} → {last_ts:%Y-%m-%d %H:%M}",
                     loc="left", pad=6, fontsize=12)
        ax.set_ylabel(unit)
        ax.grid(True)
        ax.margins(x=0.003)
        ax.legend(loc="upper right", ncol=2)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(span_sec / 3600 / 14))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.set_xlabel("wall-clock time")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.16)
        out = SUMMARY_DIR / f"corpus__{col}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  wrote {out.name}")


def main():
    csvs = sorted(CSV_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV in {CSV_DIR} — run the GPU pipeline with --emit-csv first.")

    print(f"Found {len(csvs)} CSV files in {CSV_DIR}")
    print(f"Output root: {OUT_ROOT}\n")

    for i, csv in enumerate(csvs, 1):
        info = plot_one_video(csv)
        if info:
            tag = info["start"].strftime("%H:%M") if info["start"] else "??:??"
            print(f"  [{i:2d}/{len(csvs)}] {info['video_id'][:42]:<42} "
                  f"start={tag}  n={info['n_keyframes']:4d}  span={info['span_sec']/60:5.1f}m")

    print("\n=== corpus summary ===")
    plot_corpus_summary(csvs)

    print(f"\nDone.")
    print(f"  per-video dashboards : {PER_VIDEO_DIR}")
    print(f"  single-feature PNGs  : {SINGLE_FEAT_DIR}")
    print(f"  corpus summary       : {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
