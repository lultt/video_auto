"""
plot_night_features.py — Per-night feature time-series dashboards.

Scans ALL parquet files under J:\video_auto\outputs, groups by night
(18:00–06:00), and generates one dashboard per night with all 9 features
× LEFT/RIGHT ROI.

Style: same scientific / industrial dashboard as plot_gpu_features.py.
"""

from __future__ import annotations

import re, os
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
PARQUET_ROOT = Path(r"J:\video_auto\outputs")
OUT_ROOT = Path(r"J:\video_auto\outputs\night_feature_plots")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    ("mean_brightness",    "Mean Brightness",       "intensity (0–255)"),
    ("brightness_std",     "Brightness Std",        "std"),
    ("mean_b",             "Mean B (Blue)",         "channel (0–255)"),
    ("mean_g",             "Mean G (Green)",        "channel (0–255)"),
    ("mean_r",             "Mean R (Red)",          "channel (0–255)"),
    ("motion_intensity",   "Motion Intensity",      "mean |Δ|"),
    ("entropy",            "Entropy",               "bits (0–8)"),
    ("edge_density",       "Edge Density",          "fraction"),
    ("laplacian_variance", "Laplacian Variance",    "texture proxy"),
]

COLOR_LEFT  = "#2E86AB"
COLOR_RIGHT = "#E63946"
RAW_ALPHA   = 0.18
SMOOTH_LW   = 1.65
RAW_LW      = 0.8
FIG_DPI     = 120   # slightly lower for batch speed

# Night window (inclusive)
NIGHT_START_HOUR = 18
NIGHT_END_HOUR   = 6

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
    "font.family":       ["DejaVu Sans", "Arial"],
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
    m = _FNAME_RE.search(str(video_id))
    if not m:
        return None
    return datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")

def smooth(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if n < 7:
        return y.copy()
    win = min(31, n - (1 - n % 2))
    if win < 5:
        return y.copy()
    poly = 3 if win > 5 else 2
    try:
        return savgol_filter(y, win, poly, mode="interp")
    except Exception:
        return pd.Series(y).rolling(win, center=True, min_periods=1).mean().to_numpy()

def assign_night_key(ts: pd.Timestamp) -> str:
    if ts.hour >= NIGHT_START_HOUR:
        return f"{ts.strftime('%Y-%m-%d')}_night"
    elif ts.hour < NIGHT_END_HOUR:
        return f"{(ts - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}_night"
    return "daytime"

# ---------------------------------------------------------------------------
# Load all parquets
# ---------------------------------------------------------------------------
def load_all_parquets(max_files: int = None) -> tuple:
    """Recursively scan all parquets, add abs_time, assign night keys."""
    files = sorted(PARQUET_ROOT.rglob("*.parquet"))
    # exclude debug/test dirs
    files = [f for f in files if "debug" not in str(f).lower()
             and "test" not in str(f).lower()]
    if max_files:
        files = files[:max_files]
    print(f"Scanning {len(files)} parquet files...")

    parts = []
    for i, f in enumerate(files):
        try:
            pf = pd.read_parquet(f)
        except Exception:
            continue
        if pf.empty or "roi_name" not in pf.columns:
            continue
        # keep left_deck + right_deck
        pf = pf[pf["roi_name"].isin(["left_deck", "right_deck"])]
        if pf.empty:
            continue

        vid = pf["video_id"].iloc[0]
        start = parse_video_start(vid)
        if start is None:
            continue
        pf["abs_time"] = [start + timedelta(seconds=float(s)) for s in pf["timestamp_sec"]]

        keep = ["video_id", "abs_time", "timestamp_sec", "roi_name",
                "mean_brightness", "brightness_std", "mean_b", "mean_g", "mean_r",
                "motion_intensity", "edge_density", "laplacian_variance", "entropy"]
        keep = [c for c in keep if c in pf.columns]
        parts.append(pf[keep])

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(files)}] loaded, {sum(len(p) for p in parts):,} rows so far")

    if not parts:
        raise SystemExit("No parquet files loaded.")

    big = pd.concat(parts, ignore_index=True)
    big["night"] = big["abs_time"].apply(assign_night_key)
    nights = sorted(n for n in big["night"].unique() if n != "daytime")
    print(f"  {len(big):,} rows, {big['video_id'].nunique()} videos, {len(nights)} nights")
    print(f"  Time range: {big['abs_time'].min()} → {big['abs_time'].max()}")
    return big, nights

# ---------------------------------------------------------------------------
# Plot one night
# ---------------------------------------------------------------------------
def plot_one_night(night_df: pd.DataFrame, night_label: str, out_dir: Path):
    df_l = night_df[night_df["roi_name"] == "left_deck"].sort_values("abs_time")
    df_r = night_df[night_df["roi_name"] == "right_deck"].sort_values("abs_time")
    if len(df_l) < 3 or len(df_r) < 3:
        return None

    x_l = df_l["abs_time"].values
    x_r = df_r["abs_time"].values
    t0 = x_l[0]
    t_hours_l = (x_l - t0).astype("timedelta64[s]").astype(float) / 3600.0
    t_hours_r = (x_r - t0).astype("timedelta64[s]").astype(float) / 3600.0

    n = len(FEATURES)
    fig, axes = plt.subplots(n, 1, figsize=(13.5, 1.85 * n + 1.2),
                             sharex=True, constrained_layout=False)

    for ax, (col, title, unit) in zip(axes, FEATURES):
        if col not in df_l.columns or col not in df_r.columns:
            continue
        yl = df_l[col].to_numpy()
        yr = df_r[col].to_numpy()
        yl_s = smooth(yl)
        yr_s = smooth(yr)

        ax.plot(t_hours_l, yl, color=COLOR_LEFT,  lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(t_hours_r, yr, color=COLOR_RIGHT, lw=RAW_LW, alpha=RAW_ALPHA)
        ax.plot(t_hours_l, yl_s, color=COLOR_LEFT,  lw=SMOOTH_LW, label="left")
        ax.plot(t_hours_r, yr_s, color=COLOR_RIGHT, lw=SMOOTH_LW, label="right")

        ymin = float(min(yl.min(), yr.min()))
        ymax = float(max(yl.max(), yr.max()))
        pad = max((ymax - ymin) * 0.08, 1e-3)
        ax.set_ylim(max(ymin - pad, 0.0), ymax + pad)
        ax.set_title(f"{title}   ·   {unit}", loc="left", pad=4, fontsize=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
        ax.grid(True, axis="both", which="major")
        ax.margins(x=0.005)

    axes[0].legend(loc="upper right", ncol=2, handlelength=2.2, borderpad=0.45)

    span_h = float(t_hours_l[-1])
    dur_min = span_h * 60
    axes[-1].set_xlabel(f"Time within night (hours)  —  {night_label}")
    axes[-1].set_xlim(0, max(span_h, 0.5))

    n_frames = len(df_l)
    fig.suptitle(
        f"ROI Feature Time Series — {night_label}",
        fontsize=13, fontweight="bold", y=0.995, x=0.02, ha="left",
    )
    fig.text(0.02, 0.975,
             f"span {dur_min:.0f} min   ·   {n_frames} keyframes / ROI   ·   GPU NVDEC pipeline",
             ha="left", fontsize=9, color="#555555")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.05, hspace=0.42)

    out = out_dir / f"{night_label}.png"
    fig.savefig(out)
    plt.close(fig)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None, help="limit parquet count")
    ap.add_argument("--night", type=str, default=None, help="plot only this night")
    args = ap.parse_args()

    big, nights = load_all_parquets(max_files=args.max_files)

    out_dir = OUT_ROOT / "per_night"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.night:
        nights = [n for n in nights if n == args.night]
        if not nights:
            print(f"Night '{args.night}' not found. Available: {sorted(big['night'].unique())}")
            return

    print(f"\nPlotting {len(nights)} nights → {out_dir}")
    for i, night in enumerate(nights):
        ndf = big[big["night"] == night].copy()
        out = plot_one_night(ndf, night, out_dir)
        status = f"→ {out.name}" if out else "→ SKIP (too few samples)"
        print(f"  [{i+1:3d}/{len(nights)}] {night}  {status}")

    print(f"\nDone. Output: {out_dir}")

if __name__ == "__main__":
    main()
