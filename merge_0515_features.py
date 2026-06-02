from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from pipeline.shared.io.cbf_reader import read_cbf
from pipeline.shared.utils.time_utils import parse_video_filename_timestamp, parse_video_time_range, timestamp_to_wall_time

FEATURES = [
    ("brightness", "Brightness", "mean_brightness"),
    ("brightness_std", "Brightness Std", "brightness_std"),
    ("motion", "Motion", "motion_intensity"),
    ("entropy", "Entropy", "entropy"),
]
BOUNDARY_FRAMES = 25
RAW_ALPHA = 0.18
SMOOTH_LW = 1.65
COLOR_LEFT = "#2E86AB"
COLOR_RIGHT = "#E63946"
COLOR_MEAN = "#2A9D8F"
COLOR_DIFF = "#6A4C93"

plt.rcParams.update({
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#cccccc",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#dddddd",
    "font.family": ["Inter", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "savefig.facecolor": "#fafafa",
    "savefig.bbox": "tight",
})


def smooth(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n < 7:
        return values.copy()
    win = min(31, n - (1 - n % 2))
    if win < 5:
        return values.copy()
    poly = 3 if win > 5 else 2
    try:
        return savgol_filter(values, win, poly, mode="interp")
    except Exception:
        return pd.Series(values).rolling(win, center=True, min_periods=1).mean().to_numpy()


def read_one_cbf(path: Path) -> tuple[pd.DataFrame, dict]:
    df = read_cbf(path)
    start_time = parse_video_filename_timestamp(path.name)
    time_range = parse_video_time_range(path.name)
    if start_time is None or time_range is None:
        raise ValueError(f"Cannot parse timestamp from {path}")

    end_time = time_range[1]
    fps = None
    frame_max = int(df["frame_idx"].max()) if not df.empty else -1
    frame_count = frame_max + 1 if frame_max >= 0 else 0
    if frame_count > 1:
        fps_est = (frame_count - 1) / max(float(df["timestamp_sec"].max()), 1e-6)
        fps = round(fps_est, 6)
    else:
        fps = 0.0

    df = df.copy()
    df["source_file"] = path.name
    df["wall_time"] = df["timestamp_sec"].apply(lambda s: timestamp_to_wall_time(path.name, float(s)))
    df["is_boundary"] = ((df["frame_idx"] < BOUNDARY_FRAMES) | (df["frame_idx"] >= max(frame_count - BOUNDARY_FRAMES, 0))).astype(int)

    left = df[df["roi_code"] == 0].sort_values("frame_idx").set_index("frame_idx")
    right = df[df["roi_code"] == 1].sort_values("frame_idx").set_index("frame_idx")
    merged = pd.DataFrame(index=left.index.union(right.index)).sort_index()
    merged["source_file"] = left["source_file"].reindex(merged.index).fillna(right["source_file"])
    merged["wall_time"] = left["wall_time"].reindex(merged.index).fillna(right["wall_time"])
    merged["is_boundary"] = left["is_boundary"].reindex(merged.index).fillna(right["is_boundary"]).astype(int)

    for short_name, _, cbf_col in FEATURES:
        merged[f"left_deck_{short_name}"] = left[cbf_col].reindex(merged.index)
        merged[f"right_deck_{short_name}"] = right[cbf_col].reindex(merged.index)
        merged[f"roi_mean_{short_name}"] = (merged[f"left_deck_{short_name}"] + merged[f"right_deck_{short_name}"]) / 2.0
        merged[f"roi_diff_{short_name}"] = merged[f"left_deck_{short_name}"] - merged[f"right_deck_{short_name}"]

    merged = merged.reset_index().rename(columns={"index": "frame_idx"})
    summary = {
        "video_name": path.name,
        "fps": fps,
        "frame_count": frame_count,
        "kept_frames": len(merged),
        "start_time": start_time,
        "end_time": end_time,
    }
    return merged, summary


def merge_cbfs(feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    summaries = []
    for path in sorted(feature_dir.glob("*.cbf")):
        merged, summary = read_one_cbf(path)
        parts.append(merged)
        summaries.append(summary)
    if not parts:
        raise FileNotFoundError(f"No .cbf files found in {feature_dir}")
    raw_df = pd.concat(parts, ignore_index=True).sort_values("wall_time").reset_index(drop=True)
    clean_df = raw_df[raw_df["is_boundary"] == 0].copy().reset_index(drop=True)
    summary_df = pd.DataFrame(summaries)
    return raw_df, clean_df, summary_df


def plot_single_trace(df: pd.DataFrame, col: str, color: str, title: str, out_path: Path):
    if col not in df.columns:
        return
    fig, ax = plt.subplots(1, 1, figsize=(16, 4.5))
    x = df["wall_time"]
    y = df[col].to_numpy(dtype=float)
    ax.plot(x, y, color=color, alpha=RAW_ALPHA, linewidth=0.7)
    ax.plot(x, smooth(y), color=color, linewidth=SMOOTH_LW)
    ax.set_title(title, loc="left", pad=4)
    ax.grid(True)
    ax.margins(x=0.002)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  PLOT: {out_path}")


def plot_roi_diff(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(16, 8.4), sharex=True)
    diff_features = [
        ("brightness", "Brightness diff (L-R)"),
        ("motion", "Motion diff (L-R)"),
        ("entropy", "Entropy diff (L-R)"),
    ]
    for ax, (short_name, label) in zip(axes, diff_features):
        col = f"roi_diff_{short_name}"
        if col not in df.columns:
            continue
        x = df["wall_time"]
        y = df[col].to_numpy(dtype=float)
        ax.plot(x, y, color=COLOR_DIFF, alpha=RAW_ALPHA, linewidth=0.7)
        ax.plot(x, smooth(y), color=COLOR_DIFF, linewidth=SMOOTH_LW, label=col)
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle="-")
        ax.set_title(label, loc="left", pad=4)
        ax.grid(True)
        ax.margins(x=0.002)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07, hspace=0.35)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


ZOOM_WINDOWS = [
    {
        "name": "window1_211849_211905",
        "focus_start": "2025-05-15 21:18:49",
        "focus_end": "2025-05-15 21:19:05",
        "pad_before": pd.Timedelta(minutes=2),
        "pad_after": pd.Timedelta(minutes=2),
    },
    {
        "name": "window2_221223_221300",
        "focus_start": "2025-05-15 22:12:23",
        "focus_end": "2025-05-15 22:13:00",
        "pad_before": pd.Timedelta(minutes=2),
        "pad_after": pd.Timedelta(minutes=2),
    },
]

ZOOM_FEATURES = [
    ("brightness", "Brightness", "mean_brightness"),
    ("motion", "Motion", "motion_intensity"),
    ("entropy", "Entropy", "entropy"),
]


def plot_zoom_window(df: pd.DataFrame, window: dict, out_dir: Path):
    focus_start = pd.Timestamp(window["focus_start"])
    focus_end = pd.Timestamp(window["focus_end"])
    zoom_start = focus_start - window["pad_before"]
    zoom_end = focus_end + window["pad_after"]
    zoom_df = df[(df["wall_time"] >= zoom_start) & (df["wall_time"] <= zoom_end)].copy()
    if zoom_df.empty:
        print(f"  WARNING: no data in zoom window {window['name']}")
        return

    fig, axes = plt.subplots(len(ZOOM_FEATURES), 1, figsize=(16, 8.4), sharex=True)
    for ax, (short_name, label, _) in zip(axes, ZOOM_FEATURES):
        x = zoom_df["wall_time"]
        for col, color, lbl in [
            (f"left_deck_{short_name}", COLOR_LEFT, "left_deck"),
            (f"right_deck_{short_name}", COLOR_RIGHT, "right_deck"),
            (f"roi_mean_{short_name}", COLOR_MEAN, "roi_mean"),
        ]:
            if col in zoom_df.columns:
                ax.plot(x, zoom_df[col], color=color, linewidth=1.1, label=lbl)
        ax.axvline(focus_start, color="#C1121F", linestyle="--", linewidth=1.0)
        ax.axvline(focus_end, color="#C1121F", linestyle="--", linewidth=1.0)
        ax.set_title(label, loc="left", pad=4)
        ax.grid(True)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0, 60, 10)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    title = f"0515 zoom: {window['focus_start'].split(' ')[1]} ~ {window['focus_end'].split(' ')[1]}"
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.07, hspace=0.35)
    out_path = out_dir / f"{window['name']}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  ZOOM: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge 0515 CBF outputs into raw/clean experiment artifacts")
    parser.add_argument("--feature-dir", default=r"J:\video_auto\outputs\0515_fullframe\features")
    parser.add_argument("--output-root", default=r"J:\video_auto\outputs\0515_fullframe")
    parser.add_argument("--plot-only", action="store_true", help="Skip merge, read existing parquet and re-plot")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_path = output_root / "0515_fullframe_raw.parquet"
    clean_path = output_root / "0515_fullframe_clean.parquet"
    summary_csv = output_root / "video_summary.csv"

    if args.plot_only:
        print("Loading existing parquet...")
        clean_df = pd.read_parquet(clean_path)
    else:
        raw_df, clean_df, summary_df = merge_cbfs(feature_dir)
        merged_path = output_root / "0515_fullframe_features.parquet"
        raw_df.to_parquet(raw_path, index=False)
        raw_df.to_parquet(merged_path, index=False)
        clean_df.to_parquet(clean_path, index=False)
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        print(f"RAW PARQUET: {raw_path}")
        print(f"CLEAN PARQUET: {clean_path}")
        print(f"VIDEO SUMMARY: {summary_csv}")
        print(f"TOTAL VIDEOS: {len(summary_df)}")
        print(f"TOTAL FRAMES: {int(summary_df['frame_count'].sum()) if not summary_df.empty else 0}")

    # Per-feature time series plots: left / right / mean each as separate file
    feature_plots = [
        ("brightness", "Brightness"),
        ("brightness_std", "Brightness Std"),
        ("motion", "Motion"),
        ("entropy", "Entropy"),
    ]
    for short_name, label in feature_plots:
        plot_single_trace(clean_df, f"left_deck_{short_name}", COLOR_LEFT,
                          f"{label} (left_deck)", output_root / f"{short_name}_left_time_series.png")
        plot_single_trace(clean_df, f"right_deck_{short_name}", COLOR_RIGHT,
                          f"{label} (right_deck)", output_root / f"{short_name}_right_time_series.png")
        plot_single_trace(clean_df, f"roi_mean_{short_name}", COLOR_MEAN,
                          f"{label} (roi_mean)", output_root / f"{short_name}_mean_time_series.png")

    # ROI diff plot
    diff_out = output_root / "roi_diff_time_series.png"
    plot_roi_diff(clean_df, diff_out)
    print(f"  PLOT: {diff_out}")

    # Zoom window plots
    zoom_dir = output_root / "zoom_windows"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    for window in ZOOM_WINDOWS:
        plot_zoom_window(clean_df, window, zoom_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
