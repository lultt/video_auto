from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

FFMPEG_CANDIDATES = [r"D:\ffmpeg\bin\ffmpeg.exe", "ffmpeg"]
FFPROBE_CANDIDATES = [r"D:\ffmpeg\bin\ffprobe.exe", "ffprobe"]
VIDEO_GLOB = "ch01_*.mp4"
BOUNDARY_FRAMES = 25
PLOT_RAW_ALPHA = 0.18
PLOT_LW = 1.35
ROI_COLORS = {
    "left_deck": "#2E86AB",
    "right_deck": "#E63946",
    "roi_mean": "#2A9D8F",
    "roi_diff": "#6A4C93",
}
FEATURES = [
    ("brightness", "Brightness"),
    ("motion", "Motion"),
    ("entropy", "Entropy"),
]

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


def pick_executable(candidates: list[str]) -> str:
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, "-version"], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return candidate
    raise FileNotFoundError(f"Executable not found. Tried: {candidates}")


FFMPEG_PATH = pick_executable(FFMPEG_CANDIDATES)
FFPROBE_PATH = pick_executable(FFPROBE_CANDIDATES)


def load_rois(config_path: Path) -> dict[str, dict[str, float]]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rois = cfg["rois"]
    return {
        "left_deck": rois["left_deck"],
        "right_deck": rois["right_deck"],
    }


def parse_video_range(path: Path) -> tuple[datetime, datetime]:
    stem = path.stem
    parts = stem.split("_")
    start_dt = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(parts[1] + parts[3].replace("topspeed", ""), "%Y%m%d%H%M%S")
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def probe_video(video_path: Path) -> dict:
    cmd = [
        FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,r_frame_rate,nb_frames,width,height:format=duration",
        "-of", "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {video_path}")
    payload = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return {
        "fps": float(Fraction(payload["r_frame_rate"])),
        "frame_count": int(payload.get("nb_frames") or 0),
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "duration_sec": float(payload.get("duration") or 0.0),
        "codec": payload.get("codec_name", "").lower(),
    }


def build_roi_slices(rois: dict[str, dict[str, float]], width: int, height: int) -> dict[str, tuple[slice, slice]]:
    roi_slices = {}
    for name, roi in rois.items():
        y1 = int(round(roi["y_min"] * height))
        y2 = int(round(roi["y_max"] * height))
        x1 = int(round(roi["x_min"] * width))
        x2 = int(round(roi["x_max"] * width))
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))
    return roi_slices


def entropy_bits(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= hist.sum() + 1e-12
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


def decode_command(video_path: Path, codec: str, use_gpu: bool) -> list[str]:
    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error"]
    if use_gpu:
        decoder = "hevc_cuvid" if codec == "hevc" else "h264_cuvid"
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder]
    cmd += ["-i", str(video_path)]
    if use_gpu:
        cmd += ["-vf", "hwdownload,format=nv12,format=bgr24"]
    cmd += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return cmd


def decode_frames(video_path: Path, meta: dict, use_gpu: bool):
    cmd = decode_command(video_path, meta["codec"], use_gpu)
    frame_size = meta["width"] * meta["height"] * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((meta["height"], meta["width"], 3))
    finally:
        if proc.stdout:
            proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg decode failed for {video_path}")


def smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    return pd.Series(values).rolling(31, center=True, min_periods=1).mean().to_numpy()


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_video_to_parquet(video_path: Path, rois: dict[str, dict[str, float]], out_path: Path, use_gpu: bool) -> dict:
    meta = probe_video(video_path)
    roi_slices = build_roi_slices(rois, meta["width"], meta["height"])
    start_dt, end_dt = parse_video_range(video_path)
    total_frames = meta["frame_count"]

    def run_decode(gpu_enabled: bool) -> tuple[pd.DataFrame, str]:
        prev_gray = {name: None for name in roi_slices}
        rows = []
        backend = "gpu" if gpu_enabled else "cpu-fallback"
        for frame_idx, frame in enumerate(decode_frames(video_path, meta, use_gpu=gpu_enabled)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            wall_time = start_dt + timedelta(seconds=frame_idx / meta["fps"])
            is_boundary = 1 if (frame_idx < BOUNDARY_FRAMES or frame_idx >= max(total_frames - BOUNDARY_FRAMES, 0)) else 0
            row = {
                "source_file": video_path.name,
                "frame_idx": frame_idx,
                "wall_time": wall_time,
                "is_boundary": is_boundary,
            }
            for roi_name, roi_slice in roi_slices.items():
                roi_gray = gray[roi_slice]
                brightness = float(roi_gray.mean())
                entropy = entropy_bits(roi_gray)
                if prev_gray[roi_name] is None:
                    motion = 0.0
                else:
                    motion = float(cv2.absdiff(roi_gray, prev_gray[roi_name]).mean())
                row[f"{roi_name}_brightness"] = brightness
                row[f"{roi_name}_motion"] = motion
                row[f"{roi_name}_entropy"] = entropy
                prev_gray[roi_name] = roi_gray.copy()
            for feature, _ in FEATURES:
                left = row[f"left_deck_{feature}"]
                right = row[f"right_deck_{feature}"]
                row[f"roi_mean_{feature}"] = (left + right) / 2.0
                row[f"roi_diff_{feature}"] = left - right
            rows.append(row)
        return pd.DataFrame(rows), backend

    try:
        df, backend = run_decode(use_gpu)
    except Exception:
        if not use_gpu:
            raise
        df, backend = run_decode(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    kept_frames = len(df)
    del df
    return {
        "video_name": video_path.name,
        "fps": meta["fps"],
        "frame_count": total_frames,
        "kept_frames": kept_frames,
        "start_time": start_dt,
        "end_time": end_dt,
        "backend": backend,
    }


def merge_feature_files(feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames = []
    clean_frames = []
    for path in sorted(feature_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        raw_frames.append(df)
        clean_frames.append(df[df["is_boundary"] == 0].copy())
    raw_df = pd.concat(raw_frames, ignore_index=True).sort_values("wall_time").reset_index(drop=True)
    clean_df = pd.concat(clean_frames, ignore_index=True).sort_values("wall_time").reset_index(drop=True)
    return raw_df, clean_df


def plot_dataset(df: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(16, 8.8), sharex=True)
    if len(FEATURES) == 1:
        axes = [axes]
    for ax, (feature, feature_title) in zip(axes, FEATURES):
        for key in ["left_deck", "right_deck", "roi_mean", "roi_diff"]:
            col = f"{key}_{feature}"
            y = df[col].to_numpy(dtype=float)
            x = df["wall_time"]
            ax.plot(x, y, color=ROI_COLORS[key], alpha=PLOT_RAW_ALPHA, linewidth=0.7)
            ax.plot(x, smooth(y), color=ROI_COLORS[key], linewidth=PLOT_LW, label=key)
        ax.set_title(feature_title, loc="left", pad=4)
        ax.grid(True)
        ax.margins(x=0.002)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.07, hspace=0.35)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_zoom(df: pd.DataFrame, out_path: Path):
    zoom_start = datetime(2025, 5, 15, 21, 17, 0)
    zoom_end = datetime(2025, 5, 15, 21, 21, 0)
    focus_start = datetime(2025, 5, 15, 21, 18, 49)
    focus_end = datetime(2025, 5, 15, 21, 19, 5)
    zoom_df = df[(df["wall_time"] >= zoom_start) & (df["wall_time"] <= zoom_end)].copy()
    cols = [
        ("right_deck_motion", "right_deck_motion"),
        ("right_deck_entropy", "right_deck_entropy"),
        ("right_deck_brightness", "right_deck_brightness"),
    ]
    fig, axes = plt.subplots(len(cols), 1, figsize=(16, 8.4), sharex=True)
    for ax, (col, title) in zip(axes, cols):
        ax.plot(zoom_df["wall_time"], zoom_df[col], color=ROI_COLORS["right_deck"], linewidth=1.1)
        ax.axvline(focus_start, color="#C1121F", linestyle="--", linewidth=1.0)
        ax.axvline(focus_end, color="#C1121F", linestyle="--", linewidth=1.0)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(True)
    axes[-1].xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0, 60, 30)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.suptitle("0515 zoom: 21:18:49 ~ 21:19:05", x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.07, hspace=0.35)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="0515 full-frame ROI feature experiment")
    parser.add_argument("--video-dir", default=r"C:\video\0515")
    parser.add_argument("--config-path", default=r"J:\video_auto\configs\config.yaml")
    parser.add_argument("--output-root", default=r"J:\video_auto\outputs\0515_fullframe")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    video_dir = Path(args.video_dir)
    output_root = Path(args.output_root)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rois = load_rois(Path(args.config_path))
    videos = sorted(video_dir.glob(VIDEO_GLOB))

    summary_rows = []
    print(f"SCRIPT: {Path(__file__).resolve()}", flush=True)
    print(f"TOTAL VIDEOS: {len(videos)}", flush=True)

    for idx, video_path in enumerate(videos, start=1):
        v0 = time.time()
        feature_path = feature_dir / f"{video_path.stem}.parquet"
        info = process_video_to_parquet(video_path, rois, feature_path, use_gpu=not args.cpu_only)
        summary_rows.append(info)
        elapsed = time.time() - t0
        avg = elapsed / idx
        eta = avg * (len(videos) - idx)
        print(
            f"PROGRESS {idx}/{len(videos)} | {info['video_name']} | backend={info['backend']} | "
            f"video_elapsed={fmt_seconds(time.time() - v0)} | total_elapsed={fmt_seconds(elapsed)} | eta={fmt_seconds(eta)}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_root / "video_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    raw_df, clean_df = merge_feature_files(feature_dir)
    raw_path = output_root / "0515_fullframe_raw.parquet"
    clean_path = output_root / "0515_fullframe_clean.parquet"
    merged_path = output_root / "0515_fullframe_features.parquet"
    raw_df.to_parquet(raw_path, index=False)
    raw_df.to_parquet(merged_path, index=False)
    clean_df.to_parquet(clean_path, index=False)

    raw_plot = output_root / "0515_fullframe_plot_raw.png"
    clean_plot = output_root / "0515_fullframe_plot_clean.png"
    zoom_plot = output_root / "0515_zoom_211849_211905.png"
    plot_dataset(raw_df, raw_plot, "0515 full-frame ROI features (raw)")
    plot_dataset(clean_df, clean_plot, "0515 full-frame ROI features (clean, boundary removed)")
    plot_zoom(clean_df, zoom_plot)

    total_elapsed = time.time() - t0
    total_frames = int(summary_df["frame_count"].sum()) if not summary_df.empty else 0
    print(f"RAW PARQUET: {raw_path}", flush=True)
    print(f"CLEAN PARQUET: {clean_path}", flush=True)
    print(f"RAW PLOT: {raw_plot}", flush=True)
    print(f"CLEAN PLOT: {clean_plot}", flush=True)
    print(f"ZOOM PLOT: {zoom_plot}", flush=True)
    print(f"TOTAL VIDEOS: {len(summary_df)}", flush=True)
    print(f"TOTAL FRAMES: {total_frames}", flush=True)
    print(f"TOTAL ELAPSED: {fmt_seconds(total_elapsed)}", flush=True)


if __name__ == "__main__":
    main()
