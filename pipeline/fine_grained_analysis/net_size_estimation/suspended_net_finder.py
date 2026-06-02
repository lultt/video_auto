"""
suspended_net_finder.py — 找每个 cycle 中网兜完全悬空的"黄金帧"。

核心思路：
  在 net_start_time → net_on_deck_time 窗口内
  每 3-5 秒 抽一帧
  计算 left_deck ROI 内黑色像素占比
  黑色比例最大的那一帧 = 网兜完全悬空的瞬间

因为：
  - 网兜是深色（接近黑色）
  - 背景是夜空/海面（相对均匀深色但不如网兜黑）
  - 网最大时，黑色像素占比达到峰值
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extract_keyframes import scan_source_videos


def dark_pixel_ratio(img_bgr: np.ndarray, dark_threshold: int = 35) -> float:
    """计算图像中暗像素占比 (R、G、B 都 < threshold)."""
    h, w = img_bgr.shape[:2]
    pixel_count = h * w
    if pixel_count == 0:
        return 0.0
    dark_mask = (img_bgr[:, :, 0] < dark_threshold) & (
        img_bgr[:, :, 1] < dark_threshold) & (
        img_bgr[:, :, 2] < dark_threshold)
    return float(dark_mask.sum()) / pixel_count


def get_frame_at_wall_time(
    target_wall_time: pd.Timestamp,
    video_index: list[dict],
    out_dir: Path,
    tmp_frame_path: Path,
) -> tuple[np.ndarray | None, float]:
    """
    根据绝对时间戳从 NAS 视频提取一帧。

    Returns: (bgr_image, video_offset_sec) 或 (None, 0) 失败。
    """
    # Find video covering this timestamp
    v = None
    for candidate in video_index:
        if candidate["start"] <= target_wall_time < candidate["end"]:
            v = candidate
            break
    if v is None:
        return None, 0.0

    offset_sec = max(0, (target_wall_time - v["start"]).total_seconds())

    # Extract single frame via ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset_sec:.3f}",
        "-i", str(v["path"]),
        "-frames:v", "1",
        "-q:v", "2",
        str(tmp_frame_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None, offset_sec

    img = cv2.imread(str(tmp_frame_path))
    if img is None:
        return None, offset_sec

    return img, offset_sec


def crop_left_deck_roi(img_bgr: np.ndarray, roi_cfg: dict | None = None) -> np.ndarray:
    """Crop left_deck ROI from full-size frame."""
    if roi_cfg is None:
        roi_cfg = {"x_min": 0.0, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0}
    h, w = img_bgr.shape[:2]
    x1, x2 = int(roi_cfg["x_min"] * w), int(roi_cfg["x_max"] * w)
    y1, y2 = int(roi_cfg["y_min"] * h), int(roi_cfg["y_max"] * h)
    return img_bgr[y1:y2, x1:x2]


def find_suspended_net_frame(
    cycle_row: pd.Series,
    video_index: list[dict],
    tmp_dir: Path,
    sample_interval_sec: int = 3,
    dark_threshold: int = 35,
    roi_cfg: dict | None = None,
) -> dict:
    """
    为单个 cycle 寻找悬空网的"黄金帧"。

    Returns:
        dict with frame info and peak dark ratio.
    """
    cycle_id = int(cycle_row["cycle_id"])
    net_start = pd.Timestamp(cycle_row["net_start_time"])
    net_on_deck = pd.Timestamp(cycle_row["net_on_deck_time"])
    duration_sec = (net_on_deck - net_start).total_seconds()

    print(f"  Cycle {cycle_id}: {net_start.strftime('%H:%M:%S')} → "
          f"{net_on_deck.strftime('%H:%M:%S')} ({duration_sec:.0f}s)")

    samples = []
    offsets = np.arange(0, duration_sec, sample_interval_sec).astype(float)
    # Always sample the final moment too
    if len(offsets) == 0 or abs(offsets[-1] - duration_sec) > sample_interval_sec / 2:
        offsets = np.append(offsets, duration_sec)

    cycle_tmp_dir = tmp_dir / f"cycle_{cycle_id:02d}"
    cycle_tmp_dir.mkdir(parents=True, exist_ok=True)

    for offset in offsets:
        t = net_start + timedelta(seconds=float(offset))
        frame_path = cycle_tmp_dir / f"t{int(offset):04d}.png"

        img, real_offset = get_frame_at_wall_time(t, video_index, cycle_tmp_dir, frame_path)
        if img is None:
            continue

        roi = crop_left_deck_roi(img, roi_cfg)
        dark_ratio = dark_pixel_ratio(roi, dark_threshold)

        samples.append({
            "cycle_id": cycle_id,
            "offset_sec": round(float(offset), 1),
            "wall_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "video_offset_sec": round(real_offset, 1),
            "dark_ratio": round(float(dark_ratio), 5),
            "roi_h": roi.shape[0],
            "roi_w": roi.shape[1],
            "frame_path": str(frame_path),
        })

    if not samples:
        return {"cycle_id": cycle_id, "status": "failed", "reason": "no_valid_frames"}

    # 找暗像素比例峰值
    samples_df = pd.DataFrame(samples)
    peak_idx = int(samples_df["dark_ratio"].argmax())
    peak = samples_df.iloc[peak_idx]

    return {
        "cycle_id": cycle_id,
        "status": "success",
        "peak_offset_sec": peak["offset_sec"],
        "peak_wall_time": peak["wall_time"],
        "peak_dark_ratio": peak["dark_ratio"],
        "peak_frame_path": peak["frame_path"],
        "all_samples": samples,
        "num_samples": len(samples),
    }


def run_suspended_net_finder(
    core_catch_windows_csv: Path,
    video_root: Path,
    out_dir: Path,
    cfg: dict | None = None,
) -> Path:
    """对所有 cycles 执行悬空网兜识别。"""
    if cfg is None:
        cfg = {}
    net_size_cfg = cfg.get("fine_grained", {}).get("net_size_estimation", {})
    coarse_cfg = cfg.get("coarse_features", {})

    sample_interval = net_size_cfg.get("sample_interval_sec", 3)
    dark_thresh = net_size_cfg.get("dark_threshold", 35)
    roi_cfg = coarse_cfg.get("rois", {}).get("left_deck", {
        "x_min": 0.0, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0,
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sample interval: {sample_interval}s, dark threshold: {dark_thresh}")

    # Load data
    core_df = pd.read_csv(core_catch_windows_csv)
    video_index = scan_source_videos(video_root)
    print(f"Loaded {len(video_index)} source videos, {len(core_df)} cycles")

    results = []
    all_samples = []
    tmp_dir = out_dir / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for _, cycle_row in core_df.iterrows():
        result = find_suspended_net_frame(
            cycle_row, video_index, tmp_dir,
            sample_interval_sec=sample_interval,
            dark_threshold=dark_thresh,
            roi_cfg=roi_cfg,
        )
        results.append({k: v for k, v in result.items() if k != "all_samples"})
        all_samples.extend(result.get("all_samples", []))

        # Save per-cycle samples
        cycle_samples = pd.DataFrame(result.get("all_samples", []))
        if len(cycle_samples) > 0:
            cycle_samples.to_csv(out_dir / f"cycle_{result['cycle_id']:02d}_samples.csv",
                                  index=False, encoding="utf-8-sig")

    # Save results summary
    results_df = pd.DataFrame(results)
    summary_csv = out_dir / "suspended_net_summary.csv"
    results_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # All samples combined
    all_samples_df = pd.DataFrame(all_samples)
    all_samples_df.to_csv(out_dir / "all_cycles_samples.csv", index=False, encoding="utf-8-sig")

    success = sum(1 for r in results if r.get("status") == "success")
    print(f"\nDone. {success}/{len(results)} succeeded, {len(results)-success} failed")
    if success > 0:
        print(f"  Peak dark ratio range: {results_df['peak_dark_ratio'].min():.3f} - {results_df['peak_dark_ratio'].max():.3f}")

    return summary_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Find suspended net peak frame per cycle")
    p.add_argument("--core-windows", required=True, help="core_catch_windows.csv path")
    p.add_argument("--video-root", default=r"\\DS224plus\video\0515", help="NAS video root")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--config", default=None, help="pipeline YAML config path")
    p.add_argument("--sample-interval", type=int, default=3, help="sample every N seconds")
    p.add_argument("--dark-threshold", type=int, default=35, help="RGB < this = dark pixel")
    args = p.parse_args()

    from ...shared.utils import load_pipeline_config
    cfg = load_pipeline_config(args.config) if args.config else {}

    run_suspended_net_finder(
        core_catch_windows_csv=Path(args.core_windows),
        video_root=Path(args.video_root),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )