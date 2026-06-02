"""
Pipeline 300s 吞吐测试 + feature visualization

输出：
  - parquet 文件 + row count
  - motion_curve.png
  - entropy_curve.png
  - scene_transition_curve.png
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FastVideoReader
from src.extract_features import extract_roi_features


def run_pipeline_300s(cfg, video_path=None, max_sec=300):
    from pathlib import Path

    if video_path is None:
        videos = sorted(Path(cfg["video_root"]).glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"无视频: {cfg['video_root']}")
        video_path = str(videos[0])

    resize = (cfg["resize_width"], cfg["resize_height"])
    adaptive_cfg = cfg["adaptive_sampling"]
    features_cfg = cfg["features"]
    rois_cfg = cfg["rois"]
    rh, rw = resize[1], resize[0]

    roi_slices = {}
    for name, r in rois_cfg.items():
        y1, y2 = int(r["y_min"] * rh), int(r["y_max"] * rh)
        x1, x2 = int(r["x_min"] * rw), int(r["x_max"] * rw)
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))

    reader = FastVideoReader(video_path, resize=resize)
    vid_name = os.path.basename(video_path)
    print(f"视频: {vid_name}")
    print(f"时长: {reader.duration_sec:.0f}s, 测试片段: {max_sec}s")
    print(f"分辨率: {reader.width}x{reader.height} → {resize[0]}x{resize[1]}")
    print(f"采样: {adaptive_cfg['normal_fps']}fps / burst {adaptive_cfg['burst_fps']}fps")
    print()

    # --- Pipeline ---
    t0 = time.perf_counter()
    records = []
    prev_gray_rois = {name: None for name in roi_slices}
    packet_count = 0

    for packet in reader.read_adaptive(adaptive_cfg, stabilize=True):
        if packet.timestamp_sec > max_sec:
            break
        packet_count += 1

        for roi_name, roi_slice in roi_slices.items():
            row = {
                "frame_idx": packet.frame_idx,
                "timestamp_sec": packet.timestamp_sec,
                "roi_name": roi_name,
                "sample_fps": packet.sample_fps,
                "scene_transition": packet.scene_transition,
                "global_motion_mag": packet.global_motion_mag,
            }
            feats = extract_roi_features(
                packet, roi_slice, prev_gray_rois[roi_name], features_cfg
            )
            row.update(feats)
            records.append(row)
            prev_gray_rois[roi_name] = packet.gray[roi_slice].copy()

        if packet_count % 50 == 0:
            elapsed_now = time.perf_counter() - t0
            print(f"  {packet_count} frames, t={packet.timestamp_sec:.0f}s, "
                  f"elapsed={elapsed_now:.1f}s, "
                  f"speed={packet.timestamp_sec/elapsed_now:.1f}x realtime")

    elapsed = time.perf_counter() - t0
    reader.release()

    print()
    print(f"Pipeline 完成: {packet_count} frames, {len(records)} rows, {elapsed:.1f}s")
    print(f"速度: {max_sec/elapsed:.2f}x realtime")
    print(f"吞吐: {len(records)/elapsed:.0f} rows/sec")

    # --- Parquet ---
    df = pd.DataFrame(records)
    float_cols = [c for c in df.columns if c not in ("roi_name", "frame_idx")]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)

    out_dir = "outputs/debug"
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "debug_300s.parquet")
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"\nParquet: {parquet_path}")
    print(f"  rows: {len(df)}")
    print(f"  cols: {list(df.columns)}")
    print(f"  size: {os.path.getsize(parquet_path)/1024:.1f} KB")

    # --- Visualization ---
    plot_features(df, out_dir)
    return df


def plot_features(df, out_dir):
    """生成 3 张 feature curve PNG。"""
    roi_names = df["roi_name"].unique()
    colors = {"left_deck": "#1f77b4", "center_deck": "#ff7f0e", "right_deck": "#2ca02c"}

    # 1. Motion curve
    fig, ax = plt.subplots(figsize=(14, 4))
    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        ax.plot(sub["timestamp_sec"], sub["motion_intensity"],
                label=roi, color=colors.get(roi, None), alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Motion Intensity")
    ax.set_title("Motion Intensity — 300s segment")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "motion_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")

    # 2. Entropy curve
    fig, ax = plt.subplots(figsize=(14, 4))
    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        ax.plot(sub["timestamp_sec"], sub["entropy"],
                label=roi, color=colors.get(roi, None), alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Entropy — 300s segment")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "entropy_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")

    # 3. Scene transition curve
    fig, ax = plt.subplots(figsize=(14, 4))
    sub = df[df["roi_name"] == roi_names[0]]
    ax.plot(sub["timestamp_sec"], sub["scene_transition"],
            color="#d62728", linewidth=0.8)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="burst threshold")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Scene Correlation")
    ax.set_title("Scene Transition (histogram correlation) — 300s segment")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    path = os.path.join(out_dir, "scene_transition_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--max-sec", type=float, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline_300s(cfg, video_path=args.video, max_sec=args.max_sec)
