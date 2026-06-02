"""
Phase 1: Ultra-fast coarse sampling pipeline

Keyframe-only decode (-skip_frame nokey) + subsample every Nth keyframe
Keyframe 间隔 ~4s，取每 2-3 个 → 有效 ~8-12s/帧
不做 stabilization（帧间隔太大无意义）

目标：快速提取 coarse time-series → 识别起网/分鱼/放网阶段
"""

import os
import sys
import subprocess
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FFMPEG_PATH


KF_SUBSAMPLE = 3  # 每 3 个 keyframe 取 1 个 (~12s/帧)
KF_INTERVAL_SEC = 4.0  # 视频 keyframe 间隔 (用于估算 timestamp)


def compute_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def build_keyframe_cmd(video_path, out_w, out_h):
    """ffmpeg keyframe-only decode — 跳过所有 P/B 帧。"""
    return [
        FFMPEG_PATH,
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vf", f"scale={out_w}:{out_h}",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "quiet",
        "-"
    ]


def process_video_coarse(video_path, cfg, output_dir):
    """单视频 coarse pipeline: keyframe-only decode → features → parquet。"""
    video_id = os.path.basename(video_path)
    out_w, out_h = cfg["resize_width"], cfg["resize_height"]
    rois_cfg = cfg["rois"]

    roi_slices = {}
    for name, r in rois_cfg.items():
        y1, y2 = int(r["y_min"] * out_h), int(r["y_max"] * out_h)
        x1, x2 = int(r["x_min"] * out_w), int(r["x_max"] * out_w)
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))

    cmd = build_keyframe_cmd(video_path, out_w, out_h)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_size = out_w * out_h * 3
    kf_idx = 0
    sample_idx = 0
    records = []
    prev_grays = {name: None for name in roi_slices}

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        if kf_idx % KF_SUBSAMPLE != 0:
            kf_idx += 1
            continue

        frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        timestamp_sec = kf_idx * KF_INTERVAL_SEC

        for roi_name, roi_slice in roi_slices.items():
            roi_bgr = frame_bgr[roi_slice]
            roi_gray = gray[roi_slice]

            row = {
                "video_id": video_id,
                "frame_idx": sample_idx,
                "timestamp_sec": timestamp_sec,
                "roi_name": roi_name,
            }

            row["mean_brightness"] = float(roi_gray.mean())
            row["brightness_std"] = float(roi_gray.std())
            row["mean_b"] = float(roi_bgr[:, :, 0].mean())
            row["mean_g"] = float(roi_bgr[:, :, 1].mean())
            row["mean_r"] = float(roi_bgr[:, :, 2].mean())

            if prev_grays[roi_name] is not None:
                diff = cv2.absdiff(roi_gray, prev_grays[roi_name])
                row["motion_intensity"] = float(diff.mean())
            else:
                row["motion_intensity"] = 0.0

            edges = cv2.Canny(roi_gray, 50, 150)
            row["edge_density"] = float(edges.sum() / 255.0 / edges.size)

            lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
            row["laplacian_variance"] = float(lap.var())

            row["entropy"] = compute_entropy(roi_gray)

            records.append(row)
            prev_grays[roi_name] = roi_gray.copy()

        kf_idx += 1
        sample_idx += 1

    proc.wait()

    if not records:
        return {"video_id": video_id, "status": "empty", "frames": 0, "rows": 0}

    df = pd.DataFrame(records)
    float_cols = [c for c in df.columns if c not in ("video_id", "roi_name", "frame_idx")]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, video_id.replace(".mp4", ".parquet"))
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    return {
        "video_id": video_id,
        "status": "ok",
        "frames": sample_idx,
        "rows": len(df),
        "parquet_path": parquet_path,
    }


def plot_coarse_features(df, out_dir, video_id):
    """生成 3 张独立 feature curve PNG + 1 张合并图。"""
    roi_names = df["roi_name"].unique()
    colors = {"left_deck": "#1f77b4", "center_deck": "#ff7f0e", "right_deck": "#2ca02c"}
    prefix = video_id.replace(".mp4", "")
    paths = []

    # Motion curve
    fig, ax = plt.subplots(figsize=(16, 4))
    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        ax.plot(sub["timestamp_sec"] / 60, sub["motion_intensity"],
                label=roi, color=colors.get(roi), linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Motion Intensity")
    ax.set_title(f"Motion — {video_id} (keyframe coarse)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_motion.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # Entropy curve
    fig, ax = plt.subplots(figsize=(16, 4))
    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        ax.plot(sub["timestamp_sec"] / 60, sub["entropy"],
                label=roi, color=colors.get(roi), linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title(f"Entropy — {video_id} (keyframe coarse)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_entropy.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # Scene transition (brightness std as proxy for scene complexity)
    fig, ax = plt.subplots(figsize=(16, 4))
    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        ax.plot(sub["timestamp_sec"] / 60, sub["brightness_std"],
                label=roi, color=colors.get(roi), linewidth=1.0)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Brightness Std")
    ax.set_title(f"Scene Complexity (brightness std) — {video_id} (keyframe coarse)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_scene.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    return paths


def run_coarse_pipeline(cfg, video_path=None, n_videos=1):
    out_dir = os.path.join(cfg["output_dir"], "coarse")
    os.makedirs(out_dir, exist_ok=True)

    if video_path:
        video_paths = [video_path]
    else:
        video_paths = sorted(Path(cfg["video_root"]).glob("*.mp4"))[:n_videos]
        video_paths = [str(v) for v in video_paths]

    print("=" * 60)
    print("  PHASE 1: Keyframe-only Coarse Sampling")
    print("=" * 60)
    print(f"  视频数: {len(video_paths)}")
    print(f"  方法: -skip_frame nokey + subsample 1/{KF_SUBSAMPLE}")
    print(f"  有效采样率: ~1帧/{KF_SUBSAMPLE * KF_INTERVAL_SEC:.0f}s")
    print(f"  Stabilization: OFF")
    print(f"  输出: {out_dir}")
    print()

    total_t0 = time.perf_counter()

    for vp in video_paths:
        vid_name = os.path.basename(vp)
        print(f"--- {vid_name} ---")

        t0 = time.perf_counter()
        result = process_video_coarse(vp, cfg, out_dir)
        elapsed = time.perf_counter() - t0

        if result["status"] != "ok":
            print(f"  FAILED: {result['status']}")
            continue

        df = pd.read_parquet(result["parquet_path"])
        duration_min = df["timestamp_sec"].max() / 60

        print(f"  keyframes sampled: {result['frames']}")
        print(f"  parquet rows: {result['rows']}")
        print(f"  视频覆盖: {duration_min:.1f} min")
        print(f"  处理耗时: {elapsed:.1f}s")
        print(f"  速度: {duration_min*60/elapsed:.1f}x realtime")
        print()

        png_paths = plot_coarse_features(df, out_dir, vid_name)
        for p in png_paths:
            print(f"  saved: {p}")
        print()

    total_elapsed = time.perf_counter() - total_t0
    print("=" * 60)
    print(f"  总耗时: {total_elapsed:.1f}s")
    if len(video_paths) > 0:
        avg = total_elapsed / len(video_paths)
        print(f"  单视频平均: {avg:.1f}s")
        print(f"  全量 2083 视频预估 (4 workers): {2083*avg/4/3600:.1f} 小时")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Coarse sampling pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video", default=None)
    parser.add_argument("--n-videos", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_coarse_pipeline(cfg, video_path=args.video, n_videos=args.n_videos)
