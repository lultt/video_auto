import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_timeseries(df, video_id, output_dir):
    """绘制单视频的时序特征曲线。"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"时序特征: {video_id}", fontsize=12)

    roi_names = df["roi_name"].unique()
    colors = {"left_deck": "tab:blue", "center_deck": "tab:orange", "right_deck": "tab:green"}

    for roi in roi_names:
        sub = df[df["roi_name"] == roi]
        t = sub["timestamp_sec"] / 60.0  # 转分钟
        c = colors.get(roi, "tab:gray")

        axes[0].plot(t, sub["mean_brightness"], color=c, alpha=0.7, label=roi, linewidth=0.5)
        axes[1].plot(t, sub["motion_intensity"], color=c, alpha=0.7, label=roi, linewidth=0.5)
        axes[2].plot(t, sub["local_motion_intensity"], color=c, alpha=0.7, label=roi, linewidth=0.5)
        axes[3].plot(t, sub["edge_density"], color=c, alpha=0.7, label=roi, linewidth=0.5)
        axes[4].plot(t, sub["entropy"], color=c, alpha=0.7, label=roi, linewidth=0.5)

    axes[0].set_ylabel("亮度")
    axes[1].set_ylabel("运动强度")
    axes[2].set_ylabel("局部运动(去抖)")
    axes[3].set_ylabel("边缘密度")
    axes[4].set_ylabel("信息熵")
    axes[4].set_xlabel("时间 (分钟)")

    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{video_id}_timeseries.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_scene_transitions(df, video_id, output_dir):
    """绘制场景突变检测图。"""
    sub = df[df["roi_name"] == df["roi_name"].unique()[0]]
    t = sub["timestamp_sec"] / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    fig.suptitle(f"场景突变 + 采样率: {video_id}", fontsize=12)

    ax1.plot(t, sub["scene_transition"], color="tab:red", linewidth=0.8)
    ax1.axhline(y=0.7, color="gray", linestyle="--", alpha=0.5, label="threshold=0.7")
    ax1.set_ylabel("scene_transition")
    ax1.legend()

    ax2.plot(t, sub["sample_fps"], color="tab:purple", linewidth=0.8)
    ax2.set_ylabel("sample_fps")
    ax2.set_xlabel("时间 (分钟)")

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{video_id}_transitions.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_stabilization(df, video_id, output_dir):
    """绘制全局运动 vs 局部运动对比。"""
    sub = df[df["roi_name"] == df["roi_name"].unique()[0]]
    t = sub["timestamp_sec"] / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
    fig.suptitle(f"Camera Motion vs Deck Activity: {video_id}", fontsize=12)

    ax1.plot(t, sub["global_motion_mag"], color="tab:red", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("全局运动幅度 (px)")
    ax1.set_title("Camera/Ship Motion (海浪+摇摆)")

    ax2.plot(t, sub["local_motion_intensity"], color="tab:green", linewidth=0.5, alpha=0.8)
    ax2.plot(t, sub["motion_rolling_mean_30s"], color="black", linewidth=1.5, label="30s均值")
    ax2.set_ylabel("局部运动强度")
    ax2.set_xlabel("时间 (分钟)")
    ax2.set_title("Deck Activity (作业活动)")
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{video_id}_stabilization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def run_visualization(cfg, max_videos=5):
    parquet_dir = os.path.join(cfg["output_dir"], "parquet")
    plot_dir = os.path.join(cfg["output_dir"], "plots")
    os.makedirs(plot_dir, exist_ok=True)

    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        print(f"未找到parquet文件: {parquet_dir}")
        return

    print(f"找到 {len(parquet_files)} 个parquet文件，可视化前 {max_videos} 个")

    for pf in parquet_files[:max_videos]:
        video_id = pf.stem
        print(f"  绘制: {video_id}")
        df = pd.read_parquet(pf)

        plot_timeseries(df, video_id, plot_dir)
        plot_scene_transitions(df, video_id, plot_dir)
        plot_stabilization(df, video_id, plot_dir)

    print(f"\n图表保存: {plot_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="特征可视化")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-videos", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_visualization(cfg, max_videos=args.max_videos)
