"""
Pipeline 链路调试脚本

逐阶段验证：
  Stage 1: ffmpeg pipe 打开 + 原始帧解码 (10帧)
  Stage 2: adaptive sampling + stabilization (30秒片段)
  Stage 3: ROI 特征提取
  Stage 4: parquet 写入 + 读回验证

每阶段独立 try/except，打印耗时和中间状态。
不跑完整视频，只跑 30 秒。
"""

import os
import sys
import time
import tempfile
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FastVideoReader, FFMPEG_PATH
from src.extract_features import extract_roi_features, compute_entropy


def find_first_video(cfg):
    from pathlib import Path
    videos = sorted(Path(cfg["video_root"]).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"无视频: {cfg['video_root']}")
    return str(videos[0])


def stage1_decode(video_path, cfg):
    """Stage 1: ffmpeg pipe 能否打开，能否读取原始帧。"""
    print("=" * 60)
    print("[Stage 1] ffmpeg pipe decode — 读取前 10 帧")
    print("=" * 60)

    resize = (cfg["resize_width"], cfg["resize_height"])
    print(f"  视频: {os.path.basename(video_path)}")
    print(f"  ffmpeg: {FFMPEG_PATH}")
    print(f"  resize: {resize[0]}x{resize[1]}")

    if not os.path.exists(FFMPEG_PATH):
        print(f"  FATAL: ffmpeg 不存在: {FFMPEG_PATH}")
        return False

    reader = FastVideoReader(video_path, resize=resize)
    print(f"  视频信息: {reader.width}x{reader.height} @ {reader.fps:.2f}fps, "
          f"{reader.frame_count} frames, {reader.duration_sec:.0f}s")

    t0 = time.perf_counter()
    frames_read = 0
    for idx, ts, frame in reader.read_fixed_fps(sample_fps=reader.fps, max_frames=10):
        frames_read += 1
        if frames_read == 1:
            print(f"  第1帧: shape={frame.shape}, dtype={frame.dtype}, "
                  f"mean={frame.mean():.1f}, idx={idx}")
        if frames_read == 10:
            print(f"  第10帧: shape={frame.shape}, idx={idx}, ts={ts:.3f}s")

    elapsed = time.perf_counter() - t0
    print(f"  结果: 读取 {frames_read} 帧, 耗时 {elapsed:.3f}s")

    if frames_read < 10:
        print(f"  WARNING: 只读到 {frames_read} 帧 (期望 10)")
        return frames_read > 0

    print("  PASS")
    return True


def stage2_adaptive(video_path, cfg, max_sec=30):
    """Stage 2: adaptive sampling + stabilization — 前 30 秒。"""
    print()
    print("=" * 60)
    print(f"[Stage 2] adaptive sampling + stabilization — 前 {max_sec}s")
    print("=" * 60)

    resize = (cfg["resize_width"], cfg["resize_height"])
    adaptive_cfg = cfg["adaptive_sampling"]
    print(f"  normal_fps={adaptive_cfg['normal_fps']}, "
          f"burst_fps={adaptive_cfg['burst_fps']}, "
          f"threshold={adaptive_cfg['trigger_threshold']}")

    reader = FastVideoReader(video_path, resize=resize)
    max_frames_est = int(max_sec * adaptive_cfg["burst_fps"]) + 50

    t0 = time.perf_counter()
    packets = []
    burst_count = 0

    for packet in reader.read_adaptive(adaptive_cfg, stabilize=True, max_frames=max_frames_est):
        if packet.timestamp_sec > max_sec:
            break
        packets.append(packet)
        if packet.sample_fps > adaptive_cfg["normal_fps"]:
            burst_count += 1
        if len(packets) % 10 == 0:
            print(f"    ... {len(packets)} packets, t={packet.timestamp_sec:.1f}s", end="\r")

    elapsed = time.perf_counter() - t0
    print(f"  结果: {len(packets)} packets in {elapsed:.2f}s")

    if not packets:
        print("  FATAL: 0 packets produced")
        return None

    last = packets[-1]
    print(f"  时间范围: 0 ~ {last.timestamp_sec:.1f}s")
    print(f"  BURST 帧数: {burst_count}")
    print(f"  stabilization 示例 (最后一帧): "
          f"dx={last.global_motion_x:.3f}, dy={last.global_motion_y:.3f}, "
          f"mag={last.global_motion_mag:.3f}")
    print(f"  scene_transition 范围: "
          f"[{min(p.scene_transition for p in packets):.3f}, "
          f"{max(p.scene_transition for p in packets):.3f}]")

    # 验证帧数据完整性
    sample = packets[len(packets) // 2]
    assert sample.frame_bgr.shape == (resize[1], resize[0], 3), \
        f"frame shape mismatch: {sample.frame_bgr.shape}"
    assert sample.gray.shape == (resize[1], resize[0]), \
        f"gray shape mismatch: {sample.gray.shape}"
    assert sample.stabilized_gray.shape == (resize[1], resize[0]), \
        f"stabilized shape mismatch: {sample.stabilized_gray.shape}"

    print("  PASS")
    return packets


def stage3_features(packets, cfg):
    """Stage 3: ROI 特征提取。"""
    print()
    print("=" * 60)
    print("[Stage 3] ROI 特征提取")
    print("=" * 60)

    features_cfg = cfg["features"]
    rois_cfg = cfg["rois"]
    resize = (cfg["resize_width"], cfg["resize_height"])
    rh, rw = resize[1], resize[0]

    roi_slices = {}
    for name, r in rois_cfg.items():
        y1, y2 = int(r["y_min"] * rh), int(r["y_max"] * rh)
        x1, x2 = int(r["x_min"] * rw), int(r["x_max"] * rw)
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))
        print(f"  ROI '{name}': [{y1}:{y2}, {x1}:{x2}] = {y2-y1}x{x2-x1} px")

    t0 = time.perf_counter()
    records = []
    prev_gray_rois = {name: None for name in roi_slices}

    for packet in packets:
        for roi_name, roi_slice in roi_slices.items():
            row = extract_roi_features(
                packet, roi_slice, prev_gray_rois[roi_name], features_cfg
            )
            row["video_id"] = "debug_test"
            row["frame_idx"] = packet.frame_idx
            row["timestamp_sec"] = packet.timestamp_sec
            row["roi_name"] = roi_name
            records.append(row)
            prev_gray_rois[roi_name] = packet.gray[roi_slice].copy()

    elapsed = time.perf_counter() - t0
    print(f"  结果: {len(records)} 行, 耗时 {elapsed:.2f}s")
    print(f"  每帧特征数: {len(records[0]) - 4} (排除 meta 字段)")

    sample_row = records[len(records) // 2]
    print(f"  示例行 (中间帧, {sample_row['roi_name']}):")
    for k, v in sample_row.items():
        if k not in ("video_id", "frame_idx", "timestamp_sec", "roi_name"):
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 检查 NaN/Inf
    nan_count = sum(1 for row in records for v in row.values()
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN/Inf values detected!")
    else:
        print("  无 NaN/Inf")

    print("  PASS")
    return records


def stage4_parquet(records, cfg):
    """Stage 4: parquet 写入 + 读回验证。"""
    print()
    print("=" * 60)
    print("[Stage 4] Parquet 写入 + 读回验证")
    print("=" * 60)

    import pandas as pd

    df = pd.DataFrame(records)
    print(f"  DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  dtypes: {dict(df.dtypes.value_counts())}")

    # cast
    float_cols = [c for c in df.columns if c not in ("video_id", "roi_name", "frame_idx")]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)

    tmp_dir = tempfile.mkdtemp(prefix="debug_pipeline_")
    parquet_path = os.path.join(tmp_dir, "debug_test.parquet")

    t0 = time.perf_counter()
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    write_elapsed = time.perf_counter() - t0

    file_size = os.path.getsize(parquet_path)
    print(f"  写入: {parquet_path}")
    print(f"  文件大小: {file_size / 1024:.1f} KB")
    print(f"  写入耗时: {write_elapsed:.3f}s")

    # 读回验证
    t0 = time.perf_counter()
    df_read = pd.read_parquet(parquet_path)
    read_elapsed = time.perf_counter() - t0

    assert df_read.shape == df.shape, f"shape mismatch: {df_read.shape} vs {df.shape}"
    assert list(df_read.columns) == list(df.columns), "columns mismatch"
    print(f"  读回验证: shape={df_read.shape}, 耗时 {read_elapsed:.3f}s")

    # cleanup
    os.remove(parquet_path)
    os.rmdir(tmp_dir)

    print("  PASS")
    return df


def run_debug(cfg, video_path=None, max_sec=30):
    print()
    print("*" * 60)
    print("  PIPELINE DEBUG — 链路稳定性验证")
    print(f"  测试片段: 前 {max_sec} 秒")
    print("*" * 60)
    print()

    if video_path is None:
        video_path = find_first_video(cfg)

    total_t0 = time.perf_counter()

    # Stage 1
    try:
        ok = stage1_decode(video_path, cfg)
        if not ok:
            print("\nSTOPPED: Stage 1 failed")
            return
    except Exception as e:
        print(f"\nFATAL Stage 1: {e}")
        traceback.print_exc()
        return

    # Stage 2
    try:
        packets = stage2_adaptive(video_path, cfg, max_sec=max_sec)
        if packets is None:
            print("\nSTOPPED: Stage 2 failed")
            return
    except Exception as e:
        print(f"\nFATAL Stage 2: {e}")
        traceback.print_exc()
        return

    # Stage 3
    try:
        records = stage3_features(packets, cfg)
    except Exception as e:
        print(f"\nFATAL Stage 3: {e}")
        traceback.print_exc()
        return

    # Stage 4
    try:
        df = stage4_parquet(records, cfg)
    except Exception as e:
        print(f"\nFATAL Stage 4: {e}")
        traceback.print_exc()
        return

    total_elapsed = time.perf_counter() - total_t0
    print()
    print("=" * 60)
    print(f"  ALL STAGES PASS — 全链路验证通过")
    print(f"  总耗时: {total_elapsed:.2f}s")
    print(f"  输出: {len(records)} 行 ({len(packets)} 采样帧 × {len(cfg['rois'])} ROI)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline 链路调试")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video", default=None, help="指定视频路径 (默认取第一个)")
    parser.add_argument("--max-sec", type=float, default=30, help="测试片段长度(秒)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_debug(cfg, video_path=args.video, max_sec=args.max_sec)
