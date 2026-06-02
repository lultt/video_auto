"""
全链路 Pipeline Benchmark

测量真实端到端吞吐：
  ffmpeg decode → resize → adaptive sampling → stabilization → ROI特征 → parquet写入

输出：
  - 单视频 wall-clock 时间
  - effective pipeline fps (采样帧/秒)
  - 2083视频全量预估（串行 + 并行）
  - GO/NO-GO 判定
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FastVideoReader
from src.extract_features import process_single_video


def find_test_videos(cfg, n=3):
    videos = sorted(Path(cfg["video_root"]).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"视频目录为空: {cfg['video_root']}")
    return [str(v) for v in videos[:n]]


def bench_single_video(video_path, cfg, tmp_dir):
    """单视频全链路：decode → features → parquet。返回 wall-clock 和帧数。"""
    t0 = time.perf_counter()
    result = process_single_video(video_path, cfg, tmp_dir)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def bench_parallel(video_paths, cfg, tmp_dir, num_workers):
    """多进程并行，测量真实并行吞吐。"""
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_video, vp, cfg, tmp_dir): vp
            for vp in video_paths
        }
        for fut in as_completed(futures):
            results.append(fut.result())
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_benchmark(cfg):
    print("=" * 70)
    print("  FULL PIPELINE BENCHMARK (端到端全链路)")
    print("=" * 70)

    num_workers = cfg.get("num_workers", 4)
    resize = (cfg["resize_width"], cfg["resize_height"])
    normal_fps = cfg["adaptive_sampling"]["normal_fps"]

    test_videos = find_test_videos(cfg, n=max(num_workers + 1, 4))
    print(f"  视频源: {cfg['video_root']}")
    print(f"  测试视频数: {len(test_videos)}")
    print(f"  Resize: {resize[0]}x{resize[1]}")
    print(f"  采样: {normal_fps} fps (adaptive burst: {cfg['adaptive_sampling']['burst_fps']} fps)")
    print(f"  Workers: {num_workers}")
    print()

    tmp_dir = tempfile.mkdtemp(prefix="bench_pipeline_")

    try:
        # --- Phase 1: 单视频全链路 ---
        print("-" * 70)
        print("[1] 单视频全链路 (decode → features → parquet)")
        print("-" * 70)

        single_results = []
        for i, vp in enumerate(test_videos[:3]):
            vid_name = os.path.basename(vp)
            result, elapsed = bench_single_video(vp, cfg, tmp_dir)
            frames = result["frames"]
            fps = frames / elapsed if elapsed > 0 else 0

            reader = FastVideoReader(vp, resize=resize)
            duration_sec = reader.duration_sec
            total_frames = reader.frame_count
            reader.release()

            realtime_ratio = duration_sec / elapsed if elapsed > 0 else 0

            print(f"  [{i+1}] {vid_name}")
            print(f"      视频时长: {duration_sec:.0f}s ({duration_sec/60:.1f}min)")
            print(f"      原始帧数: {total_frames}")
            print(f"      采样输出: {frames} 行 (3 ROI × 采样帧)")
            print(f"      耗时: {elapsed:.1f}s")
            print(f"      Pipeline FPS: {fps:.1f} 行/秒")
            print(f"      处理倍速: {realtime_ratio:.1f}x 实时")
            print()

            single_results.append({
                "video": vid_name,
                "duration_sec": duration_sec,
                "frames_out": frames,
                "elapsed": elapsed,
                "fps": fps,
                "realtime_ratio": realtime_ratio,
            })

        avg_elapsed = sum(r["elapsed"] for r in single_results) / len(single_results)
        avg_fps = sum(r["fps"] for r in single_results) / len(single_results)
        avg_realtime = sum(r["realtime_ratio"] for r in single_results) / len(single_results)

        print(f"  单视频平均: {avg_elapsed:.1f}s, {avg_fps:.1f} 行/秒, {avg_realtime:.1f}x 实时")
        print()

        # --- Phase 2: 并行吞吐 ---
        print("-" * 70)
        print(f"[2] 并行吞吐测试 ({num_workers} workers × {len(test_videos)} 视频)")
        print("-" * 70)

        # 清理 tmp 让 process_single_video 不跳过
        shutil.rmtree(tmp_dir)
        tmp_dir = tempfile.mkdtemp(prefix="bench_pipeline_")

        par_results, par_elapsed = bench_parallel(test_videos, cfg, tmp_dir, num_workers)
        par_ok = [r for r in par_results if r["status"] == "ok"]
        par_total_frames = sum(r["frames"] for r in par_ok)
        par_throughput = par_total_frames / par_elapsed if par_elapsed > 0 else 0
        videos_per_min = len(par_ok) / (par_elapsed / 60) if par_elapsed > 0 else 0

        print(f"  视频完成: {len(par_ok)}/{len(test_videos)}")
        print(f"  总输出行: {par_total_frames}")
        print(f"  Wall-clock: {par_elapsed:.1f}s")
        print(f"  并行吞吐: {par_throughput:.0f} 行/秒")
        print(f"  处理速度: {videos_per_min:.2f} 视频/分钟")
        print()

        # --- Phase 3: 全量预估 ---
        print("-" * 70)
        print("[3] 全量 2083 视频预估")
        print("-" * 70)

        total_videos = 2083
        all_videos_found = len(list(Path(cfg["video_root"]).glob("*.mp4")))

        # 基于并行实测
        if videos_per_min > 0:
            total_minutes = total_videos / videos_per_min
            total_hours = total_minutes / 60
        else:
            total_hours = float("inf")

        # 基于单视频平均
        serial_hours = (avg_elapsed * total_videos) / 3600
        parallel_hours_est = serial_hours / num_workers

        print(f"  NAS上实际视频数: {all_videos_found}")
        print(f"  目标处理数: {total_videos}")
        print()
        print(f"  方法A (基于并行实测):")
        print(f"    {num_workers} workers 并行: {total_hours:.1f} 小时")
        print()
        print(f"  方法B (基于单视频平均 × 并行系数):")
        print(f"    单视频平均: {avg_elapsed:.1f}s")
        print(f"    串行总时间: {serial_hours:.1f} 小时")
        print(f"    {num_workers} workers 并行: {parallel_hours_est:.1f} 小时")
        print()

        # --- GO/NO-GO ---
        print("=" * 70)
        print("[4] GO / NO-GO 判定")
        print("=" * 70)

        target_hours = 24.0
        feasible = total_hours <= target_hours

        print(f"  预估总时间: {total_hours:.1f} 小时")
        print(f"  目标上限: {target_hours:.0f} 小时")
        print()

        if feasible:
            print(f"  >>> GO — 系统具备全量处理能力 <<<")
            print(f"  {num_workers} workers 可在 {total_hours:.1f}h 内完成 {total_videos} 视频")
        else:
            needed_workers = int(total_hours / target_hours * num_workers) + 1
            print(f"  >>> NO-GO (当前配置) <<<")
            print(f"  需要 {needed_workers} workers 或优化单视频速度")
            speedup_needed = total_hours / target_hours
            print(f"  需要 {speedup_needed:.1f}x 加速")

        print()
        print(f"  补充信息:")
        print(f"    单视频平均时长: ~43 min")
        print(f"    Pipeline 处理倍速: {avg_realtime:.1f}x 实时")
        print(f"    每视频输出行数: ~{int(sum(r['frames_out'] for r in single_results) / len(single_results))}")
        print("=" * 70)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="全链路 Pipeline Benchmark")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.workers:
        cfg["num_workers"] = args.workers
    run_benchmark(cfg)
