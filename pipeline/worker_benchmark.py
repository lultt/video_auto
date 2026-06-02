"""
worker_benchmark.py — 实测不同 worker 数的真实处理速度。

 32 个视频（~21 小时），workers=1,4,6,8,12,16,32
"""
import os, sys, time, re
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from pipeline.coarse_features.feature_merge import (
    process_video_coarse, DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
    DEFAULT_RESIZE, DEFAULT_ROIS_CFG, DEFAULT_FEATURE_FLAGS,
)

VIDEO_ROOT = Path(r"\\DS224plus\video\viedeo")
# Pick 32 videos — ~21 hours total
ALL_VIDEOS = sorted([str(p) for p in VIDEO_ROOT.glob("*.mp4")])
TEST_VIDEOS = ALL_VIDEOS[:32]

OUT_DIR = Path(r"J:\video_auto\outputs\benchmark")
WORKER_COUNTS = [1, 4, 6, 8, 12, 16, 32]


def video_duration_sec(filepath):
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_(\d{6})", Path(filepath).name)
    if not m:
        return 0
    d, s, e = m.group(1), m.group(2), m.group(3)
    s = datetime.strptime(d+s, "%Y%m%d%H%M%S")
    e = datetime.strptime(d+e, "%Y%m%d%H%M%S")
    if e < s:
        e += timedelta(days=1)
    return (e - s).total_seconds()


def benchmark_workers(worker_count, test_videos):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous parquet to force re-process
    for f in OUT_DIR.glob("*.parquet"):
        f.unlink()

    t_start = time.perf_counter()
    cpu_samples = [psutil.cpu_percent(interval=None, percpu=False)]

    if worker_count == 1:
        for vp in test_videos:
            process_video_coarse(vp, str(OUT_DIR), DEFAULT_RESIZE, DEFAULT_ROIS_CFG,
                                  DEFAULT_FEATURE_FLAGS, DEFAULT_KF_SUBSAMPLE,
                                  DEFAULT_KF_INTERVAL_SEC)
            cpu_samples.append(psutil.cpu_percent(interval=None, percpu=False))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as exec:
            futures = {exec.submit(
                process_video_coarse, vp, str(OUT_DIR), DEFAULT_RESIZE, DEFAULT_ROIS_CFG,
                DEFAULT_FEATURE_FLAGS, DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
            ): vp for vp in test_videos}
            for fut in as_completed(futures):
                fut.result()
                cpu_samples.append(psutil.cpu_percent(interval=None, percpu=False))

    elapsed = time.perf_counter() - t_start
    video_sec = sum(video_duration_sec(v) for v in test_videos)
    n_frames = sum(1 for f in OUT_DIR.glob("*.parquet"))

    return {
        "worker_count": worker_count,
        "video_count": len(test_videos),
        "video_hours": round(video_sec / 3600, 2),
        "processing_seconds": round(elapsed, 1),
        "processing_minutes": round(elapsed / 60, 2),
        "realtime_ratio": round(video_sec / elapsed, 1),
        "cpu_mean_pct": round(sum(cpu_samples) / len(cpu_samples), 1),
        "parquet_files": n_frames,
    }


def main():
    test_videos = [v for v in TEST_VIDEOS if Path(v).exists()]
    if len(test_videos) < 32:
        print(f"WARNING: only {len(test_videos)} videos found")
    total_h = sum(video_duration_sec(v) for v in test_videos) / 3600
    print(f"Benchmark: {len(test_videos)} videos, {total_h:.1f} hours total")
    print(f"Worker counts: {WORKER_COUNTS}")
    print()

    # Warmup — 1 worker × 1 video
    print("=== Warmup (1 video × 1 worker) ===\n")
    benchmark_workers(1, test_videos[:1])

    all_stats = []
    for w in WORKER_COUNTS:
        print(f"=== workers={w} === ", end="", flush=True)
        stats = benchmark_workers(w, test_videos)
        stats["worker_count"] = w
        all_stats.append(stats)
        print(f"  {stats['processing_minutes']:.1f} min  {stats['realtime_ratio']:.0f}×  CPU:{stats['cpu_mean_pct']:.0f}%")

    # Report
    best = max(all_stats, key=lambda s: s["realtime_ratio"])

    report = "# Worker Benchmark — 32 videos × multiple worker counts\n\n"
    report += "Generated: " + time.strftime("%Y-%m-%d %H:%M") + "\n\n"
    report += f"Test: {len(test_videos)} videos, {best['video_hours']:.1f} hours\n\n"

    report += "| Workers | Time (min) | Realtime | CPU% | Files |\n"
    report += "|---------|-----------|----------|------|-------|\n"
    for s in all_stats:
        report += f"| {s['worker_count']:7d} | {s['processing_minutes']:9.1f} | {s['realtime_ratio']:6.0f}× | {s['cpu_mean_pct']:4.0f}% | {s['parquet_files']:5d} |\n"

    report += f"\n## Best: {best['worker_count']} workers — {best['realtime_ratio']:.0f}× realtime\n"
    report += f"  {best['video_hours']:.1f} hrs video → {best['processing_minutes']:.1f} min\n"

    # Projection
    nightly_h = 23.5
    report += "\n## Projection (23.5 hrs/night × 61 nights)\n\n"
    for s in all_stats:
        pn = nightly_h / s["realtime_ratio"] * 60
        pt = pn * 61 / 60
        report += f"- {s['worker_count']:2d} workers: {pn:.0f} min/night → {pt:.0f} hrs total\n"

    report += f"\n**Recommended: {best['worker_count']} workers**"

    out_path = Path(r"J:\video_auto\outputs\best_workers_recommendation.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
