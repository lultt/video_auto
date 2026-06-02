"""
cpp_scaling_benchmark.py — Pure feature-extraction performance benchmark for the C++ pipeline.

Tests scaling across worker counts with --no-output mode (skips all file I/O).
Measures: wall time, realtime factor, fps, CPU%, peak memory.
"""
import os, sys, subprocess, time, json, re, threading
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CPP_BIN = r"J:\video_auto\cpp_pipeline\build\Release\cpp_pipeline.exe"
VIDEO_DIR = r"C:\video\0515"
OUT_ROOT = Path(r"J:\video_auto\outputs\cpp_scaling")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

WORKER_COUNTS = [8, 12, 16, 24, 32, 36, 40, 48]


def video_duration_sec(filepath):
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_(\d{6})", Path(filepath).name)
    if not m:
        return 0
    d, s, e = m.group(1), m.group(2), m.group(3)
    s = datetime.strptime(d + s, "%Y%m%d%H%M%S")
    e = datetime.strptime(d + e, "%Y%m%d%H%M%S")
    if e < s:
        e += timedelta(days=1)
    return (e - s).total_seconds()


def run_one(num_threads, video_dir):
    """Run cpp_pipeline.exe with --no-output and sample CPU/mem."""
    cpu_samples = []
    mem_samples = []
    stop = threading.Event()

    def sampler():
        proc = psutil.Process()
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=None, percpu=False))
            mem_samples.append(psutil.virtual_memory().used / 1024**3)
            time.sleep(0.1)

    # Prime cpu_percent
    psutil.cpu_percent(interval=None)
    t0 = time.perf_counter()
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    proc = subprocess.run(
        [CPP_BIN, video_dir, "--no-output", "--threads", str(num_threads)],
        capture_output=True, text=True, timeout=600,
    )
    wall = time.perf_counter() - t0
    stop.set()
    sampler_thread.join(timeout=2)

    out = proc.stdout

    # Parse C++ banner
    rt = fps = vps = wall_cpp = peak_mem = total_video_hrs = 0.0
    total_frames = 0
    video_count = 0
    for line in out.splitlines():
        ls = line.strip()
        if "Realtime factor:" in ls:
            try: rt = float(ls.split()[-1].replace("x", ""))
            except: pass
        elif "Frames/sec:" in ls:
            try: fps = float(ls.split()[-1])
            except: pass
        elif "Videos/sec:" in ls:
            try: vps = float(ls.split()[-1])
            except: pass
        elif "Wall time:" in ls:
            try: wall_cpp = float(ls.replace("s", "").split()[-1])
            except: pass
        elif "Peak memory:" in ls:
            try: peak_mem = float(ls.replace("MB", "").split()[-1])
            except: pass
        elif "Video duration:" in ls:
            try: total_video_hrs = float(ls.replace("hrs", "").split()[-1])
            except: pass
        elif "Videos:" in ls and ":" in ls:
            # Two lines match: header "Videos: 37" and summary "Videos:          37"
            try:
                v = int(ls.split(":")[-1].strip())
                if video_count == 0:
                    video_count = v
            except: pass

    # Frames = fps * wall (compute from cpp output)
    total_frames = int(round(fps * wall_cpp)) if wall_cpp > 0 else 0

    return {
        "workers": num_threads,
        "wall_time_s": round(wall_cpp if wall_cpp > 0 else wall, 2),
        "realtime_factor": round(rt, 1),
        "fps": round(fps, 1),
        "videos_per_sec": round(vps, 3),
        "cpu_usage_pct": round(np.mean(cpu_samples) if cpu_samples else 0, 1),
        "cpu_max_pct": round(np.max(cpu_samples) if cpu_samples else 0, 1),
        "peak_memory_mb": round(peak_mem, 0),
        "mem_used_avg_gb": round(np.mean(mem_samples) if mem_samples else 0, 2),
        "video_count": video_count,
        "total_video_hours": round(total_video_hrs, 2),
        "total_feature_rows": total_frames,
    }


def main():
    # System info
    phys = psutil.cpu_count(logical=False)
    log = psutil.cpu_count(logical=True)
    mem_total = round(psutil.virtual_memory().total / 1024**3)
    print(f"System: {phys}P/{log}L cores, {mem_total} GB RAM")
    print(f"Binary: {CPP_BIN}")
    print(f"Video dir: {VIDEO_DIR}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print()

    # Warmup so OS caches stabilize and ffmpeg etc. is hot
    print("=== Warmup (1 thread on 1 video skipped — running full warmup at 16 threads) ===")
    _ = run_one(16, VIDEO_DIR)
    print(f"  Warmup wall: {_['wall_time_s']}s\n")

    results = []
    for w in WORKER_COUNTS:
        print(f"=== workers={w} ===", flush=True)
        r = run_one(w, VIDEO_DIR)
        results.append(r)
        print(f"  wall={r['wall_time_s']}s  realtime={r['realtime_factor']}x  "
              f"fps={r['fps']}  cpu={r['cpu_usage_pct']}%  peak_mem={r['peak_memory_mb']}MB  "
              f"rows={r['total_feature_rows']:,}")
        print()

    df = pd.DataFrame(results)
    df.to_csv(OUT_ROOT / "scaling_benchmark.csv", index=False)
    with open(OUT_ROOT / "scaling_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("  SCALING BENCHMARK RESULTS (pure feature extraction, no output)")
    print("="*70)
    print(df.to_string(index=False))

    # ---- Charts ----
    chart_dir = OUT_ROOT / "charts"
    chart_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    workers_arr = np.array([r["workers"] for r in results])
    wall_arr = np.array([r["wall_time_s"] for r in results])
    rt_arr = np.array([r["realtime_factor"] for r in results])
    fps_arr = np.array([r["fps"] for r in results])
    cpu_arr = np.array([r["cpu_usage_pct"] for r in results])
    mem_arr = np.array([r["peak_memory_mb"] for r in results])

    best_idx = int(np.argmax(rt_arr))
    best_w = int(workers_arr[best_idx])
    best_rt = float(rt_arr[best_idx])
    best_fps = float(fps_arr[best_idx])

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(workers_arr)))

    def save(fig, name):
        path = chart_dir / name
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved {name}")

    # 1. Wall time
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(workers_arr.astype(str), wall_arr, color=colors, edgecolor="white")
    for b, v in zip(bars, wall_arr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}s",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Workers (threads)"); ax.set_ylabel("Wall time (seconds)")
    ax.set_title("C++ Pipeline - Wall Time vs Workers (--no-output)")
    save(fig, "workers_vs_wall_time.png")

    # 2. Realtime
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(workers_arr.astype(str), rt_arr, color=colors, edgecolor="white")
    for b, v in zip(bars, rt_arr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v:.0f}x",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Workers"); ax.set_ylabel("Realtime factor (x)")
    ax.set_title("C++ Pipeline - Realtime Factor vs Workers")
    ax.axhline(best_rt, color="red", alpha=0.3, ls="--", label=f"Peak {best_rt:.0f}x @ {best_w}")
    ax.legend()
    save(fig, "workers_vs_realtime.png")

    # 3. FPS
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(workers_arr.astype(str), fps_arr, color=colors, edgecolor="white")
    for b, v in zip(bars, fps_arr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Workers"); ax.set_ylabel("Feature rows/sec")
    ax.set_title("C++ Pipeline - Throughput (FPS) vs Workers")
    save(fig, "workers_vs_fps.png")

    # 4. CPU
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(workers_arr, cpu_arr, "o-", color="#d62728", linewidth=2, markersize=10)
    for w, c in zip(workers_arr, cpu_arr):
        ax.annotate(f"{c:.0f}%", (w, c), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    ax.axhline(100, color="gray", ls=":", alpha=0.4, label="100% saturation")
    ax.set_xlabel("Workers"); ax.set_ylabel("CPU usage (%)")
    ax.set_title(f"CPU Usage vs Workers ({phys}P/{log}L cores)")
    ax.set_ylim(0, 105); ax.legend()
    save(fig, "workers_vs_cpu.png")

    # 5. Memory
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(workers_arr, mem_arr, "o-", color="#2ca02c", linewidth=2, markersize=10)
    for w, m in zip(workers_arr, mem_arr):
        ax.annotate(f"{m:.0f}MB", (w, m), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    ax.set_xlabel("Workers"); ax.set_ylabel("Peak memory (MB)")
    ax.set_title("C++ Pipeline - Peak Memory vs Workers")
    save(fig, "workers_vs_memory.png")

    # 6. Scaling efficiency
    ideal_per_w = rt_arr[0] / workers_arr[0]
    eff = (rt_arr / workers_arr) / ideal_per_w * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(workers_arr, eff, "o-", color="#9467bd", linewidth=2, markersize=10, label="Actual")
    ax.axhline(100, color="green", ls="--", alpha=0.4, label="Linear (ideal)")
    for w, e in zip(workers_arr, eff):
        ax.annotate(f"{e:.0f}%", (w, e), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    ax.set_xlabel("Workers"); ax.set_ylabel("Scaling efficiency (%)")
    ax.set_title(f"Scaling Efficiency (relative to {workers_arr[0]} workers)")
    ax.legend()
    save(fig, "scaling_efficiency.png")

    # 7. Combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (ax1, ax2), (ax3, ax4) = axes
    ax1.bar(workers_arr.astype(str), rt_arr, color=colors)
    ax1.set_title("Realtime Factor"); ax1.set_ylabel("x")
    for b, v in zip(ax1.containers[0], rt_arr):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{v:.0f}x", ha="center", fontsize=9)
    ax2.bar(workers_arr.astype(str), fps_arr, color=colors)
    ax2.set_title("Frames/sec"); ax2.set_ylabel("rows/s")
    for b, v in zip(ax2.containers[0], fps_arr):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax3.plot(workers_arr, cpu_arr, "o-", color="#d62728", lw=2, ms=8)
    ax3.set_title("CPU %"); ax3.set_ylabel("%"); ax3.set_xlabel("Workers")
    ax3.set_ylim(0, 105); ax3.axhline(100, color="gray", ls=":", alpha=0.4)
    for w, c in zip(workers_arr, cpu_arr):
        ax3.annotate(f"{c:.0f}%", (w, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax4.plot(workers_arr, eff, "o-", color="#9467bd", lw=2, ms=8)
    ax4.set_title("Scaling Efficiency"); ax4.set_ylabel("%"); ax4.set_xlabel("Workers")
    ax4.axhline(100, color="green", ls="--", alpha=0.4)
    for w, e in zip(workers_arr, eff):
        ax4.annotate(f"{e:.0f}%", (w, e), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle(f"C++ Pipeline Scaling - No Output Mode | {phys}P/{log}L cores | Best: {best_w} workers = {best_rt:.0f}x realtime",
                 fontsize=14, fontweight="bold")
    save(fig, "scaling_dashboard.png")

    # ---- Bottleneck analysis ----
    print(f"\n{'='*70}")
    print("  BOTTLENECK ANALYSIS")
    print(f"{'='*70}")
    print(f"  Peak performance: {best_w} workers -> {best_rt:.0f}x realtime, {best_fps:.0f} fps")
    print(f"  CPU at peak: {cpu_arr[best_idx]:.0f}%")
    print(f"  Memory at peak: {mem_arr[best_idx]:.0f} MB")
    print(f"  Scaling efficiency at peak: {eff[best_idx]:.0f}%")

    print(f"\n  Per-worker realtime (rt / workers):")
    for w, r in zip(workers_arr, rt_arr):
        print(f"    {w:3d} workers: {r/w:6.1f}x per worker")

    if cpu_arr[best_idx] < 70:
        verdict = "I/O or memory bandwidth bound (CPU underutilized)"
    elif cpu_arr[best_idx] < 90:
        verdict = "Mixed CPU + I/O (some headroom)"
    else:
        verdict = "CPU bound (saturated)"
    print(f"\n  -> {verdict}")

    if len(results) >= 3 and rt_arr[-1] < rt_arr[best_idx] * 0.95:
        print(f"  -> Over-subscription detected: {workers_arr[-1]} workers slower than {best_w}")
        print(f"  -> Likely thread contention, ffmpeg pipe overhead, or memory pressure")

    print(f"\nResults: {OUT_ROOT}")
    print(f"Charts:  {chart_dir}")


if __name__ == "__main__":
    main()
