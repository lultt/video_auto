"""
gpu_scaling_benchmark.py — GPU keyframe-only pipeline scaling across worker counts.

Tests NVDEC hevc_cuvid + -discard nokey (genuine sparse keyframe decode) at
8, 12, 16, 24, 32, 48 workers.  --no-output mode measures pure decode + feature
throughput; GPU decoder util, CPU%, and VRAM are sampled concurrently.
"""

import os, sys, subprocess, time, json, re, threading
from pathlib import Path

import psutil
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths & consts
# ---------------------------------------------------------------------------
CPP_BIN = r"J:\video_auto\cpp_pipeline\build\Release\cpp_pipeline.exe"
VIDEO_DIR = r"C:\video\0515"
OUT_ROOT = Path(r"J:\video_auto\outputs\gpu_scaling")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
NVIDIA_SMI = r"C:\Windows\System32\nvidia-smi.exe"

WORKER_COUNTS = [1, 2, 4, 8, 16, 24, 32]

# ---------------------------------------------------------------------------
# GPU sampling
# ---------------------------------------------------------------------------
def sample_gpu():
    """(gpu_util_pct, decoder_util_pct, mem_used_mb) — single poll via nvidia-smi."""
    try:
        r = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return -1.0, -1.0, -1.0

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(num_threads: int):
    cpu_samples, gpu_util_samples, gpu_dec_samples, gpu_mem_samples = [], [], [], []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=None, percpu=False))
            g, d, m = sample_gpu()
            gpu_util_samples.append(g)
            gpu_dec_samples.append(d)
            gpu_mem_samples.append(m)
            time.sleep(0.18)

    # prime sensors
    psutil.cpu_percent(interval=None)
    sample_gpu()

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    cmd = [CPP_BIN, VIDEO_DIR, "--no-output", "--threads", str(num_threads), "--gpu"]
    print(f"  running {num_threads:3d} workers  ({' '.join(cmd)})", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.perf_counter() - t0
    stop.set()
    sampler_thread.join(timeout=2.5)

    out = proc.stdout

    # --- parse C++ banner ---
    rt = fps = vps = wall_cpp = peak_mem = total_video_hrs = 0.0
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
            try:
                v = int(ls.split(":")[-1].strip())
                if video_count == 0:
                    video_count = v
            except: pass

    n_cpu = len(cpu_samples)
    n_gpu = len(gpu_dec_samples)
    cpu_mean  = float(np.mean(cpu_samples))  if n_cpu else 0.0
    cpu_max   = float(np.max(cpu_samples))   if n_cpu else 0.0
    gpu_util_mean = float(np.mean(gpu_util_samples)) if n_gpu else -1.0
    gpu_dec_mean  = float(np.mean(gpu_dec_samples))  if n_gpu else -1.0
    gpu_dec_max   = float(np.max(gpu_dec_samples))   if n_gpu else -1.0
    gpu_mem_max   = float(np.max([m for m in gpu_mem_samples if m > 0.0])) if n_gpu else -1.0
    total_frames  = int(round(fps * wall_cpp)) if wall_cpp > 0 else 0

    result = {
        "workers": num_threads,
        "wall_time_s": round(wall_cpp if wall_cpp > 0 else wall, 2),
        "realtime_factor": round(rt, 1),
        "fps": round(fps, 1),
        "videos_per_sec": round(vps, 3),
        "cpu_usage_pct": round(cpu_mean, 1),
        "cpu_max_pct": round(cpu_max, 1),
        "gpu_util_avg_pct": round(gpu_util_mean, 1),
        "gpu_dec_avg_pct": round(gpu_dec_mean, 1),
        "gpu_dec_max_pct": round(gpu_dec_max, 1),
        "gpu_mem_max_mb": round(gpu_mem_max, 0),
        "video_count": video_count,
        "total_video_hours": round(total_video_hrs, 2),
        "total_feature_rows": total_frames,
    }
    print(f"    wall={result['wall_time_s']:.1f}s  rt={result['realtime_factor']:.0f}x  "
          f"fps={result['fps']:.0f}  cpu={result['cpu_usage_pct']:.0f}%  "
          f"gpu_dec={result['gpu_dec_avg_pct']:.0f}%  vram={result['gpu_mem_max_mb']:.0f} MB")
    return result

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_results(results: list[dict]):
    chart_dir = OUT_ROOT / "charts"
    chart_dir.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "axes.facecolor": "#fdfdfd",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#cccccc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
    })

    workers_arr = np.array([r["workers"] for r in results], dtype=int)
    wall_arr    = np.array([r["wall_time_s"] for r in results])
    rt_arr      = np.array([r["realtime_factor"] for r in results])
    fps_arr     = np.array([r["fps"] for r in results])
    cpu_arr     = np.array([r["cpu_usage_pct"] for r in results])
    dec_arr     = np.array([r["gpu_dec_avg_pct"] for r in results])
    dec_max_arr = np.array([r["gpu_dec_max_pct"] for r in results])
    vram_arr    = np.array([r["gpu_mem_max_mb"] for r in results])

    colors = plt.cm.inferno(np.linspace(0.10, 0.85, len(workers_arr)))

    def save(fig, name):
        p = chart_dir / name
        fig.savefig(p)
        plt.close(fig)
        print(f"  wrote {p}")

    # --- scaling efficiency (relative to 8 workers) ---
    base_w     = int(workers_arr[0])
    base_rt_per_w = rt_arr[0] / base_w
    eff = (rt_arr / workers_arr) / base_rt_per_w * 100

    # --- 1. wall time ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(workers_arr.astype(str), wall_arr, color=colors, edgecolor="white")
    for b, v in zip(bars, wall_arr):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.1f}s", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Workers (threads)"); ax.set_ylabel("Wall time (seconds)")
    ax.set_title("GPU Keyframe Pipeline — Wall Time vs Workers")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    save(fig, "workers_vs_wall_time.png")

    # --- 2. realtime factor ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(workers_arr, rt_arr, "D-", color="#2E86AB", ms=10, lw=2.0, label="realtime factor")
    ax.set_xlabel("Workers"); ax.set_ylabel("Realtime Factor (×)")
    ax.set_title("GPU Keyframe Pipeline — Realtime Factor vs Workers")
    ax.legend()
    for w, v in zip(workers_arr, rt_arr):
        ax.annotate(f"{v:.0f}×", (w, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    save(fig, "workers_vs_realtime.png")

    # --- 3. feature throughput (fps) ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(workers_arr, fps_arr, "s-", color="#E63946", ms=10, lw=2.0, label="rows/s")
    ax.set_xlabel("Workers"); ax.set_ylabel("Feature rows / sec")
    ax.set_title("GPU Keyframe Pipeline — Throughput vs Workers")
    ax.legend()
    for w, v in zip(workers_arr, fps_arr):
        ax.annotate(f"{v:.0f}", (w, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10)
    save(fig, "workers_vs_fps.png")

    # --- 4. GPU decoder utilisation ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.fill_between(workers_arr, dec_arr, alpha=0.25, color="#2ca02c")
    ax.plot(workers_arr, dec_arr, "o-", color="#2ca02c", ms=10, lw=2.0, label="GPU decoder (avg)")
    ax.plot(workers_arr, dec_max_arr, "v--", color="#1f77b4", ms=8, lw=1.5, label="GPU decoder (max)")
    ax.axhline(95, color="#d62728", ls=":", alpha=0.5, label="saturation threshold")
    ax.set_xlabel("Workers"); ax.set_ylabel("Utilization (%)")
    ax.set_title("GPU Keyframe Pipeline — NVDEC Decoder Utilization")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    for w, v in zip(workers_arr, dec_arr):
        ax.annotate(f"{v:.0f}%", (w, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9)
    save(fig, "workers_vs_gpu_decoder.png")

    # --- 5. GPU memory ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(workers_arr, vram_arr / 1024, "o-", color="#9467bd", ms=10, lw=2.0)
    ax.set_xlabel("Workers"); ax.set_ylabel("VRAM (GB)")
    ax.set_title("GPU Keyframe Pipeline — Peak VRAM vs Workers")
    for w, v in zip(workers_arr, vram_arr):
        ax.annotate(f"{v/1024:.1f} GB", (w, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    save(fig, "workers_vs_vram.png")

    # --- 6. CPU usage ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(workers_arr, cpu_arr, "o-", color="#d62728", ms=10, lw=2.0, label="CPU %")
    ax.axhline(25, color="gray", ls=":", alpha=0.4, label="25% baseline")
    ax.set_xlabel("Workers"); ax.set_ylabel("CPU usage (%)")
    ax.set_title("GPU Keyframe Pipeline — CPU Offload")
    ax.set_ylim(0, max(cpu_arr.max(), 50) * 1.2)
    ax.legend()
    for w, v in zip(workers_arr, cpu_arr):
        ax.annotate(f"{v:.0f}%", (w, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    save(fig, "workers_vs_cpu.png")

    # --- 7. scaling efficiency ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(workers_arr, eff, "D-", color="#17becf", ms=10, lw=2.0, label="Actual")
    ax.axhline(100, color="green", ls="--", alpha=0.4, label="Linear (ideal)")
    ax.set_xlabel("Workers"); ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("GPU Keyframe Pipeline — Scaling Efficiency (rel. {})".format(base_w))
    ax.legend()
    for w, e in zip(workers_arr, eff):
        ax.annotate(f"{e:.0f}%", (w, e), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    save(fig, "scaling_efficiency.png")

    # --- 8. Combined dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes

    # A: wall time
    bars = ax1.bar(workers_arr.astype(str), wall_arr, color=colors)
    for b, v in zip(bars, wall_arr):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f"{v:.1f}s", ha="center", fontsize=8)
    ax1.set_title("Wall Time"); ax1.set_ylabel("seconds")

    # B: realtime
    ax2.plot(workers_arr, rt_arr, "D-", color="#2E86AB", ms=8, lw=1.8)
    ax2.set_title("Realtime Factor"); ax2.set_ylabel("×")
    for w, v in zip(workers_arr, rt_arr):
        ax2.text(w, v + 50, f"{v:.0f}×", ha="center", fontsize=8)

    # C: feature throughput
    ax3.plot(workers_arr, fps_arr, "s-", color="#E63946", ms=8, lw=1.8)
    ax3.set_title("Feature rows / sec"); ax3.set_ylabel("rows/s")

    # D: GPU decoder + CPU overlay
    ax4.plot(workers_arr, dec_arr, "o-", color="#2ca02c", ms=8, lw=1.8, label="GPU dec %")
    ax4.plot(workers_arr, dec_max_arr, "v--", color="#1f77b4", ms=6, lw=1.0, label="GPU dec max %")
    ax4.plot(workers_arr, cpu_arr, "o-", color="#d62728", ms=6, lw=1.2, label="CPU %")
    ax4.set_title("GPU Decoder / CPU Utilization"); ax4.set_ylabel("%")
    ax4.axhline(95, color="#d62728", ls=":", lw=0.8)

    # E: VRAM
    ax5.plot(workers_arr, vram_arr / 1024, "o-", color="#9467bd", ms=8, lw=1.8)
    ax5.set_title("Peak VRAM"); ax5.set_ylabel("GB"); ax5.set_xlabel("Workers")

    # F: scaling efficiency
    ax6.plot(workers_arr, eff, "D-", color="#17becf", ms=8, lw=1.8)
    ax6.axhline(100, color="green", ls="--", lw=0.8)
    ax6.set_title("Scaling Efficiency"); ax6.set_ylabel("%"); ax6.set_xlabel("Workers")

    for ax in (ax4,): ax.legend(fontsize=7.5)

    fig.suptitle(f"GPU Keyframe-Only Pipeline Scaling — NVDEC hevc_cuvid + -discard nokey\n"
                 f"37 videos  ·  25.5 h  ·  RTX A6000",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "scaling_dashboard_gpu.png")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # system info
    phys = psutil.cpu_count(logical=False)
    log  = psutil.cpu_count(logical=True)
    mem_total = psutil.virtual_memory().total / 1024**3
    try:
        gpu_name = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        gpu_name = "unknown"

    print(f"GPU Scaling Benchmark")
    print(f"  CPU:      {phys}P/{log}L cores  |  {mem_total:.0f} GB RAM")
    print(f"  GPU:      {gpu_name}")
    print(f"  Binary:   {CPP_BIN}")
    print(f"  Videos:   {VIDEO_DIR}")
    print(f"  Workers:  {WORKER_COUNTS}")
    print(f"  Out:      {OUT_ROOT}\n")

    # warmup at 4 workers so NVDEC / cuda ctx are hot but startup is quick
    print("=== warmup (4 workers, GPU) ===")
    _ = run_one(4)
    print()

    results = []
    for w in WORKER_COUNTS:
        print(f"=== {w} workers ===", flush=True)
        r = run_one(w)
        results.append(r)
        print()

    # save
    df = pd.DataFrame(results)
    df.to_csv(OUT_ROOT / "gpu_scaling_benchmark.csv", index=False)
    with open(OUT_ROOT / "gpu_scaling_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    # print table
    print("=" * 90)
    print("  GPU SCALING BENCHMARK (--gpu, --no-output)")
    print("=" * 90)
    cols = ["workers", "wall_time_s", "realtime_factor", "fps",
            "cpu_usage_pct", "gpu_dec_avg_pct", "gpu_dec_max_pct", "gpu_mem_max_mb"]
    print(df[cols].to_string(index=False))

    # --- bottleneck analysis ---
    best_idx = int(np.argmax([r["realtime_factor"] for r in results]))
    best = results[best_idx]
    dec_avg = np.array([r["gpu_dec_avg_pct"] for r in results])
    eff_vals = (np.array([r["realtime_factor"] for r in results]) /
                np.array([r["workers"] for r in results]))
    eff_vals = eff_vals / eff_vals[0] * 100

    print(f"\n  Best:  {best['workers']} workers → {best['realtime_factor']:.0f}x realtime  "
          f"{best['wall_time_s']:.1f}s wall")
    print(f"  GPU decoder saturation:  {best['gpu_dec_avg_pct']:.0f}% avg / {best['gpu_dec_max_pct']:.0f}% peak")
    print(f"  VRAM peak:  {best['gpu_mem_max_mb']:.0f} MB")
    if best['gpu_dec_avg_pct'] < 80:
        print("  → NOT decode-bound at best worker count; may be I/O or pipe overhead")
    else:
        print("  → GPU decoder saturated — this is a decode-bound pipeline")

    # scaling health
    if len(eff_vals) >= 3 and eff_vals[-1] < eff_vals[0] * 0.7:
        print(f"  → Scaling efficiency dropping: {eff_vals[-1]:.0f}% at {results[-1]['workers']} workers")
    else:
        print(f"  → Scaling still healthy across range (eff {eff_vals[-1]:.0f}% at {results[-1]['workers']} workers)")

    # charts
    print()
    plot_results(results)
    print(f"\n  Output: {OUT_ROOT}")
    print(f"  Charts: {OUT_ROOT / 'charts'}")

if __name__ == "__main__":
    main()
