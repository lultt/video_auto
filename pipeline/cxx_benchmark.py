"""
cxx_benchmark.py — High-thread C++ backend benchmark + visualization.
Single run per thread count. Auto-generates performance charts.
"""
import subprocess, time, psutil, os, sys, json
from pathlib import Path
import numpy as np

os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CPP_BIN = r"J:\video_auto\cpp_pipeline\build\Release\cpp_pipeline.exe"
VIDEO_DIR = r"C:\video"
OUT_DIR = r"J:\video_auto\outputs\cpp_bench"
RESULT_DIR = Path(r"J:\video_auto\cpp_pipeline\benchmark\results")
THREADS = [12, 16, 24, 32]

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# System info
phys_cores = psutil.cpu_count(logical=False)
log_cores  = psutil.cpu_count(logical=True)
mem_total  = round(psutil.virtual_memory().total / 1024**3)

print(f"=== C++ Backend Scaling Benchmark ===")
print(f"System: {phys_cores}P/{log_cores}L cores | {mem_total} GB RAM | SSD")
print()

results = []

for n_threads in THREADS:
    print(f"  Threads={n_threads} ...", end=" ", flush=True)

    for f in Path(OUT_DIR).glob("*"): f.unlink()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cpu_samples = []
    mem_samples = []

    proc = subprocess.Popen(
        [CPP_BIN, VIDEO_DIR, OUT_DIR, "--threads", str(n_threads)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    while proc.poll() is None:
        cpu_samples.append(psutil.cpu_percent(interval=0.1, percpu=False))
        mem_samples.append(psutil.virtual_memory().used / 1024**3)

    elapsed = time.perf_counter() - t0
    output = proc.stdout.read() if proc.stdout else ""
    cpu_m = np.mean(cpu_samples) if cpu_samples else 0
    mem_m = np.mean(mem_samples) if mem_samples else 0

    # Parse output
    rt = 0.0; fps_val = 0.0; videos_sec = 0.0; wall_time = 0.0; peak_mem = 0
    for line in output.splitlines():
        line_s = line.strip()
        if "Realtime factor:" in line_s:
            try: rt = float(line_s.split()[-1].replace("×",""))
            except: pass
        if "Frames/sec:" in line_s:
            try: fps_val = float(line_s.split()[-1])
            except: pass
        if "Videos/sec:" in line_s:
            try: videos_sec = float(line_s.split()[-1])
            except: pass
        if "Wall time:" in line_s:
            try: wall_time = float(line_s.replace("s","").split()[-1])
            except: pass
        if "Peak memory:" in line_s:
            try: peak_mem = float(line_s.replace("MB","").split()[-1])
            except: pass

    r = {
        "threads": n_threads,
        "elapsed_sec": round(elapsed, 2),
        "wall_time_sec": round(wall_time, 2),
        "realtime": round(rt, 0),
        "fps": round(fps_val, 1),
        "videos_per_sec": round(videos_sec, 3),
        "cpu_pct": round(cpu_m, 1),
        "memory_gb": round(mem_m, 1),
        "peak_memory_mb": round(peak_mem, 0),
    }
    results.append(r)
    print(f"{rt:.0f}x | {fps_val:.0f} fps | CPU {cpu_m:.0f}% | {elapsed:.1f}s")

# ---- Save raw data ----
import pandas as pd
df = pd.DataFrame(results)
df.to_csv(RESULT_DIR / "benchmark_summary.csv", index=False)
with open(RESULT_DIR / "benchmark_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULT_DIR}")

# ---- Charts ----
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

threads_arr = np.array([r["threads"] for r in results])
rt_arr      = np.array([r["realtime"] for r in results])
fps_arr     = np.array([r["fps"] for r in results])
cpu_arr     = np.array([r["cpu_pct"] for r in results])
mem_arr     = np.array([r["memory_gb"] for r in results])

best_idx = np.argmax(rt_arr)
best_t   = threads_arr[best_idx]
best_rt  = rt_arr[best_idx]
best_fps = fps_arr[best_idx]

# Color palette
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bar_colors = [colors[i % 4] for i in range(len(THREADS))]

def save_and_close(fig, name):
    path = RESULT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")

# 1. threads_vs_fps
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(threads_arr.astype(str), fps_arr, color=bar_colors, edgecolor="white")
ax.set_xlabel("Threads"); ax.set_ylabel("Frames/sec"); ax.set_title("Threads vs FPS")
ax.bar_label(bars, fmt="%.0f", fontsize=10)
ax.axvline(x=best_idx, color="red", alpha=0.3, linewidth=3, linestyle="--")
ax.annotate(f"Peak: {best_fps:.0f} fps @ {best_t} threads", xy=(best_idx, best_fps),
            xytext=(best_idx + 0.5, best_fps * 1.05), fontsize=11, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))
save_and_close(fig, "threads_vs_fps.png")

# 2. threads_vs_realtime
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(threads_arr.astype(str), rt_arr, color=bar_colors, edgecolor="white")
ax.set_xlabel("Threads"); ax.set_ylabel("Realtime Factor"); ax.set_title("Threads vs Realtime Factor")
ax.bar_label(bars, fmt="%.0f×", fontsize=10)
ax.axhline(y=best_rt, color="red", alpha=0.3, linewidth=2, linestyle="--")
ax.annotate(f"Peak: {best_rt:.0f}× @ {best_t} threads", xy=(best_idx, best_rt),
            xytext=(best_idx + 0.5, best_rt * 0.9), fontsize=11, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))
save_and_close(fig, "threads_vs_realtime.png")

# 3. threads_vs_cpu
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threads_arr, cpu_arr, "o-", color="#d62728", linewidth=2, markersize=10)
ax.set_xlabel("Threads"); ax.set_ylabel("CPU Utilization (%)"); ax.set_title("Threads vs CPU Usage")
for t, c in zip(threads_arr, cpu_arr):
    ax.annotate(f"{c:.0f}%", (t, c), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)
ax.axhline(y=100, color="gray", alpha=0.3, linewidth=1, linestyle=":", label="100% saturation")
ax.legend(fontsize=9)
ax.set_ylim(0, 105)
save_and_close(fig, "threads_vs_cpu.png")

# 4. threads_vs_memory
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threads_arr, mem_arr, "o-", color="#2ca02c", linewidth=2, markersize=10)
ax.set_xlabel("Threads"); ax.set_ylabel("Memory Usage (GB)"); ax.set_title("Threads vs Memory Usage")
for t, m in zip(threads_arr, mem_arr):
    ax.annotate(f"{m:.1f} GB", (t, m), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)
ax.axhline(y=mem_total, color="gray", alpha=0.3, linewidth=1, linestyle=":", label=f"Total ({mem_total} GB)")
ax.legend(fontsize=9)
save_and_close(fig, "threads_vs_memory.png")

# 5. scaling_efficiency
ideal_first = rt_arr[0] / threads_arr[0]  # realtime-per-thread at baseline
scaling_efficiency = []
for i in range(len(results)):
    actual_ratio = rt_arr[i] / threads_arr[i]
    scaling_efficiency.append(actual_ratio / ideal_first * 100)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threads_arr, scaling_efficiency, "o-", color="#9467bd", linewidth=2, markersize=10)
ax.set_xlabel("Threads"); ax.set_ylabel("Scaling Efficiency (%)"); ax.set_title("Thread Scaling Efficiency")
ax.axhline(y=100, color="green", alpha=0.3, linewidth=1, linestyle="--", label="Linear scaling")
for t, s in zip(threads_arr, scaling_efficiency):
    ax.annotate(f"{s:.0f}%", (t, s), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10)
ax.legend(fontsize=9)
save_and_close(fig, "scaling_efficiency.png")

# 6. Combined summary plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax1, ax2), (ax3, ax4) = axes

ax1.bar(threads_arr.astype(str), rt_arr, color=bar_colors); ax1.set_title("Realtime Factor"); ax1.set_ylabel("×")
ax1.bar_label(ax1.containers[0], fmt="%.0f×", fontsize=9)

ax2.bar(threads_arr.astype(str), fps_arr, color=bar_colors); ax2.set_title("Frames/sec")
ax2.bar_label(ax2.containers[0], fmt="%.0f", fontsize=9)

ax3.plot(threads_arr, cpu_arr, "o-", color="#d62728", linewidth=2); ax3.set_title("CPU %")
for t, c in zip(threads_arr, cpu_arr): ax3.annotate(f"{c:.0f}%", (t, c), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

ax4.plot(threads_arr, scaling_efficiency, "o-", color="#9467bd", linewidth=2); ax4.set_title("Scaling Efficiency %")
for t, s in zip(threads_arr, scaling_efficiency): ax4.annotate(f"{s:.0f}%", (t, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

fig.suptitle(f"C++ Backend Scaling Analysis\n{phys_cores}P/{log_cores}L cores | SSD | Best: {best_t} threads = {best_rt:.0f}×", fontsize=14, fontweight="bold")
save_and_close(fig, "scaling_summary.png")

# ---- Analysis ----
print(f"\n{'='*60}")
print(f"  SCALING ANALYSIS")
print(f"{'='*60}")
print(f"  Best: {best_t} threads → {best_rt:.0f}× realtime, {best_fps:.0f} fps")
print(f"  CPU at best: {cpu_arr[best_idx]:.0f}%")
print(f"  Memory at best: {mem_arr[best_idx]:.1f} GB")
print(f"  Scaling efficiency at best: {scaling_efficiency[best_idx]:.0f}%")

if len(results) >= 4 and rt_arr[-1] < rt_arr[-2]:
    print(f"\n  REGRESSION: {THREADS[-1]} threads slower than {THREADS[-2]} threads")
    print(f"  → Likely bottleneck: thread contention or memory bandwidth saturation")

# Determine bottleneck
cpu_at_best = cpu_arr[best_idx]
if cpu_at_best < 80:
    print(f"\n  CPU at peak is {cpu_at_best:.0f}% — NOT CPU bound.")
    if cpu_at_best < 50:
        print(f"  → Strong I/O or memory bandwidth bottleneck.")
    else:
        print(f"  → Moderate memory/thread contention.")
else:
    print(f"\n  CPU at peak is {cpu_at_best:.0f}% — CPU-bound.")

print(f"\n  All charts saved to: {RESULT_DIR}")
