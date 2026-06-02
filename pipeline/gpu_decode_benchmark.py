"""
gpu_decode_benchmark.py — Compare CPU vs GPU (NVDEC) decode at 32 workers.
Both runs use --no-output to isolate decode + feature extraction.
"""
import os, sys, subprocess, time, json, threading, re
from pathlib import Path
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
OUT_ROOT = Path(r"J:\video_auto\outputs\gpu_bench")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
NVIDIA_SMI = r"C:\Windows\System32\nvidia-smi.exe"
WORKERS = 32


def sample_gpu():
    """Return (gpu_util, decode_util, mem_used_mb). Decode util via nvidia-smi dmon."""
    try:
        r = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=utilization.gpu,utilization.decoder,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def run_one(use_gpu, label):
    cpu_samples, gpu_samples, dec_samples, gmem_samples = [], [], [], []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=None, percpu=False))
            g, d, m = sample_gpu()
            gpu_samples.append(g)
            dec_samples.append(d)
            gmem_samples.append(m)
            time.sleep(0.2)

    psutil.cpu_percent(interval=None)
    sample_gpu()  # prime
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    cmd = [CPP_BIN, VIDEO_DIR, "--no-output", "--threads", str(WORKERS)]
    if use_gpu:
        cmd.append("--gpu")

    print(f"\n=== {label} ===")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.perf_counter() - t0
    stop.set()
    sampler_thread.join(timeout=2)

    out = proc.stdout

    rt = fps = wall_cpp = 0.0
    total_video_hrs = 0.0
    for line in out.splitlines():
        ls = line.strip()
        if "Realtime factor:" in ls:
            try: rt = float(ls.split()[-1].replace("x", ""))
            except: pass
        elif "Frames/sec:" in ls:
            try: fps = float(ls.split()[-1])
            except: pass
        elif "Wall time:" in ls:
            try: wall_cpp = float(ls.replace("s", "").split()[-1])
            except: pass
        elif "Video duration:" in ls:
            try: total_video_hrs = float(ls.replace("hrs", "").split()[-1])
            except: pass

    cpu_mean = float(np.mean(cpu_samples)) if cpu_samples else 0
    gpu_mean = float(np.mean(gpu_samples)) if gpu_samples else 0
    dec_mean = float(np.mean(dec_samples)) if dec_samples else 0
    dec_max = float(np.max(dec_samples)) if dec_samples else 0
    gmem_max = float(np.max(gmem_samples)) if gmem_samples else 0

    result = {
        "mode": label,
        "wall_time_s": round(wall_cpp if wall_cpp > 0 else wall, 2),
        "realtime_factor": round(rt, 1),
        "fps": round(fps, 1),
        "cpu_usage_pct": round(cpu_mean, 1),
        "gpu_util_pct": round(gpu_mean, 1),
        "gpu_decode_pct": round(dec_mean, 1),
        "gpu_decode_max_pct": round(dec_max, 1),
        "gpu_mem_max_mb": round(gmem_max, 0),
        "total_video_hours": round(total_video_hrs, 2),
    }
    print(f"  wall={result['wall_time_s']}s  rt={result['realtime_factor']}x  "
          f"fps={result['fps']}  cpu={result['cpu_usage_pct']}%  "
          f"gpu_dec_avg={result['gpu_decode_pct']}%  gpu_dec_max={result['gpu_decode_max_pct']}%")
    return result, (cpu_samples, gpu_samples, dec_samples)


def main():
    phys = psutil.cpu_count(logical=False)
    log = psutil.cpu_count(logical=True)
    g, d, m = sample_gpu()
    print(f"System: {phys}P/{log}L CPU cores")
    try:
        gpu_name = subprocess.run([NVIDIA_SMI, "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
        print(f"GPU:    {gpu_name}")
    except: pass
    print(f"Workers: {WORKERS}  |  no-output mode\n")

    # Warmup
    print("=== Warmup (CPU, 32 workers) ===")
    _ = run_one(False, "warmup")

    cpu_result, cpu_samples = run_one(False, "CPU decode")
    gpu_result, gpu_samples = run_one(True, "GPU decode (NVDEC/hevc_cuvid)")

    results = [cpu_result, gpu_result]
    df = pd.DataFrame(results)
    df.to_csv(OUT_ROOT / "gpu_vs_cpu.csv", index=False)
    with open(OUT_ROOT / "gpu_vs_cpu.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("  CPU vs GPU DECODE @ 32 workers, --no-output")
    print("="*80)
    print(df.to_string(index=False))

    speedup = cpu_result["wall_time_s"] / max(gpu_result["wall_time_s"], 0.01)
    cpu_drop = cpu_result["cpu_usage_pct"] - gpu_result["cpu_usage_pct"]
    print(f"\nSpeedup (CPU→GPU): {speedup:.2f}x")
    print(f"CPU usage drop:    {cpu_drop:+.1f}%pt  ({cpu_result['cpu_usage_pct']}% -> {gpu_result['cpu_usage_pct']}%)")
    print(f"GPU decoder util:  {gpu_result['gpu_decode_pct']:.0f}% avg, {gpu_result['gpu_decode_max_pct']:.0f}% peak")
    print(f"GPU memory used:   {gpu_result['gpu_mem_max_mb']:.0f} MB peak")

    # Save sample timelines
    with open(OUT_ROOT / "samples.json", "w") as f:
        json.dump({
            "cpu_mode": {"cpu": cpu_samples[0], "gpu": cpu_samples[1], "dec": cpu_samples[2]},
            "gpu_mode": {"cpu": gpu_samples[0], "gpu": gpu_samples[1], "dec": gpu_samples[2]},
        }, f)

    # Charts
    chart_dir = OUT_ROOT / "charts"
    chart_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["CPU decode", "GPU decode\n(NVDEC)"]
    colors = ["#1f77b4", "#2ca02c"]

    ax = axes[0]
    walls = [cpu_result["wall_time_s"], gpu_result["wall_time_s"]]
    bars = ax.bar(labels, walls, color=colors)
    for b, v in zip(bars, walls):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}s",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Wall Time @ {WORKERS} workers")
    ax.set_ylabel("seconds")

    ax = axes[1]
    rts = [cpu_result["realtime_factor"], gpu_result["realtime_factor"]]
    bars = ax.bar(labels, rts, color=colors)
    for b, v in zip(bars, rts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v:.0f}x",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Realtime Factor")
    ax.set_ylabel("x")

    ax = axes[2]
    cpu_pcts = [cpu_result["cpu_usage_pct"], gpu_result["cpu_usage_pct"]]
    gpu_pcts = [0, gpu_result["gpu_decode_pct"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, cpu_pcts, w, color="#d62728", label="CPU %")
    ax.bar(x + w/2, gpu_pcts, w, color="#2ca02c", label="GPU decoder %")
    for i, (c, g) in enumerate(zip(cpu_pcts, gpu_pcts)):
        ax.text(i - w/2, c + 1, f"{c:.0f}%", ha="center", fontsize=9)
        ax.text(i + w/2, g + 1, f"{g:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("CPU vs GPU Utilization")
    ax.set_ylabel("%"); ax.set_ylim(0, 105)
    ax.legend()

    fig.suptitle(f"CPU vs GPU Decode @ {WORKERS} workers (--no-output)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(chart_dir / "cpu_vs_gpu.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Timeline chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    ax = axes[0]
    ax.plot(cpu_samples[0], color="#1f77b4", lw=1.2, label="CPU mode: CPU %")
    ax.plot(gpu_samples[0], color="#2ca02c", lw=1.2, label="GPU mode: CPU %")
    ax.set_title("CPU usage over time"); ax.set_ylabel("CPU %")
    ax.set_xlabel("sample # (~200ms each)")
    ax.legend(); ax.set_ylim(0, 105)
    ax.axhline(100, color="gray", ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(gpu_samples[2], color="#2ca02c", lw=1.2, label="GPU decoder %")
    ax.plot(gpu_samples[1], color="#9467bd", lw=1.2, label="GPU SM %")
    ax.set_title("GPU usage over time (GPU mode)"); ax.set_ylabel("%")
    ax.set_xlabel("sample # (~200ms each)")
    ax.legend(); ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(chart_dir / "timeline.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nCharts: {chart_dir}")


if __name__ == "__main__":
    main()
