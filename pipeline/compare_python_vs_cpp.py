"""
compare_python_vs_cpp.py — Python vs C++ pipeline performance benchmark.

Runs both backends on the same video directory with 16 workers, collects
per-video timing + CPU samples, and generates comparison charts.
Both pipelines output LEFT + RIGHT ROIs for apples-to-apples comparison.

Usage:
  python pipeline/compare_python_vs_cpp.py --video-dir C:/video/0515 --workers 16
"""
import subprocess, time, psutil, os, sys, json, shutil, threading, re
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from pipeline.coarse_features.feature_merge import (
    process_video_coarse, DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
    DEFAULT_RESIZE, DEFAULT_FEATURE_FLAGS,
)

CPP_BIN = r"J:\video_auto\cpp_pipeline\build\Release\cpp_pipeline.exe"
DEFAULT_VIDEO_DIR = r"C:\video\0515"
DEFAULT_OUT_ROOT = r"J:\video_auto\outputs\pipeline_comparison"

# 2-ROI config (left + right only, centre dropped for apples-to-apples comparison)
ROIS_LEFT_RIGHT = {
    "left_deck":  {"x_min": 0.000, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0},
    "right_deck": {"x_min": 0.667, "x_max": 1.000, "y_min": 0.4, "y_max": 1.0},
}


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


class CPUSampler:
    """Background thread that samples CPU % + memory."""
    def __init__(self, interval=0.2):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=None, percpu=False)
            mem = psutil.virtual_memory().used / 1024**3
            self.samples.append((time.perf_counter() - t0, cpu, mem))
            self._stop.wait(self.interval)


def run_python_benchmark(video_paths, out_dir, num_workers):
    """Run Python pipeline. Returns {wall_sec, per_video: [{video_id, t, rows, dur_sec}]}."""
    out_dir = Path(out_dir)
    # wipe previous
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sampler = CPUSampler()
    per_video = []

    t0 = time.perf_counter()
    sampler.start()

    if num_workers <= 1:
        for vp in video_paths:
            vt0 = time.perf_counter()
            res = process_video_coarse(
                vp, str(out_dir), DEFAULT_RESIZE, ROIS_LEFT_RIGHT, DEFAULT_FEATURE_FLAGS,
                DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
            )
            vt1 = time.perf_counter()
            per_video.append({
                "video_id": Path(vp).name,
                "processing_time_sec": round(vt1 - vt0, 3),
                "rows": res.get("rows", 0),
                "frames": res.get("frames", 0),
                "video_duration_sec": video_duration_sec(vp),
            })
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for vp in video_paths:
                fut = executor.submit(
                    _timed_process_video, vp, str(out_dir),
                )
                futures[fut] = vp
            for fut in as_completed(futures):
                info = fut.result()
                per_video.append(info)
                vid = info["video_id"][:40] if info["video_id"] else "?"
                print(f"  [PY] {vid}  {info['frames']} frames  {info['rows']} rows  {info['processing_time_sec']:.1f}s")

    sampler.stop()
    wall = time.perf_counter() - t0

    return {
        "wall_sec": round(wall, 2),
        "cpu_samples": sampler.samples,
        "per_video": per_video,
    }


def _timed_process_video(vp, out_dir):
    vt0 = time.perf_counter()
    res = process_video_coarse(
        vp, out_dir, DEFAULT_RESIZE, ROIS_LEFT_RIGHT, DEFAULT_FEATURE_FLAGS,
        DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
    )
    vt1 = time.perf_counter()
    return {
        "video_id": Path(vp).name,
        "processing_time_sec": round(vt1 - vt0, 3),
        "rows": res.get("rows", 0),
        "frames": res.get("frames", 0),
        "video_duration_sec": video_duration_sec(vp),
    }


def run_cpp_benchmark(video_dir, out_dir, num_workers):
    """Run C++ pipeline. Returns {wall_sec, per_video: [...]}."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    per_video_log = out_dir / "_per_video.jsonl"

    sampler = CPUSampler()
    sampler.start()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [CPP_BIN, video_dir, str(out_dir), "--threads", str(num_workers),
         "--emit-csv", "--per-video-log", str(per_video_log)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    while proc.poll() is None:
        time.sleep(0.1)

    wall = time.perf_counter() - t0
    sampler.stop()

    stdout = proc.stdout.read() if proc.stdout else ""

    # Parse C++ banner
    rt = fps = vps = wall_cpp = peak_mem = 0.0
    for line in stdout.splitlines():
        ls = line.strip()
        if "Realtime factor:" in ls:
            try: rt = float(ls.split()[-1].replace("x", ""))
            except: pass
        if "Frames/sec:" in ls:
            try: fps = float(ls.split()[-1])
            except: pass
        if "Videos/sec:" in ls:
            try: vps = float(ls.split()[-1])
            except: pass
        if "Wall time:" in ls:
            try: wall_cpp = float(ls.replace("s", "").split()[-1])
            except: pass
        if "Peak memory:" in ls:
            try: peak_mem = float(ls.replace("MB", "").split()[-1])
            except: pass
        if "Video duration:" in ls:
            try: total_dur = float(ls.replace("hrs", "").split()[-1])
            except: pass

    # Parse per-video log
    per_video = []
    if per_video_log.exists():
        with open(per_video_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    per_video.append(json.loads(line))

    print(stdout)

    return {
        "wall_sec": round(wall_cpp if wall_cpp > 0 else wall, 2),
        "realtime_factor": round(rt, 1),
        "fps": round(fps, 1),
        "videos_per_sec": round(vps, 3),
        "peak_memory_mb": round(peak_mem, 1),
        "cpu_samples": sampler.samples,
        "per_video": per_video,
    }


def compute_feature_diff(py_dir, cpp_dir, out_path):
    """Compare Python parquet vs C++ CSV per video per ROI per feature."""
    py_dir = Path(py_dir)
    cpp_dir = Path(cpp_dir)
    rows = []

    py_files = sorted(py_dir.glob("*.parquet"))
    for pf in py_files:
        vid = pf.stem
        csv_path = cpp_dir / f"{vid}.csv"
        if not csv_path.exists():
            continue

        df_py = pd.read_parquet(pf)
        df_py = df_py[df_py["roi_name"].isin(["left_deck", "right_deck"])].copy()
        df_py["roi"] = df_py["roi_name"].map({"left_deck": "left", "right_deck": "right"})

        df_cpp = pd.read_csv(csv_path)

        feature_cols = [
            "mean_brightness", "brightness_std",
            "mean_b", "mean_g", "mean_r",
            "motion_intensity", "edge_density",
            "laplacian_variance", "entropy",
        ]

        for roi in ["left", "right"]:
            py_r = df_py[df_py["roi"] == roi]
            cpp_r = df_cpp[df_cpp["roi_name"] == roi]
            if len(py_r) == 0 or len(cpp_r) == 0:
                continue
            # Align on frame_idx (C++ may have slightly different keyframe count)
            merged = py_r.merge(cpp_r, on="frame_idx", suffixes=("_py", "_cpp"), how="inner")
            if len(merged) < 2:
                continue

            for feat in feature_cols:
                py_vals = merged[f"{feat}_py"].values.astype(float)
                cpp_vals = merged[f"{feat}_cpp"].values.astype(float)
                denom = np.abs(py_vals).mean() + 1e-9
                mad_rel = np.abs(py_vals - cpp_vals).mean() / denom
                rows.append({
                    "video_id": vid,
                    "roi": roi,
                    "feature": feat,
                    "py_mean": round(float(py_vals.mean()), 6),
                    "cpp_mean": round(float(cpp_vals.mean()), 6),
                    "mad_rel": round(float(mad_rel), 6),
                })

    if not rows:
        print("  [WARN] No overlapping data for feature diff — check that --emit-csv was used")
        return

    df_diff = pd.DataFrame(rows)
    df_diff.to_csv(out_path, index=False)
    print(f"\n  Feature diff saved to {out_path}")

    # Summarize
    print("\n  Feature-diff summary (mean absolute relative diff):")
    for feat in df_diff["feature"].unique():
        sub = df_diff[df_diff["feature"] == feat]
        avg = sub["mad_rel"].mean()
        note = ""
        if feat == "edge_density":
            note = "  [EXPECTED DIVERGENCE — Python uses Canny, C++ uses Sobel]"
        print(f"    {feat:25s}: {avg:.4f}{note}")


def plot_results(py_result, cpp_result, out_dir, video_count, total_video_hours):
    """Generate comparison charts."""
    chart_dir = Path(out_dir) / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    py_wall = py_result["wall_sec"]
    cpp_wall = cpp_result["wall_sec"]
    cpp_rt = cpp_result.get("realtime_factor", total_video_hours * 3600 / max(cpp_wall, 0.1))
    py_rt = total_video_hours * 3600 / max(py_wall, 0.1)
    cpp_fps = cpp_result.get("fps", 0)
    py_fps = sum(pv["rows"] for pv in py_result["per_video"]) / max(py_wall, 0.1)

    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    colors = ["#1f77b4", "#d62728"]

    # --- Chart A: Wall time ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Python (16 procs)", "C++ (16 threads)"]
    values = [py_wall, cpp_wall]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylabel("Wall time (seconds)")
    ax.set_title(f"Processing Wall Time ({video_count} videos, {total_video_hours:.1f} hrs)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                f"{v:.0f}s ({v/60:.1f}min)", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(chart_dir / "wall_time_compare.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: wall_time_compare.png")

    # --- Chart B: Realtime factor ---
    fig, ax = plt.subplots(figsize=(8, 5))
    values_rt = [py_rt, cpp_rt]
    bars = ax.bar(labels, values_rt, color=colors, edgecolor="white")
    ax.set_ylabel("Realtime Factor (×)")
    ax.set_title("Realtime Factor Comparison")
    for b, v in zip(bars, values_rt):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.0f}×", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(chart_dir / "realtime_factor_compare.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: realtime_factor_compare.png")

    # --- Chart C: Frames/sec ---
    fig, ax = plt.subplots(figsize=(8, 5))
    values_fps = [py_fps, cpp_fps]
    bars = ax.bar(labels, values_fps, color=colors, edgecolor="white")
    ax.set_ylabel("Frames/sec (feature rows/sec)")
    ax.set_title("Throughput Comparison")
    for b, v in zip(bars, values_fps):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(chart_dir / "fps_compare.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: fps_compare.png")

    # --- Chart D: CPU timeline ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, result, col in [("Python", py_result, colors[0]), ("C++", cpp_result, colors[1])]:
        if result.get("cpu_samples"):
            ts = [s[0] for s in result["cpu_samples"]]
            cpu = [s[1] for s in result["cpu_samples"]]
            ax.plot(ts, cpu, color=col, linewidth=1.5, alpha=0.8, label=label)
    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("CPU %")
    ax.set_title("CPU Utilization Over Time")
    ax.axhline(y=100, color="gray", alpha=0.3, linestyle=":")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(chart_dir / "cpu_timeline.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: cpu_timeline.png")

    # --- Chart E: Per-video time distribution ---
    if py_result["per_video"] and cpp_result["per_video"]:
        fig, ax = plt.subplots(figsize=(14, 6))
        py_pv = sorted(py_result["per_video"], key=lambda x: x["processing_time_sec"])
        cpp_pv_map = {p["video_id"]: p["processing_time_sec"] for p in cpp_result["per_video"]}

        x = range(len(py_pv))
        py_times = [p["processing_time_sec"] for p in py_pv]
        cpp_times = [cpp_pv_map.get(p["video_id"], 0) for p in py_pv]

        ax.plot(x, py_times, "o-", color=colors[0], linewidth=1.5, markersize=5, label="Python")
        ax.plot(x, cpp_times, "s-", color=colors[1], linewidth=1.5, markersize=5, label="C++")
        ax.set_xlabel("Video index (sorted by Python time)")
        ax.set_ylabel("Processing time (s)")
        ax.set_title("Per-Video Processing Time Distribution")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(chart_dir / "per_video_times.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: per_video_times.png")

    # --- Chart F: Summary dashboard (2×2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax1, ax2), (ax3, ax4) = axes

    ax1.bar(labels, values, color=colors)
    ax1.set_title("Wall Time"); ax1.set_ylabel("seconds")
    for b, v in zip(ax1.containers[0], values):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{v:.0f}s", ha="center", fontsize=9)

    ax2.bar(labels, values_rt, color=colors)
    ax2.set_title("Realtime Factor"); ax2.set_ylabel("×")
    for b, v in zip(ax2.containers[0], values_rt):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.0f}×", ha="center", fontsize=9)

    ax3.bar(labels, values_fps, color=colors)
    ax3.set_title("Throughput"); ax3.set_ylabel("rows/sec")
    for b, v in zip(ax3.containers[0], values_fps):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, f"{v:.0f}", ha="center", fontsize=9)

    for label, result, col in [("Python", py_result, colors[0]), ("C++", cpp_result, colors[1])]:
        if result.get("cpu_samples"):
            ts = [s[0] for s in result["cpu_samples"]]
            cpu = [s[1] for s in result["cpu_samples"]]
            ax4.plot(ts, cpu, color=col, linewidth=1.5, alpha=0.8, label=label)
    ax4.set_title("CPU % Over Time"); ax4.set_ylabel("CPU %"); ax4.set_xlabel("time (s)")
    ax4.axhline(y=100, color="gray", alpha=0.3, linestyle=":")
    ax4.legend(fontsize=8)

    speedup = py_wall / max(cpp_wall, 0.01)
    fig.suptitle(f"Python vs C++ Pipeline Comparison (16 workers)\n"
                 f"{video_count} videos | {total_video_hours:.1f} hrs video | C++ is {speedup:.1f}× faster",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(chart_dir / "summary_dashboard.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: summary_dashboard.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Python vs C++ pipeline benchmark")
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-cpp", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    py_out = out_root / "python_features"
    cpp_out = out_root / "cpp_features"

    video_paths = sorted([str(p) for p in Path(args.video_dir).glob("*.mp4")])
    if not video_paths:
        print(f"No .mp4 files in {args.video_dir}")
        return 1

    total_video_hours = sum(video_duration_sec(v) for v in video_paths) / 3600
    print(f"{'='*60}")
    print(f"  Python vs C++ Pipeline Benchmark")
    print(f"{'='*60}")
    print(f"  Videos: {len(video_paths)}")
    print(f"  Total duration: {total_video_hours:.1f} hours")
    print(f"  Workers: {args.workers}")
    print(f"  Output:  {out_root}")
    print()

    # --- Python run ---
    py_result = {}
    if not args.skip_python:
        print("=" * 60)
        print("  PYTHON PIPELINE (16 workers, left+right ROIs)")
        print("=" * 60)
        py_result = run_python_benchmark(video_paths, py_out, args.workers)
        py_df = pd.DataFrame(py_result["per_video"])
        py_df.to_csv(out_root / "python_per_video.csv", index=False)
        print(f"  Python wall: {py_result['wall_sec']:.1f}s  "
              f"realtime: {total_video_hours*3600/py_result['wall_sec']:.0f}×")
        print()
    else:
        print("  Skipping Python run\n")

    # --- C++ run ---
    cpp_result = {}
    if not args.skip_cpp:
        print("=" * 60)
        print("  C++ PIPELINE (16 threads, left+right ROIs)")
        print("=" * 60)
        cpp_result = run_cpp_benchmark(args.video_dir, cpp_out, args.workers)
        if cpp_result.get("per_video"):
            cpp_df = pd.DataFrame(cpp_result["per_video"])
            cpp_df.to_csv(out_root / "cpp_per_video.csv", index=False)
        print()
    else:
        print("  Skipping C++ run\n")

    if not py_result or not cpp_result:
        print("Missing results — run both or provide skip flags")
        return 1

    # --- Summary ---
    py_wall = py_result["wall_sec"]
    cpp_wall = cpp_result["wall_sec"]
    speedup = py_wall / max(cpp_wall, 0.01)
    cpp_rt = cpp_result.get("realtime_factor", 0)
    py_rt = total_video_hours * 3600 / max(py_wall, 0.1)

    summary = {
        "pipeline": ["python", "cpp"],
        "wall_sec": [py_wall, cpp_wall],
        "realtime_factor": [round(py_rt, 1), round(cpp_rt, 1)],
        "fps": [round(sum(pv["rows"] for pv in py_result["per_video"]) / max(py_wall, 0.1), 1),
                round(cpp_result.get("fps", 0), 1)],
        "cpu_avg_pct": [
            round(np.mean([s[1] for s in py_result.get("cpu_samples", [])]) if py_result.get("cpu_samples") else 0, 1),
            round(np.mean([s[1] for s in cpp_result.get("cpu_samples", [])]) if cpp_result.get("cpu_samples") else 0, 1),
        ],
        "total_video_hours": [total_video_hours, total_video_hours],
        "speedup_vs_cpp": [round(speedup, 2), 1.0],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(out_root / "summary.csv", index=False)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Charts ---
    print("\n" + "=" * 60)
    print("  GENERATING CHARTS")
    print("=" * 60)
    plot_results(py_result, cpp_result, out_root, len(video_paths), total_video_hours)

    # --- Feature diff ---
    if not args.skip_python and not args.skip_cpp:
        print("\n" + "=" * 60)
        print("  FEATURE DIFF (Python vs C++)")
        print("=" * 60)
        compute_feature_diff(py_out, cpp_out, out_root / "feature_diff.csv")

    # --- Final ---
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Python:  {py_wall:.0f}s wall  |  {py_rt:.0f}× realtime")
    print(f"  C++:     {cpp_wall:.0f}s wall  |  {cpp_rt:.0f}× realtime")
    print(f"  Speedup: C++ is {speedup:.1f}× faster than Python")
    print(f"\n  All outputs: {out_root}")
    print(f"  Charts:      {out_root}/charts/")
    print(f"  Summary:     {out_root}/summary.csv")


if __name__ == "__main__":
    main()
