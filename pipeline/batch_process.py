"""
batch_process.py — 批量跑 61 晚 L1(粗特征)+L2(网次检测)。

用法:
  python pipeline/batch_process.py                  # 跑所有 61 晚
  python pipeline/batch_process.py --max-nights 3   # 只跑前 3 晚
  python pipeline/batch_process.py --workers 4      # 4 个并行 worker
"""
import os, sys, time
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from pipeline.coarse_features.feature_merge import (
    run_coarse_pipeline, DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
    DEFAULT_RESIZE, DEFAULT_ROIS_CFG, DEFAULT_FEATURE_FLAGS,
)
from pipeline.cycle_detection.detect_cycles import run_cycle_detection

VIDEO_ROOT = Path(r"\\DS224plus\video\viedeo")
MANIFEST = Path(r"J:\video_auto\data\video_manifest.csv")
OUTPUT_BASE = Path(r"J:\video_auto\outputs")
SUMMARY = Path(r"J:\video_auto\reports\nightly_operation_summary.csv")

NIGHT_START = "20:30"
NIGHT_END = "08:30"


def process_one_night(date, video_paths, night_out, n_workers):
    """L1 → L2 per night. Returns result dict."""
    parquet_dir = night_out
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # L1: coarse feature extraction (parallel within night)
    run_coarse_pipeline(
        video_paths, str(parquet_dir),
        DEFAULT_RESIZE, DEFAULT_ROIS_CFG, DEFAULT_FEATURE_FLAGS,
        DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC,
        num_workers=n_workers,
    )

    pq_files = list(parquet_dir.glob("*.parquet"))
    if len(pq_files) == 0:
        return {"date": date, "has_operation": False, "net_cycles_detected": 0,
                "status": "no_parquet_output"}

    # L2: cycle detection
    l2_dir = night_out / "net_cycle_detection"
    l2_dir.mkdir(parents=True, exist_ok=True)
    l2_result = run_cycle_detection(
        parquet_root=parquet_dir, out_dir=l2_dir,
        night_start=NIGHT_START, night_end=NIGHT_END,
    )
    cycles_df = l2_result["cycles_df"]

    return {
        "date": date,
        "has_operation": len(cycles_df) > 0,
        "net_cycles_detected": len(cycles_df),
        "mean_cycle_duration_min": round(float(cycles_df["duration_min"].mean()), 1) if len(cycles_df) > 0 else 0,
        "std_cycle_duration_min": round(float(cycles_df["duration_min"].std()), 1) if len(cycles_df) > 1 else 0,
        "min_cycle_duration_min": round(float(cycles_df["duration_min"].min()), 1) if len(cycles_df) > 0 else 0,
        "max_cycle_duration_min": round(float(cycles_df["duration_min"].max()), 1) if len(cycles_df) > 0 else 0,
        "cycle_times": " | ".join(cycles_df["start_time"].str[11:16] + " " + cycles_df["end_time"].str[11:16]) if len(cycles_df) > 0 else "",
        "peak_motion_mean": round(float(cycles_df["max_motion"].mean()), 3) if len(cycles_df) > 0 else 0,
        "parquet_files": len(pq_files),
        "status": "ok",
    }


def main(max_nights=None, workers=4):
    df = pd.read_csv(MANIFEST)
    df = df.dropna(subset=["start_time"])
    df["date"] = df["start_time"].str[:10]
    dates = sorted(df["date"].unique())
    if max_nights:
        dates = dates[:max_nights]

    print(f"Batch processing {len(dates)} nights, {workers} workers per night")
    print(f"  Video root: {VIDEO_ROOT}")
    print(f"  Night window: {NIGHT_START} — {NIGHT_END}")
    print()

    results = []
    t_total = time.perf_counter()

    for i, date in enumerate(dates):
        t0 = time.perf_counter()
        night_videos = df[df["date"] == date]["path"].tolist()
        night_out = OUTPUT_BASE / f"coarse_{date}"
        tag = f"[{i+1:2d}/{len(dates)}]"

        print(f"{tag} {date}: {len(night_videos)} videos → {night_out}")

        try:
            r = process_one_night(date, night_videos, night_out, n_workers=workers)
        except Exception as e:
            r = {"date": date, "has_operation": False, "net_cycles_detected": 0,
                 "status": f"error: {str(e)[:80]}"}

        elapsed = time.perf_counter() - t0
        r["processing_time_sec"] = round(elapsed, 1)
        results.append(r)

        n_nets = r.get("net_cycles_detected", "?")
        status = r.get("status", "?")
        print(f"      → {n_nets} nets, {elapsed:.0f}s ({status})")

        # save incrementally
        pd.DataFrame(results).to_csv(SUMMARY, index=False, encoding="utf-8-sig")

    total_elapsed = time.perf_counter() - t_total
    results_df = pd.DataFrame(results)
    ops = results_df[results_df["has_operation"] == True]

    print(f"\nDone. {len(results)} nights in {total_elapsed/60:.0f} min")
    print(f"  Nights with ops: {len(ops)}")
    if len(ops) > 0:
        print(f"  Mean nets/night: {ops['net_cycles_detected'].mean():.1f}")
        print(f"  Mean cycle duration: {ops['mean_cycle_duration_min'].mean():.1f} min")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-nights", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    main(max_nights=args.max_nights, workers=args.workers)