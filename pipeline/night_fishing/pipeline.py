# pipeline.py — main entry point: load → group → analyze → visualize → report

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Optional
from .config import OUTPUT_ROOT, NIGHT_OUT, CSV_DIR
from .loader import load_all
from .night_grouper import assign_night, get_night_list
from .fai import compute_fai, compute_global_stats
from .detector import analyze_night
from .visualizer import plot_night


def run_pipeline(
    parquet_root = None,
    max_files: Optional[int] = None,
    skip_plots: bool = False,
):
    """Run the full night fishing detection pipeline.

    Parameters
    ----------
    parquet_root : Path or None
        Root directory for recursive parquet scan. Defaults to OUTPUT_ROOT.
    max_files : int or None
        Limit number of parquet files to load (for quick testing).
    skip_plots : bool
        Skip per-night plot generation.
    """
    t0 = time.perf_counter()

    # Ensure output dirs exist
    NIGHT_OUT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load ----
    print("=" * 70)
    print("  STEP 1: Loading parquet files")
    print("=" * 70)
    df = load_all(root=Path(parquet_root) if parquet_root else None, max_files=max_files)

    # ---- Step 2: Night grouping ----
    print("\n" + "=" * 70)
    print("  STEP 2: Assigning night groups (18:00 ~ 06:00)")
    print("=" * 70)
    df = assign_night(df)
    nights = get_night_list(df)
    daytime_count = (df["night"] == "daytime").sum()
    print(f"  Found {len(nights)} nights  |  {daytime_count:,} daytime rows excluded")

    # ---- Step 3: Compute global reference stats for FAI scaling ----
    print("\n" + "=" * 70)
    print("  STEP 3: Computing global FAI reference statistics")
    print("=" * 70)
    night_mask = df["night"] != "daytime"
    global_stats = compute_global_stats(df[night_mask])
    print(f"  motion p50={global_stats['motion_p50']:.1f}  p90={global_stats['motion_p90']:.1f}")
    print(f"  bstd   p50={global_stats['bstd_p50']:.1f}  p90={global_stats['bstd_p90']:.1f}")
    print(f"  ent_diff p50={global_stats['entropy_diff_p50']:.3f}  p90={global_stats['entropy_diff_p90']:.3f}")

    # ---- Step 4–6: Per-night analysis ----
    print("\n" + "=" * 70)
    print(f"  STEP 4–6: Analyzing {len(nights)} nights")
    print("=" * 70)

    summaries = []
    skipped_incomplete = 0
    for i, night_label in enumerate(nights):
        night_df = df[df["night"] == night_label].sort_values("abs_time").reset_index(drop=True)
        if len(night_df) < 10:
            print(f"  [{i+1:3d}/{len(nights)}] {night_label}  →  SKIP (only {len(night_df)} samples)")
            skipped_incomplete += 1
            continue

        night_df = compute_fai(night_df, global_stats)
        info = analyze_night(night_df, night_label)

        # Status line
        if not info["is_complete"]:
            status = "INCOMPLETE"
            skipped_incomplete += 1
        elif info["is_fishing"]:
            status = "FISH"
        else:
            status = "----"

        fft_str = ""
        if info.get("fft_period_min"):
            fft_str = f"  FFT={info['fft_period_min']:.0f}m SNR={info['fft_snr']:.1f}"
        print(f"  [{i+1:3d}/{len(nights)}] {night_label}  →  {status:10s}  "
              f"net={info['net_count']:2d}  raw_seg={info['n_segments_raw']:3d}  "
              f"act={info['activity_score']:.3f}  active={info['active_fraction']:.1%}"
              f"{fft_str}  "
              f"samples={info['n_samples']:4d}  span={info['duration_hours']:.1f}h  "
              f"cover={info['coverage_ratio']:.0%}  gap={info['max_gap_minutes']:.0f}m")

        if info.get("warnings"):
            for w in info["warnings"]:
                print(f"         [WARN] {w}")

        if not skip_plots:
            plot_night(night_df, night_label, info["is_fishing"],
                       info["net_count"], info.get("peak_indices", []),
                       info["is_complete"], info.get("warnings", []),
                       info.get("segments", []), info.get("minute_info"))

        # Store summary (exclude non-serializable fields)
        summary = {}
        for k, v in info.items():
            if k in ("peak_indices", "segments", "minute_info"):
                continue
            elif k == "warnings":
                summary[k] = "; ".join(v) if v else ""
            else:
                summary[k] = v
        summaries.append(summary)

    # ---- Step 7: Summary output ----
    print("\n" + "=" * 70)
    print("  STEP 7: Writing summary")
    print("=" * 70)

    df_summary = pd.DataFrame(summaries)
    if not df_summary.empty:
        df_summary = df_summary.sort_values("night").reset_index(drop=True)

    csv_path = CSV_DIR / "night_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}  ({len(df_summary)} nights)")

    json_path = CSV_DIR / "night_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)
    print(f"  wrote {json_path}")

    # ---- Quick stats ----
    complete_nights = df_summary[df_summary["is_complete"] == True] if not df_summary.empty else pd.DataFrame()
    fishing_nights  = complete_nights[complete_nights["is_fishing"] == True] if not complete_nights.empty else pd.DataFrame()
    total_nets = int(fishing_nights["net_count"].sum()) if not fishing_nights.empty else 0

    t1 = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total nights found:    {len(nights)}")
    print(f"  Incomplete (skipped):  {skipped_incomplete}")
    print(f"  Complete nights:       {len(complete_nights)}")
    print(f"  Fishing nights:        {len(fishing_nights)}")
    print(f"  Total net cycles:      {total_nets}")
    print(f"  Wall time:             {t1 - t0:.1f} s")
    print(f"  Output:                {NIGHT_OUT}")

    return df_summary


if __name__ == "__main__":
    run_pipeline()
