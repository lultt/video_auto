"""
run_night.py — unified entry point: run the full three-layer pipeline
on a single night of deck video.

  L1: coarse feature extraction → parquet
  L2: cycle detection            → detected_cycles.csv
  L3: core window extraction     → core_catch_windows.csv

Usage:
  python pipeline/run_night.py --config pipeline/shared/configs/pipeline.yaml
  python pipeline/run_night.py --night-tag 0515 --video-root //DS224plus/video/0515 --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from pipeline.shared.utils import load_pipeline_config
from pipeline.shared.io.video_scanner import scan_videos
from pipeline.coarse_features.feature_merge import run_coarse_pipeline_from_config
from pipeline.cycle_detection.detect_cycles import run_cycle_detection
from pipeline.core_window_detection.detect_core_windows import run_core_window_detection


def run_night(
    config_path: str | None = None,
    night_tag: str | None = None,
    video_root: str | None = None,
    skip_l1: bool = False,
    max_videos: int | None = None,
    num_workers: int | None = None,
) -> dict:
    """Run Layer 1 → Layer 2 → Layer 3 for one night.

    Returns a dict with paths to all output tables.
    """
    cfg = load_pipeline_config(config_path)
    paths_cfg = cfg.get("paths", {})
    output_root = paths_cfg.get("output_root", "outputs")
    night = night_tag or cfg.get("night_tag", "default")
    video_root = video_root or paths_cfg.get("video_root", "")

    # Night-level output dir
    night_out = Path(output_root) / f"coarse_{night}"
    l1_parquet_dir = night_out
    l2_dir = night_out / "net_cycle_detection"
    l3_dir = night_out / "core_catch_windows"

    t_total = time.perf_counter()

    # ---- L1: Coarse features ----
    if not skip_l1:
        print("\n" + "=" * 60)
        print(f"  LAYER 1: Coarse Feature Extraction (night={night})")
        print("=" * 60)
        run_coarse_pipeline_from_config(
            config_path=config_path,
            video_root=video_root,
            out_dir=str(l1_parquet_dir),
            max_videos=max_videos,
            num_workers=num_workers,
        )
    else:
        print(f"\n  L1 skipped — reusing parquet at {l1_parquet_dir}")

    # ---- L2: Cycle detection ----
    print("\n" + "=" * 60)
    print(f"  LAYER 2: Cycle Detection")
    print("=" * 60)
    l2_params = cfg.get("cycle_detection", {})
    l2_result = run_cycle_detection(
        parquet_root=l1_parquet_dir,
        out_dir=l2_dir,
        **{k: v for k, v in l2_params.items() if k not in ("enabled",)},
    )
    cycles_csv = l2_dir / "detected_cycles.csv"

    # ---- L3: Core window extraction ----
    print("\n" + "=" * 60)
    print(f"  LAYER 3: Core Window Extraction")
    print("=" * 60)
    l3_params = cfg.get("core_window_detection", {})
    results_df = run_core_window_detection(
        parquet_root=l1_parquet_dir,
        cycles_csv=cycles_csv,
        out_dir=l3_dir,
        **{k: v for k, v in l3_params.items() if k not in ("enabled",)},
    )
    core_csv = l3_dir / "core_catch_windows.csv"

    elapsed = time.perf_counter() - t_total
    print("\n" + "=" * 60)
    print(f"  Pipeline complete: {elapsed:.0f}s total")
    print(f"  Cycles:  {l2_dir / 'detected_cycles.csv'}")
    print(f"  Windows: {core_csv}")
    print("=" * 60)

    return {
        "parquet_dir": str(l1_parquet_dir),
        "cycles_csv": str(cycles_csv),
        "core_windows_csv": str(core_csv),
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full night pipeline (L1→L2→L3)")
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml")
    parser.add_argument("--night-tag", default=None, help="Night identifier (e.g. 0515)")
    parser.add_argument("--video-root", default=None, help="Path to night video directory")
    parser.add_argument("--skip-l1", action="store_true", help="Skip feature extraction (use existing parquet)")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_night(
        config_path=args.config, night_tag=args.night_tag,
        video_root=args.video_root, skip_l1=args.skip_l1,
        max_videos=args.max_videos, num_workers=args.workers,
    )