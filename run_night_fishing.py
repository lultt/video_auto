"""
run_night_fishing.py — Run the full night fishing detection pipeline.

Usage:
  python run_night_fishing.py                          # full scan (all parquets)
  python run_night_fishing.py --max-files 200          # quick test
  python run_night_fishing.py --threshold 0.10         # override FAI variance threshold
  python run_night_fishing.py --prominence 0.15        # override peak prominence
  python run_night_fishing.py --skip-plots             # no per-night PNGs (faster)

All parameters can also be edited in pipeline/night_fishing/config.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.night_fishing.pipeline import run_pipeline
from pipeline.night_fishing import config


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Night Fishing Detection Pipeline")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of parquet files loaded (for quick testing)")
    ap.add_argument("--skip-plots", action="store_true",
                    help="Skip per-night plot generation (much faster)")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"FAI variance threshold for fishing classification "
                         f"(default: {config.FAI_VAR_THRESHOLD})")
    ap.add_argument("--prominence", type=float, default=None,
                    help=f"Peak prominence for net cycle detection "
                         f"(default: {config.PEAK_PROMINENCE})")
    ap.add_argument("--distance", type=int, default=None,
                    help=f"Min samples between peaks (default: {config.PEAK_DISTANCE})")
    args = ap.parse_args()

    # Override config if CLI flags provided
    if args.threshold is not None:
        config.FAI_VAR_THRESHOLD = args.threshold
    if args.prominence is not None:
        config.PEAK_PROMINENCE = args.prominence
    if args.distance is not None:
        config.PEAK_DISTANCE = args.distance

    print(f"Configuration:")
    print(f"  FAI_VAR_THRESHOLD = {config.FAI_VAR_THRESHOLD}")
    print(f"  PEAK_PROMINENCE   = {config.PEAK_PROMINENCE}")
    print(f"  PEAK_DISTANCE     = {config.PEAK_DISTANCE}")
    print()

    run_pipeline(max_files=args.max_files, skip_plots=args.skip_plots)


if __name__ == "__main__":
    main()
