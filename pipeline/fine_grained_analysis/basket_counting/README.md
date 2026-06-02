"""
Basket counting — PRIORITY 2 of Layer 4.

Count fish baskets visible on the right deck during the fish-transfer stage.
This is a direct proxy for catch volume (commercial KPI).

Approach (planned, not implemented):
  1. For each cycle, sample frames evenly within fish_start→fish_end.
  2. Crop to right_deck ROI.
  3. Run YOLOv8n (or light detector) for "basket"/"box"/"container".
  4. Track basket IDs across frames to avoid double-counting.
  5. Output: basket_count_per_cycle.csv (cycle_id, max_baskets, mean_baskets).

INPUT:    core_catch_windows.csv  + source videos
OUTPUT:   outputs/<night>/fine/baskets/basket_count_per_cycle.csv
COST:     ~15 frames per cycle × YOLOv8n inference (fast)
STATUS:   stub
"""

import argparse


def run_basket_counting(core_windows_csv, video_root, out_dir):
    raise NotImplementedError(
        "Basket counting planned — run YOLOv8n on right_deck ROI during fish transfer."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-windows", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_basket_counting(args.core_windows, args.video_root, args.out_dir)