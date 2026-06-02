"""
extract_keyframes.py — sample sparse keyframes from core catch windows.

Reads core_catch_windows.csv, matches wall-clock timestamps to source video
files on NAS, and extracts single frames via ffmpeg (no full decode).

Sampling strategy (3 groups per cycle, ~10 frames total):
  net_landing    — around net_on_deck_time, captures max net area
  net_settled    — between net_on_deck and fish_start, most stable
  fish_transfer  — during fish emptying, tracks area shrinkage

Output: keyframe_index.csv + per-cycle PNG directories.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__:
    from ...shared.utils import load_pipeline_config, parse_video_time_range
else:
    # Allow running as standalone script
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parent.parent.parent.parent))
    from pipeline.shared.utils import load_pipeline_config, parse_video_time_range

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

# ---------------------------------------------------------------------------
# Default sampling config (overridden by pipeline.yaml)
# ---------------------------------------------------------------------------
DEFAULT_SAMPLING = {
    "net_landing":   {"window_before_sec": 5,  "window_after_sec": 30, "interval_sec": 15},
    "net_settled":   {"center_offset_pct": 0.5, "count": 2, "interval_sec": 8},
    "fish_transfer": {"edge_trim_sec": 30, "count": 5},
}


def scan_source_videos(video_root: Path) -> list[dict]:
    """Index all source .mp4 files with their wall-clock time ranges."""
    videos = []
    for p in sorted(video_root.rglob("*.mp4")):
        times = parse_video_time_range(p)
        if times:
            videos.append({"path": p, "start": times[0], "end": times[1], "name": p.name})
    return sorted(videos, key=lambda v: v["start"])


def find_video_for_time(wall_time: datetime, video_index: list[dict]) -> dict | None:
    """Return the video whose time range covers *wall_time*, or None."""
    for v in video_index:
        if v["start"] <= wall_time < v["end"]:
            return v
    # If wall_time is exactly at a boundary, try nearby
    for v in video_index:
        if abs((wall_time - v["start"]).total_seconds()) < 120:
            return v
        if abs((wall_time - v["end"]).total_seconds()) < 120:
            return v
    return None


def compute_sample_timestamps(cycle_row: pd.Series, sampling_cfg: dict | None = None) -> list[dict]:
    """
    Per cycle, return a list of {group, wall_time, label} dicts.

    Groups:
      - net_landing:   around net_on_deck_time
      - net_settled:   between net_on_deck and fish_start
      - fish_transfer: during fish emptying
    """
    if sampling_cfg is None:
        sampling_cfg = DEFAULT_SAMPLING

    net_on_deck = pd.Timestamp(cycle_row["net_on_deck_time"]).to_pydatetime()
    fish_start = pd.Timestamp(cycle_row["fish_start_time"]).to_pydatetime()
    fish_end = pd.Timestamp(cycle_row["fish_end_time"]).to_pydatetime()

    timestamps = []

    # --- net_landing ---
    nl = sampling_cfg.get("net_landing", DEFAULT_SAMPLING["net_landing"])
    t0 = net_on_deck - timedelta(seconds=nl.get("window_before_sec", 5))
    t1 = net_on_deck + timedelta(seconds=nl.get("window_after_sec", 30))
    interval = timedelta(seconds=nl.get("interval_sec", 15))
    t = t0
    while t <= t1:
        timestamps.append({"group": "net_landing", "wall_time": t})
        t += interval

    # --- net_settled ---
    if fish_start > net_on_deck:
        ns = sampling_cfg.get("net_settled", DEFAULT_SAMPLING["net_settled"])
        midpoint = net_on_deck + (fish_start - net_on_deck) * ns.get("center_offset_pct", 0.5)
        count = ns.get("count", 2)
        interval_s = ns.get("interval_sec", 8)
        for i in range(count):
            offset = (i - (count - 1) / 2) * interval_s
            timestamps.append({"group": "net_settled", "wall_time": midpoint + timedelta(seconds=offset)})

    # --- fish_transfer ---
    ft = sampling_cfg.get("fish_transfer", DEFAULT_SAMPLING["fish_transfer"])
    edge_trim = timedelta(seconds=ft.get("edge_trim_sec", 30))
    t_start = fish_start + edge_trim
    t_end = fish_end - edge_trim
    if t_end > t_start:
        count = ft.get("count", 5)
        span = (t_end - t_start).total_seconds()
        step = max(1, span / (count - 1)) if count > 1 else span
        for i in range(count):
            timestamps.append({
                "group": "fish_transfer",
                "wall_time": t_start + timedelta(seconds=i * step),
            })

    return timestamps


def extract_one_frame(
    video_path: Path,
    offset_sec: float,
    out_full: Path,
    out_roi: Path,
    roi_cfg: dict,
    resize_width: int = 0,
) -> dict | None:
    """
    Extract a single frame from *video_path* at *offset_sec*.

    Saves the full resized frame to *out_full* and the cropped left_deck
    ROI to *out_roi*. Returns metadata dict or None on failure.
    """
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_roi.parent.mkdir(parents=True, exist_ok=True)

    # Extract full frame via ffmpeg
    cmd = [
        FFMPEG_PATH, "-y",
        "-ss", f"{offset_sec:.3f}",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",
    ]
    if resize_width > 0:
        cmd += ["-vf", f"scale={resize_width}:-1"]
    cmd.append(str(out_full))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not out_full.exists():
        return None

    img = cv2.imread(str(out_full))
    if img is None:
        return None

    h, w = img.shape[:2]

    # Crop left_deck ROI
    x_min = roi_cfg.get("x_min", 0.0)
    x_max = roi_cfg.get("x_max", 0.333)
    y_min = roi_cfg.get("y_min", 0.4)
    y_max = roi_cfg.get("y_max", 1.0)

    x1, x2 = int(x_min * w), int(x_max * w)
    y1, y2 = int(y_min * h), int(y_max * h)
    roi = img[y1:y2, x1:x2]
    cv2.imwrite(str(out_roi), roi)

    return {
        "image_w": w,
        "image_h": h,
        "roi_w": x2 - x1,
        "roi_h": y2 - y1,
    }


def run_extract_keyframes(
    core_windows_csv: Path,
    video_root: Path,
    out_dir: Path,
    cfg: dict | None = None,
    resize_width: int | None = None,
) -> Path:
    """
    Extract keyframes for all cycles in *core_windows_csv*.

    Returns path to keyframe_index.csv.
    """
    if cfg is None:
        cfg = {}
    coarse_cfg = cfg.get("coarse_features", {})
    roi_cfg = coarse_cfg.get("rois", {}).get("left_deck", {
        "x_min": 0.0, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0,
    })
    # Use explicit resize_width override, or read from config, default to 0 (native)
    if resize_width is None:
        resize_width = coarse_cfg.get("resize_width", 0)

    fine_cfg = cfg.get("fine_grained", {}).get("net_size_estimation", {})
    sampling_cfg = fine_cfg.get("sampling", DEFAULT_SAMPLING)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycles_df = pd.read_csv(core_windows_csv)
    video_index = scan_source_videos(Path(video_root))
    print(f"Found {len(video_index)} source videos under {video_root}")

    if not video_index:
        raise FileNotFoundError(f"No parsable .mp4 files under {video_root}")

    index_rows = []
    total_extracted = 0
    total_attempted = 0

    for _, cycle_row in cycles_df.iterrows():
        cycle_id = int(cycle_row["cycle_id"])
        timestamps = compute_sample_timestamps(cycle_row, sampling_cfg)

        cycle_dir = out_dir / f"cycle_{cycle_id:02d}"
        ok_count = 0

        for i, ts in enumerate(timestamps):
            total_attempted += 1
            video = find_video_for_time(ts["wall_time"], video_index)
            if video is None:
                index_rows.append({
                    "cycle_id": cycle_id, "frame_id": f"{cycle_id:02d}_{i:03d}",
                    "group": ts["group"],
                    "wall_time": ts["wall_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "video_path": "", "video_offset_sec": float("nan"),
                    "full_frame_path": "", "roi_path": "",
                    "status": "no_video",
                })
                continue

            offset_sec = max(0, (ts["wall_time"] - video["start"]).total_seconds())
            frame_id = f"{cycle_id:02d}_{i:03d}"
            full_path = cycle_dir / f"frame_{frame_id}_full.png"
            roi_path = cycle_dir / f"frame_{frame_id}_roi.png"

            meta = extract_one_frame(video["path"], offset_sec, full_path, roi_path, roi_cfg, resize_width)

            if meta is None:
                index_rows.append({
                    "cycle_id": cycle_id, "frame_id": frame_id,
                    "group": ts["group"],
                    "wall_time": ts["wall_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "video_path": str(video["path"]),
                    "video_offset_sec": round(offset_sec, 2),
                    "full_frame_path": "", "roi_path": "",
                    "status": "extract_failed",
                })
                continue

            index_rows.append({
                "cycle_id": cycle_id, "frame_id": frame_id,
                "group": ts["group"],
                "wall_time": ts["wall_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "video_path": str(video["path"]),
                "video_offset_sec": round(offset_sec, 2),
                "full_frame_path": str(full_path),
                "roi_path": str(roi_path),
                "image_w": meta["image_w"], "image_h": meta["image_h"],
                "roi_w": meta["roi_w"], "roi_h": meta["roi_h"],
                "status": "ok",
            })
            ok_count += 1

        print(f"  Cycle {cycle_id:2d}: {ok_count}/{len(timestamps)} frames extracted "
              f"({cycle_row['core_start_time'][11:16]} → {cycle_row['core_end_time'][11:16]})")
        total_extracted += ok_count

    index_df = pd.DataFrame(index_rows)
    index_csv = out_dir / "keyframe_index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8-sig")
    ok = (index_df["status"] == "ok").sum()
    no_vid = (index_df["status"] == "no_video").sum()
    failed = (index_df["status"] == "extract_failed").sum()

    print(f"\nDone. {ok} ok, {no_vid} no-video, {failed} failed "
          f"(of {len(index_df)} attempted keyframes across {len(cycles_df)} cycles)")
    print(f"Index: {index_csv}")

    return index_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract keyframes from core catch windows")
    p.add_argument("--core-windows", default="outputs/coarse_0515/core_catch_windows/core_catch_windows.csv")
    p.add_argument("--video-root", default="//DS224plus/video/0515")
    p.add_argument("--out-dir", default="outputs/coarse_0515/fine/net_size/keyframes")
    p.add_argument("--config", default=None, help="path to pipeline.yaml")
    p.add_argument("--resize-width", type=int, default=None,
                   help="override resize width for keyframe extraction (0 = native resolution)")
    args = p.parse_args()

    cfg = load_pipeline_config(args.config) if args.config else {}

    # Override video root from config if available
    if not cfg:
        cfg = {}
    paths = cfg.get("paths", {})
    video_root = args.video_root or paths.get("night_video_root", args.video_root)
    night_tag = cfg.get("night_tag", "0515")
    if args.out_dir == "outputs/coarse_0515/fine/net_size/keyframes":
        out_root = paths.get("output_root", "outputs")
        out_dir = f"{out_root}/coarse_{night_tag}/fine/net_size/keyframes"
    else:
        out_dir = args.out_dir

    run_extract_keyframes(
        core_windows_csv=Path(args.core_windows),
        video_root=Path(video_root),
        out_dir=Path(out_dir),
        cfg=cfg,
        resize_width=args.resize_width,
    )