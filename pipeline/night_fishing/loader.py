# loader.py — recursive parquet scan & efficient loading

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .config import OUTPUT_ROOT, ROIS

# ch01_YYYYMMDD_HHMMSS_HHMMSS
_FNAME_RE = re.compile(r"ch\d+_(\d{8})_(\d{6})_(\d{6})")


def parse_video_start(video_id: str) -> Optional[datetime]:
    """Extract the recording start time from the video filename."""
    m = _FNAME_RE.search(str(video_id))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def scan_parquet_files(root: Optional[Path] = None) -> list:
    """Recursively find all .parquet files under the output tree."""
    root = Path(root) if root else OUTPUT_ROOT
    files = sorted(root.rglob("*.parquet"))
    # exclude debug / test dirs
    files = [f for f in files if "debug" not in str(f).lower()
             and "test" not in str(f).lower()]
    return files


def load_one_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Load a single parquet, keep only left_deck + right_deck, add abs_time."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [SKIP] {path.name}: {e}")
        return None
    if df.empty:
        return None
    # filter ROIs
    if "roi_name" in df.columns:
        df = df[df["roi_name"].isin(ROIS)].copy()
    if df.empty:
        return None

    video_id = df["video_id"].iloc[0]
    start = parse_video_start(video_id)
    if start is not None:
        df["abs_time"] = [
            start + timedelta(seconds=float(s))
            for s in df["timestamp_sec"]
        ]
    else:
        df["abs_time"] = pd.NaT

    # keep only essential columns
    keep = [
        "video_id", "abs_time", "timestamp_sec", "roi_name",
        "mean_brightness", "brightness_std", "mean_b", "mean_g", "mean_r",
        "motion_intensity", "edge_density", "laplacian_variance", "entropy",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def aggregate_rois(df: pd.DataFrame) -> pd.DataFrame:
    """Average left_deck + right_deck into a single row per timestamp.

    Returns a DataFrame indexed by abs_time with averaged feature columns.
    """
    feat_cols = [
        "mean_brightness", "brightness_std", "mean_b", "mean_g", "mean_r",
        "motion_intensity", "edge_density", "laplacian_variance", "entropy",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    grouped = df.groupby(["abs_time", "video_id", "timestamp_sec"], dropna=False)
    avg = grouped[feat_cols].mean().reset_index()
    avg = avg.sort_values("abs_time").reset_index(drop=True)
    return avg


def load_all(root: Optional[Path] = None, max_files: Optional[int] = None) -> pd.DataFrame:
    """Load all parquets, aggregate ROIs, return a single sorted DataFrame."""
    files = scan_parquet_files(root)
    if max_files:
        files = files[:max_files]
    print(f"Scanning {len(files)} parquet files...")

    parts = []
    for i, f in enumerate(files):
        df = load_one_parquet(f)
        if df is not None:
            parts.append(df)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(files)}] loaded, {sum(len(p) for p in parts):,} rows so far")

    if not parts:
        raise SystemExit("No parquet files loaded.")

    big = pd.concat(parts, ignore_index=True)
    print(f"\n  Total raw rows: {len(big):,}  |  videos: {big['video_id'].nunique()}")
    print(f"  Time range: {big['abs_time'].min()}  →  {big['abs_time'].max()}")

    # Aggregate left + right ROIs
    avg = aggregate_rois(big)
    print(f"  After ROI averaging: {len(avg):,} rows")
    return avg
