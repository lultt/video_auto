"""video_scanner.py — scan NAS directory for MP4 files, produce a manifest CSV."""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_filename(filename: str) -> dict | None:
    pattern = r"ch(\d+)_(\d{8})_(\d{6})_(\d{6})topspeed\.mp4"
    m = re.match(pattern, filename)
    if not m:
        return None
    ch, date_str, start_str, end_str = m.groups()
    try:
        start_time = datetime.strptime(f"{date_str}_{start_str}", "%Y%m%d_%H%M%S")
        end_time = datetime.strptime(f"{date_str}_{end_str}", "%Y%m%d_%H%M%S")
        if end_time < start_time:
            end_time += timedelta(days=1)
    except ValueError:
        return None
    return {
        "channel": int(ch),
        "start_time": start_time,
        "end_time": end_time,
        "duration_est_min": (end_time - start_time).total_seconds() / 60.0,
    }


def scan_videos(video_root: str, output_path: str = "data/video_manifest.csv") -> pd.DataFrame:
    print(f"Scanning video directory: {video_root}")
    records = []
    video_root = Path(video_root)

    for f in sorted(video_root.rglob("*.mp4")):
        info = parse_filename(f.name)
        size_mb = f.stat().st_size / (1024 * 1024)
        record = {
            "filename": f.name,
            "path": str(f),
            "size_mb": round(size_mb, 1),
            "channel": info["channel"] if info else None,
            "start_time": info["start_time"] if info else None,
            "end_time": info["end_time"] if info else None,
            "duration_est_min": round(info["duration_est_min"], 1) if info else None,
        }
        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Scan complete: {len(df)} videos")
    print(f"Total size: {df['size_mb'].sum() / 1024:.1f} GB")
    valid = df["duration_est_min"].dropna()
    if len(valid):
        print(f"Duration range: {valid.min():.0f} – {valid.max():.0f} min")
    print(f"Manifest saved: {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scan video directory for MP4 files")
    parser.add_argument("video_root", help="Root directory to scan")
    parser.add_argument("--output", "-o", default="data/video_manifest.csv")
    args = parser.parse_args()
    scan_videos(args.video_root, args.output)
