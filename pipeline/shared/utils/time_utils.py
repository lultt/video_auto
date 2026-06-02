"""Time helpers: parse video filenames, wall-clock conversion, night mask."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_video_filename_timestamp(path: Path | str) -> datetime | None:
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_", Path(path).stem)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def parse_video_time_range(path: Path | str) -> tuple[datetime, datetime] | None:
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_(\d{6})", Path(path).stem)
    if not m:
        return None
    date = m.group(1)
    start = datetime.strptime(date + m.group(2), "%Y%m%d%H%M%S")
    end = datetime.strptime(date + m.group(3), "%Y%m%d%H%M%S")
    if end < start:
        end += timedelta(days=1)
    return start, end


def timestamp_to_wall_time(path: Path | str, timestamp_sec: float) -> datetime:
    start = parse_video_filename_timestamp(path)
    if start is None:
        raise ValueError(f"Cannot parse timestamp from filename: {path}")
    return start + timedelta(seconds=timestamp_sec)


def build_night_mask(wall_times, night_start_str: str, night_end_str: str):
    if len(wall_times) == 0:
        return pd.Series(False, index=wall_times)
    first = wall_times[0]
    date_str = first.strftime("%Y-%m-%d") if hasattr(first, "strftime") else str(first)[:10]
    t_start = pd.Timestamp(f"{date_str} {night_start_str}:00")
    t_end = pd.Timestamp(f"{date_str} {night_end_str}:00")
    if t_end <= t_start:
        t_end += timedelta(days=1)
    return (wall_times >= t_start) & (wall_times <= t_end)