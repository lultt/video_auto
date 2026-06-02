"""Shared I/O helpers: load Parquet files, build unified DataFrames, save tables."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..utils import parse_video_filename_timestamp


def load_parquet_folder(folder: Path, add_wall_time: bool = True) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(folder).glob("*.parquet")):
        df = pd.read_parquet(path)
        if add_wall_time and "wall_time" not in df.columns:
            start = parse_video_filename_timestamp(path)
            if start is not None:
                df["wall_time"] = df["timestamp_sec"].apply(
                    lambda v: start + timedelta(seconds=float(v))
                )
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files in {folder}")
    return pd.concat(frames, ignore_index=True).sort_values("wall_time")


def build_minute_series(
    df: pd.DataFrame,
    roi_order: list[str] | None = None,
    features: list[str] | None = None,
    freq: str = "1min",
    interp_limit: int = 5,
) -> pd.DataFrame:
    if roi_order is None:
        roi_order = ["left_deck", "center_deck", "right_deck"]
    if features is None:
        features = ["motion_intensity", "edge_density", "mean_brightness", "entropy"]

    pivots = []
    for roi in roi_order:
        sub = df[df["roi_name"] == roi]
        if sub.empty:
            continue
        sub = sub.set_index("wall_time").sort_index()
        minute = sub[features].resample(freq).mean()
        minute.columns = [f"{roi}_{c}" for c in minute.columns]
        pivots.append(minute)

    wide = pd.concat(pivots, axis=1)
    idx = pd.date_range(
        wide.index.min().floor(freq), wide.index.max().ceil(freq), freq=freq
    )
    return wide.reindex(idx).interpolate(limit=interp_limit).ffill().bfill()


def save_table(df: pd.DataFrame, path: Path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", **kwargs)
