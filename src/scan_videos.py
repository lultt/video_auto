import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config


def parse_filename(filename):
    pattern = r"ch(\d+)_(\d{8})_(\d{6})_(\d{6})topspeed\.mp4"
    m = re.match(pattern, filename)
    if not m:
        return None
    ch, date_str, start_str, end_str = m.groups()
    try:
        start_time = datetime.strptime(f"{date_str}_{start_str}", "%Y%m%d_%H%M%S")
        end_time = datetime.strptime(f"{date_str}_{end_str}", "%Y%m%d_%H%M%S")
        if end_time < start_time:
            end_date = datetime.strptime(date_str, "%Y%m%d")
            from datetime import timedelta
            end_time = datetime.strptime(f"{end_str}", "%H%M%S").replace(
                year=end_date.year, month=end_date.month, day=end_date.day
            ) + timedelta(days=1)
    except ValueError:
        return None
    return {
        "channel": int(ch),
        "start_time": start_time,
        "end_time": end_time,
        "duration_est_min": (end_time - start_time).total_seconds() / 60.0,
    }


def scan_videos(video_root, output_path="data/video_manifest.csv"):
    print(f"扫描视频目录: {video_root}")
    records = []
    video_root = Path(video_root)

    for f in sorted(video_root.iterdir()):
        if not f.suffix.lower() == ".mp4":
            continue
        info = parse_filename(f.name)
        size_mb = f.stat().st_size / (1024 * 1024)
        record = {
            "filename": f.name,
            "path": str(f),
            "size_mb": round(size_mb, 1),
        }
        if info:
            record.update(info)
        else:
            record.update({"channel": None, "start_time": None, "end_time": None, "duration_est_min": None})
        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"扫描完成: {len(df)} 个视频")
    print(f"总大小: {df['size_mb'].sum() / 1024:.1f} GB")
    if "duration_est_min" in df.columns:
        valid = df["duration_est_min"].dropna()
        print(f"时长范围: {valid.min():.0f} - {valid.max():.0f} 分钟")
    print(f"清单保存: {output_path}")
    return df


if __name__ == "__main__":
    cfg = load_config()
    scan_videos(cfg["video_root"], os.path.join(cfg["data_dir"], "video_manifest.csv"))
