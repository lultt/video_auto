"""
Scan all NAS videos, compute durations, estimate processing time.
Outputs processing_benchmark.csv + nights_summary.csv + estimated_total_runtime.txt
"""
import re, time, os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

VIDEO_ROOT = Path(r"\\DS224plus\video\viedeo")
OUT_DIR = Path(r"J:\video_auto\reports")
REALTIME_RATIO_COARSE = 60   # coarse pipeline ~60x realtime (verified)


def parse_filename(filepath: Path):
    """Parse ch{N}_{date}_{start}_{end}topspeed.mp4 → start, end, duration_sec."""
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_(\d{6})", filepath.stem)
    if not m:
        return None, None, None
    date = m.group(1)
    start = datetime.strptime(date + m.group(2), "%Y%m%d%H%M%S")
    end = datetime.strptime(date + m.group(3), "%Y%m%d%H%M%S")
    if end < start:
        end += timedelta(days=1)
    duration_sec = (end - start).total_seconds()
    return start, end, duration_sec


print(f"Scanning {VIDEO_ROOT} ...")
t0 = time.time()

records = []
for f in sorted(VIDEO_ROOT.glob("*.mp4")):
    start, end, dur = parse_filename(f)
    size_mb = round(f.stat().st_size / (1024*1024), 1)
    est_proc_min = round(dur / 60 / REALTIME_RATIO_COARSE, 2) if dur else 0
    records.append({
        "video": f.name,
        "channel": int(f.name[2:4]) if re.match(r"ch\d{2}_", f.name) else 0,
        "start_time": start.strftime("%Y-%m-%d %H:%M:%S") if start else "",
        "end_time": end.strftime("%Y-%m-%d %H:%M:%S") if end else "",
        "duration_hours": round(dur / 3600, 3) if dur else 0,
        "duration_min": round(dur / 60, 1) if dur else 0,
        "size_mb": size_mb,
        "estimated_proc_min": est_proc_min,
        "realtime_ratio": round(REALTIME_RATIO_COARSE, 0),
        "date": start.strftime("%Y%m%d") if start else "",
    })

elapsed = time.time() - t0
print(f"Scanned {len(records)} videos in {elapsed:.1f}s")

df = pd.DataFrame(records)

# ---- processing_benchmark.csv ----
benchmark = df[["video", "duration_hours", "estimated_proc_min", "realtime_ratio"]].copy()
benchmark.columns = ["video", "duration_hours", "processing_minutes", "realtime_ratio"]
benchmark.to_csv(OUT_DIR / "processing_benchmark.csv", index=False, encoding="utf-8-sig")
print(f"Saved: {OUT_DIR / 'processing_benchmark.csv'} ({len(benchmark)} rows)")

# ---- nights_summary.csv ----
nightly = df.groupby("date").agg(
    videos=("video", "count"),
    total_hours=("duration_hours", "sum"),
    total_size_gb=("size_mb", lambda x: round(x.sum() / 1024, 1)),
    first_video=("start_time", "min"),
    last_video=("end_time", "max"),
).reset_index()
nightly["estimated_proc_hours"] = round(nightly["total_hours"] / REALTIME_RATIO_COARSE, 2)
nightly["estimated_proc_hours_4workers"] = round(nightly["estimated_proc_hours"] / 4, 2)
nightly.columns = [
    "date", "videos", "total_hours", "total_size_gb",
    "first_video", "last_video", "est_proc_hours_1worker", "est_proc_hours_4workers",
]
nightly.to_csv(OUT_DIR / "nights_summary.csv", index=False, encoding="utf-8-sig")
print(f"Saved: {OUT_DIR / 'nights_summary.csv'} ({len(nightly)} nights)")

# ---- estimated_total_runtime.txt ----
total_hours = df["duration_hours"].sum()
total_days = total_hours / 24
est_1worker = total_hours / REALTIME_RATIO_COARSE
est_4workers = est_1worker / 4
total_size_gb = df["size_mb"].sum() / 1024

report = f"""# Video Processing Estimate

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Source: {VIDEO_ROOT}

## Raw Video Stats
  Total videos:       {len(df):,}
  Total duration:     {total_hours:,.1f} hours ({total_days:.1f} days)
  Total size:         {total_size_gb:,.0f} GB
  Date range:         {nightly['date'].iloc[0]} → {nightly['date'].iloc[-1]}
  Nights with video:  {len(nightly)}

## Processing Estimate (coarse pipeline, ~60x realtime)
  1 worker:           {est_1worker:.1f} hours ({est_1worker/24:.1f} days)
  4 workers:           {est_4workers:.1f} hours ({est_4workers/24:.1f} days)

## Nightly Summary
  {"date":10s} {"videos":>7s} {"hours":>8s} {"est_4w":>7s}
"""
for _, n in nightly.iterrows():
    report += f"  {n['date']:10s} {n['videos']:7d} {n['total_hours']:8.1f} {n['est_proc_hours_4workers']:7.1f}\n"

report += f"""
## Recommendation
  Run 4 workers → {est_4workers:.1f} hours total
  Can process ~{len(nightly)/est_4workers*24 if est_4workers > 0 else 0:.1f} nights/day
"""

with open(OUT_DIR / "estimated_total_runtime.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"Saved: {OUT_DIR / 'estimated_total_runtime.txt'}")
print(report)
