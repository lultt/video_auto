"""
review_video_generator.py — cut review clips for core windows.

Given core_catch_windows.csv + source video index, produce:
  - cycle_{N}_core.mp4     (real-time, with overlays)
  - cycle_{N}_core_4x.mp4  (4× speed)
  - all_core_windows_4x.mp4 (concatenated with title cards)
  - manual_review_report.md
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_video_time_range(filepath: Path):
    m = re.search(r"ch\d+_(\d{8})_(\d{6})_(\d{6})", filepath.name)
    if not m:
        return None
    date = m.group(1)
    start = datetime.strptime(date + m.group(2), "%Y%m%d%H%M%S")
    end = datetime.strptime(date + m.group(3), "%Y%m%d%H%M%S")
    if end < start:
        end += timedelta(days=1)
    return start, end


def scan_source_videos(video_root: Path) -> list[dict]:
    """Index all source .mp4 files with time ranges."""
    videos = []
    for p in sorted(video_root.glob("*.mp4")):
        times = parse_video_time_range(p)
        if times:
            videos.append({"path": p, "start": times[0], "end": times[1], "name": p.name})
    return sorted(videos, key=lambda v: v["start"])


def find_videos_for_window(t_start: datetime, t_end: datetime, video_index: list[dict]) -> list[dict]:
    result = []
    for v in video_index:
        if t_start < v["end"] and t_end > v["start"]:
            result.append(v)
    return sorted(result, key=lambda v: v["start"])


def cut_core_window(cycle_row, video_index, out_dir: Path, fps: float = 25.0) -> str | None:
    cycle_id = int(cycle_row["cycle_id"])
    t_start = pd.Timestamp(cycle_row["core_start_time"]).to_pydatetime()
    t_end = pd.Timestamp(cycle_row["core_end_time"]).to_pydatetime()

    videos = find_videos_for_window(t_start, t_end, video_index)
    if not videos:
        print(f"  Cycle {cycle_id}: no source video found!")
        return None

    temp_dir = out_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    clips = []

    for i, v in enumerate(videos):
        clip_start = max(t_start, v["start"])
        clip_end = min(t_end, v["end"])
        start_offset = (clip_start - v["start"]).total_seconds()
        clip_duration = (clip_end - clip_start).total_seconds()

        clip_path = temp_dir / f"cycle_{cycle_id:02d}_part{i}.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_offset), "-t", str(clip_duration),
            "-i", str(v["path"]),
            "-c:v", "libx264", "-crf", "27", "-preset", "ultrafast", "-an", str(clip_path),
        ]
        subprocess.run(cmd, capture_output=True)
        if clip_path.exists():
            clips.append(str(clip_path))

    if not clips:
        return None

    if len(clips) > 1:
        concat_file = temp_dir / f"cycle_{cycle_id:02d}_concat.txt"
        with open(concat_file, "w") as f:
            for c in clips:
                f.write(f"file '{Path(c).absolute()}'\n")
        merged = temp_dir / f"cycle_{cycle_id:02d}_uncut.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(merged)], capture_output=True)
    else:
        merged = Path(clips[0])

    final_path = out_dir / f"cycle_{cycle_id:02d}_core.mp4"
    overlay = (
        f"drawtext=text='Cycle {cycle_id}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5,"
        f"drawtext=text='%{{localtime:%H\\:%M\\:%S}}':x=10:y=40:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5,"
        f"drawtext=text='Core Window':x=10:y=70:fontsize=18:fontcolor=yellow:box=1:boxcolor=black@0.5"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", str(merged),
        "-vf", overlay, "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-an", str(final_path),
    ], capture_output=True)
    print(f"  Cycle {cycle_id}: core clip saved, {(t_end - t_start).total_seconds() / 60:.1f} min")
    return str(final_path)


def make_4x_version(core_path: str) -> str | None:
    if not core_path or not Path(core_path).exists():
        return None
    p = Path(core_path)
    out = p.with_name(p.stem + "_4x.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", core_path, "-filter:v", "setpts=0.25*PTS",
        "-an", "-c:v", "libx264", "-crf", "23", str(out),
    ], capture_output=True)
    return str(out)


def generate_concat_overview(fourx_paths: list[str | None], cycles_df: pd.DataFrame, out_dir: Path) -> str | None:
    temp_dir = out_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    segments = []

    for i, clip_path in enumerate(fourx_paths):
        if not clip_path or not Path(clip_path).exists():
            continue
        cycle_idx = i + 1
        row = cycles_df[cycles_df["cycle_id"] == cycle_idx]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        title_video = temp_dir / f"title_{cycle_idx:02d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=black:s=1920x1080:d=2",
            "-vf",
            f"drawtext=text='Cycle {cycle_idx}':x=(w-text_w)/2:y=(h-text_h)/2-40:fontsize=48:fontcolor=white,"
            f"drawtext=text='{row['core_start_time'][11:16]} - {row['core_end_time'][11:16]}':"
            f"x=(w-text_w)/2:y=(h-text_h)/2+20:fontsize=32:fontcolor=white",
            str(title_video),
        ]
        subprocess.run(cmd, capture_output=True)
        segments.append(str(title_video))
        segments.append(clip_path)

    concat_file = temp_dir / "concat_all.txt"
    with open(concat_file, "w") as f:
        for s in segments:
            f.write(f"file '{Path(s).absolute()}'\n")

    out_path = out_dir / "all_core_windows_4x.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(out_path)], capture_output=True)
    print(f"  Concatenated overview: {out_path.name}")
    return str(out_path)


def generate_review_report(cycles_df: pd.DataFrame, out_dir: Path, confidence_rules: dict | None = None):
    if confidence_rules is None:
        confidence_rules = {
            "core_window_short_min": 15,
            "core_window_long_min": 30,
            "right_edge_increase_weak": 1.05,
            "cycle_long_min": 60,
        }

    scores = []
    for _, row in cycles_df.iterrows():
        issues = []
        core_dur = row["duration_core_window_min"]
        if core_dur < confidence_rules["core_window_short_min"]:
            issues.append("core window too short")
        if core_dur > confidence_rules["core_window_long_min"]:
            issues.append("core window too long")
        if row["right_edge_increase_vs_pre"] < confidence_rules["right_edge_increase_weak"]:
            issues.append("weak right-deck signal")
        if row["duration_total_cycle_min"] > confidence_rules["cycle_long_min"]:
            issues.append("abnormally long cycle")
        conf = "HIGH" if len(issues) == 0 else "MEDIUM" if len(issues) == 1 else "LOW"
        scores.append({"cycle_id": int(row["cycle_id"]), "confidence": conf, "issues": issues, "row": row})

    lines = [
        "# Net-cycle stage segmentation — manual review report",
        "",
        "## Summary",
        "",
        f"- Total cycles: {len(cycles_df)}",
        f"- Mean core-window duration: {cycles_df['duration_core_window_min'].mean():.1f} min",
        f"- HIGH confidence: {sum(1 for s in scores if s['confidence'] == 'HIGH')}",
        f"- MEDIUM confidence: {sum(1 for s in scores if s['confidence'] == 'MEDIUM')}",
        f"- LOW confidence: {sum(1 for s in scores if s['confidence'] == 'LOW')}",
        "",
        "## Per-cycle details",
        "",
    ]

    for cs in scores:
        r = cs["row"]
        lines.append(f"### Cycle {cs['cycle_id']}")
        lines.append(f"- **Confidence**: {cs['confidence']}")
        lines.append(f"- **Cycle duration**: {r['duration_total_cycle_min']:.0f} min")
        lines.append(f"- **Core window**: {r['duration_core_window_min']:.0f} min")
        lines.append(f"- **Net on deck**: {r['net_on_deck_time'][11:16]}")
        lines.append(f"- **Fish transfer end**: {r['fish_end_time'][11:16]}")
        lines.append(f"- **Right-edge increase**: x{r['right_edge_increase_vs_pre']:.2f}")
        if cs["issues"]:
            lines.append(f"- **Flags**: {', '.join(cs['issues'])}")
        lines.append("")

    lines.append("## Cycles needing review")
    for cs in scores:
        if cs["confidence"] != "HIGH":
            lines.append(f"- **Cycle {cs['cycle_id']}**: {', '.join(cs['issues'])}")

    with open(out_dir / "manual_review_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Report saved: manual_review_report.md")


def run_review_generator(
    core_windows_csv: Path,
    video_root: Path,
    out_dir: Path,
    confidence_rules: dict | None = None,
):
    """Main entry point: cut clips + generate overview + report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning source videos...")
    video_index = scan_source_videos(Path(video_root))
    print(f"  {len(video_index)} source videos found")

    cycles_df = pd.read_csv(core_windows_csv)
    print(f"  {len(cycles_df)} cycles loaded")

    print("Cutting core window clips...")
    core_paths = []
    for _, row in cycles_df.iterrows():
        core_paths.append(cut_core_window(row, video_index, out_dir))

    print("Creating 4x versions...")
    fourx_paths = [make_4x_version(p) for p in core_paths]

    print("Building concatenated overview...")
    generate_concat_overview(fourx_paths, cycles_df, out_dir)

    print("Generating review report...")
    generate_review_report(cycles_df, out_dir, confidence_rules)

    print(f"\nAll outputs in: {out_dir}")