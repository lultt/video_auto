import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

FFMPEG_CANDIDATES = [
    r"D:\ffmpeg\bin\ffmpeg.exe",
    "ffmpeg",
]
FFPROBE_CANDIDATES = [
    r"D:\ffmpeg\bin\ffprobe.exe",
    "ffprobe",
]
VIDEO_PATTERN = re.compile(r"ch(\d+)_(\d{8})_(\d{6})_(\d{6})topspeed\.mp4$", re.IGNORECASE)
THUMB_WIDTH = 320
THUMB_HEIGHT = 180
DEFAULT_GRID_COLS = 20
DEFAULT_GRID_ROWS = 20


def pick_executable(candidates):
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            return candidate
    raise FileNotFoundError(f"Executable not found. Tried: {candidates}")


FFMPEG_PATH = pick_executable(FFMPEG_CANDIDATES)
FFPROBE_PATH = pick_executable(FFPROBE_CANDIDATES)


def parse_video_name(path):
    match = VIDEO_PATTERN.search(path.name)
    if not match:
        return None
    channel, date_str, start_str, end_str = match.groups()
    start_dt = datetime.strptime(f"{date_str}{start_str}", "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(f"{date_str}{end_str}", "%Y%m%d%H%M%S")
    if end_dt < start_dt:
        end_dt += timedelta(days=1)
    return {
        "channel": int(channel),
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


def find_video_for_window(video_dir, window_start_dt, window_end_dt, channel=1):
    candidates = []
    for path in sorted(Path(video_dir).glob("*.mp4")):
        parsed = parse_video_name(path)
        if not parsed or parsed["channel"] != channel:
            continue
        if parsed["start_dt"] <= window_start_dt and parsed["end_dt"] >= window_end_dt:
            candidates.append({"path": path, **parsed})
    if not candidates:
        raise FileNotFoundError("No source video fully covers the requested window.")
    return candidates[0]


def parse_rate(rate_str):
    return float(Fraction(rate_str))


def probe_video(video_path):
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,r_frame_rate,nb_frames,width,height:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}

    fps = parse_rate(stream["r_frame_rate"])
    frame_count = int(stream["nb_frames"])
    width = int(stream["width"])
    height = int(stream["height"])
    duration_sec = float(fmt.get("duration") or 0.0)
    codec = (stream.get("codec_name") or "").upper()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "codec": codec,
    }


def format_timestamp(dt):
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def ensure_clean_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def compute_laplacian_variance(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def build_contact_sheet(image_paths, output_path, cols=20, rows=20):
    max_images = cols * rows
    selected = image_paths[:max_images]
    canvas = np.zeros((rows * THUMB_HEIGHT, cols * THUMB_WIDTH, 3), dtype=np.uint8)

    for idx, image_path in enumerate(selected):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        thumb = cv2.resize(frame, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
        label = Path(image_path).stem.replace("frame_", "")
        cv2.putText(
            thumb,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        row = idx // cols
        col = idx % cols
        y0 = row * THUMB_HEIGHT
        x0 = col * THUMB_WIDTH
        canvas[y0:y0 + THUMB_HEIGHT, x0:x0 + THUMB_WIDTH] = thumb

    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def export_frames(args):
    window_start_dt = datetime.strptime(f"{args.date} {args.window_start}", "%Y-%m-%d %H:%M:%S")
    window_end_dt = datetime.strptime(f"{args.date} {args.window_end}", "%Y-%m-%d %H:%M:%S")
    source = find_video_for_window(args.video_dir, window_start_dt, window_end_dt, channel=args.channel)
    video_info = probe_video(source["path"])

    fps = video_info["fps"]
    offset_start_sec = (window_start_dt - source["start_dt"]).total_seconds()
    offset_end_sec = (window_end_dt - source["start_dt"]).total_seconds()
    start_frame_idx = int(math.floor(offset_start_sec * fps + 1e-9))
    end_frame_exclusive = int(math.floor(offset_end_sec * fps + 1e-9))
    expected_frames = end_frame_exclusive - start_frame_idx

    ensure_clean_dir(args.output_raw)
    ensure_clean_dir(args.output_top50)

    cap = cv2.VideoCapture(str(source["path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source['path']}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    timestamps_path = Path(args.output_raw) / "timestamps.csv"
    quality_path = Path(args.output_raw) / "frame_quality.csv"
    contact_sheet_path = Path(args.output_raw) / "contact_sheet.jpg"
    summary_path = Path(args.output_raw) / "summary.txt"

    records = []
    exported_paths = []
    actual_frame_indices = []

    with open(timestamps_path, "w", newline="", encoding="utf-8") as ts_file:
        writer = csv.writer(ts_file)
        writer.writerow(["frame_id", "timestamp", "filename"])

        current_frame_idx = start_frame_idx
        frame_id = 1
        while current_frame_idx < end_frame_exclusive:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_dt = source["start_dt"] + timedelta(seconds=current_frame_idx / fps)
            filename = f"frame_{frame_id:06d}.jpg"
            output_path = Path(args.output_raw) / filename
            sharpness = compute_laplacian_variance(frame)

            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            writer.writerow([frame_id, format_timestamp(timestamp_dt), filename])

            records.append(
                {
                    "frame_id": frame_id,
                    "timestamp": format_timestamp(timestamp_dt),
                    "filename": filename,
                    "sharpness": sharpness,
                    "source_frame_idx": current_frame_idx,
                }
            )
            exported_paths.append(output_path)
            actual_frame_indices.append(current_frame_idx)
            frame_id += 1
            current_frame_idx += 1

    cap.release()

    records_sorted = sorted(records, key=lambda item: item["sharpness"], reverse=True)
    top_records = records_sorted[: min(args.top_n, len(records_sorted))]
    top_names = {item["filename"] for item in top_records}

    with open(quality_path, "w", newline="", encoding="utf-8") as quality_file:
        writer = csv.writer(quality_file)
        writer.writerow(["frame_id", "sharpness"])
        for item in records:
            writer.writerow([item["frame_id"], f"{item['sharpness']:.6f}"])

    for item in top_records:
        src = Path(args.output_raw) / item["filename"]
        dst = Path(args.output_top50) / item["filename"]
        shutil.copy2(src, dst)

    build_contact_sheet(exported_paths, contact_sheet_path, cols=args.grid_cols, rows=args.grid_rows)

    actual_frames = len(records)
    actual_start = records[0]["timestamp"] if records else "N/A"
    actual_end = records[-1]["timestamp"] if records else "N/A"
    sharpest = records_sorted[0] if records_sorted else None
    blurriest = records_sorted[-1] if records_sorted else None

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"视频文件: {source['path']}\n")
        summary_file.write(f"fps: {fps:.6f}\n")
        summary_file.write(f"总帧数: {video_info['frame_count']}\n")
        summary_file.write(f"窗口时长: {(window_end_dt - window_start_dt).total_seconds():.3f}s\n")
        summary_file.write(f"理论帧数: {expected_frames}\n")
        summary_file.write(f"实际导出帧数: {actual_frames}\n")
        if sharpest:
            summary_file.write(
                f"最清晰帧: {sharpest['filename']} ({sharpest['sharpness']:.6f})\n"
            )
        if blurriest:
            summary_file.write(
                f"最模糊帧: {blurriest['filename']} ({blurriest['sharpness']:.6f})\n"
            )

    print(f"Video: {source['path'].name}")
    print(f"Window duration: {(window_end_dt - window_start_dt).total_seconds():.3f}s")
    print(f"FPS: {fps:.6f}")
    print(f"Expected frame count: {expected_frames}")
    print(f"Frames exported: {actual_frames}")
    print("")
    print(f"Start: {actual_start}")
    print(f"End: {actual_end}")
    print(f"FPS: {fps:.6f}")
    print(f"contact_sheet: {contact_sheet_path}")
    print(f"top50_dir: {Path(args.output_top50)}")

    return {
        "video": str(source["path"]),
        "fps": fps,
        "expected_frames": expected_frames,
        "actual_frames": actual_frames,
        "start": actual_start,
        "end": actual_end,
        "contact_sheet": str(contact_sheet_path),
        "top50_dir": str(Path(args.output_top50)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Export full-resolution frames for net dataset extraction")
    parser.add_argument("--video-dir", default=r"C:\video\0515")
    parser.add_argument("--output-raw", default=r"C:\video\net_dataset\0515_first_net_raw")
    parser.add_argument("--output-top50", default=r"C:\video\net_dataset\0515_first_net_top50")
    parser.add_argument("--date", default="2025-05-15")
    parser.add_argument("--window-start", default="21:18:49")
    parser.add_argument("--window-end", default="21:19:05")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_ROWS)
    return parser


if __name__ == "__main__":
    export_frames(build_parser().parse_args())
