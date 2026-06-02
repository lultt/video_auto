"""Phase 4: Run netbag detection on known time windows in video files."""
import os
import cv2
import csv
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
OUT_DIR = "J:/video_auto/netbag/video_test"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(WEIGHTS)

# Video files and their wallclock start/end times (HHMMSS)
WINDOWS = [
    {
        "video": "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4",
        "wall_start": "205339",  # video covers 20:53:39 to 21:40:53
        "wall_end": "214053",
        "target_start": "211849",  # 21:18:49
        "target_end": "211905",    # 21:19:05
        "label": "window1",
    },
    {
        "video": "C:/video/0515/ch01_20250515_214053_222634topspeed.mp4",
        "wall_start": "214053",  # video covers 21:40:53 to 22:26:34
        "wall_end": "222634",
        "target_start": "221223",  # 22:12:23
        "target_end": "221300",    # 22:13:00
        "label": "window2",
    },
]


def hhmmss_to_sec(t):
    h, m, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
    return h * 3600 + m * 60 + s


def sec_to_hhmmss(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_window(win):
    video_path = win["video"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    wall_start_sec = hhmmss_to_sec(win["wall_start"])
    wall_end_sec = hhmmss_to_sec(win["wall_end"])
    wall_duration = wall_end_sec - wall_start_sec

    speed_ratio = wall_duration / video_duration

    target_start_sec = hhmmss_to_sec(win["target_start"])
    target_end_sec = hhmmss_to_sec(win["target_end"])

    offset_start = (target_start_sec - wall_start_sec) / speed_ratio
    offset_end = (target_end_sec - wall_start_sec) / speed_ratio

    frame_start = int(offset_start * fps)
    frame_end = int(offset_end * fps)

    print(f"\n{'='*60}")
    print(f"Window: {win['label']}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {video_duration:.1f}s")
    print(f"Speed ratio: {speed_ratio:.2f}x")
    print(f"Target: {sec_to_hhmmss(target_start_sec)} ~ {sec_to_hhmmss(target_end_sec)}")
    print(f"Frame range: {frame_start} ~ {frame_end}")
    print(f"{'='*60}")

    # Sample every 5 frames within the window
    step = 5
    results_data = []

    for frame_idx in range(frame_start, frame_end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        wall_time_sec = wall_start_sec + (frame_idx / fps) * speed_ratio
        frame_time = sec_to_hhmmss(wall_time_sec)

        preds = model.predict(frame, imgsz=1280, conf=0.25, verbose=False)

        for r in preds:
            if len(r.boxes) > 0:
                for box in r.boxes:
                    conf = box.conf[0].item()
                    xyxy = box.xyxy[0].tolist()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    results_data.append({
                        "frame_idx": frame_idx,
                        "frame_time": frame_time,
                        "confidence": round(conf, 4),
                        "bbox_area": round(area, 1),
                        "x1": round(xyxy[0], 1),
                        "y1": round(xyxy[1], 1),
                        "x2": round(xyxy[2], 1),
                        "y2": round(xyxy[3], 1),
                    })
            else:
                results_data.append({
                    "frame_idx": frame_idx,
                    "frame_time": frame_time,
                    "confidence": 0,
                    "bbox_area": 0,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                })

    cap.release()

    csv_path = os.path.join(OUT_DIR, f"{win['label']}_detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "frame_time", "confidence", "bbox_area", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(results_data)

    detected = [r for r in results_data if r["confidence"] > 0]
    print(f"\nResults: {len(detected)}/{len(results_data)} frames with detections")
    if detected:
        max_area = max(detected, key=lambda x: x["bbox_area"])
        print(f"Max bbox_area: {max_area['bbox_area']:.0f} at {max_area['frame_time']} (conf={max_area['confidence']:.3f})")
    print(f"Saved to: {csv_path}")

    return results_data


if __name__ == "__main__":
    for win in WINDOWS:
        process_window(win)
