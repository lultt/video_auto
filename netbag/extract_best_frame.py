"""Phase 5: Extract the frame with max bbox_area from each window."""
import os
import cv2
import csv
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
OUT_DIR = "J:/video_auto/netbag/video_test"
model = YOLO(WEIGHTS)

WINDOWS = [
    {
        "video": "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4",
        "wall_start": "205339",
        "csv": os.path.join(OUT_DIR, "window1_detections.csv"),
        "label": "window1",
    },
    {
        "video": "C:/video/0515/ch01_20250515_214053_222634topspeed.mp4",
        "wall_start": "214053",
        "csv": os.path.join(OUT_DIR, "window2_detections.csv"),
        "label": "window2",
    },
]


def hhmmss_to_sec(t):
    h, m, s = int(t[0:2]), int(t[2:4]), int(t[4:6])
    return h * 3600 + m * 60 + s


def process(win):
    with open(win["csv"], "r") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if float(r["confidence"]) > 0]

    if not rows:
        print(f"{win['label']}: No detections found")
        return

    best = max(rows, key=lambda r: float(r["bbox_area"]))
    frame_idx = int(best["frame_idx"])
    frame_time = best["frame_time"]
    bbox_area = float(best["bbox_area"])
    confidence = float(best["confidence"])

    cap = cv2.VideoCapture(win["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"{win['label']}: Failed to read frame {frame_idx}")
        return

    # Save raw frame
    raw_path = os.path.join(OUT_DIR, f"{win['label']}_best_frame.jpg")
    cv2.imwrite(raw_path, frame)

    # Save annotated frame
    preds = model.predict(frame, imgsz=1280, conf=0.25, verbose=False)
    annotated = preds[0].plot()
    ann_path = os.path.join(OUT_DIR, f"{win['label']}_best_frame_annotated.jpg")
    cv2.imwrite(ann_path, annotated)

    print(f"\n{win['label']}:")
    print(f"  timestamp:  {frame_time}")
    print(f"  frame_idx:  {frame_idx}")
    print(f"  bbox_area:  {bbox_area:.0f}")
    print(f"  confidence: {confidence:.4f}")
    print(f"  saved:      {raw_path}")
    print(f"  annotated:  {ann_path}")


if __name__ == "__main__":
    for win in WINDOWS:
        process(win)
