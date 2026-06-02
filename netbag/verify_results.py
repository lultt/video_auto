"""Verify YOLO results: Top10 by score, annotated images, scatter plot."""
import os
import csv
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
OUT_DIR = "J:/video_auto/netbag/video_test"
model = YOLO(WEIGHTS)

WINDOWS = [
    {
        "video": "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4",
        "csv": os.path.join(OUT_DIR, "window1_detections.csv"),
        "label": "window1",
    },
    {
        "video": "C:/video/0515/ch01_20250515_214053_222634topspeed.mp4",
        "csv": os.path.join(OUT_DIR, "window2_detections.csv"),
        "label": "window2",
    },
]


def process(win):
    with open(win["csv"], "r") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if float(r["confidence"]) > 0]

    for r in rows:
        r["bbox_area"] = float(r["bbox_area"])
        r["confidence"] = float(r["confidence"])
        r["score"] = r["bbox_area"] * r["confidence"]
        r["frame_idx"] = int(r["frame_idx"])

    rows.sort(key=lambda x: x["score"], reverse=True)
    top10 = rows[:10]

    # 1. Save Top10 CSV
    top10_csv = os.path.join(OUT_DIR, f"{win['label']}_top10.csv")
    with open(top10_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "timestamp", "frame_idx", "bbox_area", "confidence", "score"])
        for i, r in enumerate(top10, 1):
            writer.writerow([i, r["frame_time"], r["frame_idx"],
                             f"{r['bbox_area']:.0f}", f"{r['confidence']:.4f}", f"{r['score']:.0f}"])

    print(f"\n{'='*60}")
    print(f"{win['label']} Top10 by score (bbox_area * confidence)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Timestamp':<12} {'bbox_area':>10} {'confidence':>11} {'score':>10}")
    print("-" * 55)
    for i, r in enumerate(top10, 1):
        print(f"{i:<5} {r['frame_time']:<12} {r['bbox_area']:>10.0f} {r['confidence']:>11.4f} {r['score']:>10.0f}")

    # 2. Export Top10 annotated images
    img_dir = os.path.join(OUT_DIR, f"{win['label']}_top10")
    os.makedirs(img_dir, exist_ok=True)

    cap = cv2.VideoCapture(win["video"])
    for i, r in enumerate(top10, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_idx"])
        ret, frame = cap.read()
        if not ret:
            continue
        preds = model.predict(frame, imgsz=1280, conf=0.25, verbose=False)
        annotated = preds[0].plot()
        filename = f"rank{i:02d}_{r['frame_time'].replace(':', '')}_{r['score']:.0f}.jpg"
        cv2.imwrite(os.path.join(img_dir, filename), annotated)
    cap.release()
    print(f"\nAnnotated images saved to: {img_dir}")

    # 3. Scatter plot: bbox_area vs confidence
    areas = [r["bbox_area"] for r in rows]
    confs = [r["confidence"] for r in rows]
    scores = [r["score"] for r in rows]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sc = ax.scatter(areas, confs, c=scores, cmap="viridis", alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, label="score (area * conf)")
    ax.set_xlabel("bbox_area (pixels)")
    ax.set_ylabel("confidence")
    ax.set_title(f"{win['label']}: bbox_area vs confidence")

    # Mark top10
    for i, r in enumerate(top10[:3], 1):
        ax.annotate(f"#{i} {r['frame_time']}", (r["bbox_area"], r["confidence"]),
                    fontsize=8, ha="left", va="bottom")

    plot_path = os.path.join(OUT_DIR, f"{win['label']}_scatter.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to: {plot_path}")


if __name__ == "__main__":
    for win in WINDOWS:
        process(win)
