"""Generate prediction previews on 20 random val images."""
import os
import random
from ultralytics import YOLO

random.seed(123)

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
VAL_DIR = "J:/video_auto/netbag/dataset/images/val"
OUT_DIR = "J:/video_auto/netbag/predictions_preview"

os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(WEIGHTS)

images = sorted(os.listdir(VAL_DIR))
sample = random.sample(images, min(20, len(images)))
sample_paths = [os.path.join(VAL_DIR, f) for f in sample]

results = model.predict(
    source=sample_paths,
    imgsz=1280,
    conf=0.25,
    save=True,
    project=OUT_DIR,
    name="val_20",
    exist_ok=True,
)

print(f"\nPredictions saved to: {OUT_DIR}/val_20")
print(f"Total images: {len(sample_paths)}")
for r in results:
    boxes = r.boxes
    if len(boxes) > 0:
        conf = boxes.conf[0].item()
        xyxy = boxes.xyxy[0].tolist()
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        print(f"  {os.path.basename(r.path)}: conf={conf:.3f}, area={area:.0f}")
    else:
        print(f"  {os.path.basename(r.path)}: NO DETECTION")
