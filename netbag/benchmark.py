"""Benchmark YOLO netbag model inference speed."""
import os
import time
import torch
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
IMG_DIR = "J:/video_auto/netbag/0515_first_net_raw"

model = YOLO(WEIGHTS)

images = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
print(f"Total images: {len(images)}")
print(f"Model: {WEIGHTS}")
print(f"imgsz: 1280")
print(f"Device: CUDA:{torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Warmup
for _ in range(5):
    model.predict(images[0], imgsz=1280, conf=0.25, verbose=False)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
for img in images:
    model.predict(img, imgsz=1280, conf=0.25, verbose=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

peak_mem = torch.cuda.max_memory_allocated() / 1024**3

n = len(images)
ms_per_image = (elapsed / n) * 1000
fps = n / elapsed

print(f"\n{'='*50}")
print(f"BENCHMARK RESULTS")
print(f"{'='*50}")
print(f"Images processed:  {n}")
print(f"Total time:        {elapsed:.2f} s")
print(f"ms/image:          {ms_per_image:.2f} ms")
print(f"FPS:               {fps:.1f} images/s")
print(f"Peak GPU memory:   {peak_mem:.2f} GB")
print(f"{'='*50}")

# Estimates
frames_per_window = 1500
windows_per_day = 9
total_frames = frames_per_window * windows_per_day

time_one_window = frames_per_window * ms_per_image / 1000
time_all = total_frames * ms_per_image / 1000

print(f"\nESTIMATES (based on measured {ms_per_image:.2f} ms/image):")
print(f"  60s window (1500 frames): {time_one_window:.1f} s ({time_one_window/60:.2f} min)")
print(f"  9 windows (13500 frames): {time_all:.1f} s ({time_all/60:.1f} min)")
print(f"{'='*50}")
