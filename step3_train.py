# -*- coding: utf-8 -*-
"""Step 3: train baseline classifier (yolov8s-cls) on fishing_status dataset."""
import torch
from ultralytics import YOLO

DATA   = r"J:\video_auto\dataset"
MODEL  = "yolov8s-cls.pt"
PROJECT = r"J:\video_auto\runs"
NAME    = "fishing_status_v1"

def main():
    gpu = torch.cuda.is_available()
    dev = 0 if gpu else "cpu"
    print("CUDA:", gpu, "| device:", torch.cuda.get_device_name(0) if gpu else "CPU")

    # batch/workers auto-adjusted for available hardware
    batch   = 64 if gpu else 16
    workers = 8 if gpu else 4

    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        task="classify",
        imgsz=640,
        epochs=50,
        batch=batch,
        patience=10,
        workers=workers,
        device=dev,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    print("\nTRAIN_DONE")
    print("save_dir:", results.save_dir)
    print("top1:", getattr(results, "top1", None))
    print("top5:", getattr(results, "top5", None))

if __name__ == "__main__":
    main()
