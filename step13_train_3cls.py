# -*- coding: utf-8 -*-
"""Train 3-class business classifier (Background/OnDeck/Sorting) on production-path 640 frames."""
import torch
from ultralytics import YOLO

DATA    = r"J:\video_auto\dataset_v2_3cls"
MODEL   = "yolov8s-cls.pt"
PROJECT = r"J:\video_auto\runs"
NAME    = "fishing_3cls_v1"

def main():
    gpu = torch.cuda.is_available()
    dev = 0 if gpu else "cpu"
    print("CUDA:", gpu, "| device:", torch.cuda.get_device_name(0) if gpu else "CPU")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA, task="classify",
        imgsz=640, epochs=50, batch=64, patience=10,
        workers=8, device=dev,
        project=PROJECT, name=NAME, exist_ok=True,
        verbose=True, plots=True,
    )
    print("\nTRAIN_DONE")
    print("save_dir:", results.save_dir)
    print("top1:", getattr(results, "top1", None))

if __name__ == "__main__":
    main()
