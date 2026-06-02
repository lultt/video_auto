from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="J:/video_auto/netbag/dataset.yaml",
        imgsz=1280,
        epochs=20,
        batch=16,
        workers=0,
        device=0,
        cache=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()
