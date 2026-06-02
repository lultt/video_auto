"""Production mode YOLO netbag inference: batch=32, video stream, speed test."""
import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
VIDEO_DIR = "C:/video/0515"
OUT_DIR = "J:/video_auto/netbag/production_test"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOWS = [
    {
        "label": "window1",
        "video": "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4",
        "wall_start": "205339",
        "wall_end": "214053",
        "target_start": "211849",
        "target_end": "211905",
    },
    {
        "label": "window2",
        "video": "C:/video/0515/ch01_20250515_214053_222634topspeed.mp4",
        "wall_start": "214053",
        "wall_end": "222634",
        "target_start": "221223",
        "target_end": "221300",
    },
]


def hhmmss_to_sec(t):
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def sec_to_hhmmss(sec):
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"


def extract_window_frames(win):
    """Extract all frames in the target window from video."""
    cap = cv2.VideoCapture(win["video"])
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
    print(f"{win['label']}: {sec_to_hhmmss(target_start_sec)} ~ {sec_to_hhmmss(target_end_sec)}")
    print(f"Video: {os.path.basename(win['video'])}")
    print(f"FPS: {fps:.2f}, Speed ratio: {speed_ratio:.3f}x")
    print(f"Frame range: {frame_start} ~ {frame_end} ({frame_end - frame_start} frames)")
    print(f"{'='*60}")

    frames = []
    frame_indices = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    for idx in range(frame_start, frame_end):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_indices.append(idx)
    cap.release()

    timestamps = []
    for idx in frame_indices:
        wall_sec = wall_start_sec + (idx / fps) * speed_ratio
        timestamps.append(sec_to_hhmmss(wall_sec))

    return frames, frame_indices, timestamps, fps, speed_ratio, wall_start_sec


def batch_inference(model, frames, batch_size=32):
    """Run batch inference on pre-loaded frames."""
    all_results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        results = model.predict(batch, imgsz=1280, conf=0.25, verbose=False, batch=batch_size)
        all_results.extend(results)
    return all_results


def main():
    model = YOLO(WEIGHTS)
    print(f"Model: {WEIGHTS}")
    print(f"Device: CUDA:{torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Mode: batch=32, imgsz=1280")

    # Warmup
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, imgsz=1280, conf=0.25, verbose=False)
    torch.cuda.synchronize()

    report_lines = []
    summary_rows = []
    total_frames_all = 0
    total_time_all = 0

    for win in WINDOWS:
        frames, frame_indices, timestamps, fps, speed_ratio, wall_start_sec = extract_window_frames(win)
        n_frames = len(frames)
        total_frames_all += n_frames

        # Batch inference with timing
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        results = batch_inference(model, frames, batch_size=32)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time_all += elapsed

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        # Analyze results
        detections = []
        for i, r in enumerate(results):
            if len(r.boxes) > 0:
                best_box_idx = r.boxes.conf.argmax().item()
                conf = r.boxes.conf[best_box_idx].item()
                xyxy = r.boxes.xyxy[best_box_idx].tolist()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                detections.append({
                    "idx": i,
                    "frame_idx": frame_indices[i],
                    "timestamp": timestamps[i],
                    "confidence": conf,
                    "bbox_area": area,
                    "xyxy": xyxy,
                })

        det_rate = len(detections) / n_frames * 100
        avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0
        max_conf = max([d["confidence"] for d in detections]) if detections else 0
        avg_area = np.mean([d["bbox_area"] for d in detections]) if detections else 0
        max_area = max([d["bbox_area"] for d in detections]) if detections else 0

        # Best confidence frame
        best_conf_det = max(detections, key=lambda x: x["confidence"]) if detections else None
        # Max area frame
        max_area_det = max(detections, key=lambda x: x["bbox_area"]) if detections else None

        ms_per_frame = (elapsed / n_frames) * 1000
        window_fps = n_frames / elapsed

        print(f"\n--- {win['label']} Results ---")
        print(f"Total frames:     {n_frames}")
        print(f"Detected frames:  {len(detections)} ({det_rate:.1f}%)")
        print(f"Avg confidence:   {avg_conf:.4f}")
        print(f"Max confidence:   {max_conf:.4f}")
        print(f"Avg bbox_area:    {avg_area:.0f}")
        print(f"Max bbox_area:    {max_area:.0f}")
        print(f"Inference time:   {elapsed:.2f}s ({ms_per_frame:.2f} ms/frame, {window_fps:.1f} FPS)")
        print(f"Peak GPU memory:  {peak_mem:.2f} GB")

        # Save best confidence frame
        if best_conf_det:
            idx = best_conf_det["idx"]
            annotated = results[idx].plot()
            path = os.path.join(OUT_DIR, f"{win['label']}_best_conf.jpg")
            cv2.imwrite(path, annotated)
            print(f"\nBest conf frame: {best_conf_det['timestamp']}, conf={best_conf_det['confidence']:.4f}, area={best_conf_det['bbox_area']:.0f}")
            print(f"  Saved: {path}")

        # Save max area frame
        if max_area_det:
            idx = max_area_det["idx"]
            annotated = results[idx].plot()
            path = os.path.join(OUT_DIR, f"{win['label']}_max_area.jpg")
            cv2.imwrite(path, annotated)
            print(f"Max area frame:  {max_area_det['timestamp']}, conf={max_area_det['confidence']:.4f}, area={max_area_det['bbox_area']:.0f}")
            print(f"  Saved: {path}")

        # Collect for report
        report_lines.append(f"\n{'='*60}")
        report_lines.append(f"{win['label'].upper()}")
        report_lines.append(f"{'='*60}")
        report_lines.append(f"Time window:      {timestamps[0]} ~ {timestamps[-1]}")
        report_lines.append(f"Total frames:     {n_frames}")
        report_lines.append(f"Detected frames:  {len(detections)} ({det_rate:.1f}%)")
        report_lines.append(f"Avg confidence:   {avg_conf:.4f}")
        report_lines.append(f"Max confidence:   {max_conf:.4f}")
        report_lines.append(f"Avg bbox_area:    {avg_area:.0f}")
        report_lines.append(f"Max bbox_area:    {max_area:.0f}")
        report_lines.append(f"Inference:        {ms_per_frame:.2f} ms/frame, {window_fps:.1f} FPS")
        report_lines.append(f"Peak GPU mem:     {peak_mem:.2f} GB")
        if best_conf_det:
            report_lines.append(f"Best conf frame:  {best_conf_det['timestamp']} conf={best_conf_det['confidence']:.4f} area={best_conf_det['bbox_area']:.0f}")
        if max_area_det:
            report_lines.append(f"Max area frame:   {max_area_det['timestamp']} conf={max_area_det['confidence']:.4f} area={max_area_det['bbox_area']:.0f}")

        summary_rows.append({
            "window": win["label"],
            "total_frames": n_frames,
            "detected": len(detections),
            "det_rate": f"{det_rate:.1f}%",
            "avg_conf": f"{avg_conf:.4f}",
            "max_conf": f"{max_conf:.4f}",
            "avg_area": f"{avg_area:.0f}",
            "max_area": f"{max_area:.0f}",
            "best_conf_ts": best_conf_det["timestamp"] if best_conf_det else "",
            "best_conf_val": f"{best_conf_det['confidence']:.4f}" if best_conf_det else "",
            "best_conf_area": f"{best_conf_det['bbox_area']:.0f}" if best_conf_det else "",
            "max_area_ts": max_area_det["timestamp"] if max_area_det else "",
            "max_area_conf": f"{max_area_det['confidence']:.4f}" if max_area_det else "",
            "max_area_val": f"{max_area_det['bbox_area']:.0f}" if max_area_det else "",
        })

    # Overall performance
    overall_fps = total_frames_all / total_time_all
    overall_ms = (total_time_all / total_frames_all) * 1000

    # Production estimates
    frames_9_windows = 13500
    est_time = frames_9_windows / overall_fps

    print(f"\n{'='*60}")
    print(f"PRODUCTION PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total frames tested: {total_frames_all}")
    print(f"Total inference time: {total_time_all:.2f} s")
    print(f"FPS:              {overall_fps:.1f}")
    print(f"ms/frame:         {overall_ms:.2f}")
    print(f"Peak GPU memory:  {peak_mem:.2f} GB")
    print(f"")
    print(f"--- Production Estimate ---")
    print(f"9 windows x 1500 frames = 13500 frames")
    print(f"Estimated time:   {est_time:.1f} s ({est_time/60:.2f} min)")
    print(f"{'='*60}")

    # Write performance_report.txt
    report_lines.insert(0, "NETBAG YOLO PRODUCTION TEST REPORT")
    report_lines.insert(1, f"Model: {WEIGHTS}")
    report_lines.insert(2, f"GPU: {torch.cuda.get_device_name(0)}")
    report_lines.insert(3, f"Mode: batch=32, imgsz=1280")
    report_lines.insert(4, f"Date: 2025-06-01")
    report_lines.append(f"\n{'='*60}")
    report_lines.append("PERFORMANCE")
    report_lines.append(f"{'='*60}")
    report_lines.append(f"FPS:              {overall_fps:.1f}")
    report_lines.append(f"ms/frame:         {overall_ms:.2f}")
    report_lines.append(f"Peak GPU memory:  {peak_mem:.2f} GB")
    report_lines.append(f"9 windows (13500 frames): {est_time:.1f} s ({est_time/60:.2f} min)")

    with open(os.path.join(OUT_DIR, "performance_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Write summary.csv
    import csv
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nFiles saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
