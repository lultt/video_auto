"""Test 1: batch=128 speed test. Test 2: 30-min negative window FP check."""
import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

WEIGHTS = "J:/video_auto/netbag/runs/detect/train/weights/best.pt"
OUT_DIR_FP = "J:/video_auto/netbag/false_positive_test"


def hhmmss_to_sec(t):
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def sec_to_hhmmss(sec):
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"


def sec_to_filename(sec):
    return f"{int(sec//3600):02d}{int((sec%3600)//60):02d}{int(sec%60):02d}"


def log(msg):
    print(msg, flush=True)


def test1_batch128(model):
    log("\n" + "=" * 60)
    log("TEST 1: batch=128 SPEED TEST")
    log("=" * 60)

    VIDEO1 = "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4"
    cap = cv2.VideoCapture(VIDEO1)
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    wall_start_sec = hhmmss_to_sec("205339")
    target_start_sec = hhmmss_to_sec("211500")
    offset = target_start_sec - wall_start_sec
    frame_start = int(offset * fps_video)

    N_TEST = 1500
    log(f"Loading {N_TEST} frames from video...")
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    for i in range(N_TEST):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    log(f"Loaded {len(frames)} frames")

    # Warmup
    log("Warmup (3 batches)...")
    for _ in range(3):
        model.predict(frames[:128], imgsz=1280, conf=0.25, verbose=False)
    torch.cuda.synchronize()
    log("Warmup done")

    # Timed run
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(frames), 128):
        batch = frames[i:i + 128]
        model.predict(batch, imgsz=1280, conf=0.25, verbose=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    fps_val = len(frames) / elapsed
    ms_val = (elapsed / len(frames)) * 1000

    log(f"\nbatch=128 Results:")
    log(f"  Frames:         {len(frames)}")
    log(f"  Total time:     {elapsed:.2f} s")
    log(f"  FPS:            {fps_val:.1f}")
    log(f"  ms/frame:       {ms_val:.2f}")
    log(f"  Peak GPU mem:   {peak_mem:.2f} GB")
    log(f"\nComparison:")
    log(f"  batch=32:  79.5 FPS, 12.58 ms/frame, 1.51 GB")
    log(f"  batch=128: {fps_val:.1f} FPS, {ms_val:.2f} ms/frame, {peak_mem:.2f} GB")
    log(f"  Speedup:   {fps_val / 79.5:.2f}x")

    del frames
    torch.cuda.empty_cache()
    return fps_val, ms_val, peak_mem


def test2_negative_window(model):
    log("\n" + "=" * 60)
    log("TEST 2: NEGATIVE WINDOW (20:40:00 ~ 21:10:00)")
    log("=" * 60)

    os.makedirs(OUT_DIR_FP, exist_ok=True)

    SEGMENTS = [
        {
            "video": "C:/video/0515/ch01_20250515_200910_205339topspeed.mp4",
            "wall_start": "200910",
            "wall_end": "205339",
            "target_start": "204000",
            "target_end": "205339",
        },
        {
            "video": "C:/video/0515/ch01_20250515_205339_214053topspeed.mp4",
            "wall_start": "205339",
            "wall_end": "214053",
            "target_start": "205339",
            "target_end": "211000",
        },
    ]

    all_detections = []
    total_frames_neg = 0

    for seg in SEGMENTS:
        log(f"\nProcessing: {os.path.basename(seg['video'])}")
        log(f"  Target: {seg['target_start']} ~ {seg['target_end']}")

        cap = cv2.VideoCapture(seg["video"])
        fps_v = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur = total_f / fps_v

        ws = hhmmss_to_sec(seg["wall_start"])
        we = hhmmss_to_sec(seg["wall_end"])
        wall_dur = we - ws
        speed_ratio = wall_dur / video_dur

        ts = hhmmss_to_sec(seg["target_start"])
        te = hhmmss_to_sec(seg["target_end"])

        offset_start = (ts - ws) / speed_ratio
        offset_end = (te - ws) / speed_ratio

        fs = int(offset_start * fps_v)
        fe = int(offset_end * fps_v)
        n_frames_seg = fe - fs

        log(f"  Frame range: {fs} ~ {fe} ({n_frames_seg} frames)")
        log(f"  Speed ratio: {speed_ratio:.3f}x, FPS: {fps_v:.2f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, fs)

        batch_frames = []
        batch_indices = []
        seg_frame_count = 0
        batch_num = 0

        for frame_idx in range(fs, fe):
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            seg_frame_count += 1

            if len(batch_frames) == 128 or frame_idx == fe - 1:
                results = model.predict(batch_frames, imgsz=1280, conf=0.25, verbose=False)
                for bi, r in enumerate(results):
                    if len(r.boxes) > 0:
                        best_idx = r.boxes.conf.argmax().item()
                        conf = r.boxes.conf[best_idx].item()
                        xyxy = r.boxes.xyxy[best_idx].tolist()
                        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                        fidx = batch_indices[bi]
                        wall_sec = ws + (fidx / fps_v) * speed_ratio
                        all_detections.append({
                            "frame_idx": fidx,
                            "wall_sec": wall_sec,
                            "timestamp": sec_to_hhmmss(wall_sec),
                            "confidence": conf,
                            "bbox_area": area,
                            "frame": batch_frames[bi],
                        })
                batch_num += 1
                if batch_num % 10 == 0:
                    log(f"  Progress: {seg_frame_count}/{n_frames_seg} frames ({seg_frame_count*100//n_frames_seg}%)")
                batch_frames = []
                batch_indices = []

        cap.release()
        total_frames_neg += seg_frame_count
        log(f"  Done: {seg_frame_count} frames processed")

    # Results
    fp_count = len(all_detections)
    fp_rate = fp_count / total_frames_neg * 100 if total_frames_neg > 0 else 0
    avg_conf = np.mean([d["confidence"] for d in all_detections]) if all_detections else 0
    max_conf = max([d["confidence"] for d in all_detections]) if all_detections else 0

    log(f"\n{'='*60}")
    log("NEGATIVE WINDOW RESULTS")
    log(f"{'='*60}")
    log(f"Total frames:         {total_frames_neg}")
    log(f"False positive count: {fp_count}")
    log(f"False positive rate:  {fp_rate:.4f}%")
    log(f"Avg FP confidence:    {avg_conf:.4f}")
    log(f"Max FP confidence:    {max_conf:.4f}")

    # Export Top20
    if all_detections:
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)
        top20 = all_detections[:20]

        log(f"\nTop20 false positives:")
        log(f"{'Rank':<5} {'Timestamp':<12} {'Confidence':>11} {'bbox_area':>10}")
        log("-" * 45)
        for i, d in enumerate(top20, 1):
            log(f"{i:<5} {d['timestamp']:<12} {d['confidence']:>11.4f} {d['bbox_area']:>10.0f}")
            results = model.predict(d["frame"], imgsz=1280, conf=0.25, verbose=False)
            annotated = results[0].plot()
            ts_fname = sec_to_filename(d["wall_sec"])
            filename = f"20250515_{ts_fname}_conf{d['confidence']:.2f}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR_FP, filename), annotated)

        log(f"\nTop20 images saved to: {OUT_DIR_FP}")
    else:
        log("\nNo false positives! Model is clean.")

    return total_frames_neg, fp_count, fp_rate, max_conf


def main():
    model = YOLO(WEIGHTS)
    log(f"Model: {WEIGHTS}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    fps_128, ms_128, mem_128 = test1_batch128(model)
    total_neg, fp_count, fp_rate, max_fp_conf = test2_negative_window(model)

    # Final summary
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    log(f"1. batch=128 FPS:          {fps_128:.1f}")
    log(f"   batch=128 ms/frame:     {ms_128:.2f}")
    log(f"   batch=128 GPU mem:      {mem_128:.2f} GB")
    est_9win = 13500 / fps_128
    log(f"   9 windows (13500 fr):   {est_9win:.1f} s ({est_9win/60:.2f} min)")
    log(f"")
    log(f"2. Negative window (30 min):")
    log(f"   Total frames:           {total_neg}")
    log(f"   False positives:        {fp_count}")
    log(f"   FP rate:                {fp_rate:.4f}%")
    log(f"   Max FP confidence:      {max_fp_conf:.4f}")
    log(f"")
    log(f"3. Production ready:       {'YES' if fp_rate < 1.0 else 'NEEDS MORE WORK'}")


if __name__ == "__main__":
    main()
