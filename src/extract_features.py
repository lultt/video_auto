"""
ROI 时序特征提取

职责：
- 从 video_reader 的 FramePacket 中裁剪 ROI
- 计算瞬时特征 + rolling 统计
- 输出 parquet

不做：解码、stabilization（由 video_reader 完成）
"""

import os
import sys
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_validator import load_config
from src.video_reader import FastVideoReader


def compute_entropy(gray_roi):
    hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def extract_roi_features(packet, roi_slice, prev_gray_roi, features_cfg):
    """从 FramePacket 提取单个 ROI 的全部特征。"""
    roi_bgr = packet.frame_bgr[roi_slice]
    roi_gray = packet.gray[roi_slice]
    stabilized_roi = packet.stabilized_gray[roi_slice]

    row = {}

    if features_cfg.get("brightness", True):
        row["mean_brightness"] = float(roi_gray.mean())
        row["brightness_std"] = float(roi_gray.std())

    if features_cfg.get("color", True):
        row["mean_b"] = float(roi_bgr[:, :, 0].mean())
        row["mean_g"] = float(roi_bgr[:, :, 1].mean())
        row["mean_r"] = float(roi_bgr[:, :, 2].mean())

    if features_cfg.get("motion", True):
        if prev_gray_roi is not None:
            diff = cv2.absdiff(roi_gray, prev_gray_roi)
            row["motion_intensity"] = float(diff.mean())
        else:
            row["motion_intensity"] = 0.0

    if features_cfg.get("edge", True):
        edges = cv2.Canny(roi_gray, 50, 150)
        row["edge_density"] = float(edges.sum() / 255.0 / edges.size)

    if features_cfg.get("texture", True):
        lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        row["laplacian_variance"] = float(lap.var())

    if features_cfg.get("entropy", True):
        row["entropy"] = compute_entropy(roi_gray)

    # local_motion from stabilized frame
    if features_cfg.get("stabilization", True):
        if prev_gray_roi is not None:
            local_diff = cv2.absdiff(stabilized_roi, prev_gray_roi)
            row["local_motion_intensity"] = float(local_diff.mean())
        else:
            row["local_motion_intensity"] = 0.0
    else:
        row["local_motion_intensity"] = 0.0

    return row


def process_single_video(video_path, cfg, output_dir):
    """处理单个视频，输出 parquet。"""
    video_id = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_id.replace(".mp4", ".parquet"))

    if os.path.exists(output_path):
        return {"video_id": video_id, "status": "skipped", "frames": 0}

    features_cfg = cfg["features"]
    adaptive_cfg = cfg["adaptive_sampling"]
    rois_cfg = cfg["rois"]
    resize = (cfg["resize_width"], cfg["resize_height"])
    rolling_window = cfg["rolling"]["window_sec"]

    try:
        reader = FastVideoReader(video_path, resize=resize, gpu_decode=cfg.get("gpu_decode", False))
    except IOError as e:
        return {"video_id": video_id, "status": f"error: {e}", "frames": 0}

    rh, rw = resize[1], resize[0]
    roi_slices = {}
    for name, r in rois_cfg.items():
        y1, y2 = int(r["y_min"] * rh), int(r["y_max"] * rh)
        x1, x2 = int(r["x_min"] * rw), int(r["x_max"] * rw)
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))

    records = []
    prev_gray_rois = {name: None for name in roi_slices}

    for packet in reader.read_adaptive(adaptive_cfg, stabilize=True):
        for roi_name, roi_slice in roi_slices.items():
            row = {
                "video_id": video_id,
                "frame_idx": packet.frame_idx,
                "timestamp_sec": packet.timestamp_sec,
                "roi_name": roi_name,
                "sample_fps": packet.sample_fps,
                "scene_transition": packet.scene_transition,
                "global_motion_x": packet.global_motion_x,
                "global_motion_y": packet.global_motion_y,
                "global_motion_mag": packet.global_motion_mag,
            }
            feats = extract_roi_features(
                packet, roi_slice, prev_gray_rois[roi_name], features_cfg
            )
            row.update(feats)
            records.append(row)
            prev_gray_rois[roi_name] = packet.gray[roi_slice].copy()

    reader.release()

    if not records:
        return {"video_id": video_id, "status": "empty", "frames": 0}

    df = pd.DataFrame(records)

    # rolling statistics per ROI
    for roi_name in roi_slices:
        mask = df["roi_name"] == roi_name
        subset = df.loc[mask]
        df.loc[mask, "brightness_rolling_mean_30s"] = (
            subset["mean_brightness"].rolling(rolling_window, min_periods=1).mean().values
        )
        df.loc[mask, "brightness_rolling_std_30s"] = (
            subset["mean_brightness"].rolling(rolling_window, min_periods=1).std(ddof=0).fillna(0).values
        )
        df.loc[mask, "motion_rolling_mean_30s"] = (
            subset["local_motion_intensity"].rolling(rolling_window, min_periods=1).mean().values
        )
        df.loc[mask, "motion_rolling_std_30s"] = (
            subset["local_motion_intensity"].rolling(rolling_window, min_periods=1).std(ddof=0).fillna(0).values
        )

    # enforce schema dtypes
    float32_cols = [c for c in df.columns if c not in ("video_id", "roi_name", "frame_idx")]
    for c in float32_cols:
        df[c] = df[c].astype(np.float32)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)

    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    return {"video_id": video_id, "status": "ok", "frames": len(df)}


def run_extraction(cfg, video_list=None, max_videos=None):
    output_dir = os.path.join(cfg["output_dir"], "parquet")
    manifest_path = os.path.join(cfg["data_dir"], "video_manifest.csv")

    if video_list is None:
        if not os.path.exists(manifest_path):
            print(f"清单不存在: {manifest_path}，请先运行 scan_videos.py")
            return
        manifest = pd.read_csv(manifest_path)
        video_list = manifest["path"].tolist()

    if max_videos:
        video_list = video_list[:max_videos]

    print(f"待处理: {len(video_list)} 个视频")
    os.makedirs(output_dir, exist_ok=True)

    num_workers = cfg.get("num_workers", 4)
    results = []
    t0 = time.time()

    if num_workers <= 1:
        for vp in tqdm(video_list, desc="提取特征"):
            r = process_single_video(vp, cfg, output_dir)
            results.append(r)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for vp in video_list:
                fut = executor.submit(process_single_video, vp, cfg, output_dir)
                futures[fut] = vp
            for fut in tqdm(as_completed(futures), total=len(futures), desc="提取特征"):
                results.append(fut.result())

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    err_count = sum(1 for r in results if "error" in r["status"])
    total_frames = sum(r["frames"] for r in results)

    print(f"\n完成: {ok_count} 成功, {skip_count} 跳过, {err_count} 失败")
    print(f"总帧数: {total_frames}, 耗时: {elapsed:.1f}s")
    if elapsed > 0 and total_frames > 0:
        print(f"速度: {total_frames / elapsed:.0f} 行/秒")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频时序特征提取")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.workers:
        cfg["num_workers"] = args.workers
    run_extraction(cfg, max_videos=args.max_videos)
