"""
suspended_net_geometry.py — 通过几何特征找"网兜完全悬空"的黄金帧。

核心逻辑（替代暗像素法）：
  1. 在 net_rise 窗口内每 3-5s 抽一帧
  2. SAM auto-mode 生成 masks
  3. 几何过滤：
     - vertical_ratio: height/width > threshold   → 瘦高形状
     - suspension_height: (bottom - top) / roi_h   → 纵向覆盖大
     - center_right_region: bbox 在中右部区域
     - compactness: mask 面积 / bbox 面积          → 形状紧凑
  4. 找到 geometric_net_score 峰值帧

这是真正稳定的信号：网兜悬空时的形状永远是瘦高下垂。
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from .extract_keyframes import scan_source_videos


# Fix PIL DLL issue
os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")


def init_sam_auto(checkpoint_path: Path, points_per_side: int = 24) -> SamAutomaticMaskGenerator:
    """初始化 SAM auto-mode。"""
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    sam.to("cuda")
    sam.eval()
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.70,
        stability_score_thresh=0.75,
        min_mask_region_area=500,
    )


def get_frame_at_wall_time(
    target_wall_time: pd.Timestamp,
    video_index: list[dict],
    tmp_path: Path,
) -> tuple[np.ndarray | None, float]:
    """从 NAS 视频提取指定时间戳的一帧。"""
    v = None
    for candidate in video_index:
        if candidate["start"] <= target_wall_time < candidate["end"]:
            v = candidate
            break
    if v is None:
        return None, 0.0

    offset_sec = max(0, (target_wall_time - v["start"]).total_seconds())

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{offset_sec:.3f}",
        "-i", str(v["path"]),
        "-frames:v", "1",
        "-q:v", "2",
        str(tmp_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)

    img = cv2.imread(str(tmp_path))
    if img is None:
        return None, offset_sec
    return img, offset_sec


def crop_left_deck_roi(img_bgr: np.ndarray, roi_cfg: dict | None = None) -> np.ndarray:
    """裁剪左甲板 ROI。"""
    if roi_cfg is None:
        roi_cfg = {"x_min": 0.0, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0}
    h, w = img_bgr.shape[:2]
    x1, x2 = int(roi_cfg["x_min"] * w), int(roi_cfg["x_max"] * w)
    y1, y2 = int(roi_cfg["y_min"] * h), int(roi_cfg["y_max"] * h)
    return img_bgr[y1:y2, x1:x2]


def compute_geometric_net_score(
    masks: list[dict],
    roi_w: int,
    roi_h: int,
    vertical_ratio_min: float = 1.5,
    suspension_height_min: float = 0.25,
    center_right_region: tuple[float, float] = (0.4, 0.9),
    compactness_min: float = 0.3,
) -> dict:
    """
    从所有 masks 中过滤出最像悬空网的 mask，计算分数。

    Returns:
        {score, best_mask, area_px, vertical_ratio, suspension_height, compactness}
    """
    best_score = 0.0
    best_mask = None
    best_meta = {}

    for m in masks:
        area_px = m["area"]
        bbox = m["bbox"]  # x, y, w, h
        bx, by, bw, bh = bbox
        center_x = bx + bw / 2

        # 形状几何特征
        vertical_ratio = bh / max(bw, 1)               # 高宽比
        suspension_height = bh / roi_h                   # 纵向覆盖比例
        center_x_norm = center_x / roi_w                 # 归一化 x 中心
        compactness = area_px / max(bw * bh, 1)          # 紧凑度

        # 过滤：中右部、瘦高、纵向跨度足够、紧凑
        if (vertical_ratio >= vertical_ratio_min and
            suspension_height >= suspension_height_min and
            center_right_region[0] <= center_x_norm <= center_right_region[1] and
            compactness >= compactness_min):

            # 综合得分 = 纵向跨度 * 垂直比 * (1 - 过宽惩罚)
            # 越大越像是完整悬空网
            width_penalty = max(0, 1.0 - bw / roi_w)
            score = suspension_height * vertical_ratio * width_penalty * compactness

            if score > best_score:
                best_score = score
                best_mask = m["segmentation"]
                best_meta = {
                    "area_px": area_px,
                    "vertical_ratio": round(vertical_ratio, 3),
                    "suspension_height": round(suspension_height, 3),
                    "compactness": round(compactness, 3),
                    "center_x_norm": round(center_x_norm, 3),
                    "bbox": bbox,
                }

    return {
        "net_detected": best_score > 0,
        "geometric_score": round(best_score, 5),
        "mask": best_mask,
        **best_meta,
    }


def analyze_frame_for_suspended_net(
    roi_bgr: np.ndarray,
    sam_auto: SamAutomaticMaskGenerator,
    geometry_cfg: dict | None = None,
) -> dict:
    """分析单帧 ROI 是否存在悬空网。"""
    if geometry_cfg is None:
        geometry_cfg = {}

    # 运行 SAM auto-mode
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    masks = sam_auto.generate(roi_rgb)

    # 几何过滤
    roi_h, roi_w = roi_bgr.shape[:2]
    result = compute_geometric_net_score(
        masks, roi_w, roi_h,
        vertical_ratio_min=geometry_cfg.get("vertical_ratio_min", 1.5),
        suspension_height_min=geometry_cfg.get("suspension_height_min", 0.25),
        center_right_region=tuple(geometry_cfg.get("center_right_region", (0.4, 0.9))),
        compactness_min=geometry_cfg.get("compactness_min", 0.3),
    )
    result["num_masks_total"] = len(masks)

    return result


def find_suspended_net_peak(
    cycle_row: pd.Series,
    video_index: list[dict],
    sam_auto: SamAutomaticMaskGenerator,
    tmp_dir: Path,
    sample_interval_sec: int = 3,
    geometry_cfg: dict | None = None,
    roi_cfg: dict | None = None,
) -> dict:
    """对单个 cycle 遍历 net_rise 窗口，寻找几何分数峰值帧。"""
    cycle_id = int(cycle_row["cycle_id"])
    net_start = pd.Timestamp(cycle_row["net_start_time"])
    net_on_deck = pd.Timestamp(cycle_row["net_on_deck_time"])
    duration_sec = (net_on_deck - net_start).total_seconds()

    print(f"  Cycle {cycle_id}: {net_start.strftime('%H:%M:%S')} → "
          f"{net_on_deck.strftime('%H:%M:%S')} ({duration_sec:.0f}s)")

    samples = []
    offsets = np.arange(0, duration_sec, sample_interval_sec).astype(float)
    if len(offsets) == 0 or abs(offsets[-1] - duration_sec) > sample_interval_sec / 2:
        offsets = np.append(offsets, duration_sec)

    cycle_tmp = tmp_dir / f"cycle_{cycle_id:02d}"
    cycle_tmp.mkdir(parents=True, exist_ok=True)

    for offset in offsets:
        t = net_start + timedelta(seconds=float(offset))
        frame_path = cycle_tmp / f"t{int(offset):04d}.png"

        img, real_offset = get_frame_at_wall_time(t, video_index, frame_path)
        if img is None:
            continue

        roi = crop_left_deck_roi(img, roi_cfg)
        result = analyze_frame_for_suspended_net(roi, sam_auto, geometry_cfg)

        samples.append({
            "cycle_id": cycle_id,
            "offset_sec": round(float(offset), 1),
            "wall_time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "video_offset_sec": round(real_offset, 1),
            "geometric_score": result["geometric_score"],
            "net_detected": result["net_detected"],
            "area_px": result.get("area_px", 0),
            "vertical_ratio": result.get("vertical_ratio", 0),
            "suspension_height": result.get("suspension_height", 0),
            "compactness": result.get("compactness", 0),
            "center_x_norm": result.get("center_x_norm", 0),
            "num_masks_total": result["num_masks_total"],
            "frame_path": str(frame_path),
        })

    if not samples:
        return {"cycle_id": cycle_id, "status": "failed", "reason": "no_valid_frames"}

    samples_df = pd.DataFrame(samples)
    peak_idx = int(samples_df["geometric_score"].argmax())
    peak = samples_df.iloc[peak_idx]

    return {
        "cycle_id": cycle_id,
        "status": "success",
        "peak_offset_sec": peak["offset_sec"],
        "peak_wall_time": peak["wall_time"],
        "peak_geometric_score": peak["geometric_score"],
        "peak_net_detected": bool(peak["net_detected"]),
        "peak_area_px": int(peak["area_px"]),
        "peak_vertical_ratio": peak["vertical_ratio"],
        "peak_suspension_height": peak["suspension_height"],
        "peak_compactness": peak["compactness"],
        "peak_frame_path": peak["frame_path"],
        "all_samples": samples,
        "num_samples": len(samples),
    }


def run_suspended_net_geometry(
    core_catch_windows_csv: Path,
    video_root: Path,
    checkpoint_path: Path,
    out_dir: Path,
    cfg: dict | None = None,
) -> Path:
    """对所有 9 个 cycles 运行几何检测，找每个 cycle 的悬空网黄金帧。"""
    if cfg is None:
        cfg = {}
    net_size_cfg = cfg.get("fine_grained", {}).get("net_size_estimation", {})
    geometry_cfg = net_size_cfg.get("suspended_net_geometry", {})

    sample_interval = geometry_cfg.get("sample_interval_sec", 3)
    roi_cfg = cfg.get("coarse_features", {}).get("rois", {}).get("left_deck", {
        "x_min": 0.0, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0,
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Suspended Net Geometry Detector ===")
    print(f"  Sample interval: {sample_interval}s")
    print(f"  Min vertical_ratio: {geometry_cfg.get('vertical_ratio_min', 1.5)}")
    print(f"  Min suspension_height: {geometry_cfg.get('suspension_height_min', 0.25)}")
    print()

    # Load data
    core_df = pd.read_csv(core_catch_windows_csv)
    video_index = scan_source_videos(video_root)
    print(f"Loaded {len(video_index)} source videos, {len(core_df)} cycles")

    # Initialize SAM
    print("Loading SAM model...")
    sam_auto = init_sam_auto(checkpoint_path, points_per_side=24)

    # Run all cycles
    results = []
    all_samples = []
    tmp_dir = out_dir / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for _, cycle_row in core_df.iterrows():
        result = find_suspended_net_peak(
            cycle_row, video_index, sam_auto, tmp_dir,
            sample_interval_sec=sample_interval,
            geometry_cfg=geometry_cfg,
            roi_cfg=roi_cfg,
        )
        results.append({k: v for k, v in result.items() if k != "all_samples"})
        all_samples.extend(result.get("all_samples", []))

        cycle_samples = pd.DataFrame(result.get("all_samples", []))
        if len(cycle_samples) > 0:
            cycle_samples.to_csv(out_dir / f"cycle_{result['cycle_id']:02d}_samples.csv",
                                  index=False, encoding="utf-8-sig")

    # Save results
    results_df = pd.DataFrame(results)
    summary_csv = out_dir / "suspended_net_geometry_summary.csv"
    results_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    all_samples_df = pd.DataFrame(all_samples)
    all_samples_df.to_csv(out_dir / "all_cycles_samples.csv", index=False, encoding="utf-8-sig")

    success = sum(1 for r in results if r.get("status") == "success")
    net_detected = sum(1 for r in results if r.get("peak_net_detected"))

    print(f"\nDone. {success}/{len(results)} succeeded")
    print(f"  Net detected in peak frame: {net_detected}/{success}")
    if success > 0:
        print(f"  Peak geometric score range: {results_df['peak_geometric_score'].min():.4f} - "
              f"{results_df['peak_geometric_score'].max():.4f}")
        print(f"  Peak area_px range: {results_df['peak_area_px'].min()} - {results_df['peak_area_px'].max()}")
        print(f"  Peak vertical_ratio range: {results_df['peak_vertical_ratio'].min():.2f} - "
              f"{results_df['peak_vertical_ratio'].max():.2f}")

    return summary_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Suspended net detection via geometric signature")
    p.add_argument("--core-windows", required=True, help="core_catch_windows.csv path")
    p.add_argument("--video-root", default=r"\\DS224plus\video\0515", help="NAS video root")
    p.add_argument("--checkpoint", default=r"J:\video_auto\third_party\sam2\checkpoints\sam_vit_b_01ec64.pth",
                   help="SAM checkpoint path")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--config", default=None, help="pipeline YAML config path")
    p.add_argument("--sample-interval", type=int, default=3, help="sample every N seconds")
    args = p.parse_args()

    from ...shared.utils import load_pipeline_config
    cfg = load_pipeline_config(args.config) if args.config else {}

    run_suspended_net_geometry(
        core_catch_windows_csv=Path(args.core_windows),
        video_root=Path(args.video_root),
        checkpoint_path=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )