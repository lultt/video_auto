"""
sam_segmentation.py — SAM1 box+point prompt segmentation of net in left_deck ROI.

Segmentation approach:
  1. Bounding box prompt covering the center ~60% of left_deck ROI
  2. Single positive point at ROI center-bottom (where net pile sits)
  3. Multi-mask output, pick highest scoring mask
  4. Save PNG mask + JSON metadata per frame
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Fix PIL DLL issue
os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")


def get_sam_predictor(checkpoint_path: str | Path):
    """Load SAM1 ViT-B model and return predictor instance."""
    from segment_anything import sam_model_registry, SamPredictor

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    try:
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM model: {e}") from e

    sam.to("cuda")
    sam.eval()
    return SamPredictor(sam)


def segment_net_roi(
    predictor,
    roi_bgr: np.ndarray,
    box_rel: tuple[float, float, float, float] = (0.15, 0.20, 0.85, 0.75),
    point_rel: tuple[float, float] = (0.5, 0.45),
) -> dict:
    """
    Segment the net in a single left_deck ROI frame.

    Args:
        predictor: loaded SamPredictor
        roi_bgr: BGR image array
        box_rel: relative box coords (x1, y1, x2, y2) normalized to [0,1]
        point_rel: relative point coords (x, y) normalized to [0,1]

    Returns:
        dict with mask, area_px, score, bbox, etc.
    """
    h, w = roi_bgr.shape[:2]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    predictor.set_image(roi_rgb)

    # Box prompt (absolute pixel coords)
    box_abs = np.array([
        int(box_rel[0] * w), int(box_rel[1] * h),
        int(box_rel[2] * w), int(box_rel[3] * h),
    ])
    # Point prompt (absolute pixel coords)
    point_abs = np.array([[int(point_rel[0] * w), int(point_rel[1] * h)]])
    point_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point_abs,
        point_labels=point_label,
        box=box_abs[None, :],
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    best_score = float(scores[best_idx])
    area_px = int(best_mask.sum())

    # Compute mask bbox
    ys, xs = np.where(best_mask)
    if len(ys) > 0:
        mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        mask_bbox = (0, 0, 0, 0)

    return {
        "mask": best_mask,
        "score": best_score,
        "area_px": area_px,
        "area_frac": area_px / (w * h) if (w * h) > 0 else 0,
        "mask_bbox": mask_bbox,
        "prompt_box": box_abs.tolist(),
        "prompt_point": point_abs[0].tolist(),
        "roi_w": w,
        "roi_h": h,
    }


def run_segmentation(
    keyframe_index_csv: Path,
    out_dir: Path,
    checkpoint_path: str | Path,
    cfg: dict | None = None,
) -> Path:
    """
    Run SAM segmentation on all keyframes in *keyframe_index_csv*.

    Saves per-frame PNG masks and JSON metadata. Returns path to
    segmentation_summary.csv.
    """
    if cfg is None:
        cfg = {}
    net_size_cfg = cfg.get("fine_grained", {}).get("net_size_estimation", {})

    # Prompt params from config (default if not set)
    box_rel = tuple(net_size_cfg.get("box_prompt_rel", (0.15, 0.20, 0.85, 0.75)))
    point_rel = tuple(net_size_cfg.get("point_prompt_rel", (0.5, 0.45)))
    print(f"Prompt config: box_rel={box_rel}, point_rel={point_rel}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load keyframe index
    index_df = pd.read_csv(keyframe_index_csv)
    total_frames = len(index_df)
    print(f"Loaded {total_frames} keyframes")

    # Initialize SAM
    print("Loading SAM model...")
    predictor = get_sam_predictor(checkpoint_path)

    # Process each frame
    results = []
    for idx, row in index_df.iterrows():
        if row.get("status") != "ok" or pd.isna(row.get("roi_path", None)):
            results.append({
                "frame_id": row.get("frame_id", ""),
                "cycle_id": int(row.get("cycle_id", 0)),
                "group": row.get("group", ""),
                "wall_time": row.get("wall_time", ""),
                "segmented": False,
                "skip_reason": "keyframe_status_not_ok",
            })
            continue

        roi_path = Path(str(row["roi_path"]))
        if not roi_path.exists():
            results.append({
                "frame_id": row["frame_id"],
                "cycle_id": int(row["cycle_id"]),
                "group": row["group"],
                "wall_time": row["wall_time"],
                "segmented": False,
                "skip_reason": "roi_file_not_found",
            })
            continue

        cycle_id = int(row["cycle_id"])
        frame_id = str(row["frame_id"])

        try:
            roi_bgr = cv2.imread(str(roi_path))
            if roi_bgr is None:
                raise ValueError("cv2.imread failed")

            seg_result = segment_net_roi(predictor, roi_bgr, box_rel, point_rel)

            # Save mask PNG
            mask_dir = out_dir / f"cycle_{cycle_id:02d}"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / f"mask_{frame_id}.png"
            mask_uint8 = (seg_result["mask"].astype(np.uint8) * 255)
            cv2.imwrite(str(mask_path), mask_uint8)

            # Save meta JSON
            meta = {k: v for k, v in seg_result.items() if k != "mask"}
            meta.update({
                "frame_id": frame_id,
                "cycle_id": cycle_id,
                "group": row.get("group", ""),
                "wall_time": row.get("wall_time", ""),
                "roi_path": str(roi_path),
                "source_full_path": str(row.get("full_frame_path", "")),
                "video_path": str(row.get("video_path", "")),
                "video_offset_sec": float(row.get("video_offset_sec", 0)),
            })
            meta_path = mask_dir / f"meta_{frame_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            results.append({
                "frame_id": frame_id,
                "cycle_id": cycle_id,
                "group": row.get("group", ""),
                "wall_time": row.get("wall_time", ""),
                "segmented": True,
                "score": seg_result["score"],
                "area_px": seg_result["area_px"],
                "area_frac": seg_result["area_frac"],
                "mask_bbox": str(seg_result["mask_bbox"]),
                "mask_path": str(mask_path),
                "meta_path": str(meta_path),
            })

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{total_frames}")

        except Exception as e:
            results.append({
                "frame_id": frame_id,
                "cycle_id": cycle_id,
                "group": row.get("group", ""),
                "wall_time": row.get("wall_time", ""),
                "segmented": False,
                "skip_reason": f"error: {str(e)}",
            })

    # Save summary CSV
    summary_df = pd.DataFrame(results)
    summary_csv = out_dir / "segmentation_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    succeeded = sum(1 for r in results if r["segmented"])
    failed = total_frames - succeeded
    print(f"\nDone. {succeeded} succeeded, {failed} failed")
    if failed > 0:
        failures = [r for r in results if not r["segmented"]]
        for f in failures[:5]:
            print(f"  {f.get('frame_id', '')}: {f.get('skip_reason', 'unknown')}")

    return summary_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SAM1 segmentation of net in left_deck ROI keyframes")
    p.add_argument("--keyframe-index", required=True, help="path to keyframe_index.csv")
    p.add_argument("--out-dir", required=True, help="output dir for masks and metadata")
    p.add_argument("--checkpoint", default=r"J:\video_auto\third_party\sam2\checkpoints\sam_vit_b_01ec64.pth",
                   help="SAM1 ViT-B checkpoint path")
    p.add_argument("--config", default=None, help="pipeline YAML config path")
    args = p.parse_args()

    from ...shared.utils import load_pipeline_config
    cfg = load_pipeline_config(args.config) if args.config else {}

    run_segmentation(
        keyframe_index_csv=Path(args.keyframe_index),
        out_dir=Path(args.out_dir),
        checkpoint_path=Path(args.checkpoint),
        cfg=cfg,
    )