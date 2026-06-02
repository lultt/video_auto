"""
feature_merge.py — Layer 1 coarse feature extraction, main entry point.

Orchestrates per-frame feature computation (motion, edge, entropy, brightness)
via sub-module extractors and writes a single Parquet file per video.

Keyframe-only decode (-skip_frame nokey), subsampled every N keyframes
for throughput (~60× realtime).
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extract_entropy import compute_entropy
from .extract_edge import compute_edge_density, compute_laplacian_variance
from .extract_motion import compute_motion_intensity
from ..shared.utils import load_pipeline_config

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", r"D:\ffmpeg\bin\ffmpeg.exe")

# ---------------------------------------------------------------------------
# defaults — overridden by pipeline.yaml if loaded
# ---------------------------------------------------------------------------
DEFAULT_KF_SUBSAMPLE = 3
DEFAULT_KF_INTERVAL_SEC = 4.0
DEFAULT_RESIZE = (640, 360)
DEFAULT_ROIS_CFG = {
    "left_deck":   {"x_min": 0.000, "x_max": 0.333, "y_min": 0.4, "y_max": 1.0},
    "center_deck": {"x_min": 0.333, "x_max": 0.667, "y_min": 0.4, "y_max": 1.0},
    "right_deck":  {"x_min": 0.667, "x_max": 1.000, "y_min": 0.4, "y_max": 1.0},
}
DEFAULT_FEATURE_FLAGS = {
    "brightness": True, "color": True, "motion": True,
    "edge": True, "texture": True, "entropy": True,
}


def _params_from_config(cfg: dict | None):
    """Extract coarse-feature parameters from unified pipeline config."""
    if cfg is None:
        return DEFAULT_KF_SUBSAMPLE, DEFAULT_KF_INTERVAL_SEC, DEFAULT_RESIZE, DEFAULT_ROIS_CFG, DEFAULT_FEATURE_FLAGS

    coarse = cfg.get("coarse_features", {})
    kf_subsample = coarse.get("kf_subsample", DEFAULT_KF_SUBSAMPLE)
    kf_interval = coarse.get("kf_interval_sec", DEFAULT_KF_INTERVAL_SEC)
    resize = (coarse.get("resize_width", DEFAULT_RESIZE[0]),
              coarse.get("resize_height", DEFAULT_RESIZE[1]))
    rois_cfg = coarse.get("rois", DEFAULT_ROIS_CFG)
    feature_flags = coarse.get("features", DEFAULT_FEATURE_FLAGS)
    return kf_subsample, kf_interval, resize, rois_cfg, feature_flags


def build_keyframe_cmd(video_path: str, out_w: int, out_h: int) -> list[str]:
    return [
        FFMPEG_PATH,
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vf", f"scale={out_w}:{out_h}",
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-v", "quiet",
        "-",
    ]


def extract_all_features_for_frame(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    roi_slices: dict[str, tuple],
    prev_grays: dict[str, np.ndarray | None],
    feature_flags: dict[str, bool],
    canny_low: int = 50,
    canny_high: int = 150,
) -> list[dict]:
    rows = []
    for roi_name, roi_sl in roi_slices.items():
        roi_bgr = frame_bgr[roi_sl]
        roi_gray = gray[roi_sl]

        row: dict = {"roi_name": roi_name}

        if feature_flags.get("brightness", True):
            row["mean_brightness"] = float(roi_gray.mean())
            row["brightness_std"] = float(roi_gray.std())

        if feature_flags.get("color", True):
            row["mean_b"] = float(roi_bgr[:, :, 0].mean())
            row["mean_g"] = float(roi_bgr[:, :, 1].mean())
            row["mean_r"] = float(roi_bgr[:, :, 2].mean())

        if feature_flags.get("motion", True):
            row["motion_intensity"] = compute_motion_intensity(roi_gray, prev_grays.get(roi_name))

        if feature_flags.get("edge", True):
            row["edge_density"] = compute_edge_density(roi_gray, canny_low, canny_high)

        if feature_flags.get("texture", True):
            row["laplacian_variance"] = compute_laplacian_variance(roi_gray)

        if feature_flags.get("entropy", True):
            row["entropy"] = compute_entropy(roi_gray)

        rows.append(row)
        prev_grays[roi_name] = roi_gray.copy()
    return rows


def process_video_coarse(
    video_path: str,
    out_dir: str,
    resize: tuple[int, int] = DEFAULT_RESIZE,
    rois_cfg: dict[str, dict] | None = None,
    feature_flags: dict[str, bool] | None = None,
    kf_subsample: int = DEFAULT_KF_SUBSAMPLE,
    kf_interval_sec: float = DEFAULT_KF_INTERVAL_SEC,
) -> dict:
    video_id = os.path.basename(video_path)
    out_w, out_h = resize

    if rois_cfg is None:
        rois_cfg = DEFAULT_ROIS_CFG
    if feature_flags is None:
        feature_flags = DEFAULT_FEATURE_FLAGS

    roi_slices = {}
    for name, r in rois_cfg.items():
        y1, y2 = int(r["y_min"] * out_h), int(r["y_max"] * out_h)
        x1, x2 = int(r["x_min"] * out_w), int(r["x_max"] * out_w)
        roi_slices[name] = (slice(y1, y2), slice(x1, x2))

    cmd = build_keyframe_cmd(video_path, out_w, out_h)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    frame_size = out_w * out_h * 3
    kf_idx = 0
    sample_idx = 0
    records = []
    prev_grays: dict[str, np.ndarray | None] = {n: None for n in roi_slices}

    canny_low = feature_flags.get("canny_low", 50)
    canny_high = feature_flags.get("canny_high", 150)

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        if kf_idx % kf_subsample != 0:
            kf_idx += 1
            continue

        frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        timestamp_sec = kf_idx * kf_interval_sec

        rows = extract_all_features_for_frame(
            frame_bgr, gray, roi_slices, prev_grays, feature_flags,
            canny_low=canny_low, canny_high=canny_high,
        )
        for r in rows:
            r["video_id"] = video_id
            r["frame_idx"] = sample_idx
            r["timestamp_sec"] = timestamp_sec
        records.extend(rows)

        kf_idx += 1
        sample_idx += 1

    proc.wait()

    if not records:
        return {"video_id": video_id, "status": "empty", "frames": 0, "rows": 0}

    df = pd.DataFrame(records)
    float_cols = [c for c in df.columns if c not in ("video_id", "roi_name", "frame_idx")]
    for c in float_cols:
        df[c] = df[c].astype(np.float32)
    df["frame_idx"] = df["frame_idx"].astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, video_id.replace(".mp4", ".parquet"))
    df.to_parquet(parquet_path, index=False, engine="pyarrow")

    return {
        "video_id": video_id,
        "status": "ok",
        "frames": sample_idx,
        "rows": len(df),
        "parquet_path": parquet_path,
    }


def run_coarse_pipeline(
    video_paths: list[str],
    out_dir: str,
    resize: tuple[int, int] = DEFAULT_RESIZE,
    rois_cfg: dict | None = None,
    feature_flags: dict | None = None,
    kf_subsample: int = DEFAULT_KF_SUBSAMPLE,
    kf_interval_sec: float = DEFAULT_KF_INTERVAL_SEC,
    num_workers: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    if num_workers <= 1:
        for vp in video_paths:
            result = process_video_coarse(
                vp, out_dir, resize, rois_cfg, feature_flags, kf_subsample, kf_interval_sec
            )
            if result["status"] == "ok":
                print(f"  {result['video_id']}: {result['frames']} frames, {result['rows']} rows")
            elif "error" in result["status"]:
                print(f"  {result['video_id']}: {result['status']}")
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for vp in video_paths:
                fut = executor.submit(
                    process_video_coarse, vp, out_dir, resize, rois_cfg, feature_flags,
                    kf_subsample, kf_interval_sec,
                )
                futures[fut] = os.path.basename(vp)
            for fut in as_completed(futures):
                result = fut.result()
                if result["status"] == "ok":
                    print(f"  {result['video_id']}: {result['frames']} frames, {result['rows']} rows")
                else:
                    print(f"  {result['video_id']}: {result['status']}")

    elapsed = time.perf_counter() - t0
    print(f"Done {len(video_paths)} videos in {elapsed:.0f}s")
    return elapsed


def run_coarse_pipeline_from_config(
    config_path: str | None = None,
    video_root: str | None = None,
    out_dir: str | None = None,
    max_videos: int | None = None,
    num_workers: int | None = None,
):
    """Run Layer 1 using the unified pipeline config.

    Reads paths + params from pipeline.yaml, with optional overrides
    for video_root / out_dir. This is the recommended entry point.
    """
    cfg = load_pipeline_config(config_path)
    kf_subsample, kf_interval, resize, rois_cfg, feature_flags = _params_from_config(cfg)

    paths = cfg.get("paths", {})
    video_root = video_root or paths.get("video_root", "")
    out_dir = out_dir or os.path.join(
        paths.get("output_root", "outputs"),
        f"coarse_{cfg.get('night_tag', 'default')}",
    )
    if num_workers is None:
        num_workers = cfg.get("num_workers", 4)

    video_paths = sorted(Path(video_root).glob("*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"No mp4 files under {video_root}")
    video_paths = [str(v) for v in video_paths]
    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"Layer 1: extracting coarse features from {len(video_paths)} videos")
    print(f"  resize={resize}  kf_subsample={kf_subsample}  kf_interval={kf_interval}s")
    return run_coarse_pipeline(
        video_paths, out_dir, resize, rois_cfg, feature_flags,
        kf_subsample, kf_interval, num_workers,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Layer 1: Coarse feature extraction")
    parser.add_argument("--config", default=None, help="Path to pipeline.yaml")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_coarse_pipeline_from_config(
        config_path=args.config, video_root=args.video_root,
        out_dir=args.out_dir, max_videos=args.max_videos, num_workers=args.workers,
    )