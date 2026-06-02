"""
extract_motion.py — frame-difference motion intensity for coarse pipeline.

Motion = mean(absdiff(this_gray - prev_gray)) for each ROI slice.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_motion_intensity(roi_gray: np.ndarray, prev_gray_roi: np.ndarray | None) -> float:
    """
    Per-frame motion: pixelwise absdiff mean between this ROI and the previous one.
    Returns 0.0 for the first frame of a segment (no predecessor).
    """
    if prev_gray_roi is None:
        return 0.0
    diff = cv2.absdiff(roi_gray, prev_gray_roi)
    return float(diff.mean())
