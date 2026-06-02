"""
extract_entropy.py — Shannon entropy of grayscale histogram.

High entropy = complex texture (e.g. net mesh, busy deck).
Low entropy  = uniform surface (e.g. empty deck, still water).
"""

import cv2
import numpy as np


def compute_entropy(roi_gray: np.ndarray) -> float:
    """Shannon entropy (bits) of the 8-bit grayscale histogram."""
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    nonzero = hist[hist > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))