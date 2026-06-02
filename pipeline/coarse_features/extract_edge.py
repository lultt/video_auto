"""
extract_edge.py — Canny edge density + Laplacian variance for coarse pipeline.

edge_density   ≈ fraction of pixels classified as edge (Canny 50/150).
laplacian_var  ≈ texture sharpness proxy.
"""

import cv2
import numpy as np


def compute_edge_density(roi_gray: np.ndarray, canny_low: int = 50, canny_high: int = 150) -> float:
    """Fraction of pixels above the Canny edge threshold."""
    edges = cv2.Canny(roi_gray, canny_low, canny_high)
    return float(edges.sum() / 255.0 / edges.size)


def compute_laplacian_variance(roi_gray: np.ndarray) -> float:
    """Variance of the Laplacian (focus / texture sharpness proxy)."""
    return float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
