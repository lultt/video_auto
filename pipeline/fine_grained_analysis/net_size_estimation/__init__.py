"""Net size estimation package — Layer 4 fine-grained visual analysis.

Estimates the visible net size (pixel area) on the left deck during the
core catch window (net on deck → fish transfer end) for each haul cycle.

Modules:
  extract_keyframes.py   — sample sparse keyframes from core windows
  sam2_segmentation.py   — SAM2 point-prompted segmentation (B)
  net_mask_tracking.py   — IoU-based mask consistency filter (D)
  net_area_estimation.py — pixel-based area computation (C)
  temporal_smoothing.py  — rolling median + interpolation (C)
  volume_proxy_estimation.py — area × duration × motion → size class (E)
  visualization.py       — overlay videos, area curves, report (E)
"""

from .extract_keyframes import run_extract_keyframes

__all__ = ["run_extract_keyframes"]