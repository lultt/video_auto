"""Run suspended net geometry on all 9 cycles."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
os.environ["PATH"] = r"C:\Users\ljj\anaconda3\envs\yolonew\Library\bin;" + os.environ.get("PATH", "")
from pipeline.fine_grained_analysis.net_size_estimation.suspended_net_geometry import run_suspended_net_geometry

summary = run_suspended_net_geometry(
    core_catch_windows_csv=Path(r"J:\video_auto\outputs\coarse_0515\core_catch_windows\core_catch_windows.csv"),
    video_root=Path(r"\\DS224plus\video\0515"),
    checkpoint_path=Path(r"J:\video_auto\third_party\sam2\checkpoints\sam_vit_b_01ec64.pth"),
    out_dir=Path(r"J:\video_auto\outputs\coarse_0515\fine\net_size\suspended_net"),
)
print(f"\nFinal summary: {summary}")