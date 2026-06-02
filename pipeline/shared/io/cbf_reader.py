"""
CBF reader — read Columnar Binary Format (.cbf) files from C++ pipeline.

Supports:
  v1 (CBF\x01): 11 columns, no roi_code — whole-frame features
  v2 (CBF\x02): 12 columns, includes roi_code — per-ROI features (left/right)
"""
import struct
import numpy as np
import pandas as pd
from pathlib import Path


_V2_COL_NAMES = [
    "frame_idx", "roi_code", "timestamp_sec",
    "mean_brightness", "brightness_std",
    "mean_b", "mean_g", "mean_r",
    "motion_intensity", "edge_density",
    "laplacian_variance", "entropy",
]

_V1_COL_NAMES = [
    "frame_idx", "timestamp_sec",
    "mean_brightness", "brightness_std",
    "mean_b", "mean_g", "mean_r",
    "motion_intensity", "edge_density",
    "laplacian_variance", "entropy",
]


def read_cbf(filepath):
    with open(filepath, "rb") as f:
        magic = f.read(4).decode("ascii")
        if magic == "CBF\x02":
            return _read_cbf_v2(f)
        elif magic == "CBF\x01":
            return _read_cbf_v1(f)
        else:
            raise ValueError(f"Not a CBF v1/v2 file: {magic}")


def _read_cbf_v1(f):
    """Read CBF v1: 76-byte header, 11 columns, no roi_code."""
    # remaining header after magic
    tail = f.read(72)
    n_rows = struct.unpack_from("<I", tail, 0)[0]
    n_cols = struct.unpack_from("<I", tail, 4)[0]
    offsets = struct.unpack_from("<12I", tail, 8)

    if n_cols != 11:
        raise ValueError(f"Expected 11 columns, got {n_cols}")

    col_names = _V1_COL_NAMES
    col_dtypes = [np.uint32] + [np.float32] * 10

    data = {}
    for i in range(n_cols):
        size = offsets[i + 1] - offsets[i]
        f.seek(offsets[i])
        buf = f.read(size)
        arr = np.frombuffer(buf, dtype=col_dtypes[i])
        if len(arr) != n_rows:
            arr = arr[:n_rows]
        data[col_names[i]] = arr

    df = pd.DataFrame(data)
    return df


def _read_cbf_v2(f):
    """Read CBF v2: 80-byte header, 12 columns, includes roi_code."""
    # remaining header after magic
    tail = f.read(76)
    n_rows = struct.unpack_from("<I", tail, 0)[0]
    n_cols = struct.unpack_from("<I", tail, 4)[0]
    offsets = struct.unpack_from("<13I", tail, 8)

    if n_cols != 12:
        raise ValueError(f"Expected 12 columns, got {n_cols}")

    col_dtypes = [np.uint32, np.uint32] + [np.float32] * 10

    data = {}
    for i in range(n_cols):
        size = offsets[i + 1] - offsets[i]
        f.seek(offsets[i])
        buf = f.read(size)
        arr = np.frombuffer(buf, dtype=col_dtypes[i])
        if len(arr) != n_rows:
            arr = arr[:n_rows]
        data[_V2_COL_NAMES[i]] = arr

    df = pd.DataFrame(data)
    df["roi_name"] = df["roi_code"].map({0: "left", 1: "right"})
    return df


if __name__ == "__main__":
    import sys
    for fp in sys.argv[1:]:
        df = read_cbf(fp)
        print(f"{fp}: {len(df)} rows x {len(df.columns)} cols")
        print(df.head(5))
        print()
