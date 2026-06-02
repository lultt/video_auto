# Net Size Estimation — Layer 4 Priority 1

Estimates the visible net size (pixel area) on the left deck during the core
catch window for each haul cycle.

**Core principle:** only runs inside the core window (net on deck → fish end).
Never processes full-night video.

## Pipeline position

```
L3 core_catch_windows.csv  +  source videos (NAS)
        │
        ▼
  extract_keyframes.py       ← Phase A (this module)
        │
        ▼
  sam2_segmentation.py       ← Phase B (SAM2 required)
        │
        ▼
  net_area_estimation.py  →  temporal_smoothing.py  →  volume_proxy_estimation.py
        │
        ▼
  net_size_per_cycle.csv     ← headline output (area_px, size_class, volume_index)
```

## Install requirements

### SAM2 (one-time)

```bash
conda activate yolonew

# 1. Clone & install
mkdir -p third_party
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
cd third_party/sam2
pip install -e .

# 2. Download checkpoint (~80 MB)
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt

# 3. Verify
python -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"
```

Other deps already in `yolonew`: PyTorch 2.0.1, OpenCV, pandas, numpy.

## Usage

### Phase A: extract keyframes (no SAM2 needed)

```bash
python pipeline/fine_grained_analysis/net_size_estimation/extract_keyframes.py \
  --core-windows outputs/coarse_0515/core_catch_windows/core_catch_windows.csv \
  --video-root //DS224plus/video/0515 \
  --out-dir outputs/coarse_0515/fine/net_size/keyframes
```

Output: `keyframe_index.csv` + per-cycle PNG directories
- `cycle_NN/frame_XX_NNN_full.png` — full frame at that timestamp
- `cycle_NN/frame_XX_NNN_roi.png` — left_deck ROI crop (SAM2 input)

### Phase B–F (after SAM2 installed, TODO)

```bash
python pipeline/fine_grained_analysis/net_size_estimation/run_net_size.py \
  --night-tag 0515
```

## Outputs

```
outputs/coarse_{night}/fine/net_size/
├── keyframes/
│   ├── keyframe_index.csv
│   └── cycle_NN/
│       ├── frame_NN_NNN_full.png
│       └── frame_NN_NNN_roi.png
├── masks_raw/           (Phase B)
├── masks_tracked/       (Phase D)
├── area_raw/            (Phase C)
├── area_smoothed/       (Phase C)
├── net_size_per_cycle.csv   (Phase E — headline KPI)
├── visualizations/      (Phase E)
└── net_size_report.md   (Phase E)
```

## Configuration

All params in `pipeline/shared/configs/pipeline.yaml` → `fine_grained.net_size_estimation`:

| Key | Purpose |
|-----|---------|
| `sam2_checkpoint` | Path to SAM2 model weights |
| `sampling.*` | Which frames to extract per cycle (net_landing/settled/fish_transfer) |
| `segmentation.*` | SAM2 prompt mode, point position, area bounds |
| `tracking.*` | IoU thresholds for mask consistency filter |
| `smoothing.*` | Temporal smoothing method |
| `volume_proxy.*` | Size classification method (terciles/fixed) |

## Known limits (V1)

- **Area in pixels only** — m² calibration requires on-deck reference measurement (deferred)
- **Within-night size classes** — no cross-night normalization yet
- **Weak cycles (7, 9)** flagged but not dropped; expect lower confidence
- **SAM2 single-frame mode** — not using the video predictor (frames are too sparse)

## Next after V1

- On-deck reference measurement for m² calibration
- Cross-night absolute size classes
- Correlation analysis: area vs. fish_transfer_duration vs. catch weight
- SAM2 video predictor for denser temporal sampling
