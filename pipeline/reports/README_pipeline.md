# Pipeline — 三层式智慧渔业分析系统

The pipeline analyses an **entire night** of fishing-vessel deck video
and outputs: net-cycle count, haul/set times, core catch windows, and
(planned) catch-volume KPIs.

## Design principle

This is **not** a video classification system. It is a
**periodic-activity-behaviour analysis** system:

```
    state machine  +  time-series features  +  targeted visual analysis
```

Heavy deep-learning models (SAM2, YOLO, etc.) are restricted to the
**core catch window** (net-on-deck → fish-transfer-end), dramatically
reducing cost compared to whole-night frame-by-frame inference.

## Three-layer architecture

| Layer | Name                       | Input                  | Output                    | Status   |
| ----- | -------------------------- | ---------------------- | ------------------------- | -------- |
| 1     | Coarse feature extraction  | Night video (.mp4)     | `*.parquet` per video     | ✅ Done  |
| 2     | Cycle detection            | Parquet from Layer 1   | `detected_cycles.csv`     | ✅ Done  |
| 3     | Core-window extraction     | Cycles + Parquet       | `core_catch_windows.csv`  | ✅ Done  |
| 4     | Fine-grained analysis      | Core windows + video   | KPIs (catch, baskets...)  | ⏳ Next  |

## Get started

```bash
cd J:/video_auto

# One-shot: run L1 → L2 → L3 for the night specified in pipeline.yaml
python pipeline/run_night.py

# Or override per-run:
python pipeline/run_night.py --night-tag 0515 --video-root //DS224plus/video/0515 --workers 4

# Skip L1 (reuse existing parquet), only re-run L2 + L3:
python pipeline/run_night.py --skip-l1 --night-tag 0515

# Individual layers (programmatic):
python -c "
from pipeline.coarse_features.feature_merge import run_coarse_pipeline_from_config
run_coarse_pipeline_from_config(num_workers=4)
"

python -c "
from pipeline.cycle_detection.detect_cycles import run_cycle_detection
from pathlib import Path
run_cycle_detection(
    parquet_root=Path('outputs/coarse_0515'),
    out_dir=Path('outputs/coarse_0515/net_cycle_detection'),
)
"

python -c "
from pipeline.core_window_detection.detect_core_windows import run_core_window_detection
from pathlib import Path
run_core_window_detection(
    parquet_root=Path('outputs/coarse_0515'),
    cycles_csv=Path('outputs/coarse_0515/net_cycle_detection/detected_cycles.csv'),
    out_dir=Path('outputs/coarse_0515/core_catch_windows'),
)
"

# Generate review videos (requires ffmpeg + source videos on NAS)
python pipeline/core_window_detection/review_video_generator.py
```

## Directory layout

```
pipeline/
├── coarse_features/        ← Layer 1: keyframe-only feature extraction
│   ├── extract_motion.py
│   ├── extract_edge.py
│   ├── extract_entropy.py
│   └── feature_merge.py
├── cycle_detection/        ← Layer 2: valley-based cycle detection
│   ├── detect_cycles.py
│   ├── valley_detection.py
│   ├── fft_analysis.py
│   └── cycle_visualization.py
├── core_window_detection/  ← Layer 3: net-on-deck → fish-end windows
│   ├── detect_core_windows.py
│   ├── phase_segmentation.py
│   ├── change_point_detection.py
│   └── review_video_generator.py
├── fine_grained_analysis/  ← Layer 4: SAM2, YOLO, regression (stubs)
│   ├── net_size_estimation/
│   ├── basket_counting/
│   ├── catch_volume_estimation/
│   ├── fish_species_detection/
│   └── catch_density_analysis/
├── visualization/          ← Dashboards, debug renders (stubs)
├── shared/
│   ├── configs/pipeline.yaml  ← *the* unified config
│   ├── utils/
│   └── io/
└── reports/
    ├── README_pipeline.md     ← you are here
    └── pipeline_status.md     ← completion status per module
```

## Key outputs (0515 night, validated)

| Metric                        | Value        |
| ----------------------------- | ------------ |
| Cycles detected               | 9            |
| Mean cycle period (FFT)       | ~50 min      |
| Mean core window              | 20.9 min     |
| Mean net-rising duration      | 2.6 min      |
| Mean fish-transfer duration   | 15.2 min     |
| High-confidence cycles        | 4 of 9       |
| Low-confidence cycles         | 2 of 9 (7, 9)|

## Legacy code

- `src/` — original Phase-1 implementation (video reader, full-rate
  feature extraction, benchmarks). Kept for reference.
- `outputs/coarse_0515/` — results + scripts from the 0515 night
  analysis. These are the reference outputs against which the new
  pipeline modules were validated.
- `configs/config.yaml` — old config schema. The canonical config is
  now `pipeline/shared/configs/pipeline.yaml`.

## What to build next

See `pipeline_status.md` for priority-ordered next steps.
Priority 1: **net size estimation** with SAM2 on the left-deck ROI
at net_on_deck_time.