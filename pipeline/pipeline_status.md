# Pipeline Status — Fishing-vessel Deck-Video Analysis

_Last updated: 2026-05-26_

This document tracks what is **done**, what is **partial**, and what is
**planned** across the 3-layer pipeline. It supersedes the original
single-stage description in `/README.md`.

---

## Architecture at a glance

```
[ entire night video ]
        │
        ▼
┌──────────────────────────────────────────┐
│ Layer 1 — Coarse Feature Extraction      │   ✅ DONE
│   coarse_features/                    │
│   keyframe-only decode, ROI features     │
└──────────────────────────────────────────┘
        │   *.parquet per video
        ▼
┌──────────────────────────────────────────┐
│ Layer 2 — Cycle Detection                │   ✅ DONE
│   cycle_detection/                    │
│   valley-to-valley boundaries on motion  │
└──────────────────────────────────────────┘
        │   detected_cycles.csv
        ▼
┌──────────────────────────────────────────┐
│ Layer 3 — Core-Window Detection          │   ✅ DONE (1.0)
│   core_window_detection/              │
│   net-on-deck → fish-transfer-end        │
└──────────────────────────────────────────┘
        │   core_catch_windows.csv
        ▼
┌──────────────────────────────────────────┐
│ Layer 4 — Fine-grained Visual Analysis   │   ⏳ NEXT
│   fine_grained_analysis/              │
│   SAM2 / YOLO / regression — CORE ONLY   │
└──────────────────────────────────────────┘
```

---

## Layer 1 — Coarse feature extraction

**Status:** ✅ COMPLETE — currently the best version of the system.

| Item                          | Path                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| Per-feature extractors        | `pipeline/coarse_features/extract_{motion,edge,entropy}.py`    |
| Orchestrator                  | `pipeline/coarse_features/feature_merge.py`                    |
| Legacy single-file version    | `src/coarse_pipeline.py` (kept for reference)                     |
| Sample output (one night)     | `outputs/coarse_0515/*.parquet`                                   |

**Best current method:**
- ffmpeg `-skip_frame nokey` → only I-frames decoded
- `kf_subsample = 3` → effective sampling ≈ 1 frame / 12 s
- No frame stabilization (interval too large to be useful)
- 3 ROIs (`left_deck`, `center_deck`, `right_deck`)
- Features computed: brightness mean/std, color BGR means,
  motion intensity (absdiff), edge density (Canny),
  Laplacian variance, Shannon entropy.
- Throughput: ~60× realtime per worker.

**Known issues / not-yet:**
- No GPU decode path enabled (NVDEC available but disabled).
- No per-feature unit tests.
- The full-night batch runner (`run_extraction(...)`) lives in
  `src/extract_features.py` but uses the old config schema.
  → TODO: port to `pipeline/coarse_features/feature_merge.py`.

---

## Layer 2 — Cycle detection

**Status:** ✅ COMPLETE — stable on 0515 night (9 cycles found).

| Item                       | Path                                                  |
| -------------------------- | ----------------------------------------------------- |
| Valley/peak primitives     | `pipeline/cycle_detection/valley_detection.py`     |
| FFT periodicity helpers    | `pipeline/cycle_detection/fft_analysis.py`         |
| Orchestrator               | `pipeline/cycle_detection/detect_cycles.py`        |
| Diagnostic plots           | `pipeline/cycle_detection/cycle_visualization.py`  |
| Sample output (0515 night) | `outputs/coarse_0515/net_cycle_detection/`            |
| Legacy peak-based v1       | `outputs/coarse_0515/net_cycle_detection_script.py`   |

**Best current method:** valley-to-valley segmentation (v2).
- Smooth motion signal (Savitzky-Golay, window=11).
- Find valleys (negated peaks) with `prominence=1.2`, `distance=25 min`,
  filtered to be below the 35th percentile of motion (rejects daytime
  pseudo-valleys).
- Each valley→valley span = 1 cycle.
- Strongest motion peak within each segment = activity anchor.

**Verified on:** 0515 night, 9 net cycles, FFT-confirmed period ≈ 50 min.

**Known issues:**
- 2 of 9 cycles span > 60 min — possibly merged (cycles 3 & 4).
- Hardcoded night-window date in legacy v2; the new
  `detect_cycles.py` derives it from the data.

---

## Layer 3 — Core-window detection

**Status:** ✅ FUNCTIONAL — 1.0; most cycles correct, weak-signal cycles
have ±2 min boundary error.

| Item                  | Path                                                          |
| --------------------- | ------------------------------------------------------------- |
| Phase segmentation    | `pipeline/core_window_detection/phase_segmentation.py`     |
| Change-point helpers  | `pipeline/core_window_detection/change_point_detection.py` |
| Orchestrator          | `pipeline/core_window_detection/detect_core_windows.py`    |
| Review-video clipper  | `pipeline/core_window_detection/review_video_generator.py` |
| Per-cycle stage plots | `outputs/coarse_0515/core_catch_windows/cycle_*_stages.png`   |
| Stage logic doc       | `outputs/coarse_0515/core_catch_windows/core_window_detection_logic.md` |

**Best current method:**
- `left_deck_edge_smooth` rise → "net rising"
- `left_deck_edge_smooth` peak (within first 70 % of cycle) → "net on deck"
- `right_deck_edge_smooth` rise (after net_on_deck) → "fish transfer start"
- `right_deck_edge_smooth` fall back to baseline → "fish transfer end"
- Core window = `net_on_deck_time` → `fish_end_time`.

**Output statistics (0515 night, 9 cycles):**
- core window: mean 21 min, range 11–27 min
- net-rising:   mean 2.6 min, range 1–5 min
- fish transfer: mean 15.2 min, range 11–21 min
- right-edge increase vs pre-window: mean ×1.10, range ×1.01–×1.17

**Known issues / failure cases:**
- Cycle 7 and Cycle 9 have `right_edge_increase_vs_pre ≈ 1.01–1.03`
  → very weak fish-transfer signal, boundary may be unreliable.
- Cycle 3 spans 71 min total — likely two cycles merged at Layer 2.

**Engineering rule (going forward):** _do NOT keep tuning the boundaries._
9 cycles are stably recovered; further tuning yields diminishing returns.

---

## Layer 4 — Fine-grained visual analysis (NEXT)

**Status:** ⏳ NOT STARTED — design priorities below.

> **All Layer-4 models run ONLY inside the core window, ideally only
> inside the fish-transfer sub-window.** This is the central
> cost-control principle of the new architecture.

| Priority | Module                                              | Status | Why                                      |
| -------- | --------------------------------------------------- | ------ | ---------------------------------------- |
| **P1**   | `net_size_estimation/` (SAM2)                       | stub   | Easiest, strong catch correlation        |
| **P2**   | `basket_counting/` (YOLOv8n)                        | stub   | Direct production KPI                    |
| **P3**   | `catch_volume_estimation/` (regression)             | stub   | Fuses P1, P2, durations                  |
| **P4**   | `fish_species_detection/`                           | stub   | Hardest; requires labelled data          |
| —        | `catch_density_analysis/` (empty-haul heuristic)    | stub   | Cheap, helps QC                          |

---

## Layer 5 — Visualization

**Status:** Existing exploratory plots in
`outputs/coarse_0515/explore_plots/`. The pipeline's own per-stage
diagnostic plots are produced by Layer 2 and Layer 3 directly.

No new visualization modules yet — `pipeline/visualization/`
currently contains only a README; specific dashboards will be added
as Layer 4 produces structured KPIs.

---

## What changed in this restructure (2026-05-26 v2)

### v1 (earlier today)
- New canonical layout under `pipeline/` with `01_` / `02_` numeric prefixes.
- Single config: `pipeline/shared/configs/pipeline.yaml`.
- Shared utilities (`shared/utils/`, `shared/io/`).
- Stage-specific scripts split into single-responsibility modules.

### v2 (now)
- **Dropped numeric prefix** directories — `coarse_features/`, `cycle_detection/`,
  `core_window_detection/`, `fine_grained_analysis/`, `visualization/`.
  These are now proper Python packages with relative imports.
- **`shared/utils/` split** from monolithic `__init__.py` into `time_utils.py`,
  `config_loader.py`, clean `__init__.py` re-export.
- **`shared/io/__init__.py` fixed** — missing imports (`pd`, `Path`, `timedelta`,
  `parse_video_filename_timestamp` naming bug) resolved.
- **All `sys.path` hacks removed** from L1 / L2 / L3 modules — replaced with
  proper relative imports (`from ..shared.io import ...`).
- **`feature_merge.py` wired to unified config** — reads `coarse_features.*`
  section from `pipeline.yaml` instead of hardcoding params.
- **Multiprocessing added** — `run_coarse_pipeline()` supports `num_workers`
  via `ProcessPoolExecutor`.
- **`pipeline/run_night.py`** — single entry point for L1 → L2 → L3.
- **`pipeline/shared/io/video_scanner.py`** — video manifest scanner ported
  from `src/scan_videos.py`.
- **`src/`** is kept as legacy reference (no pipeline imports it).

---

## What to work on next

1. **Port `coarse_features/feature_merge.py` to support parallel
   processing** (current `feature_merge.process_video_coarse()` is
   single-process).
2. **Implement `fine_grained_analysis/net_size_estimation/`** —
   SAM2 on the left-deck frame at `net_on_deck_time`.
3. After P1 lands: implement basket counting (P2).
4. Build a small `visualization/dashboards/` page showing
   per-night cycle table + KPI overlays.

---

## What NOT to do

- Don't add full-night deep-learning inference. All heavy CV runs
  inside the core window.
- Don't keep refining Layer-2 / Layer-3 boundary detection. 9-out-of-9
  cycles are recovered; weak signals are inherent to those cycles.
- Don't add a transformer-style end-to-end video classifier. The
  project is a periodic-behaviour analysis system, not a video
  understanding system.