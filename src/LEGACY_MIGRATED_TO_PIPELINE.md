# LEGACY CODE — DO NOT USE

All logic here has been superseded by `pipeline/`.

Pipe visual reference:
  src/coarse_pipeline.py     →  pipeline/coarse_features/feature_merge.py
  src/extract_features.py    →  pipeline/coarse_features/feature_merge.py  (multiprocessing)
  src/video_reader.py        →  pipeline/shared/io/  (video_scanner.py)
  src/scan_videos.py         →  pipeline/shared/io/video_scanner.py
  src/visualize_features.py  →  pipeline/cycle_detection/cycle_visualization.py

You can safely delete this file. Keep for reference only.