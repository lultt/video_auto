# config.py — central parameters for night fishing detection pipeline
from pathlib import Path

# ---- Paths ----
OUTPUT_ROOT = Path(r"J:\video_auto\outputs")
NIGHT_OUT   = Path(r"J:\video_auto\outputs\night_fishing")
PLOT_DIR    = NIGHT_OUT / "plots"
CSV_DIR     = NIGHT_OUT / "csv"

# ---- Night definition ----
NIGHT_START_HOUR = 18   # day boundary: 18:00
NIGHT_END_HOUR   = 6    # next day 06:00

# ---- FAI (Fishing Activity Index) weights ----
W_MOTION   = 0.50
W_BSTD     = 0.25
W_ENTROPY  = 0.25

# ---- Smoothing ----
SG_WINDOW = 7   # Savitzky-Golay window (odd, in samples)
SG_POLY   = 3   # polynomial order

# ---- Fishing detection ----
ROLLING_WINDOW = 20    # samples for rolling statistics
FAI_VAR_THRESHOLD = 0.15  # min rolling std of FAI to classify as fishing

# ---- Activity segment detection (net cycle = continuous high-FAI segment) ----
ACTIVITY_THRESHOLD   = 0.55   # FAI_smooth > this = "active" in the binary mask
MERGE_GAP_MINUTES    = 25     # merge segments separated by <= this many minutes
MIN_EVENT_DURATION_MINUTES = 12  # min segment duration to count as a real net event
MIN_EVENT_AUC        = 5.0    # min area under FAI curve within the segment

# ---- Valley-based cycle detection (primary method) ----
VALLEY_DISTANCE     = 30     # min samples (minutes) between valleys
VALLEY_PROMINENCE   = 1.5    # min depth of a valley (motion units)
VALLEY_DEPTH_PCT    = 40.0   # valley must be below this motion percentile
SG_MOTION_WINDOW    = 11     # Savitzky-Golay window for 1-min motion
SG_MOTION_POLYORDER = 2      # SG polynomial order
FFT_PERIOD_LOW      = 30.0   # min plausible net-cycle period (minutes)
FFT_PERIOD_HIGH     = 120.0  # max plausible net-cycle period (minutes)

# ---- Plotting ----
PLOT_DPI = 150

# ---- ROI selection ----
# Use left_deck + right_deck only (center_deck is mast/obstructed)
ROIS = ["left_deck", "right_deck"]

# ---- Night completeness checks ----
MIN_DURATION_HOURS = 4.0        # night must span at least 4 hours
MIN_SAMPLE_COUNT   = 30         # at least 30 keyframes
MAX_GAP_MINUTES    = 180        # warn if any gap > 180 min (allow May 16's 148min gap)
MIN_COVERAGE_RATIO = 0.33       # actual span / expected 12h span must be > 0.33
