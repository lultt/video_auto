# detector.py — valley-to-valley net cycle detection (v3)

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import rfft, rfftfreq
from .config import (
    ROLLING_WINDOW, FAI_VAR_THRESHOLD, ACTIVITY_THRESHOLD,
    MIN_DURATION_HOURS, MIN_SAMPLE_COUNT, MAX_GAP_MINUTES, MIN_COVERAGE_RATIO,
    VALLEY_DISTANCE, VALLEY_PROMINENCE, VALLEY_DEPTH_PCT,
    SG_MOTION_WINDOW, SG_MOTION_POLYORDER,
    FFT_PERIOD_LOW, FFT_PERIOD_HIGH,
)


# ---------------------------------------------------------------------------
# Valley-based net cycle detection
# ---------------------------------------------------------------------------
def _smooth(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    w = min(window, len(values) - (1 - len(values) % 2))
    if w < 5:
        return values.copy()
    try:
        return savgol_filter(values, w, polyorder, mode="interp")
    except Exception:
        return pd.Series(values).rolling(w, center=True, min_periods=1).mean().values


def _build_minute_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1-minute grid, average motion across left+right ROI."""
    df = df.set_index("abs_time").sort_index()
    # Average left+right motion for each timestamp first
    minute = df["motion_intensity"].resample("1min").mean()
    idx = pd.date_range(minute.index.min().floor("1min"),
                         minute.index.max().ceil("1min"), freq="1min")
    minute = minute.reindex(idx).interpolate(limit=5).ffill().bfill()
    return minute.to_frame(name="motion")


def detect_valleys(motion: np.ndarray, min_distance: int = 20,
                   prominence: float = 1.0, depth_percentile: float = 30.0) -> np.ndarray:
    """Find deep valleys in the motion signal (cycle boundaries)."""
    if len(motion) < min_distance * 2:
        return np.array([], dtype=int)
    valleys, _ = find_peaks(-motion, distance=min_distance, prominence=prominence)
    threshold = np.percentile(motion, depth_percentile)
    valleys = np.array([v for v in valleys if motion[v] < threshold])
    return valleys


def segment_cycles(valleys: np.ndarray, motion: np.ndarray,
                   times: pd.DatetimeIndex) -> list[dict]:
    """Build cycle entries from valley→valley segments."""
    cycles = []
    for i in range(len(valleys) - 1):
        v_start = valleys[i]
        v_end   = valleys[i + 1]

        seg_motion = motion[v_start:v_end + 1]
        peak_offset = int(np.argmax(seg_motion))
        peak_idx = v_start + peak_offset
        peak_val = float(motion[peak_idx])

        dur_min = (times[v_end] - times[v_start]).total_seconds() / 60.0
        auc = float(np.trapz(seg_motion))

        cycles.append({
            "net_index": i + 1,
            "start_idx": int(v_start),
            "peak_idx": peak_idx,
            "end_idx": int(v_end),
            "start_time": times[v_start],
            "peak_time": times[peak_idx],
            "end_time": times[v_end],
            "duration_minutes": round(dur_min, 1),
            "peak_motion": round(peak_val, 1),
            "auc": round(auc, 1),
            "mean_motion": round(float(np.mean(seg_motion)), 1),
        })
    return cycles


def fft_periodicity(motion: np.ndarray, period_low: float = 15.0,
                    period_high: float = 180.0) -> tuple:
    """Return (best_period_min, peak_power, mean_power, snr).
    snr = peak_power / mean_power; high snr → strong periodicity.
    """
    motion = motion - np.nanmean(motion)
    if len(motion) < 8:
        return float("nan"), 0.0, 0.0, 0.0
    freqs = rfftfreq(len(motion), d=60.0)
    power = np.abs(rfft(motion))
    period_min = np.full_like(freqs, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = freqs > 0
        period_min[mask] = 1.0 / freqs[mask] / 60.0
    valid = (period_min >= period_low) & (period_min <= period_high) & np.isfinite(period_min)
    if not np.any(valid):
        return float("nan"), 0.0, 0.0, 0.0
    p_valid = power[valid]
    best_idx = np.argmax(p_valid)
    best_period = float(period_min[valid][best_idx])
    peak_power = float(p_valid[best_idx])
    mean_power = float(np.mean(p_valid))
    snr = peak_power / max(mean_power, 1e-9)
    return best_period, peak_power, mean_power, snr


def detect_net_cycles(df: pd.DataFrame) -> tuple:
    """FFT-guided valley-to-valley net cycle detection.

    Two-pass: first estimates cycle period via FFT, then uses period/2
    as valley distance to avoid splitting real cycles.
    """
    minute = _build_minute_motion(df)
    if len(minute) < 30:
        return [], None

    motion_raw = minute["motion"].values
    motion_smooth = _smooth(motion_raw, window=SG_MOTION_WINDOW, polyorder=SG_MOTION_POLYORDER)
    times = minute.index

    # Pass 1: lenient valleys → estimate cycle period via FFT
    valleys_1 = detect_valleys(motion_smooth, min_distance=15,
                                prominence=0.8, depth_percentile=40.0)
    period, peak_pow, mean_pow, snr = fft_periodicity(
        motion_smooth, period_low=FFT_PERIOD_LOW, period_high=FFT_PERIOD_HIGH
    )

    # If no clear FFT period (SNR < 3.0), it's likely non-fishing — still
    # run valley detection but expect few cycles.  We'll let the classifier
    # decide based on actual valley count.
    if np.isnan(period) or snr < 3.0:
        if len(valleys_1) < 2:
            return [], {"times": times, "motion_raw": motion_raw,
                        "motion_smooth": motion_smooth, "valleys": np.array([]),
                        "fft_period_min": period, "fft_snr": snr}
        # Fall through to Pass 2 with conservative defaults
        valley_dist = VALLEY_DISTANCE
    else:
        # Pass 2: FFT-guided valley distance (capped)
        valley_dist = max(VALLEY_DISTANCE, min(int(period * 0.40), 45))

    valleys = detect_valleys(motion_smooth, min_distance=valley_dist,
                             prominence=VALLEY_PROMINENCE, depth_percentile=VALLEY_DEPTH_PCT)
    if len(valleys) < 2:
        return [], {"times": times, "motion_raw": motion_raw,
                    "motion_smooth": motion_smooth, "valleys": np.array([]),
                    "fft_period_min": period, "fft_snr": snr}

    cycles = segment_cycles(valleys, motion_smooth, times)

    minute_info = {
        "times": times,
        "motion_raw": motion_raw,
        "motion_smooth": motion_smooth,
        "valleys": valleys,
        "fft_period_min": period,
        "fft_snr": snr,
    }
    return cycles, minute_info


# ---------------------------------------------------------------------------
# Night completeness
# ---------------------------------------------------------------------------
def check_night_completeness(df: pd.DataFrame, night_label: str) -> dict:
    n = len(df)
    start = df["abs_time"].min()
    end   = df["abs_time"].max()
    dur_h = 0.0
    if pd.notna(start) and pd.notna(end):
        dur_h = (end - start).total_seconds() / 3600
    coverage = dur_h / 12.0
    max_gap_min = 0.0
    if n >= 2:
        gaps = df["abs_time"].diff().dropna()
        if len(gaps) > 0:
            max_gap_min = gaps.max().total_seconds() / 60.0
    warnings = []
    if dur_h < MIN_DURATION_HOURS:
        warnings.append(f"duration={dur_h:.1f}h < {MIN_DURATION_HOURS}h minimum")
    if n < MIN_SAMPLE_COUNT:
        warnings.append(f"samples={n} < {MIN_SAMPLE_COUNT} minimum")
    if max_gap_min > MAX_GAP_MINUTES:
        warnings.append(f"large gap={max_gap_min:.0f}min > {MAX_GAP_MINUTES}min threshold")
    if coverage < MIN_COVERAGE_RATIO:
        warnings.append(f"coverage={coverage:.1%} < {MIN_COVERAGE_RATIO:.0%} minimum")
    return {
        "is_complete": len(warnings) == 0,
        "duration_hours": round(dur_h, 1),
        "n_samples": n,
        "max_gap_minutes": round(max_gap_min, 0),
        "coverage_ratio": round(coverage, 2),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------
def classify_night(df: pd.DataFrame) -> tuple:
    n = len(df)
    if n < ROLLING_WINDOW:
        return False, 0.0, 0, 0.0
    fai = df["fai_smooth"].values
    t_start = df["abs_time"].min()
    t_end   = df["abs_time"].max()
    dur_h = (t_end - t_start).total_seconds() / 3600 if pd.notna(t_start) and pd.notna(t_end) else n / 300.0
    active_frac = float(np.mean(fai > ACTIVITY_THRESHOLD))
    rolling_std = pd.Series(fai).rolling(
        window=min(ROLLING_WINDOW, max(n // 4, 3)),
        center=True, min_periods=1
    ).std().values
    var_score = min(float(np.percentile(rolling_std, 90)) / 0.15, 1.0)
    active = fai > ACTIVITY_THRESHOLD
    n_segments_raw = 0
    in_seg = False
    for v in active:
        if v and not in_seg:
            n_segments_raw += 1; in_seg = True
        elif not v:
            in_seg = False
    density_score = min(n_segments_raw / max(dur_h, 0.5) / 1.5, 1.0)
    active_score  = min(active_frac / 0.15, 1.0)
    activity_score = 0.5 * density_score + 0.3 * active_score + 0.2 * var_score
    is_fishing = (n_segments_raw >= 8) and (active_frac >= 0.08)
    return is_fishing, round(activity_score, 4), n_segments_raw, round(active_frac, 3)


# ---------------------------------------------------------------------------
# Per-night analysis
# ---------------------------------------------------------------------------
def analyze_night(df: pd.DataFrame, night_label: str) -> dict:
    """Full analysis: completeness → FAI metrics → valley detection → classify."""
    completeness = check_night_completeness(df, night_label)

    is_fishing = False
    activity_score = 0.0
    net_count = 0
    n_segments_raw = 0
    active_fraction = 0.0
    segments = []
    minute_info = None
    fft_period = float("nan")
    fft_snr = 0.0

    if completeness["is_complete"]:
        # FAI-based activity metrics (always computed)
        _, activity_score, n_segments_raw, active_fraction = classify_night(df)

        # Valley-based cycle detection (always attempted)
        segments, minute_info = detect_net_cycles(df)
        net_count = len(segments)

        if minute_info is not None:
            fft_period = minute_info.get("fft_period_min", float("nan"))
            fft_snr = minute_info.get("fft_snr", 0.0)

        # Classification: >= 3 valleys AND active_fraction >= 0.18 → fishing
        # Non-fishing nights have few valleys AND low active fraction (flat FAI)
        if net_count >= 3 and active_fraction >= 0.18:
            is_fishing = True
        elif net_count >= 2 and not np.isnan(fft_snr) and fft_snr >= 5.0:
            is_fishing = True
        else:
            is_fishing = False

    start = df["abs_time"].min()
    end   = df["abs_time"].max()

    return {
        "night": night_label,
        "is_complete": completeness["is_complete"],
        "is_fishing": is_fishing,
        "net_count": net_count,
        "n_segments_raw": n_segments_raw,
        "active_fraction": round(active_fraction, 3),
        "activity_score": round(activity_score, 4),
        "fft_period_min": round(fft_period, 1) if not np.isnan(fft_period) else None,
        "fft_snr": round(fft_snr, 1),
        "start_time": str(start),
        "end_time": str(end),
        "duration_hours": completeness["duration_hours"],
        "n_samples": completeness["n_samples"],
        "max_gap_minutes": completeness["max_gap_minutes"],
        "coverage_ratio": completeness["coverage_ratio"],
        "warnings": completeness["warnings"],
        "segments": segments,
        "minute_info": minute_info,
    }
