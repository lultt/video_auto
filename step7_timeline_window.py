# -*- coding: utf-8 -*-
"""Timeline v1 on a time WINDOW of a video. raw + 5-frame majority vote. With timing."""
import os, csv, subprocess, shutil, time
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

FFMPEG  = r"C:\Users\ljj\anaconda3\envs\yolonew\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
VIDEO   = r"C:\video\0515\ch01_20250515_214053_222634topspeed.mp4"
WEIGHTS = r"J:\video_auto\runs\fishing_status_v1\weights\best.pt"
OUTDIR  = r"J:\video_auto\timeline_test"
TMP     = os.path.join(OUTDIR, "_frames_w2")
RAW_CSV = os.path.join(OUTDIR, "timeline_w2_raw.csv")
SM_CSV  = os.path.join(OUTDIR, "timeline_w2_smoothed.csv")
PLOT    = os.path.join(OUTDIR, "timeline_w2.png")
STEP    = 5
WIN     = 5
# video real start, and the window we want to analyze
VIDEO_START  = datetime.strptime("20250515_214053", "%Y%m%d_%H%M%S")
WIN_START    = datetime.strptime("20250515_215500", "%Y%m%d_%H%M%S")
WIN_END      = datetime.strptime("20250515_222300", "%Y%m%d_%H%M%S")
SS    = (WIN_START - VIDEO_START).total_seconds()      # seek offset
DUR   = (WIN_END - WIN_START).total_seconds()          # window length

NAMES = {0: "hauling", 1: "netdown", 2: "on_deck", 3: "sorting", 4: "waiting"}
NAME2IDX = {v: k for k, v in NAMES.items()}
ORDER = ["waiting", "hauling", "on_deck", "sorting", "netdown"]
ALLOWED = {
    "waiting": {"hauling"},
    "hauling": {"on_deck"},
    "on_deck": {"sorting"},
    "sorting": {"waiting", "netdown"},
    "netdown": {"waiting"},
}


def extract_frames():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-an",
           "-ss", str(SS), "-i", VIDEO, "-t", str(DUR),
           "-vf", "fps=1/%d" % STEP, "-q:v", "2", "-start_number", "0",
           os.path.join(TMP, "f_%06d.jpg")]
    subprocess.run(cmd, check=True)
    return sorted(os.path.join(TMP, f) for f in os.listdir(TMP) if f.endswith(".jpg"))


def predict(paths):
    model = YOLO(WEIGHTS)
    out = []
    B = 256
    for i in range(0, len(paths), B):
        for r in model(paths[i:i + B], imgsz=640, verbose=False):
            out.append(r.probs.data.cpu().numpy())
    return np.array(out)


def majority_vote(labels, probs, win=WIN):
    n = len(labels); half = win // 2; sm = []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        votes = Counter(labels[a:b]); top = max(votes.values())
        cands = [c for c, v in votes.items() if v == top]
        if len(cands) == 1:
            sm.append(cands[0])
        else:
            sm.append(max(cands, key=lambda c: sum(probs[j][NAME2IDX[c]] for j in range(a, b))))
    return sm


def find_illegal(states):
    bad = []
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        if b != a and b not in ALLOWED.get(a, set()):
            bad.append((i, a, b))
    return bad


def durations(states):
    d = Counter()
    for s in states:
        d[s] += STEP
    return d


def fmt(dt):
    return dt.strftime("%H:%M:%S") if dt else "-"


def collapse(states):
    seq = [states[0]]
    for s in states[1:]:
        if s != seq[-1]:
            seq.append(s)
    return seq


def state_window(states, times, target):
    idx = [i for i, s in enumerate(states) if s == target]
    return (times[idx[0]], times[idx[-1]]) if idx else (None, None)


def write_csv(path, times, states, confs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["timestamp", "status", "confidence"])
        for t, s, c in zip(times, states, confs):
            w.writerow([t.strftime("%H:%M:%S"), s, "%.4f" % c])


def plot(times, raw, sm, bad):
    yidx = {c: i for i, c in enumerate(ORDER)}; xs = list(range(len(times)))
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for ax, seq, title in ((axes[0], raw, "RAW predictions (argmax)  21:55-22:23"),
                           (axes[1], sm, "SMOOTHED (5-frame majority vote)")):
        ax.step(xs, [yidx[s] for s in seq], where="post", lw=1.6)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    for i, _, _ in bad:
        axes[1].axvline(i, color="red", alpha=0.4, lw=0.9)
    ticks = xs[::max(1, len(xs) // 12)]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([times[t].strftime("%H:%M") for t in ticks])
    axes[1].set_xlabel("time of day")
    plt.tight_layout(); plt.savefig(PLOT, dpi=110)


def report(tag, states, times):
    print("\n--- %s timeline ---" % tag)
    d = durations(states)
    for c in ORDER:
        m, s = divmod(d.get(c, 0), 60)
        print("  %-9s %4ds  (%dm%02ds)" % (c, d.get(c, 0), m, s))
    o0, o1 = state_window(states, times, "on_deck")
    print("  on_deck window: %s -> %s" % (fmt(o0), fmt(o1)))
    print("  sequence:", " -> ".join(collapse(states)))
    return o0, o1


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Window: %s -> %s  (seek=%ds, dur=%ds)" % (fmt(WIN_START), fmt(WIN_END), SS, DUR))

    t0 = time.time()
    paths = extract_frames()
    t_extract = time.time() - t0
    times = [WIN_START + timedelta(seconds=i * STEP) for i in range(len(paths))]
    print("Frames: %d  (%s -> %s)" % (len(paths), fmt(times[0]), fmt(times[-1])))

    t1 = time.time()
    probs = predict(paths)
    t_infer = time.time() - t1

    raw = [NAMES[int(np.argmax(p))] for p in probs]
    raw_conf = [float(np.max(p)) for p in probs]
    sm = majority_vote(raw, probs)
    sm_conf = [float(probs[i][NAME2IDX[sm[i]]]) for i in range(len(sm))]
    changed = sum(1 for a, b in zip(raw, sm) if a != b)
    print("\nMajority vote changed %d/%d frames (%.1f%%)" % (changed, len(sm), 100*changed/len(sm)))

    write_csv(RAW_CSV, times, raw, raw_conf)
    write_csv(SM_CSV, times, sm, sm_conf)

    bad_raw = find_illegal(raw); bad_sm = find_illegal(sm)
    print("\nIllegal transitions (vs permissive set) -- RAW: %d, SMOOTHED: %d  [observation only]"
          % (len(bad_raw), len(bad_sm)))
    for i, a, b in bad_sm[:20]:
        print("   %s : %s -> %s" % (fmt(times[i]), a, b))

    report("RAW", raw, times)
    o0, o1 = report("SMOOTHED", sm, times)
    plot(times, raw, sm, bad_sm)

    total = time.time() - t0
    print("\n=== TIMING ===")
    print("  frame extraction : %6.1fs" % t_extract)
    print("  inference (%d f) : %6.1fs  (%.1f ms/frame)" % (len(paths), t_infer, 1000*t_infer/len(paths)))
    print("  total            : %6.1fs" % total)
    print("\n=== OUTPUT ===")
    print("  raw CSV      :", RAW_CSV)
    print("  smoothed CSV :", SM_CSV)
    print("  plot         :", PLOT)
    print("on_deck window (smoothed): %s -> %s" % (fmt(o0), fmt(o1)))
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
