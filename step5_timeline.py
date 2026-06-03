# -*- coding: utf-8 -*-
"""Video state-timeline inference test + cyclic state-machine constraint (Viterbi)."""
import os, csv, subprocess, shutil
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

FFMPEG  = r"C:\Users\ljj\anaconda3\envs\yolonew\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
VIDEO   = r"C:\video\0515\ch01_20250515_205339_214053topspeed.mp4"
WEIGHTS = r"J:\video_auto\runs\fishing_status_v1\weights\best.pt"
OUTDIR  = r"J:\video_auto\timeline_test"
TMP     = os.path.join(OUTDIR, "_frames")
RAW_CSV = os.path.join(OUTDIR, "timeline_raw.csv")
CON_CSV = os.path.join(OUTDIR, "timeline_constrained.csv")
PLOT    = os.path.join(OUTDIR, "timeline.png")
STEP    = 5                                   # seconds per sampled frame
VIDEO_START = datetime.strptime("20250515_205339", "%Y%m%d_%H%M%S")

NAMES = {0: "hauling", 1: "netdown", 2: "on_deck", 3: "sorting", 4: "waiting"}
CYCLE = ["waiting", "hauling", "on_deck", "sorting", "netdown"]   # legal order
NAME2MODEL = {v: k for k, v in NAMES.items()}
NEXT = {CYCLE[i]: CYCLE[(i + 1) % len(CYCLE)] for i in range(len(CYCLE))}


def extract_frames():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-an", "-i", VIDEO,
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
    return np.array(out)                       # (N,5) model-index order


def viterbi(probs):
    """Decode most-likely path under cyclic state machine (stay or advance)."""
    eps = 1e-9
    S = len(CYCLE)
    N = len(probs)
    # emission in cycle order
    emit = np.log(np.stack([probs[:, NAME2MODEL[c]] for c in CYCLE], axis=1) + eps)
    # transition matrix (cycle order): stay or +1
    LSTAY, LADV, LNO = np.log(0.85), np.log(0.15), -1e9
    trans = np.full((S, S), LNO)
    for p in range(S):
        trans[p, p] = LSTAY
        trans[p, (p + 1) % S] = LADV
    dp = np.full((N, S), -1e18)
    bp = np.zeros((N, S), dtype=int)
    dp[0] = emit[0] + np.log(1.0 / S)
    for i in range(1, N):
        for s in range(S):
            cand = dp[i - 1] + trans[:, s]
            bp[i, s] = int(np.argmax(cand))
            dp[i, s] = cand[bp[i, s]] + emit[i, s]
    path = [int(np.argmax(dp[-1]))]
    for i in range(N - 1, 0, -1):
        path.append(bp[i, path[-1]])
    path.reverse()
    return [CYCLE[p] for p in path]


def find_illegal(states):
    bad = []
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        if b != a and b != NEXT[a]:
            bad.append((i, a, b))
    return bad


def durations(states):
    d = Counter()
    for s in states:
        d[s] += STEP
    return d


def state_window(states, times, target):
    idx = [i for i, s in enumerate(states) if s == target]
    if not idx:
        return None, None
    return times[idx[0]], times[idx[-1]]


def fmt(dt):
    return dt.strftime("%H:%M:%S") if dt else "-"


def write_csv(path, times, states, confs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "status", "confidence"])
        for t, s, c in zip(times, states, confs):
            w.writerow([t.strftime("%H:%M:%S"), s, "%.4f" % c])


def plot(times, raw, con, bad):
    yidx = {c: i for i, c in enumerate(CYCLE)}
    xs = [t.timestamp() for t in times]
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for ax, seq, title in ((axes[0], raw, "RAW predictions"),
                           (axes[1], con, "CONSTRAINED (cyclic state machine / Viterbi)")):
        ax.step(xs, [yidx[s] for s in seq], where="post", lw=1.6)
        ax.set_yticks(range(len(CYCLE)))
        ax.set_yticklabels(CYCLE)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    for i, _, _ in bad:
        axes[0].axvline(xs[i], color="red", alpha=0.35, lw=0.8)
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M", tz=None))
    ticks = xs[::max(1, len(xs) // 12)]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([datetime.fromtimestamp(x).strftime("%H:%M") for x in ticks])
    axes[1].set_xlabel("time of day")
    plt.tight_layout()
    plt.savefig(PLOT, dpi=110)


def report(tag, states, times):
    print("\n--- %s timeline ---" % tag)
    d = durations(states)
    for c in CYCLE:
        m, s = divmod(d.get(c, 0), 60)
        print("  %-9s %4ds  (%dm%02ds)" % (c, d.get(c, 0), m, s))
    o0, o1 = state_window(states, times, "on_deck")
    print("  on_deck window: %s -> %s" % (fmt(o0), fmt(o1)))
    # collapsed sequence
    seq = [states[0]]
    for s in states[1:]:
        if s != seq[-1]:
            seq.append(s)
    print("  sequence:", " -> ".join(seq))
    return o0, o1, seq


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Extracting frames at 1/%ds ..." % STEP)
    paths = extract_frames()
    times = [VIDEO_START + timedelta(seconds=i * STEP) for i in range(len(paths))]
    print("Frames: %d  (%s -> %s)" % (len(paths), fmt(times[0]), fmt(times[-1])))

    probs = predict(paths)
    raw = [NAMES[int(np.argmax(p))] for p in probs]
    raw_conf = [float(np.max(p)) for p in probs]

    bad = find_illegal(raw)
    print("\nIllegal transitions in RAW: %d" % len(bad))
    for i, a, b in bad[:15]:
        print("   %s : %s -> %s" % (fmt(times[i]), a, b))

    con = viterbi(probs)
    con_conf = [float(probs[i][NAME2MODEL[con[i]]]) for i in range(len(con))]
    changed = sum(1 for a, b in zip(raw, con) if a != b)
    print("\nViterbi changed %d/%d frames (%.1f%%)" % (changed, len(con), 100*changed/len(con)))
    print("Illegal transitions after constraint: %d" % len(find_illegal(con)))

    write_csv(RAW_CSV, times, raw, raw_conf)
    write_csv(CON_CSV, times, con, con_conf)

    report("RAW", raw, times)
    o0, o1, seq = report("CONSTRAINED", con, times)
    plot(times, raw, con, bad)

    # workflow validity
    valid_cycle = all(seq[i+1] == NEXT[seq[i]] for i in range(len(seq)-1)) if len(seq) > 1 else True
    print("\n=== OUTPUT FILES ===")
    print("  raw CSV        :", RAW_CSV)
    print("  constrained CSV:", CON_CSV)
    print("  timeline plot  :", PLOT)
    print("on_deck window (constrained): %s -> %s" % (fmt(o0), fmt(o1)))
    print("follows legal workflow order:", valid_cycle)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()

