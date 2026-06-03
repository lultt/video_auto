# -*- coding: utf-8 -*-
"""Business re-run of 9 nets with the 3-class model via production path
(NVDEC -> scale_cuda -> batch256 -> YOLO). No disk. Reuses step8 decode primitives."""
import os, re, csv
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from step8_analyzer_gpu import VideoStatusAnalyzerGPU

SRC     = r"C:\video\0515"
WEIGHTS = r"J:\video_auto\runs\fishing_3cls_v1\weights\best.pt"
CSVOUT  = r"J:\video_auto\timeline_3cls_summary.csv"
KF      = 4.0
CLASSES = ["Background", "OnDeck", "Sorting"]
# legal cyclic transitions (self-loops implicit)
ALLOWED = {"Background": {"OnDeck"}, "OnDeck": {"Sorting"}, "Sorting": {"Background"}}

def D(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
NETS = [("Net1", D("2025-05-15 21:10:00"), D("2025-05-15 21:40:00")),
        ("Net2", D("2025-05-15 21:50:00"), D("2025-05-15 22:20:00")),
        ("Net3", D("2025-05-15 22:55:00"), D("2025-05-15 23:25:00")),
        ("Net4", D("2025-05-16 00:05:00"), D("2025-05-16 00:35:00")),
        ("Net5", D("2025-05-16 00:50:00"), D("2025-05-16 01:20:00")),
        ("Net6", D("2025-05-16 01:40:00"), D("2025-05-16 02:10:00")),
        ("Net7", D("2025-05-16 02:40:00"), D("2025-05-16 03:10:00")),
        ("Net8", D("2025-05-16 03:40:00"), D("2025-05-16 04:10:00")),
        ("Net9", D("2025-05-16 04:30:00"), D("2025-05-16 05:00:00"))]

def scan_videos():
    out = []
    for fn in sorted(os.listdir(SRC)):
        m = re.search(r"(\d{8})_(\d{6})_(\d{6})", fn)
        if fn.lower().endswith(".mp4") and m:
            d, s, e = m.groups()
            st = datetime.strptime(d+s, "%Y%m%d%H%M%S"); en = datetime.strptime(d+e, "%Y%m%d%H%M%S")
            if en <= st: en += timedelta(days=1)
            out.append((os.path.join(SRC, fn), st, en))
    return out

def vstart_of(videos, path):
    for p, st, en in videos:
        if p == path: return st
    return None

def fmt(t): return t.strftime("%H:%M:%S") if t else "-"
def first_ts(seq, times, st):
    for s, t in zip(seq, times):
        if s == st: return t
    return None
def last_ts(seq, times, st):
    r = None
    for s, t in zip(seq, times):
        if s == st: r = t
    return r
def collapse(seq):
    out = [seq[0]]
    for s in seq[1:]:
        if s != out[-1]: out.append(s)
    return out


def majority_vote(labels, probs, names_idx, win=5):
    n = len(labels); half = win // 2; sm = []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        votes = Counter(labels[a:b]); top = max(votes.values())
        cands = [c for c, v in votes.items() if v == top]
        if len(cands) == 1:
            sm.append(cands[0])
        else:
            sm.append(max(cands, key=lambda c: sum(probs[j][names_idx[c]] for j in range(a, b))))
    return sm


def decode_infer_seg(az, names, video, ss, dur, real_start):
    """Production path: NVDEC+keyframe+scale_cuda -> 640 -> 3-class YOLO. Returns records."""
    az._probe(video)
    sw, sh, cw, ch = az._out_dims(True)
    proc = az._spawn(video, ss, dur, step=5, keyframe=True, scale640=True)
    frames = []
    try:
        while True:
            fr = az._read_one(proc, cw, ch)
            if fr is None: break
            frames.append(fr)
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: pass
    n = len(frames)
    if n == 0: return [], 0.0, 0.0
    import time
    probs_all = []
    t1 = time.time()
    for i in range(0, n, az.batch):
        for r in az.model(frames[i:i+az.batch], imgsz=az.imgsz, device=az.device, verbose=False):
            probs_all.append(r.probs.data.cpu().numpy())
    t_inf = time.time() - t1
    recs = []
    for j in range(n):
        p = probs_all[j]; k = int(np.argmax(p))
        recs.append({"timestamp": real_start + timedelta(seconds=j * dur / n),
                     "status": names[k], "probs": p})
    return recs, t_inf, n


def analyze_net(az, names, names_idx, net, ws, we, videos):
    segs = [(p, max(ws, st), min(we, en)) for p, st, en in videos if min(we, en) > max(ws, st)]
    recs, t_inf = [], 0.0
    for path, a, b in segs:
        ss = (a - vstart_of(videos, path)).total_seconds()
        dur = (b - a).total_seconds()
        r, ti, n = decode_infer_seg(az, names, path, ss, dur, a)
        recs += r; t_inf += ti
    recs.sort(key=lambda x: x["timestamp"])
    labels = [r["status"] for r in recs]
    probs = [r["probs"] for r in recs]
    sm = majority_vote(labels, probs, names_idx, win=5)
    times = [r["timestamp"] for r in recs]

    illegal = conf_os = conf_so = 0
    for i in range(1, len(sm)):
        x, y = sm[i-1], sm[i]
        if y != x and y not in ALLOWED.get(x, set()):
            illegal += 1
        if x == "OnDeck" and y == "Sorting":  conf_os += 1
        if x == "Sorting" and y == "OnDeck":  conf_so += 1

    durs = Counter()
    for s in sm: durs[s] += KF
    return {
        "net": net, "n": len(recs), "seq": collapse(sm), "durs": durs,
        "ondeck_start": first_ts(sm, times, "OnDeck"), "ondeck_end": last_ts(sm, times, "OnDeck"),
        "sorting_start": first_ts(sm, times, "Sorting"), "sorting_end": last_ts(sm, times, "Sorting"),
        "has_ondeck": "OnDeck" in sm, "has_sorting": "Sorting" in sm,
        "illegal": illegal, "conf_os": conf_os, "conf_so": conf_so, "t_inf": t_inf,
    }


def main():
    from ultralytics import YOLO
    videos = scan_videos()
    az = VideoStatusAnalyzerGPU(WEIGHTS, imgsz=640, batch=256, gpu_decode=True)
    names = az.model.names                       # {0:Background,1:OnDeck,2:Sorting}
    names_idx = {v: k for k, v in names.items()}
    print("model.names:", names)

    rows, results = [], []
    for net, ws, we in NETS:
        res = analyze_net(az, names, names_idx, net, ws, we, videos)
        results.append(res)
        odd = ((res["ondeck_end"] - res["ondeck_start"]).total_seconds() + KF) if res["ondeck_start"] else 0
        sod = ((res["sorting_end"] - res["sorting_start"]).total_seconds() + KF) if res["sorting_start"] else 0
        print("%-5s kf=%3d OD=%s..%s S=%s..%s illegal=%d os->so=%d so->os=%d [%s]" % (
            net, res["n"], fmt(res["ondeck_start"]), fmt(res["ondeck_end"]),
            fmt(res["sorting_start"]), fmt(res["sorting_end"]),
            res["illegal"], res["conf_os"], res["conf_so"], " -> ".join(res["seq"])))
        rows.append([net, fmt(res["ondeck_start"]), fmt(res["ondeck_end"]),
                     fmt(res["sorting_start"]), fmt(res["sorting_end"]), int(odd), int(sod)])

    with open(CSVOUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Net_ID", "OnDeck_Start", "OnDeck_End", "Sorting_Start", "Sorting_End",
                    "OnDeck_Duration", "Sorting_Duration"])
        w.writerows(rows)

    od_ok = sum(1 for r in results if r["has_ondeck"])
    so_ok = sum(1 for r in results if r["has_sorting"])
    tot_il = sum(r["illegal"] for r in results)
    tot_os = sum(r["conf_os"] for r in results)
    tot_so = sum(r["conf_so"] for r in results)
    alld = Counter()
    for r in results:
        for k, v in r["durs"].items(): alld[k] += v
    print("\n================ AGGREGATE (9 nets, 3-class) ================")
    print("OnDeck  detected in : %d / 9 nets" % od_ok)
    print("Sorting detected in : %d / 9 nets" % so_ok)
    print("Illegal transitions total : %d" % tot_il)
    print("OnDeck->Sorting transitions: %d" % tot_os)
    print("Sorting->OnDeck confusion  : %d" % tot_so)
    print("\nTotal duration per state (min):")
    for c in CLASSES:
        print("  %-11s %6.1f min" % (c, alld.get(c, 0)/60))
    print("\nCSV:", CSVOUT)


if __name__ == "__main__":
    main()
