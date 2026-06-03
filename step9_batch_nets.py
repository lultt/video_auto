# -*- coding: utf-8 -*-
"""Batch-analyze 9 net cycles. NVDEC + -discard nokey -> memory -> YOLO. No disk.
Reuses VideoStatusAnalyzerGPU.analyze(keyframe=True) from step8."""
import os, re, csv
from datetime import datetime, timedelta
from collections import Counter
from step8_analyzer_gpu import VideoStatusAnalyzerGPU, majority_vote, collapse, NAME2IDX

SRC     = r"C:\video\0515"
WEIGHTS = r"J:\video_auto\runs\fishing_status_v1\weights\best.pt"
CSVOUT  = r"J:\video_auto\timeline_summary.csv"
KF      = 4.0   # nominal keyframe interval (s), for duration reporting
ALLOWED = {"waiting": {"hauling"}, "hauling": {"on_deck"}, "on_deck": {"sorting"},
           "sorting": {"waiting", "netdown"}, "netdown": {"waiting"}}

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

def first_ts(seq, times, st):
    for s, t in zip(seq, times):
        if s == st: return t
    return None
def last_ts(seq, times, st):
    r = None
    for s, t in zip(seq, times):
        if s == st: r = t
    return r
def fmt(t): return t.strftime("%H:%M:%S") if t else "-"


def analyze_net(az, name, ws, we, videos):
    segs = [(p, max(ws, st), min(we, en)) for p, st, en in videos if min(we, en) > max(ws, st)]
    recs, t_dec, t_inf = [], 0.0, 0.0
    for path, a, b in segs:
        # offset into this video file
        for p, st, en in videos:
            if p == path:
                vstart = st; break
        ss = (a - vstart).total_seconds()
        dur = (b - a).total_seconds()
        r, tm = az.analyze(path, a, ss=ss, dur=dur, step=5, keyframe=True, scale640=True)
        recs += r; t_dec += tm["decode"]; t_inf += tm["infer"]
    recs.sort(key=lambda x: x["timestamp"])
    smoothed = majority_vote(recs, win=5)
    times = [r["timestamp"] for r in recs]

    illegal = conf_os = conf_hw = 0
    for i in range(1, len(smoothed)):
        x, y = smoothed[i-1], smoothed[i]
        if y != x and y not in ALLOWED.get(x, set()):
            illegal += 1
        if x == "sorting" and y == "on_deck":  conf_os += 1   # on_deck<->sorting (illegal dir)
        if x == "hauling" and y == "waiting":  conf_hw += 1   # hauling<->waiting (illegal dir)

    durs = Counter()
    for s in smoothed: durs[s] += KF

    return {
        "name": name, "ws": ws, "we": we, "n": len(recs),
        "hauling_start": first_ts(smoothed, times, "hauling"),
        "ondeck_start":  first_ts(smoothed, times, "on_deck"),
        "ondeck_end":    last_ts(smoothed, times, "on_deck"),
        "sorting_start": first_ts(smoothed, times, "sorting"),
        "illegal": illegal, "conf_os": conf_os, "conf_hw": conf_hw,
        "has_ondeck": "on_deck" in smoothed,
        "seq": collapse(smoothed), "durs": durs,
        "t_dec": t_dec, "t_inf": t_inf,
    }


def main():
    videos = scan_videos()
    az = VideoStatusAnalyzerGPU(WEIGHTS, imgsz=640, batch=256, gpu_decode=True)
    rows, results = [], []
    total_dec = total_inf = 0.0
    for name, ws, we in NETS:
        res = analyze_net(az, name, ws, we, videos)
        results.append(res); total_dec += res["t_dec"]; total_inf += res["t_inf"]
        od = (res["ondeck_end"] - res["ondeck_start"]).total_seconds() + KF if res["ondeck_start"] and res["ondeck_end"] else 0
        print("%-5s kf=%3d  H=%s OD=%s..%s S=%s  illegal=%d  [%s]" % (
            name, res["n"], fmt(res["hauling_start"]), fmt(res["ondeck_start"]),
            fmt(res["ondeck_end"]), fmt(res["sorting_start"]), res["illegal"],
            " -> ".join(res["seq"])))
        rows.append([name, fmt(ws), fmt(we), fmt(res["hauling_start"]), fmt(res["ondeck_start"]),
                     fmt(res["ondeck_end"]), fmt(res["sorting_start"]), int(od), res["illegal"]])

    with open(CSVOUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Net_ID", "Start_Time", "End_Time", "Hauling_Start", "OnDeck_Start",
                    "OnDeck_End", "Sorting_Start", "OnDeck_Duration", "Illegal_Transitions"])
        w.writerows(rows)

    # aggregate
    od_ok = sum(1 for r in results if r["has_ondeck"])
    tot_os = sum(r["conf_os"] for r in results)
    tot_hw = sum(r["conf_hw"] for r in results)
    alld = Counter()
    for r in results:
        for k, v in r["durs"].items(): alld[k] += v
    print("\n================ AGGREGATE (9 nets) ================")
    print("on_deck detected in : %d / 9 nets" % od_ok)
    print("on_deck<->sorting confusion (sorting->on_deck) : %d" % tot_os)
    print("hauling<->waiting confusion (hauling->waiting)  : %d" % tot_hw)
    print("\nTotal duration per state (min):")
    for s in ["waiting", "hauling", "on_deck", "sorting", "netdown"]:
        print("  %-9s %6.1f min" % (s, alld.get(s, 0)/60))
    print("\nHOT PATH timing (NVDEC+keyframe -> memory -> YOLO, no disk):")
    hot = total_dec + total_inf
    print("  GPU decode : %6.1fs (%.0f%%)" % (total_dec, 100*total_dec/hot))
    print("  YOLO infer : %6.1fs (%.0f%%)" % (total_inf, 100*total_inf/hot))
    print("  total      : %6.1fs" % hot)
    print("\nCSV:", CSVOUT)


if __name__ == "__main__":
    main()
