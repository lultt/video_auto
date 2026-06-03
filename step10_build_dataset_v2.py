# -*- coding: utf-8 -*-
"""Build v2 review dataset from the 9 nets via the PRODUCTION path (NVDEC -> scale_cuda -> YOLO).
Saves the exact 640x640 frames the model sees, sorted by model prediction, plus a hard_cases/
copy of frames near abnormal transitions. Reuses step8 primitives (step8 stays frozen)."""
import os, re, csv, shutil
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
import cv2
from step8_analyzer_gpu import VideoStatusAnalyzerGPU, majority_vote, collapse, NAMES, NAME2IDX

SRC     = r"C:\video\0515"
WEIGHTS = r"J:\video_auto\runs\fishing_status_v1\weights\best.pt"
OUT     = r"J:\video_auto\dataset_v2_review"
HARD    = r"J:\video_auto\hard_cases"
CLASSES = ["waiting", "hauling", "on_deck", "sorting", "netdown"]
PAD     = 30      # seconds neighborhood around an abnormal transition
JPEGQ   = 95
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

def vstart_of(videos, path):
    for p, st, en in videos:
        if p == path:
            return st
    return None


def save_frame(frame, status, name):
    p = os.path.join(OUT, status, name)
    cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEGQ])
    return p


def process_segment(az, video, ss, dur, real_start, net, base_idx):
    """Production path: NVDEC + keyframe + scale_cuda -> 640x640 BGR -> YOLO -> save by pred."""
    az._probe(video)
    sw, sh, cw, ch = az._out_dims(True)            # scale640 dims (640x640 after crop)
    proc = az._spawn(video, ss, dur, step=5, keyframe=True, scale640=True)
    frames = []
    try:
        while True:
            fr = az._read_one(proc, cw, ch)
            if fr is None:
                break
            frames.append(fr)
    finally:
        proc.terminate()
        try: proc.wait(timeout=5)
        except Exception: pass
    n = len(frames)
    if n == 0:
        return []
    probs_all = []
    for i in range(0, n, az.batch):
        for r in az.model(frames[i:i+az.batch], imgsz=az.imgsz, device=az.device, verbose=False):
            probs_all.append(r.probs.data.cpu().numpy())
    recs = []
    for j in range(n):
        p = probs_all[j]; k = int(np.argmax(p))
        ts = real_start + timedelta(seconds=j * dur / n)
        gi = base_idx + j
        name = "%s_%03d_%s.jpg" % (net, gi, ts.strftime("%H%M%S"))
        path = save_frame(frames[j], NAMES[k], name)
        recs.append({"idx": gi, "timestamp": ts, "status": NAMES[k],
                     "conf": float(p[k]), "probs": p, "path": path, "net": net})
    return recs


def process_net(az, net, ws, we, videos):
    segs = [(p, max(ws, st), min(we, en)) for p, st, en in videos if min(we, en) > max(ws, st)]
    recs, base = [], 0
    for path, a, b in segs:
        ss = (a - vstart_of(videos, path)).total_seconds()
        dur = (b - a).total_seconds()
        r = process_segment(az, path, ss, dur, a, net, base)
        recs += r; base += len(r)
    recs.sort(key=lambda x: x["timestamp"])
    return recs


def hard_neighbors(recs):
    """Mark frames within +-PAD of any abnormal transition (raw preds). Returns {idx: tag}."""
    marks = {}
    def add(i, tag):
        t0 = recs[i]["timestamp"]
        for r in recs:
            if abs((r["timestamp"] - t0).total_seconds()) <= PAD:
                marks[r["idx"]] = marks.get(r["idx"], tag) if r["idx"] in marks else tag
    for i in range(1, len(recs)):
        a, b = recs[i-1]["status"], recs[i]["status"]
        if a == b:
            continue
        if {a, b} == {"on_deck", "sorting"}:
            add(i, "os")
        elif {a, b} == {"hauling", "waiting"}:
            add(i, "hw")
        elif b not in ALLOWED.get(a, set()):
            add(i, "ill")
    return marks


def main():
    for d in (OUT, HARD):
        if os.path.exists(d):
            shutil.rmtree(d)
    for c in CLASSES:
        os.makedirs(os.path.join(OUT, c), exist_ok=True)
    os.makedirs(HARD, exist_ok=True)

    videos = scan_videos()
    az = VideoStatusAnalyzerGPU(WEIGHTS, imgsz=640, batch=256, gpu_decode=True)

    all_rows, hard_rows = [], []
    per_class = Counter()
    for net, ws, we in NETS:
        recs = process_net(az, net, ws, we, videos)
        marks = hard_neighbors(recs)
        idx2rec = {r["idx"]: r for r in recs}
        for r in recs:
            per_class[r["status"]] += 1
            all_rows.append([os.path.relpath(r["path"], OUT).replace("\\", "/"),
                             r["net"], r["timestamp"].strftime("%H:%M:%S"),
                             r["status"], "%.4f" % r["conf"]])
        for gi, tag in marks.items():
            r = idx2rec[gi]; p = r["probs"]
            hn = "%s_%s_%s_pred-%s_od%.2f_so%.2f_ha%.2f_wa%.2f.jpg" % (
                tag, r["net"], r["timestamp"].strftime("%H%M%S"), r["status"],
                p[NAME2IDX["on_deck"]], p[NAME2IDX["sorting"]],
                p[NAME2IDX["hauling"]], p[NAME2IDX["waiting"]])
            shutil.copy2(r["path"], os.path.join(HARD, hn))
            hard_rows.append([hn, r["net"], r["timestamp"].strftime("%H:%M:%S"), tag, r["status"]])
        print("%-5s frames=%3d  hard=%d" % (net, len(recs), len(marks)))

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_path", "net", "timestamp", "model_pred", "confidence"])
        w.writerows(all_rows)
    with open(os.path.join(HARD, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filename", "net", "timestamp", "trigger", "model_pred"])
        w.writerows(hard_rows)

    print("\n================ dataset_v2_review ================")
    tot = sum(per_class.values())
    for c in CLASSES:
        print("  %-9s %4d  (%.1f%%)" % (c, per_class[c], 100*per_class[c]/tot))
    print("  %-9s %4d" % ("TOTAL", tot))
    print("  hard_cases : %d frames" % len(hard_rows))
    print("\n  review dir :", OUT)
    print("  hard dir   :", HARD)


if __name__ == "__main__":
    main()
# __MAIN__
