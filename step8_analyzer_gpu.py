# -*- coding: utf-8 -*-
"""VideoStatusAnalyzerGPU: NVDEC -> in-memory BGR frames -> YOLO. No per-frame JPEG.

Reuses phase-1 FastVideoReader pattern (ffmpeg pipe -> stdout.read -> numpy),
but decode backend is the validated NVDEC path (hevc_cuvid + hwdownload).
Frames go straight to YOLO as numpy arrays; disk is touched ONLY for selected
frames (hard cases / on_deck windows / on_deck<->sorting transitions).
"""
import os, subprocess, time
from datetime import datetime, timedelta
import numpy as np
import cv2
from ultralytics import YOLO

FFMPEG = r"C:\Users\ljj\anaconda3\envs\yolonew\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
NAMES = {0: "hauling", 1: "netdown", 2: "on_deck", 3: "sorting", 4: "waiting"}
NAME2IDX = {v: k for k, v in NAMES.items()}


class VideoStatusAnalyzerGPU:
    def __init__(self, weights, imgsz=640, batch=64, gpu_decode=True, device=0):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.batch = batch
        self.gpu_decode = gpu_decode
        self.device = device
        self.W = self.H = None

    def _probe(self, video):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise IOError("cannot open %s" % video)
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def _out_dims(self, scale640):
        """Output frame dims fed to YOLO. scale640: keep aspect (short side=imgsz) + center crop."""
        if scale640:
            sw = int(round(self.W * self.imgsz / self.H)); sw -= sw % 2   # scaled width (even)
            return sw, self.imgsz, self.imgsz, self.imgsz                  # sw, sh, crop_w, crop_h
        return self.W, self.H, self.W, self.H

    def _cmd(self, video, ss, dur, step, keyframe=False, scale640=False):
        c = [FFMPEG, "-hide_banner", "-loglevel", "error"]
        if self.gpu_decode:
            c += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "hevc_cuvid"]
        if keyframe:
            c += ["-discard", "nokey"]          # drop non-key packets at demux (fast)
        c += ["-ss", str(ss), "-i", video]
        if dur is not None:
            c += ["-t", str(dur)]
        sw, sh, cw, ch = self._out_dims(scale640)
        if self.gpu_decode:
            chain = []
            if not keyframe:
                chain.append("fps=1/%g" % step)         # frame select on GPU frames
            if scale640:
                chain.append("scale_cuda=%d:%d" % (sw, sh))   # GPU-side downscale BEFORE download
            chain += ["hwdownload", "format=nv12", "format=bgr24"]
            if scale640:
                chain.append("crop=%d:%d" % (cw, ch))         # center crop to imgsz (cheap, CPU)
            vf = ",".join(chain)
        else:
            vf = None if keyframe else ("fps=1/%g" % step)
        if vf:
            c += ["-vf", vf]
        c += ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-v", "quiet", "-"]
        return c

    def _spawn(self, video, ss, dur, step, keyframe=False, scale640=False):
        return subprocess.Popen(self._cmd(video, ss, dur, step, keyframe, scale640),
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    def _read_one(self, proc, w, h):
        fsize = w * h * 3
        buf = proc.stdout.read(fsize)
        if not buf or len(buf) < fsize:
            return None
        return np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()

    def analyze(self, video, real_start, ss=0, dur=None, step=5, keyframe=False, scale640=False):
        """Stream NVDEC frames in batches -> YOLO. NO disk. Returns (records, timing).
        keyframe=True: decode only keyframes (-discard nokey).
        scale640=True: GPU-side downscale (short side=imgsz) + center crop BEFORE download,
                       so only ~imgsz^2 frames cross PCIe (not full-res). Matches YOLO preproc."""
        self._probe(video)
        sw, sh, cw, ch = self._out_dims(scale640)
        proc = self._spawn(video, ss, dur, step, keyframe, scale640)
        records, t_decode, t_infer, idx, done = [], 0.0, 0.0, 0, False
        try:
            while not done:
                t0 = time.time()
                frames, idxs = [], []
                for _ in range(self.batch):
                    fr = self._read_one(proc, cw, ch)
                    if fr is None:
                        done = True; break
                    frames.append(fr); idxs.append(idx); idx += 1
                t_decode += time.time() - t0
                if not frames:
                    break
                t1 = time.time()
                results = self.model(frames, imgsz=self.imgsz, device=self.device, verbose=False)
                t_infer += time.time() - t1
                for j, r in enumerate(results):
                    p = r.probs.data.cpu().numpy()
                    k = int(np.argmax(p))
                    records.append({"idx": idxs[j], "status": NAMES[k],
                                    "confidence": float(p[k]), "probs": p})
        finally:
            proc.terminate()
            try: proc.wait(timeout=5)
            except Exception: pass
        n = len(records)
        for r in records:
            if keyframe and dur and n > 1:
                r["timestamp"] = real_start + timedelta(seconds=r["idx"] * dur / n)
            else:
                r["timestamp"] = real_start + timedelta(seconds=r["idx"] * step)
        return records, {"decode": t_decode, "infer": t_infer, "frames": n}

    def decode_region(self, video, ss, dur, step=1):
        """Decode one small window densely into memory (no disk). Returns (frames, t_decode).
        Full-resolution (no scale640) — hard_cases export needs original frames."""
        if self.W is None:
            self._probe(video)
        proc = self._spawn(video, ss, dur, step)
        frames = []
        t0 = time.time()
        try:
            while True:
                fr = self._read_one(proc, self.W, self.H)
                if fr is None:
                    break
                frames.append(fr)
        finally:
            proc.terminate()
            try: proc.wait(timeout=5)
            except Exception: pass
        return frames, time.time() - t0

    def export_regions(self, video, real_start, regions, outdir, step=1):
        """Re-decode ONLY the given regions, save JPEGs with model prediction in name.
        regions: list of (start_dt, end_dt, tag). Returns (rows, timing)."""
        os.makedirs(outdir, exist_ok=True)
        t_dec = t_inf = t_io = 0.0
        rows = []
        for rs, re_, tag in regions:
            ss = max(0.0, (rs - real_start).total_seconds())
            dur = (re_ - rs).total_seconds() + step
            frames, dt = self.decode_region(video, ss, dur, step)
            t_dec += dt
            if not frames:
                continue
            t1 = time.time()
            results = self.model(frames, imgsz=self.imgsz, device=self.device, verbose=False)
            t_inf += time.time() - t1
            for j, r in enumerate(results):
                p = r.probs.data.cpu().numpy()
                k = int(np.argmax(p))
                ts = rs + timedelta(seconds=j * step)
                name = "%s_%s_pred-%s_od%.2f_so%.2f.jpg" % (
                    tag, ts.strftime("%H%M%S"), NAMES[k],
                    p[NAME2IDX["on_deck"]], p[NAME2IDX["sorting"]])
                tio = time.time()
                cv2.imwrite(os.path.join(outdir, name), frames[j],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                t_io += time.time() - tio
                rows.append((name, ts.strftime("%H:%M:%S"), NAMES[k], tag))
        return rows, {"decode": t_dec, "infer": t_inf, "io": t_io, "saved": len(rows)}


# ---------------- module-level helpers ----------------
from collections import Counter


def majority_vote(records, win=5):
    labels = [r["status"] for r in records]
    half = win // 2
    sm = []
    for i in range(len(labels)):
        a, b = max(0, i - half), min(len(labels), i + half + 1)
        votes = Counter(labels[a:b])
        top = max(votes.values())
        cands = [c for c, v in votes.items() if v == top]
        if len(cands) == 1:
            sm.append(cands[0])
        else:
            sm.append(max(cands, key=lambda c: sum(
                records[j]["probs"][NAME2IDX[c]] for j in range(a, b))))
    return sm


def collapse(states):
    seq = [states[0]]
    for s in states[1:]:
        if s != seq[-1]:
            seq.append(s)
    return seq


def find_regions(records, smoothed, pad_sec=30):
    """Regions to save: on_deck<->sorting transition neighborhoods + on_deck windows."""
    raw = [r["status"] for r in records]
    ts = [r["timestamp"] for r in records]
    regs = []
    PAIR = {"on_deck", "sorting"}
    for i in range(1, len(raw)):
        if raw[i] != raw[i-1] and {raw[i], raw[i-1]} == PAIR:
            regs.append((ts[i] - timedelta(seconds=pad_sec),
                         ts[i] + timedelta(seconds=pad_sec), "trans"))
    i = 0
    while i < len(smoothed):
        if smoothed[i] == "on_deck":
            j = i
            while j < len(smoothed) and smoothed[j] == "on_deck":
                j += 1
            regs.append((ts[i] - timedelta(seconds=pad_sec),
                         ts[j-1] + timedelta(seconds=pad_sec), "ondeck"))
            i = j
        else:
            i += 1
    regs.sort(key=lambda r: r[0])
    merged = []
    for rs, re_, tag in regs:
        if merged and rs <= merged[-1][1]:
            ps, pe, pt = merged[-1]
            tags = pt if tag in pt else pt + "+" + tag
            merged[-1] = (ps, max(pe, re_), tags)
        else:
            merged.append((rs, re_, tag))
    return merged


def main():
    import csv
    WEIGHTS = r"J:\video_auto\runs\fishing_status_v1\weights\best.pt"
    VIDEO   = r"C:\video\0515\ch01_20250515_214053_222634topspeed.mp4"
    V_START = datetime.strptime("20250515_214053", "%Y%m%d_%H%M%S")
    W_START = datetime.strptime("20250515_215500", "%Y%m%d_%H%M%S")
    W_END   = datetime.strptime("20250515_222300", "%Y%m%d_%H%M%S")
    OUTDIR  = r"J:\video_auto\hard_cases"
    CSVOUT  = r"J:\video_auto\hard_cases\manifest.csv"
    ss  = (W_START - V_START).total_seconds()
    dur = (W_END - W_START).total_seconds()

    t_load0 = time.time()
    az = VideoStatusAnalyzerGPU(WEIGHTS, imgsz=640, batch=64, gpu_decode=True)
    t_load = time.time() - t_load0

    wall0 = time.time()
    records, tm = az.analyze(VIDEO, W_START, ss=ss, dur=dur, step=5)
    wall = time.time() - wall0

    smoothed = majority_vote(records, win=5)
    seq = collapse(smoothed)

    # hot-path timing breakdown (NVDEC -> memory -> YOLO, no disk)
    hot = tm["decode"] + tm["infer"]
    print("=== HOT PATH (no disk): %d frames, %s -> %s ===" % (
        tm["frames"], records[0]["timestamp"].strftime("%H:%M:%S"),
        records[-1]["timestamp"].strftime("%H:%M:%S")))
    print("  model load   : %6.2fs" % t_load)
    print("  GPU decode   : %6.2fs  (%4.1f%%)  [%.1f ms/frame]" % (
        tm["decode"], 100*tm["decode"]/hot, 1000*tm["decode"]/tm["frames"]))
    print("  YOLO infer   : %6.2fs  (%4.1f%%)  [%.1f ms/frame]" % (
        tm["infer"], 100*tm["infer"]/hot, 1000*tm["infer"]/tm["frames"]))
    print("  per-frame IO : %6.2fs  (0 JPEG written in hot path)" % 0.0)
    print("  hot wall     : %6.2fs" % wall)
    print("  sequence:", " -> ".join(seq))

    # selective save: ONLY transition neighborhoods + on_deck windows
    regions = find_regions(records, smoothed, pad_sec=30)
    print("\n=== SELECTIVE EXPORT: %d region(s) ===" % len(regions))
    for rs, re_, tag in regions:
        print("  [%s] %s -> %s" % (tag, rs.strftime("%H:%M:%S"), re_.strftime("%H:%M:%S")))
    rows, em = az.export_regions(VIDEO, V_START, regions, OUTDIR, step=1)

    with open(CSVOUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "timestamp", "model_pred", "region_tag"])
        w.writerows(rows)

    exp = em["decode"] + em["infer"] + em["io"]
    print("\n=== EXPORT timing (saved %d frames @1fps) ===" % em["saved"])
    print("  GPU decode : %6.2fs  (%4.1f%%)" % (em["decode"], 100*em["decode"]/exp if exp else 0))
    print("  YOLO infer : %6.2fs  (%4.1f%%)" % (em["infer"], 100*em["infer"]/exp if exp else 0))
    print("  disk IO    : %6.2fs  (%4.1f%%)" % (em["io"], 100*em["io"]/exp if exp else 0))
    print("\n  hard_cases dir :", OUTDIR)
    print("  manifest       :", CSVOUT)


if __name__ == "__main__":
    main()
