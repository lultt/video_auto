# -*- coding: utf-8 -*-
"""Extract 1fps frames for given real-world time windows -> flat Unsorted dataset."""
import os, re, csv, time, shutil, subprocess
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

FFMPEG    = r"C:\Users\ljj\anaconda3\envs\yolonew\lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
SRC_DIR   = r"C:\video\0515"
DS_DIR    = r"C:\video\0515\dataset_frames"
OUT_DIR   = os.path.join(DS_DIR, "Unsorted")
TMP_ROOT  = os.path.join(DS_DIR, "_tmp")
CSV_PATH  = os.path.join(DS_DIR, "frames.csv")
CHUNK_SECONDS = 300          # frames per ffmpeg job
JPEG_Q    = "1"              # mjpeg quality: 1 = highest
VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov")

DATE    = "20250515"
WINDOWS = [("211000", "214000"),    # 21:10:00 - 21:40:00
           ("215000", "223000")]    # 21:50:00 - 22:30:00


def parse_video(fn):
    m = re.search(r"(\d{8})_(\d{6})_(\d{6})", fn)
    if not m:
        return None
    d, s, e = m.groups()
    start = datetime.strptime(d + s, "%Y%m%d%H%M%S")
    end   = datetime.strptime(d + e, "%Y%m%d%H%M%S")
    if end <= start:
        end += timedelta(days=1)
    return start, end


def win_dt(hms):
    return datetime.strptime(DATE + hms, "%Y%m%d%H%M%S")


def build_segments():
    segs = []
    for fn in sorted(os.listdir(SRC_DIR)):
        if not fn.lower().endswith(VIDEO_EXT):
            continue
        pv = parse_video(fn)
        if not pv:
            continue
        vstart, vend = pv
        stem = os.path.splitext(fn)[0]
        for ws, we in WINDOWS:
            wstart, wend = win_dt(ws), win_dt(we)
            ist, ien = max(vstart, wstart), min(vend, wend)
            if ien <= ist:
                continue
            segs.append({
                "stem": stem, "path": os.path.join(SRC_DIR, fn),
                "offset": (ist - vstart).total_seconds(),
                "nframes": int((ien - ist).total_seconds()),
                "real_start": ist,
            })
    return segs


def build_chunks(segs):
    segs.sort(key=lambda s: s["real_start"])
    chunks = []
    for s in segs:
        done = 0
        while done < s["nframes"]:
            n = min(CHUNK_SECONDS, s["nframes"] - done)
            chunks.append({
                "stem": s["stem"], "path": s["path"],
                "ss": s["offset"] + done, "n": n,
                "real_start": s["real_start"] + timedelta(seconds=done),
            })
            done += n
    return chunks


def run_chunk(ch, idx):
    tmp = os.path.join(TMP_ROOT, "chunk_%d" % idx)
    os.makedirs(tmp, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-an",
           "-ss", str(ch["ss"]), "-i", ch["path"],
           "-vf", "fps=1", "-frames:v", str(ch["n"]),
           "-q:v", JPEG_Q, "-start_number", "0",
           os.path.join(tmp, "f_%06d.jpg")]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    rows = []
    for j, f in enumerate(sorted(os.listdir(tmp))):
        ts   = ch["real_start"] + timedelta(seconds=j)
        name = ts.strftime("%Y%m%d_%H%M%S") + ".jpg"
        os.replace(os.path.join(tmp, f), os.path.join(OUT_DIR, name))
        rows.append(("Unsorted/%s" % name, ch["stem"], ts.strftime("%H:%M:%S")))
    shutil.rmtree(tmp, ignore_errors=True)
    return ch["stem"], len(rows), time.time() - t0, rows


def main():
    segs = build_segments()
    chunks = build_chunks(segs)
    if not chunks:
        print("No video overlaps the requested windows."); return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_ROOT, exist_ok=True)
    workers = max(2, (os.cpu_count() or 4) // 2)
    print("Videos: %d | chunks: %d | workers: %d"
          % (len({c['stem'] for c in chunks}), len(chunks), workers))

    all_rows, stats = [], defaultdict(lambda: [0, 0.0])
    wall0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_chunk, c, i): c for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            stem, n, dt, rows = fut.result()
            stats[stem][0] += n
            stats[stem][1] += dt
            all_rows += rows
    wall = time.time() - wall0
    shutil.rmtree(TMP_ROOT, ignore_errors=True)

    all_rows.sort(key=lambda r: r[0])
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "video_name", "timestamp"])
        w.writerows(all_rows)

    total = sum(v[0] for v in stats.values())
    print("\n================ SUMMARY ================")
    print("Total images : %d" % total)
    print("Total time   : %.1fs" % wall)
    print("-----------------------------------------")
    for stem in sorted(stats):
        n, dt = stats[stem]
        print("  %s : %d frames (decode %.1fs)" % (stem, n, dt))
    print("-----------------------------------------")
    print("Output : %s" % OUT_DIR)
    print("CSV    : %s" % CSV_PATH)


if __name__ == "__main__":
    main()
