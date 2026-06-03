# -*- coding: utf-8 -*-
"""Step 1: dataset statistics for fishing_status classification dataset."""
import os
from collections import Counter, defaultdict
from PIL import Image

SRC = r"J:\video_auto\fishing_status"
CLASSES = ["hauling", "on_deck", "sorting", "netdown", "waiting"]
EXT = (".jpg", ".jpeg", ".png", ".bmp")

counts = {}
sizes = defaultdict(Counter)     # class -> Counter((w,h))
all_sizes = Counter()
corrupt = []
total = 0

for c in CLASSES:
    d = os.path.join(SRC, c)
    files = [f for f in os.listdir(d) if f.lower().endswith(EXT)]
    counts[c] = len(files)
    total += len(files)
    for f in files:
        p = os.path.join(d, f)
        try:
            with Image.open(p) as im:
                im.verify()          # detect truncation/corruption
            with Image.open(p) as im:
                w, h = im.size       # re-open: verify() leaves file unusable
            sizes[c][(w, h)] += 1
            all_sizes[(w, h)] += 1
        except Exception as e:
            corrupt.append((p, repr(e)))

print("============ DATASET STATISTICS ============")
print("Source:", SRC)
print("Total images:", total, "| classes:", len(CLASSES))
print("\n-- Per-class counts --")
mx = max(counts.values()); mn = min(counts.values())
for c in CLASSES:
    n = counts[c]
    bar = "#" * int(40 * n / mx)
    print("  %-9s %5d  (%5.1f%%)  %s" % (c, n, 100*n/total, bar))
print("\n  max/min ratio: %.2fx  (%s=%d vs %s=%d)" % (
    mx/mn, max(counts, key=counts.get), mx, min(counts, key=counts.get), mn))

print("\n-- Image size distribution (overall) --")
for (w, h), n in all_sizes.most_common(10):
    print("  %dx%d : %d  (%.1f%%)" % (w, h, n, 100*n/total))

print("\n-- Size distribution per class --")
for c in CLASSES:
    top = sizes[c].most_common(3)
    s = ", ".join("%dx%d:%d" % (w, h, n) for (w, h), n in top)
    print("  %-9s %s" % (c, s))

print("\n-- Corrupt / unreadable images --")
if corrupt:
    print("  FOUND %d:" % len(corrupt))
    for p, e in corrupt[:20]:
        print("   ", p, e)
else:
    print("  none")

print("\n-- Imbalance assessment --")
ratio = mx/mn
verdict = ("balanced" if ratio < 1.5 else
           "mild imbalance" if ratio < 3 else
           "moderate imbalance" if ratio < 10 else "severe imbalance")
print("  ratio=%.2fx -> %s" % (ratio, verdict))
