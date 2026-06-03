# -*- coding: utf-8 -*-
"""Step 2: stratified 80/20 train/val split into dataset/ (copy, classification layout)."""
import os, shutil, random
from collections import defaultdict

SRC = r"J:\video_auto\fishing_status"
DST = r"J:\video_auto\dataset"
CLASSES = ["hauling", "on_deck", "sorting", "netdown", "waiting"]
EXT = (".jpg", ".jpeg", ".png", ".bmp")
VAL_RATIO = 0.20
SEED = 42

random.seed(SEED)

# fresh dataset dir
if os.path.exists(DST):
    shutil.rmtree(DST)
for split in ("train", "val"):
    for c in CLASSES:
        os.makedirs(os.path.join(DST, split, c), exist_ok=True)

summary = []
total_tr = total_va = 0
for c in CLASSES:
    sd = os.path.join(SRC, c)
    files = sorted(f for f in os.listdir(sd) if f.lower().endswith(EXT))
    random.shuffle(files)
    n_val = round(len(files) * VAL_RATIO)
    val_files = set(files[:n_val])
    tr = va = 0
    for f in files:
        split = "val" if f in val_files else "train"
        shutil.copy2(os.path.join(sd, f), os.path.join(DST, split, c, f))
        if split == "val": va += 1
        else: tr += 1
    total_tr += tr; total_va += va
    summary.append((c, len(files), tr, va))

print("============ STRATIFIED SPLIT (80/20, seed=%d) ============" % SEED)
print("%-10s %6s %7s %6s %8s" % ("class", "total", "train", "val", "val%"))
for c, tot, tr, va in summary:
    print("%-10s %6d %7d %6d %7.1f%%" % (c, tot, tr, va, 100*va/tot))
print("-" * 45)
print("%-10s %6d %7d %6d %7.1f%%" % ("TOTAL", total_tr+total_va, total_tr, total_va,
                                     100*total_va/(total_tr+total_va)))
print("\nOutput:", DST)
