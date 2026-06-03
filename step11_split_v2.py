# -*- coding: utf-8 -*-
"""Stratified 80/20 split of corrected dataset_v2_review -> dataset_v2 (classification layout)."""
import os, shutil, random

SRC = r"J:\video_auto\dataset_v2_review"
DST = r"J:\video_auto\dataset_v2"
CLASSES = ["hauling", "on_deck", "sorting", "netdown", "waiting"]
EXT = (".jpg", ".jpeg", ".png")
VAL_RATIO = 0.20
SEED = 42

random.seed(SEED)
if os.path.exists(DST):
    shutil.rmtree(DST)
for split in ("train", "val"):
    for c in CLASSES:
        os.makedirs(os.path.join(DST, split, c), exist_ok=True)

summary = []
tot_tr = tot_va = 0
for c in CLASSES:
    sd = os.path.join(SRC, c)
    files = sorted(f for f in os.listdir(sd) if f.lower().endswith(EXT))
    random.shuffle(files)
    n_val = round(len(files) * VAL_RATIO)
    val_set = set(files[:n_val])
    tr = va = 0
    for f in files:
        split = "val" if f in val_set else "train"
        shutil.copy2(os.path.join(sd, f), os.path.join(DST, split, c, f))
        if split == "val": va += 1
        else: tr += 1
    tot_tr += tr; tot_va += va
    summary.append((c, len(files), tr, va))

print("=== dataset_v2 stratified split (80/20, seed=%d) ===" % SEED)
print("%-10s %6s %7s %6s %7s" % ("class", "total", "train", "val", "val%"))
for c, t, tr, va in summary:
    print("%-10s %6d %7d %6d %6.1f%%" % (c, t, tr, va, 100*va/t if t else 0))
print("-" * 40)
print("%-10s %6d %7d %6d %6.1f%%" % ("TOTAL", tot_tr+tot_va, tot_tr, tot_va,
                                     100*tot_va/(tot_tr+tot_va)))
print("Output:", DST)
