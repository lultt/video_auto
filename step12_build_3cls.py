# -*- coding: utf-8 -*-
"""Reorganize corrected dataset_v2_review (5 classes) -> 3-class business dataset.
  Background = waiting + hauling + netdown
  OnDeck     = on_deck
  Sorting    = sorting
Stratified 80/20 per ORIGINAL subclass (val keeps true distribution).
Train set: OnDeck oversampled 5x. Val set: untouched (true distribution)."""
import os, shutil, random

SRC = r"J:\video_auto\dataset_v2_review"
DST = r"J:\video_auto\dataset_v2_3cls"
EXT = (".jpg", ".jpeg", ".png")
VAL_RATIO = 0.20
SEED = 42
OVERSAMPLE = {"OnDeck": 5}            # train-only oversample factor

MAP = {"waiting": "Background", "hauling": "Background", "netdown": "Background",
       "on_deck": "OnDeck", "sorting": "Sorting"}
TARGETS = ["Background", "OnDeck", "Sorting"]

random.seed(SEED)
if os.path.exists(DST):
    shutil.rmtree(DST)
for split in ("train", "val"):
    for c in TARGETS:
        os.makedirs(os.path.join(DST, split, c), exist_ok=True)

# stratify per ORIGINAL subclass so Background's internal mix is preserved in val
train_counts = {c: 0 for c in TARGETS}
val_counts = {c: 0 for c in TARGETS}
sub_detail = []
for sub, tgt in MAP.items():
    sd = os.path.join(SRC, sub)
    files = sorted(f for f in os.listdir(sd) if f.lower().endswith(EXT))
    random.shuffle(files)
    n_val = round(len(files) * VAL_RATIO)
    val_files = set(files[:n_val])
    tr = va = 0
    for f in files:
        src = os.path.join(sd, f)
        if f in val_files:
            shutil.copy2(src, os.path.join(DST, "val", tgt, f)); va += 1; val_counts[tgt] += 1
        else:
            shutil.copy2(src, os.path.join(DST, "train", tgt, f)); tr += 1; train_counts[tgt] += 1
            k = OVERSAMPLE.get(tgt, 1)
            for r in range(1, k):     # extra copies (train only)
                stem, ext = os.path.splitext(f)
                shutil.copy2(src, os.path.join(DST, "train", tgt, "%s_os%d%s" % (stem, r, ext)))
                train_counts[tgt] += 1
    sub_detail.append((sub, tgt, len(files), tr, va))

print("=== source subclass -> target (stratified 80/20) ===")
print("%-9s -> %-11s %6s %6s %5s" % ("subclass", "target", "total", "train", "val"))
for sub, tgt, t, tr, va in sub_detail:
    print("%-9s -> %-11s %6d %6d %5d" % (sub, tgt, t, tr, va))
print("\n=== dataset_v2_3cls (train oversampled, val TRUE distribution) ===")
print("%-11s %12s %18s" % ("class", "val(true)", "train(after os)"))
for c in TARGETS:
    print("%-11s %12d %18d" % (c, val_counts[c], train_counts[c]))
print("\nOnDeck oversample = %dx (train only)" % OVERSAMPLE.get("OnDeck", 1))
print("Output:", DST)
