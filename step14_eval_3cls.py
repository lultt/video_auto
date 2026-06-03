# -*- coding: utf-8 -*-
"""Evaluate 3-class model on val set (true distribution). Confusion matrix + per-class P/R/F1.
Exports misclassified frames to confusions/ subdirs."""
import os, csv, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = r"J:\video_auto\runs\fishing_3cls_v1\weights\best.pt"
VAL     = r"J:\video_auto\dataset_v2_3cls\val"
CONF    = r"J:\video_auto\confusions"
PLOT    = r"J:\video_auto\runs\fishing_3cls_v1\confusion_eval.png"
CLASSES = ["Background", "OnDeck", "Sorting"]   # alphabetical = model index order
IDX = {c: i for i, c in enumerate(CLASSES)}
EXPORT = [("OnDeck", "Sorting"), ("Sorting", "OnDeck"),
          ("Background", "OnDeck"), ("Background", "Sorting")]


def main():
    model = YOLO(WEIGHTS)
    print("model.names:", model.names)
    for true_c, pred_c in EXPORT:
        os.makedirs(os.path.join(CONF, "%s_pred_%s" % (true_c, pred_c)), exist_ok=True)

    cm = np.zeros((3, 3), dtype=int)   # rows=true, cols=pred
    items = []
    for c in CLASSES:
        d = os.path.join(VAL, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, f), c, f))

    paths = [p for p, _, _ in items]
    preds = []
    B = 256
    for i in range(0, len(paths), B):
        for r in model(paths[i:i+B], imgsz=640, verbose=False):
            preds.append(int(r.probs.top1))

    for (path, true_c, fn), pk in zip(items, preds):
        tk = IDX[true_c]; pred_c = CLASSES[pk]
        cm[tk, pk] += 1
        if (true_c, pred_c) in EXPORT:
            dst = os.path.join(CONF, "%s_pred_%s" % (true_c, pred_c), fn)
            shutil.copy2(path, dst)

    # metrics
    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print("%-12s %10s %8s %8s | %6s" % ("true\\pred", *CLASSES, "recall"))
    for i, c in enumerate(CLASSES):
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        print("%-12s %10d %8d %8d | %5.1f%%" % (c, cm[i,0], cm[i,1], cm[i,2], 100*rec))
    print("%-12s" % "precision", end="")
    for j in range(3):
        prec = cm[j, j] / cm[:, j].sum() if cm[:, j].sum() else 0
        print(" %9.1f%%" % (100*prec), end="")
    print()

    print("\n=== Per-class P / R / F1 ===")
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        print("  %-11s P=%.3f  R=%.3f  F1=%.3f  (support=%d)" % (c, prec, rec, f1, cm[i].sum()))

    # focus stats
    od, so = IDX["OnDeck"], IDX["Sorting"]
    od_total, so_total = cm[od].sum(), cm[so].sum()
    print("\n=== FOCUS ===")
    print("  OnDeck recall : %.3f  (%d/%d)" % (cm[od,od]/od_total if od_total else 0, cm[od,od], od_total))
    print("  Sorting recall: %.3f  (%d/%d)" % (cm[so,so]/so_total if so_total else 0, cm[so,so], so_total))
    print("  OnDeck->Sorting confusion : %d  (%.1f%% of OnDeck)" % (cm[od,so], 100*cm[od,so]/od_total if od_total else 0))
    print("  Sorting->OnDeck confusion : %d  (%.1f%% of Sorting)" % (cm[so,od], 100*cm[so,od]/so_total if so_total else 0))

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title("3-class confusion (val, true dist)")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(PLOT, dpi=120)
    print("\nconfusions dir:", CONF)
    print("plot:", PLOT)


if __name__ == "__main__":
    main()
