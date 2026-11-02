"""Shared GT-loading + COCO-style single-class AP/precision/recall eval."""
import os

import numpy as np
from PIL import Image


def load_yolo_gt(labels_dir, images_dir, stems):
    """Returns dict: stem -> (width, height, [xyxy boxes])"""
    gt = {}
    for stem in stems:
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(images_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        boxes = []
        lbl_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = parts
                    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                    boxes.append([x0, y0, x1, y1])
        gt[stem] = {"img_path": img_path, "width": w, "height": h, "boxes": boxes}
    return gt


def iou_xyxy(a, b):
    xa0, ya0, xa1, ya1 = a
    xb0, yb0, xb1, yb1 = b
    ix0, iy0 = max(xa0, xb0), max(ya0, yb0)
    ix1, iy1 = min(xa1, xb1), min(ya1, yb1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
    area_b = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap_at_iou(all_preds, all_gts, iou_thr):
    """all_preds: list of (stem, score, box); all_gts: dict stem -> list of boxes.
    Standard single-class greedy-matching AP (COCO-style, 101-point interpolation)."""
    preds_sorted = sorted(all_preds, key=lambda x: -x[1])
    gt_matched = {stem: [False] * len(boxes) for stem, boxes in all_gts.items()}
    n_gt = sum(len(b) for b in all_gts.values())
    if n_gt == 0:
        return 0.0, [], []

    tp, fp = [], []
    for stem, score, box in preds_sorted:
        gts = all_gts.get(stem, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts):
            if gt_matched[stem][j]:
                continue
            iou = iou_xyxy(box, gbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            gt_matched[stem][best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(n_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # 101-point interpolation (COCO style)
    recall_thrs = np.linspace(0, 1, 101)
    precisions_interp = np.zeros(101)
    for i, r in enumerate(recall_thrs):
        mask = recalls >= r
        precisions_interp[i] = precisions[mask].max() if mask.any() else 0.0
    ap = precisions_interp.mean()
    return float(ap), recalls, precisions


def compute_pr_f1_at_threshold(all_preds, all_gts, score_thr, iou_thr=0.5):
    kept = [(s, sc, b) for s, sc, b in all_preds if sc >= score_thr]
    gt_matched = {stem: [False] * len(boxes) for stem, boxes in all_gts.items()}
    n_gt = sum(len(b) for b in all_gts.values())
    tp, fp = 0, 0
    for stem, score, box in sorted(kept, key=lambda x: -x[1]):
        gts = all_gts.get(stem, [])
        best_iou, best_j = 0.0, -1
        for j, gbox in enumerate(gts):
            if gt_matched[stem][j]:
                continue
            iou = iou_xyxy(box, gbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            gt_matched[stem][best_j] = True
            tp += 1
        else:
            fp += 1
    fn = n_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def evaluate(all_preds, all_gts, score_thr=0.25):
    ap50, _, _ = compute_ap_at_iou(all_preds, all_gts, 0.5)
    aps = [compute_ap_at_iou(all_preds, all_gts, t)[0] for t in np.arange(0.5, 1.0, 0.05)]
    ap5095 = float(np.mean(aps))
    precision, recall, f1 = compute_pr_f1_at_threshold(all_preds, all_gts, score_thr, iou_thr=0.5)
    n_gt = sum(len(b) for b in all_gts.values())
    n_pred = len(all_preds)
    return {
        "AP50": ap50, "AP50:95": ap5095,
        "precision": precision, "recall": recall, "f1": f1,
        "n_gt": n_gt, "n_pred_total": n_pred,
    }
