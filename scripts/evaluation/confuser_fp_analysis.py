#!/usr/bin/env python3
"""
Specificity analysis on the all-negative confuser set.

Metrics (per locked protocol clarifications):
  - Image-level FPR: % of images with >=1 prediction above the locked
    score threshold (0.25). Reported once (collapses the originally
    duplicated "image-level FP rate" / "% images with >=1 pred > 0.25"
    into a single metric).
  - FP/100 images: 100 * (total predictions above 0.25) / N
  - Median / highest max-confidence per image
  - All broken out per category, with raw counts alongside rates since
    several categories are small (explicitly descriptive, not inferential).

No AP/recall is computed: all 114 images are negatives (no ground truth
excavators), so this evaluates specificity against visually related
confusers, not detection accuracy or recall. A future positive-control
group is a separate localization evaluation and is NOT included here.

Raw predictions were saved by run_grounding_dino.py / run_yolo_world.py
at their existing low default floors (box_thr=0.1/text_thr=0.1 for
Grounding DINO; conf=0.001 for YOLO-World) -- i.e. already below the
locked 0.25 threshold, so no re-inference is needed to change the
score floor; all metrics below are computed post-hoc from the raw JSON.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

SCORE_THR = 0.25
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PRED_DIR = os.path.join(ROOT, "predictions")
MANIFEST = os.path.join(ROOT, "data", "confusers", "manifest.csv")
OUT_MD = os.path.join(ROOT, "results", "confuser_fp_report.md")

CONFIGS = [
    ("grounding_dino", "canonical/selected (identical: \"excavator.\")",
     "grounding_dino__confuser__canonical.json"),
    ("yolo_world", "canonical (\"excavator\")",
     "yolo_world__confuser__canonical.json"),
    ("yolo_world", "dev-selected (\"construction excavator\")",
     "yolo_world__confuser__selected.json"),
]


def load_manifest():
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    stem_to_cat = {}
    for r in rows:
        stem = os.path.splitext(r["filename"])[0]
        stem_to_cat[stem] = r["category"]
    return stem_to_cat


def analyze(pred_json, stem_to_cat):
    with open(pred_json) as f:
        raw = json.load(f)

    per_cat = defaultdict(lambda: {"images": 0, "images_with_fp": 0,
                                    "total_fp": 0, "max_confs": []})
    overall = {"images": 0, "images_with_fp": 0, "total_fp": 0, "max_confs": []}

    for stem, d in raw.items():
        if stem not in stem_to_cat:
            # excluded (e.g. audited out for containing a genuine excavator) --
            # skip rather than counting under "UNKNOWN"
            continue
        cat = stem_to_cat[stem]
        scores = d.get("scores", [])
        kept = [s for s in scores if s >= SCORE_THR]
        max_conf = max(scores) if scores else 0.0

        bucket = per_cat[cat]
        bucket["images"] += 1
        bucket["total_fp"] += len(kept)
        bucket["max_confs"].append(max_conf)
        if len(kept) > 0:
            bucket["images_with_fp"] += 1

        overall["images"] += 1
        overall["total_fp"] += len(kept)
        overall["max_confs"].append(max_conf)
        if len(kept) > 0:
            overall["images_with_fp"] += 1

    return per_cat, overall


def fmt_row(name, b):
    n = b["images"]
    with_fp = b["images_with_fp"]
    fpr = 100.0 * with_fp / n if n else 0.0
    fp_per_100 = 100.0 * b["total_fp"] / n if n else 0.0
    med_conf = float(np.median(b["max_confs"])) if b["max_confs"] else 0.0
    max_conf = float(np.max(b["max_confs"])) if b["max_confs"] else 0.0
    return (f"| {name} | {n} | {with_fp}/{n} ({fpr:.1f}%) | {fpr:.1f}% | "
            f"{b['total_fp']} | {fp_per_100:.1f} | {med_conf:.3f} | {max_conf:.3f} |")


def main():
    stem_to_cat = load_manifest()
    all_cats = sorted(set(stem_to_cat.values()))

    lines = [
        "# Confuser-set specificity results, CORRECTED (110 images, all negatives)",
        "",
        "**Correction applied:** 4 of the original 114 images "
        "(`bulldozer_03`, `dump_truck_07`, `dump_truck_09`, `dump_truck_12`) "
        "were found on visual review to contain a genuine excavator in the "
        "background, violating the exclusion rule. They were moved to "
        "`audit_contains_excavator/` for separate triage (possible future "
        "positive-control candidates) and removed from this negative set. "
        "No re-inference was needed -- these results are recomputed from "
        "the existing prediction JSONs with those 4 stems excluded.",
        "",
        "**This experiment evaluates specificity against visually related "
        "confusers, not recall or detection accuracy.** No AP is computed "
        "(no positive ground truth in this set). A future positive-control "
        "group will be evaluated separately as a localization task.",
        "",
        f"Locked score threshold for FP counting: **{SCORE_THR}**. Raw "
        "predictions were saved at each model's existing low default "
        "floor (Grounding DINO box_thr=0.1/text_thr=0.1; YOLO-World "
        "conf=0.001), so thresholding below is purely post-hoc -- no "
        "re-inference occurred.",
        "",
        "Checkpoint / prompt / input size / confidence threshold / NMS / "
        "preprocessing are identical to the original MOCS/ACID/CIS "
        "benchmark runs (same `run_grounding_dino.py` / `run_yolo_world.py` "
        "scripts, unchanged, same model checkpoints: "
        "`IDEA-Research/grounding-dino-base`, `yolov8s-worldv2.pt`).",
        "",
    ]

    overall_summary = []

    for model, variant, fname in CONFIGS:
        pred_json = os.path.join(PRED_DIR, fname)
        per_cat, overall = analyze(pred_json, stem_to_cat)

        lines.append(f"## {model} -- {variant}")
        lines.append("")
        lines.append("| Category | Images | Images with \u2265 1 FP | Image-level FPR | "
                      "Total FPs | FPs/100 images | Median max conf | Highest conf |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat in all_cats:
            b = per_cat.get(cat, {"images": 0, "images_with_fp": 0, "total_fp": 0, "max_confs": []})
            lines.append(fmt_row(cat, b))
        lines.append(fmt_row("**ALL (overall)**", overall))
        lines.append("")

        overall_summary.append((model, variant, overall))

    lines.append("## Overall comparison across the three locked configurations")
    lines.append("")
    lines.append("| Config | Images | Image-level FPR | FPs/100 images | Median max conf | Highest conf |")
    lines.append("|---|---|---|---|---|---|")
    for model, variant, overall in overall_summary:
        n = overall["images"]
        fpr = 100.0 * overall["images_with_fp"] / n if n else 0.0
        fp100 = 100.0 * overall["total_fp"] / n if n else 0.0
        med = float(np.median(overall["max_confs"])) if overall["max_confs"] else 0.0
        mx = float(np.max(overall["max_confs"])) if overall["max_confs"] else 0.0
        lines.append(f"| {model} ({variant}) | {n} | {overall['images_with_fp']}/{n} ({fpr:.1f}%) | "
                      f"{fp100:.1f} | {med:.3f} | {mx:.3f} |")
    lines.append("")
    lines.append("Note: category-level rates are **descriptive only** -- several categories "
                  "(e.g. isolated_attachment n=6, bulldozer n=12) are small, so single-image "
                  "swings move the rate substantially. Counts are shown alongside every rate "
                  "for that reason; no confidence intervals are computed given the small, "
                  "non-random samples per category.")

    table = "\n".join(lines)
    print(table)
    with open(OUT_MD, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {OUT_MD}")


if __name__ == "__main__":
    main()
