#!/usr/bin/env python3
"""Build recall/specificity operating curves from frozen positive and confuser sets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


CONFIGS = {
    "grounding_dino": {
        "label": "Grounding DINO · excavator.",
        "positive": "grounding_dino__poscontrol__canonical_floor001.json",
        "negative": "grounding_dino__confuser__canonical_floor001.json",
        "color": "#0072B2",
        "marker": "o",
        "floor": 0.01,
    },
    "yolo_world_canonical": {
        "label": "YOLO-World · excavator",
        "positive": "yolo_world__poscontrol__canonical.json",
        "negative": "yolo_world__confuser__canonical.json",
        "color": "#D55E00",
        "marker": "s",
        "floor": 0.001,
    },
    "yolo_world_selected": {
        "label": "YOLO-World · construction excavator",
        "positive": "yolo_world__poscontrol__selected.json",
        "negative": "yolo_world__confuser__selected.json",
        "color": "#009E73",
        "marker": "^",
        "floor": 0.001,
    },
}


def iou_xyxy(a, b):
    xa0, ya0, xa1, ya1 = a
    xb0, yb0, xb1, yb1 = b
    ix0, iy0 = max(xa0, xb0), max(ya0, yb0)
    ix1, iy1 = min(xa1, xb1), min(ya1, yb1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
    ab = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def load_yolo_gt(images_dir: Path, labels_dir: Path):
    gt = {}
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        boxes = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            cx, cy = cx * width, cy * height
            bw, bh = bw * width, bh * height
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        gt[image_path.stem] = boxes
    return gt


def flatten_predictions(raw, allowed_stems=None):
    rows = []
    for stem, record in raw.items():
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        for score, box in zip(record.get("scores", []), record.get("boxes", [])):
            rows.append((stem, float(score), [float(v) for v in box]))
    return rows


def prepare_positive_prefix(predictions, gt, iou_threshold=0.5):
    matched = {stem: [False] * len(boxes) for stem, boxes in gt.items()}
    scores, tp_flags, fp_flags = [], [], []
    for stem, score, box in sorted(predictions, key=lambda row: -row[1]):
        boxes = gt.get(stem, [])
        best_iou, best_index = 0.0, -1
        for index, target in enumerate(boxes):
            if matched[stem][index]:
                continue
            overlap = iou_xyxy(box, target)
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[stem][best_index] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
        scores.append(score)
    return {
        "scores": np.asarray(scores, dtype=float),
        "cum_tp": np.cumsum(tp_flags),
        "cum_fp": np.cumsum(fp_flags),
        "n_gt": sum(len(boxes) for boxes in gt.values()),
    }


def prepare_negative_scores(raw, allowed_stems):
    all_scores, max_scores = [], []
    for stem in sorted(allowed_stems):
        scores = [float(score) for score in raw.get(stem, {}).get("scores", [])]
        all_scores.extend(scores)
        max_scores.append(max(scores) if scores else 0.0)
    return np.sort(np.asarray(all_scores, dtype=float))[::-1], np.asarray(max_scores, dtype=float)


def safe_div(a, b):
    return a / b if b else 0.0


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_manifests(root: Path):
    positive_manifest = root / "positive_control/manifest_final.csv"
    positive_frozen = json.loads((root / "positive_control/manifest_final.FROZEN.json").read_text())
    confuser_manifest = root / "confuser_set/manifest_corrected.csv"
    confuser_frozen = json.loads((root / "confuser_set/FROZEN_manifest_corrected.json").read_text())
    actual_positive = sha256(positive_manifest)
    actual_confuser = sha256(confuser_manifest)
    if actual_positive != positive_frozen["manifest_file_sha256"]:
        raise RuntimeError("Positive-control manifest no longer matches its frozen hash")
    if actual_confuser != confuser_frozen["manifest_sha256"]:
        raise RuntimeError("Confuser manifest no longer matches its frozen hash")
    return {
        "positive_manifest": {"path": str(positive_manifest), "sha256": actual_positive},
        "confuser_manifest": {"path": str(confuser_manifest), "sha256": actual_confuser},
    }


def prefix_value(cumulative, count):
    return int(cumulative[count - 1]) if count else 0


def evaluate_curve(pos_predictions, pos_gt, neg_raw, neg_stems, thresholds):
    positive = prepare_positive_prefix(pos_predictions, pos_gt)
    negative_scores, negative_max_scores = prepare_negative_scores(neg_raw, neg_stems)
    rows = []
    for threshold in thresholds:
        n_pos_kept = int(np.searchsorted(-positive["scores"], -threshold, side="right"))
        tp = prefix_value(positive["cum_tp"], n_pos_kept)
        fp_pos = prefix_value(positive["cum_fp"], n_pos_kept)
        fn = positive["n_gt"] - tp
        fp_neg = int(np.searchsorted(-negative_scores, -threshold, side="right"))
        images_with_fp = int(np.count_nonzero(negative_max_scores >= threshold))
        recall = safe_div(tp, tp + fn)
        pos_precision = safe_div(tp, tp + fp_pos)
        combined_precision = safe_div(tp, tp + fp_pos + fp_neg)
        combined_f1 = safe_div(2 * combined_precision * recall, combined_precision + recall)
        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fn": fn,
            "positive_fp": fp_pos,
            "confuser_fp": fp_neg,
            "positive_recall": recall,
            "positive_precision": pos_precision,
            "combined_precision": combined_precision,
            "combined_f1": combined_f1,
            "confuser_image_fpr": safe_div(images_with_fp, len(neg_stems)),
            "confuser_fppi": safe_div(fp_neg, len(neg_stems)),
        })
    return rows


def nearest_threshold(rows, target=0.25):
    return min(rows, key=lambda row: abs(row["threshold"] - target))


def under_fpr(rows, cap):
    eligible = [row for row in rows if row["confuser_image_fpr"] <= cap + 1e-12]
    return max(eligible, key=lambda row: (row["positive_recall"], -row["confuser_fppi"], row["threshold"])) if eligible else None


def under_fppi(rows, cap):
    eligible = [row for row in rows if row["confuser_fppi"] <= cap + 1e-12]
    return max(eligible, key=lambda row: (row["positive_recall"], -row["confuser_image_fpr"], row["threshold"])) if eligible else None


def over_recall(rows, target):
    eligible = [row for row in rows if row["positive_recall"] >= target - 1e-12]
    return min(eligible, key=lambda row: (row["confuser_image_fpr"], row["confuser_fppi"], -row["threshold"])) if eligible else None


def write_csv(path, curves):
    fields = ["config", "label", *next(iter(curves.values()))[0].keys()]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, rows in curves.items():
            for row in rows:
                writer.writerow({"config": config, "label": CONFIGS[config]["label"], **row})


def write_summary_csv(path, curves):
    fields = ["config", "label", "criterion", "target", "threshold", "positive_recall",
              "positive_precision", "combined_precision", "combined_f1",
              "confuser_image_fpr", "confuser_fppi", "tp", "fn", "positive_fp", "confuser_fp"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config, rows in curves.items():
            selected = [("fixed_threshold", 0.25, nearest_threshold(rows, 0.25))]
            selected += [("max_recall_at_fpr_cap", cap, under_fpr(rows, cap)) for cap in (0.05, 0.10, 0.15, 0.25, 0.50)]
            selected += [("max_recall_at_fppi_cap", cap, under_fppi(rows, cap)) for cap in (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)]
            selected += [("min_fpr_at_recall_target", target, over_recall(rows, target)) for target in (0.25, 0.50, 0.60, 0.75, 0.80, 0.85)]
            for criterion, target, row in selected:
                if row is None:
                    continue
                writer.writerow({"config": config, "label": CONFIGS[config]["label"],
                                 "criterion": criterion, "target": target, **row})


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })


def plot_tradeoff(path_base, curves, x_key, x_label, title, x_max=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for config, rows in curves.items():
        info = CONFIGS[config]
        ordered = sorted(rows, key=lambda row: row[x_key])
        ax.plot([row[x_key] for row in ordered], [row["positive_recall"] for row in ordered],
                color=info["color"], linewidth=2.2, label=info["label"])
        fixed = nearest_threshold(rows, 0.25)
        ax.scatter([fixed[x_key]], [fixed["positive_recall"]], s=58, marker=info["marker"],
                   color=info["color"], edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate("0.25", (fixed[x_key], fixed["positive_recall"]), xytext=(5, 5),
                    textcoords="offset points", color=info["color"], fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Positive-control recall at IoU ≥ 0.50")
    ax.set_xlim(0, x_max) if x_max is not None else ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_threshold_profiles(path_base, curves):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    specs = [
        ("positive_recall", "Positive-control recall", (0, 1.02)),
        ("confuser_image_fpr", "Confuser image-level FPR", (0, 1.02)),
        ("confuser_fppi", "Confuser false positives / image", None),
    ]
    for config, rows in curves.items():
        info = CONFIGS[config]
        for ax, (key, ylabel, ylim) in zip(axes, specs):
            ax.plot([row["threshold"] for row in rows], [row[key] for row in rows],
                    color=info["color"], linewidth=2.0, label=info["label"])
            ax.axvline(0.25, color="#777777", linestyle="--", linewidth=0.8)
            ax.set_ylabel(ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    for ax in axes:
        ax.set_xlabel("Score threshold")
        ax.set_xscale("log")
        ax.set_xlim(0.001, 0.95)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Threshold profiles on frozen positive-control and confuser benchmarks", y=1.02)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(path, curves, n_pos_images, n_gt, n_neg_images):
    lines = [
        "# Operating-point analysis",
        "",
        f"Frozen evaluation inputs: **{n_pos_images} positive images / {n_gt} GT boxes** and "
        f"**{n_neg_images} corrected confuser images**. All curves use IoU ≥ 0.50 for a true positive. "
        "No prompt, annotation, inclusion rule, or model weight was changed.",
        "",
        "The threshold sweep is descriptive and was not used to retune the locked deployment threshold. "
        "The matched operating points below are post-hoc diagnostic summaries, not newly selected deployment settings.",
        "",
        "## Locked threshold 0.25",
        "",
        "| Configuration | Recall | Positive-only precision | Combined precision | Confuser image FPR | Confuser FP/image |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config, rows in curves.items():
        row = nearest_threshold(rows, 0.25)
        lines.append(f"| {CONFIGS[config]['label']} | {row['positive_recall']:.3f} | "
                     f"{row['positive_precision']:.3f} | {row['combined_precision']:.3f} | "
                     f"{row['confuser_image_fpr']:.3f} | {row['confuser_fppi']:.3f} |")
    lines += ["", "## Maximum recall subject to a confuser-FPR cap", "",
              "| Configuration | FPR cap | Threshold | Recall | Observed FPR | Confuser FP/image |",
              "|---|---:|---:|---:|---:|---:|"]
    for cap in (0.05, 0.10, 0.15, 0.25, 0.50):
        for config, rows in curves.items():
            row = under_fpr(rows, cap)
            if row:
                lines.append(f"| {CONFIGS[config]['label']} | {cap:.2f} | {row['threshold']:.3f} | "
                             f"{row['positive_recall']:.3f} | {row['confuser_image_fpr']:.3f} | "
                             f"{row['confuser_fppi']:.3f} |")
    lines += ["", "## Maximum recall subject to a confuser-FPPI cap", "",
              "| Configuration | FP/image cap | Threshold | Recall | Confuser FPR | Observed FP/image |",
              "|---|---:|---:|---:|---:|---:|"]
    for cap in (0.05, 0.10, 0.15, 0.25, 0.50, 1.00):
        for config, rows in curves.items():
            row = under_fppi(rows, cap)
            if row:
                lines.append(f"| {CONFIGS[config]['label']} | {cap:.2f} | {row['threshold']:.3f} | "
                             f"{row['positive_recall']:.3f} | {row['confuser_image_fpr']:.3f} | "
                             f"{row['confuser_fppi']:.3f} |")
    lines += ["", "## Minimum confuser FPR while meeting a recall target", "",
              "| Configuration | Recall target | Threshold | Observed recall | Confuser FPR | Confuser FP/image |",
              "|---|---:|---:|---:|---:|---:|"]
    for target in (0.25, 0.50, 0.60, 0.75, 0.80, 0.85):
        for config, rows in curves.items():
            row = over_recall(rows, target)
            if row:
                lines.append(f"| {CONFIGS[config]['label']} | {target:.2f} | {row['threshold']:.3f} | "
                             f"{row['positive_recall']:.3f} | {row['confuser_image_fpr']:.3f} | "
                             f"{row['confuser_fppi']:.3f} |")
            else:
                lines.append(f"| {CONFIGS[config]['label']} | {target:.2f} | — | not reached | — | — |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The positive-control and confuser sets are confirmatory, not threshold-selection data.",
        "- Curves show achievable post-hoc trade-offs on these frozen sets; they do not authorize choosing a new threshold from them.",
        "- Raw confidence scores are not calibrated across architectures, so comparisons should use matched recall/FPR points rather than equal thresholds alone.",
        "- Grounding DINO curves are valid down to 0.01 after rerunning the corrected confuser set at that floor; YOLO-World curves extend to their saved 0.001 floor.",
        "- The confuser benchmark is curated and relatively small; category-level population claims require additional confirmation.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = verify_frozen_manifests(args.root)

    pos_images = args.root / "positive_control/eval_set/images"
    pos_labels = args.root / "positive_control/eval_set/labels"
    pos_gt = load_yolo_gt(pos_images, pos_labels)
    with (args.root / "confuser_set/manifest_corrected.csv").open(newline="") as handle:
        neg_stems = {Path(row["filename"]).stem for row in csv.DictReader(handle)}

    loaded = {}
    for config, info in CONFIGS.items():
        pos_path = args.root / "predictions" / info["positive"]
        neg_path = args.root / "predictions" / info["negative"]
        pos_raw = json.loads(pos_path.read_text())
        neg_raw = json.loads(neg_path.read_text())
        provenance[config] = {
            "positive_predictions": {"path": str(pos_path), "sha256": sha256(pos_path)},
            "confuser_predictions": {"path": str(neg_path), "sha256": sha256(neg_path)},
            "minimum_valid_score": info["floor"],
        }
        if set(pos_raw) != set(pos_gt):
            raise RuntimeError(f"{config}: positive prediction/GT stem mismatch")
        if not neg_stems.issubset(neg_raw):
            raise RuntimeError(f"{config}: missing corrected confuser predictions")
        pos_predictions = flatten_predictions(pos_raw)
        floor = info["floor"]
        observed_scores = [floor, 0.25, 1.000001]
        observed_scores.extend(score for _, score, _ in pos_predictions if score >= floor)
        observed_scores.extend(float(score) for stem in neg_stems
                               for score in neg_raw[stem].get("scores", []) if float(score) >= floor)
        loaded[config] = (pos_predictions, neg_raw, np.unique(np.asarray(observed_scores, dtype=float)))

    # Exact empirical step curves: every observed score is a threshold boundary.
    curves = {}
    for config, (pos_predictions, neg_raw, thresholds) in loaded.items():
        curves[config] = evaluate_curve(pos_predictions, pos_gt, neg_raw, neg_stems, thresholds)

    write_csv(args.out / "operating_curve_points.csv", curves)
    write_summary_csv(args.out / "matched_operating_points.csv", curves)
    write_report(args.out / "operating_point_report.md", curves, len(pos_gt),
                 sum(len(v) for v in pos_gt.values()), len(neg_stems))
    setup_style()
    plot_tradeoff(args.out / "recall_vs_confuser_fppi", curves, "confuser_fppi",
                  "Confuser false positives per image",
                  "Recall–false-positive trade-off", x_max=3.0)
    plot_tradeoff(args.out / "recall_vs_confuser_fpr", curves, "confuser_image_fpr",
                  "Confuser images with ≥1 false detection",
                  "Recall–specificity trade-off")
    plot_threshold_profiles(args.out / "threshold_operating_profiles", curves)
    provenance["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["metric_definition"] = "score-descending greedy one-to-one matching at IoU >= 0.50"
    (args.out / "operating_curve_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote operating-point artifacts to {args.out}")


if __name__ == "__main__":
    main()
