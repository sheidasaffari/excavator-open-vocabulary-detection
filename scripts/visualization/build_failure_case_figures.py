#!/usr/bin/env python3
"""Build the three qualitative figures used in the repository README.

The case identities are frozen below. Images and raw predictions are intentionally
not distributed; pass the local benchmark and evaluation roots explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

RED = "#D62728"
GREEN = "#00A651"
BLUE = "#0072B2"
THRESHOLD = 0.25

GD_CASES = [
    ("backhoe_loader_11", "Backhoe loader", "Whole-machine category confusion"),
    ("bulldozer_07", "Bulldozer", "Whole-machine category confusion"),
    ("wheel_loader_04", "Wheel loader", "Whole-machine category confusion"),
    ("isolated_attachment_01", "Detached boom and bucket", "Part-versus-whole confusion"),
    ("crane_pilot_01", "Mobile crane", "Whole-machine category confusion"),
]

YW_CASES = [
    ("0008085", 1, "Near-threshold suppression", "Correctly localized, but score < 0.25"),
    ("0006277", 3, "Small/partial instance", "Correct candidate is severely under-scored"),
    ("0006018", 3, "Localization failure", "Best candidate does not reach IoU 0.50"),
    ("0004214", 2, "Multi-instance miss", "One excavator has no valid proposal"),
    ("0001575", 0, "Small/distant proposal failure", "Prediction overlaps only part of the target"),
]

RECOVERED_CASES = [
    ("mocs_dev", "0005609", 0, "Typical earthmoving scene"),
    ("mocs_dev", "0001929", 0, "Multi-excavator scene"),
    ("cis_test", "excavator_003025", 0, "Unusual viewpoint and attachment"),
    ("positive_control", "pc3_076", 0, "Rear/partial appearance"),
    ("positive_control", "pc3_108", 0, "Small, distant excavator"),
]


def load_json(path: Path):
    return json.loads(path.read_text())


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image for {stem} in {directory}")


def yolo_gt(image_path: Path, label_path: Path):
    width, height = Image.open(image_path).size
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append([
            (cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height,
        ])
    return boxes


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def best_candidate(pred, gt, require_iou=None):
    candidates = []
    for box, score in zip(pred.get("boxes", []), pred.get("scores", [])):
        overlap = iou(box, gt)
        if require_iou is None or overlap >= require_iou:
            candidates.append((float(score), overlap, box))
    if not candidates:
        return 0.0, 0.0, None
    if require_iou is None:
        return max(candidates, key=lambda row: row[1])
    return max(candidates, key=lambda row: row[0])


def draw_box(ax, box, color, label, linestyle="-", label_position="top"):
    x0, y0, x1, y1 = box
    ax.add_patch(patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, linewidth=2.5,
        edgecolor=color, facecolor="none", linestyle=linestyle,
    ))
    label_y = max(3, y0 - 4) if label_position == "top" else y1 + 4
    vertical_alignment = "bottom" if label_position == "top" else "top"
    ax.text(
        x0, label_y, label, color="white", fontsize=8, weight="bold",
        va=vertical_alignment, ha="left",
        bbox=dict(facecolor=color, edgecolor="none", alpha=0.92, pad=1.5),
    )


def panel(ax, image_path):
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    ax.set_axis_off()
    return image


def save(fig, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grounding_dino_figure(args, manifest):
    audit_path = args.foundation_root / "results/minimal_error_decomposition/gd/manual_audit_decisions.csv"
    audit = {row["image_id"]: row for row in csv.DictReader(audit_path.open())}
    predictions = load_json(args.foundation_root / "predictions/grounding_dino__confuser__canonical_floor001.json")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    for index, (stem, visible_class, failure) in enumerate(GD_CASES):
        ax = axes[index]
        row = audit[stem]
        image_path = Path(row["image_path"])
        panel(ax, image_path)
        kept = [(b, s) for b, s in zip(predictions[stem]["boxes"], predictions[stem]["scores"]) if s >= THRESHOLD]
        for box, score in kept:
            draw_box(ax, box, RED, f'excavator {score:.2f}')
        ax.set_title(f"({chr(97 + index)}) {visible_class}\n{failure}", fontsize=12, weight="bold")
        ax.text(0.01, 0.01, "Ground truth: negative (no excavator)", transform=ax.transAxes,
                fontsize=9, color="white", weight="bold",
                bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2))
        manifest.append(["grounding_dino_confusion", stem, visible_class, failure,
                         "negative", row["top_score"], row["n_predictions_ge_025"]])
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.66, "Box legend", fontsize=15, weight="bold")
    axes[-1].add_patch(patches.Rectangle((0.05, 0.52), 0.12, 0.08, transform=axes[-1].transAxes,
                                         fill=False, edgecolor=RED, linewidth=3))
    axes[-1].text(0.21, 0.56, "False Grounding DINO detection", transform=axes[-1].transAxes, va="center", fontsize=12)
    axes[-1].text(0.05, 0.35, "Prompt: “excavator.”\nOperating threshold: 0.25\nAll five images contain no excavator.",
                       transform=axes[-1].transAxes, fontsize=11, linespacing=1.6)
    fig.suptitle("Grounding DINO: varied excavator-category confusions", fontsize=20, weight="bold")
    save(fig, args.output, "grounding_dino_confusion_examples")


def yolo_failure_figure(args, manifest):
    predictions = load_json(args.foundation_root / "predictions/yolo_world__mocs__construction.json")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()
    for index, (stem, gt_index, failure, explanation) in enumerate(YW_CASES):
        ax = axes[index]
        image_path = find_image(args.mocs_root / "images", stem)
        panel(ax, image_path)
        gt = yolo_gt(image_path, args.mocs_root / "labels" / f"{stem}.txt")[gt_index]
        score, overlap, box = best_candidate(predictions[stem], gt)
        draw_box(ax, gt, GREEN, "GT excavator", label_position="bottom")
        if box is not None:
            draw_box(ax, box, RED, f"candidate {score:.3f} | IoU {overlap:.2f}")
        ax.set_title(f"({chr(97 + index)}) {failure}\n{explanation}", fontsize=12, weight="bold")
        manifest.append(["yolo_world_failure", stem, "excavator", failure,
                         f"gt_index={gt_index}; best_iou={overlap:.4f}", score, 1])
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.72, "Box legend", fontsize=15, weight="bold")
    for y, color, label in [(0.58, GREEN, "Ground-truth excavator"), (0.44, RED, "Best low-floor candidate")]:
        axes[-1].add_patch(patches.Rectangle((0.05, y), 0.12, 0.08, transform=axes[-1].transAxes,
                                             fill=False, edgecolor=color, linewidth=3))
        axes[-1].text(0.21, y + 0.04, label, transform=axes[-1].transAxes, va="center", fontsize=12)
    axes[-1].text(0.05, 0.18, "Prompt: “construction excavator”\nPrediction floor: 0.001\nOperating threshold: 0.25\nIoU success criterion: 0.50",
                       transform=axes[-1].transAxes, fontsize=11, linespacing=1.5)
    fig.suptitle("YOLO-World: representative zero-shot failure cases", fontsize=20, weight="bold")
    save(fig, args.output, "yolo_world_failure_examples")


def recovered_figure(args, manifest):
    dataset_paths = {
        "mocs_dev": (args.mocs_root / "images", args.mocs_root / "labels"),
        "cis_test": (args.foundation_root / "data_links/cis_combined/images", args.foundation_root / "data_links/cis_combined/labels"),
        "positive_control": (args.foundation_root / "positive_control/eval_set/images", args.foundation_root / "positive_control/eval_set/labels"),
    }
    evaluation = args.foundation_root / "adaptation/evaluation_selected_multiseed_seededloader"
    baselines, adapted = {}, {}
    for dataset in dataset_paths:
        baselines[dataset] = load_json(evaluation / "zeroshot_common_config" / f"{dataset}_predictions.json")
        adapted[dataset] = [load_json(evaluation / f"selected_random_100k_seed{seed}" / f"{dataset}_predictions.json") for seed in range(3)]

    fig, axes = plt.subplots(5, 2, figsize=(15, 22), constrained_layout=True)
    for row_index, (dataset, stem, gt_index, description) in enumerate(RECOVERED_CASES):
        image_dir, label_dir = dataset_paths[dataset]
        image_path = find_image(image_dir, stem)
        gt = yolo_gt(image_path, label_dir / f"{stem}.txt")[gt_index]
        before = best_candidate(baselines[dataset][stem], gt, require_iou=0.50)
        after = [best_candidate(run[stem], gt, require_iou=0.50) for run in adapted[dataset]]
        for column, (title, pred, color) in enumerate([
            ("No adaptation", before, RED), ("Random-100K adaptation (seed 0 shown)", after[0], BLUE)
        ]):
            ax = axes[row_index, column]
            panel(ax, image_path)
            draw_box(ax, gt, GREEN, "GT excavator", label_position="bottom")
            score, overlap, box = pred
            if box is not None:
                draw_box(ax, box, color, f"score {score:.3f} | IoU {overlap:.2f}")
            ax.set_title(f"{title}\n{description}", fontsize=12, weight="bold")
        scores = [item[0] for item in after]
        axes[row_index, 0].text(0.01, 0.01, f"{dataset} · {stem} · below 0.25", transform=axes[row_index, 0].transAxes,
                                fontsize=9, color="white", weight="bold",
                                bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2))
        axes[row_index, 1].text(0.01, 0.01, f"all seeds recovered: {scores[0]:.2f}, {scores[1]:.2f}, {scores[2]:.2f}",
                                transform=axes[row_index, 1].transAxes, fontsize=9, color="white", weight="bold",
                                bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2))
        manifest.append(["yolo_world_recovered", stem, "excavator", description,
                         f"dataset={dataset}; gt_index={gt_index}; all_3_seeds=true",
                         f"{before[0]:.6f}->{after[0][0]:.6f}", 3])
    fig.suptitle("YOLO-World detections recovered after synthetic-only selective adaptation",
                 fontsize=20, weight="bold")
    fig.text(0.5, -0.005,
             "Green: ground truth · Red: zero-shot candidate · Blue: adapted candidate. "
             "Every example changes from below 0.25 to a valid IoU≥0.50 detection in all three adapted seeds.",
             ha="center", fontsize=11)
    save(fig, args.output, "yolo_world_recovered_after_adaptation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundation-root", type=Path, required=True)
    parser.add_argument("--mocs-root", type=Path, required=True,
                        help="Directory containing MOCS images/ and labels/.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = []
    grounding_dino_figure(args, manifest)
    yolo_failure_figure(args, manifest)
    recovered_figure(args, manifest)
    with (args.output / "failure_case_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "image_id", "visible_class", "failure_or_slice",
                         "selection_evidence", "score", "n_or_seeds"])
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
