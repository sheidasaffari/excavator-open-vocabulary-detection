#!/usr/bin/env python3
"""Create additional qualitative recoveries for the frozen Random-5K model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


FOUNDATIONAL = Path("/home/ssaffari/Data/foundational")
MOCS = Path("/home/ssaffari/Data/real/mocs/selected")
BASELINE = (
    FOUNDATIONAL
    / "adaptation/evaluation_selected_multiseed_seededloader/zeroshot_common_config"
)
MOCS_EVAL = FOUNDATIONAL / "adaptation/evaluation_random_size_multiseed_mocs"
HELDOUT_EVAL = FOUNDATIONAL / "adaptation/evaluation_selected_5k_multiseed"
OUT = Path(__file__).resolve().parents[2] / "results/failure_cases"
THRESHOLD = 0.25
GREEN = "#00A651"
RED = "#D62728"
BLUE = "#0072B2"

DATASETS = {
    "mocs_dev": (MOCS / "images", MOCS / "labels", "MOCS development"),
    "cis_test": (
        FOUNDATIONAL / "data_links/cis_combined/images",
        FOUNDATIONAL / "data_links/cis_combined/labels",
        "CIS held out",
    ),
    "positive_control": (
        FOUNDATIONAL / "positive_control/eval_set/images",
        FOUNDATIONAL / "positive_control/eval_set/labels",
        "Independent positive control",
    ),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def find_image(directory: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(stem)


def gt_boxes(image_path: Path, label_path: Path) -> list[list[float]]:
    width, height = Image.open(image_path).size
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def best_valid(prediction: dict, gt: list[float]) -> tuple[float, float, list[float] | None]:
    candidates = []
    for box, score in zip(prediction.get("boxes", []), prediction.get("scores", [])):
        overlap = iou(box, gt)
        if overlap >= 0.50:
            candidates.append((float(score), overlap, box))
    return max(candidates, default=(0.0, 0.0, None), key=lambda row: row[0])


def predictions(dataset: str) -> tuple[dict, list[dict]]:
    baseline = load(BASELINE / f"{dataset}_predictions.json")
    if dataset == "mocs_dev":
        adapted = [
            load(MOCS_EVAL / f"random_5k_seed{seed}/mocs_dev_predictions.json")
            for seed in range(3)
        ]
    else:
        adapted = [
            load(
                HELDOUT_EVAL
                / f"selected_random_5k_seed{seed}/{dataset}_predictions.json"
            )
            for seed in range(3)
        ]
    return baseline, adapted


def recovered_candidates(dataset: str) -> list[dict]:
    image_dir, label_dir, _ = DATASETS[dataset]
    baseline, adapted = predictions(dataset)
    candidates = []
    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem
        if stem not in baseline or any(stem not in run for run in adapted):
            continue
        image_path = find_image(image_dir, stem)
        width, height = Image.open(image_path).size
        boxes = gt_boxes(image_path, label_path)
        for gt_index, gt in enumerate(boxes):
            before = best_valid(baseline[stem], gt)
            after = [best_valid(run[stem], gt) for run in adapted]
            if before[0] >= THRESHOLD or min(item[0] for item in after) < THRESHOLD:
                continue
            area_ratio = ((gt[2] - gt[0]) * (gt[3] - gt[1])) / (width * height)
            candidates.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "gt_index": gt_index,
                    "gt": gt,
                    "before": before,
                    "after": after,
                    "area_ratio": area_ratio,
                    "n_instances": len(boxes),
                    "image_path": image_path,
                }
            )
    return candidates


def choose(candidates: list[dict]) -> list[tuple[dict, str]]:
    by_dataset = {
        dataset: [row for row in candidates if row["dataset"] == dataset]
        for dataset in DATASETS
    }
    chosen = []
    used = set()

    def add(row: dict, description: str) -> None:
        key = (row["dataset"], row["stem"], row["gt_index"])
        if key not in used:
            chosen.append((row, description))
            used.add(key)

    for dataset in DATASETS:
        pool = by_dataset[dataset]
        multi = [row for row in pool if row["n_instances"] > 1]
        if multi:
            add(max(multi, key=lambda row: min(item[0] for item in row["after"])), "Multi-instance scene")
        add(min(pool, key=lambda row: row["area_ratio"]), "Small/distant excavator")

    if len(chosen) < 6:
        remaining = sorted(
            candidates,
            key=lambda row: min(item[0] for item in row["after"]) - row["before"][0],
            reverse=True,
        )
        for row in remaining:
            add(row, "Low-confidence zero-shot recovery")
            if len(chosen) == 6:
                break
    return chosen[:6]


def draw_box(ax, box, color, label, label_below=False):
    x0, y0, x1, y1 = box
    ax.add_patch(
        patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor=color, linewidth=2.5,
        )
    )
    y = y1 + 4 if label_below else max(2, y0 - 4)
    ax.text(
        x0, y, label, color="white", fontsize=8, weight="bold",
        va="top" if label_below else "bottom",
        bbox=dict(facecolor=color, edgecolor="none", alpha=0.92, pad=1.5),
    )


def main() -> None:
    candidates = [row for dataset in DATASETS for row in recovered_candidates(dataset)]
    selected = choose(candidates)
    if len(selected) != 6:
        raise RuntimeError(f"Expected six recoveries, found {len(selected)}")

    fig, axes = plt.subplots(6, 2, figsize=(15, 25), constrained_layout=True)
    manifest_rows = []
    for row_index, (case, description) in enumerate(selected):
        dataset_label = DATASETS[case["dataset"]][2]
        displays = [
            ("No adaptation", case["before"], RED),
            ("Random-5K adaptation (seed 0 shown)", case["after"][0], BLUE),
        ]
        for column, (model_label, prediction, color) in enumerate(displays):
            ax = axes[row_index, column]
            ax.imshow(Image.open(case["image_path"]).convert("RGB"))
            ax.set_axis_off()
            draw_box(ax, case["gt"], GREEN, "GT excavator", label_below=True)
            score, overlap, box = prediction
            if box is not None:
                draw_box(ax, box, color, f"score {score:.3f} | IoU {overlap:.2f}")
            ax.set_title(f"{model_label}\n{dataset_label} · {description}", fontsize=11, weight="bold")
        scores = [item[0] for item in case["after"]]
        axes[row_index, 0].text(
            0.01, 0.01, f"{case['stem']} · below 0.25", transform=axes[row_index, 0].transAxes,
            fontsize=8, color="white", weight="bold",
            bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2),
        )
        axes[row_index, 1].text(
            0.01, 0.01,
            f"all seeds recovered: {scores[0]:.2f}, {scores[1]:.2f}, {scores[2]:.2f}",
            transform=axes[row_index, 1].transAxes, fontsize=8, color="white", weight="bold",
            bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=2),
        )
        manifest_rows.append(
            {
                "figure": "yolo_world_selected_random5k_recoveries",
                "dataset": case["dataset"],
                "image_id": case["stem"],
                "gt_index": case["gt_index"],
                "description": description,
                "baseline_score": case["before"][0],
                "seed0_score": scores[0],
                "seed1_score": scores[1],
                "seed2_score": scores[2],
                "all_three_seeds_recovered": True,
            }
        )

    fig.suptitle(
        "Additional YOLO-World detections recovered by frozen Random-5K adaptation",
        fontsize=19, weight="bold",
    )
    fig.text(
        0.5, -0.004,
        "Green: ground truth · Red: zero-shot candidate · Blue: adapted candidate. "
        "Every case crosses 0.25 with IoU≥0.50 in all three adapted seeds.",
        ha="center", fontsize=10,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"yolo_world_selected_random5k_recoveries.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)
    with (OUT / "selected_random5k_recovery_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Recovered candidates: {len(candidates)}")
    print(OUT / "yolo_world_selected_random5k_recoveries.png")


if __name__ == "__main__":
    main()
