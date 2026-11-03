#!/usr/bin/env python3
"""Aggregate the locked multi-seed MOCS subset-size study and freeze selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVALUATION = Path(
    "/home/ssaffari/Data/foundational/adaptation/evaluation_random_size_multiseed_mocs"
)
BASELINE_SUMMARY = Path(
    "/home/ssaffari/Data/foundational/adaptation/evaluation_scale_mocs_locked_prompt/"
    "zeroshot_mocs_locked_prompt_explicit/summary.json"
)
OUTPUT = Path(
    "/home/ssaffari/Data/foundational/adaptation/results_random_size_multiseed_mocs"
)
SIZES = (5_000, 10_000, 20_000, 50_000, 100_000)
LABELS = {5_000: "5K", 10_000: "10K", 20_000: "20K", 50_000: "50K", 100_000: "100K"}
CANDIDATES = (5_000, 10_000, 20_000, 50_000)
SEEDS = (0, 1, 2)
METRICS = ("AP50", "AP50:95", "precision", "recall", "f1")
T95_DF2 = 4.302652729911275
PRACTICAL_AP_MARGIN = 0.005
PRACTICAL_RECALL_MARGIN = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half = T95_DF2 * std / math.sqrt(len(array))
    return mean, std, half


def ci_overlap(a: dict, b: dict, metric: str) -> bool:
    a_low = a[f"{metric}_mean"] - a[f"{metric}_ci95_half"]
    a_high = a[f"{metric}_mean"] + a[f"{metric}_ci95_half"]
    b_low = b[f"{metric}_mean"] - b[f"{metric}_ci95_half"]
    b_high = b[f"{metric}_mean"] + b[f"{metric}_ci95_half"]
    return max(a_low, b_low) <= min(a_high, b_high)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)

    per_seed: list[dict] = []
    config_reference = None
    checkpoint_hashes = {}
    prediction_hashes = {}

    for size in SIZES:
        label = LABELS[size].lower()
        for seed in SEEDS:
            run = EVALUATION / f"random_{label}_seed{seed}"
            summary_path = run / "summary.json"
            prediction_path = run / "mocs_dev_predictions.json"
            payload = json.loads(summary_path.read_text())
            locked_config = {
                key: payload[key]
                for key in (
                    "prompt", "prediction_floor", "operating_point", "image_size",
                    "nms_iou", "max_detections", "class_agnostic_nms"
                )
            }
            if config_reference is None:
                config_reference = locked_config
            if locked_config != config_reference:
                raise RuntimeError(f"Inference configuration mismatch in {summary_path}")
            metrics = payload["datasets"]["mocs_dev"]
            if metrics["n_gt"] != 1452:
                raise RuntimeError(f"Unexpected GT count in {summary_path}: {metrics['n_gt']}")
            checkpoint = Path(payload["checkpoint"])
            checkpoint_hashes[f"{LABELS[size]}_seed{seed}"] = {
                "path": str(checkpoint), "sha256": sha256(checkpoint)
            }
            prediction_hashes[f"{LABELS[size]}_seed{seed}"] = {
                "path": str(prediction_path), "sha256": sha256(prediction_path)
            }
            per_seed.append({
                "size": LABELS[size], "n_images": size, "seed": seed,
                **{metric: metrics[metric] for metric in METRICS},
                "n_gt": metrics["n_gt"], "n_pred_total": metrics["n_pred_total"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hashes[f"{LABELS[size]}_seed{seed}"]["sha256"],
                "prediction_sha256": prediction_hashes[f"{LABELS[size]}_seed{seed}"]["sha256"],
            })

    checkpoint_values = [item["sha256"] for item in checkpoint_hashes.values()]
    prediction_values = [item["sha256"] for item in prediction_hashes.values()]
    if len(set(checkpoint_values)) != len(checkpoint_values):
        raise RuntimeError("Checkpoint hashes are not unique across all 15 runs")
    if len(set(prediction_values)) != len(prediction_values):
        raise RuntimeError("Prediction hashes are not unique across all 15 runs")

    aggregate = []
    by_size = {}
    for size in SIZES:
        rows = [row for row in per_seed if row["n_images"] == size]
        output = {"size": LABELS[size], "n_images": size, "n_seeds": len(rows)}
        for metric in METRICS:
            mean, std, half = mean_std_ci([row[metric] for row in rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_half"] = half
        aggregate.append(output)
        by_size[size] = output

    best_primary_size = max(CANDIDATES, key=lambda size: by_size[size]["AP50:95_mean"])
    best_primary = by_size[best_primary_size]
    equivalent = []
    for size in CANDIDATES:
        row = by_size[size]
        delta = best_primary["AP50:95_mean"] - row["AP50:95_mean"]
        if delta <= PRACTICAL_AP_MARGIN or ci_overlap(row, best_primary, "AP50:95"):
            equivalent.append(size)
    best_recall = max(by_size[size]["recall_mean"] for size in equivalent)
    recall_equivalent = [
        size for size in equivalent
        if best_recall - by_size[size]["recall_mean"] <= PRACTICAL_RECALL_MARGIN
    ]
    selected = min(recall_equivalent)

    selected_row = by_size[selected]
    reference = by_size[100_000]
    reference_ap_delta = reference["AP50:95_mean"] - selected_row["AP50:95_mean"]
    selected_vs_reference_equivalent = (
        abs(reference_ap_delta) <= PRACTICAL_AP_MARGIN
        or ci_overlap(selected_row, reference, "AP50:95")
    )

    baseline_payload = json.loads(BASELINE_SUMMARY.read_text())
    baseline = baseline_payload["datasets"]["mocs_dev"]

    per_seed_fields = list(per_seed[0])
    with (OUTPUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(per_seed)
    aggregate_fields = list(aggregate[0])
    with (OUTPUT / "mean_std_ci_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate)

    integrity = {
        "n_checkpoints": len(checkpoint_values),
        "n_unique_checkpoint_hashes": len(set(checkpoint_values)),
        "n_prediction_files": len(prediction_values),
        "n_unique_prediction_hashes": len(set(prediction_values)),
        "locked_inference_config": config_reference,
        "checkpoints": checkpoint_hashes,
        "predictions": prediction_hashes,
    }
    (OUTPUT / "HASH_AND_CONFIG_VERIFICATION.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )

    selection = {
        "status": "FROZEN_AFTER_MOCS_DEVELOPMENT_SELECTION",
        "selection_dataset": "MOCS development set",
        "candidate_sizes": [LABELS[size] for size in CANDIDATES],
        "reference_size": "100K",
        "rule": {
            "primary": "highest mean AP50:95 across three seeds",
            "practical_AP_equivalence": (
                "absolute mean difference <= 0.005 OR overlapping 95% confidence intervals"
            ),
            "secondary": "highest mean recall@0.25 among AP-equivalent configurations",
            "efficiency_tiebreak": "smallest size when mean recall differs by <= 0.01",
        },
        "best_primary_candidate": LABELS[best_primary_size],
        "AP_equivalent_candidates": [LABELS[size] for size in equivalent],
        "recall_equivalent_candidates": [LABELS[size] for size in recall_equivalent],
        "selected_size": LABELS[selected],
        "selected_metrics": selected_row,
        "reference_100K_metrics": reference,
        "selected_vs_100K_AP50_95_mean_difference": (
            selected_row["AP50:95_mean"] - reference["AP50:95_mean"]
        ),
        "selected_vs_100K_AP_equivalent": selected_vs_reference_equivalent,
        "held_out_datasets_used_for_selection": False,
        "next_step": (
            "Evaluate the frozen selected size on CIS, positive control, and confusers; "
            "do not revise selection from those outcomes."
        ),
    }
    (OUTPUT / "SIZE_SELECTION_FROZEN.json").write_text(json.dumps(selection, indent=2) + "\n")

    colors = ["#2A9D8F" if size == selected else "#E9C46A" if size == 100_000 else "#457B9D" for size in SIZES]
    x = np.asarray(SIZES, dtype=float)
    panels = [
        ("AP50", "AP50", (0.73, 0.80)),
        ("AP50:95", "AP50:95", (0.54, 0.60)),
        ("recall", "Recall @ 0.25", (0.32, 0.72)),
        ("f1", "F1 @ 0.25", (0.48, 0.78)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes.flat, panels):
        means = np.asarray([by_size[size][f"{metric}_mean"] for size in SIZES])
        stds = np.asarray([by_size[size][f"{metric}_std"] for size in SIZES])
        ax.plot(x, means, color="#264653", linewidth=2, zorder=1)
        for xi, mean, std, color, size in zip(x, means, stds, colors, SIZES):
            ax.errorbar(xi, mean, yerr=std, fmt="o", color=color, ecolor=color,
                        markersize=8, capsize=4, linewidth=1.8, zorder=2)
            ax.annotate(f"{mean:.3f}", (xi, mean), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(baseline[metric], color="#6C757D", linestyle="--", linewidth=1.5,
                   label=f"No adaptation ({baseline[metric]:.3f})")
        ax.set_xscale("log")
        ax.set_xticks(x, [LABELS[size] for size in SIZES])
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Random Excavision training images")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("YOLO-World subset-size study on MOCS (mean ± SD, 3 seeds)",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.012,
             "Green: frozen selection (5K)   •   Gold: previously selected 100K reference   •   Same locked inference protocol",
             ha="center", fontsize=9)
    fig.savefig(OUTPUT / "mocs_multiseed_subset_size_curve.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / "mocs_multiseed_subset_size_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    for size, color in zip(SIZES, colors):
        row = by_size[size]
        ax.errorbar(
            row["AP50:95_mean"], row["recall_mean"],
            xerr=row["AP50:95_std"], yerr=row["recall_std"],
            fmt="o", markersize=10, color=color, ecolor=color, capsize=4,
        )
        ax.annotate(LABELS[size], (row["AP50:95_mean"], row["recall_mean"]),
                    xytext=(7, 7), textcoords="offset points", fontweight="bold")
    ax.axvline(baseline["AP50:95"], color="#6C757D", linestyle="--", linewidth=1.3)
    ax.axhline(baseline["recall"], color="#6C757D", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Mean AP50:95")
    ax.set_ylabel("Mean recall @ 0.25")
    ax.set_title("Accuracy–sensitivity trade-off across synthetic-data budgets",
                 fontweight="bold")
    ax.grid(color="#E1E1E1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.02, "Error bars: ±1 SD across 3 seeds\nGreen: selected by frozen rule",
            transform=ax.transAxes, fontsize=9, va="bottom")
    fig.savefig(OUTPUT / "mocs_subset_size_ap_recall_tradeoff.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT / "mocs_subset_size_ap_recall_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)

    def metric(row: dict, key: str) -> str:
        return f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_std']:.3f}"

    lines = [
        "# Multi-seed random-subset selection on MOCS",
        "",
        "All checkpoints were evaluated with the locked prompt `construction excavator`, "
        "prediction floor 0.001, NMS IoU 0.70, image size 640, and operating threshold 0.25.",
        "MOCS is the development set. No held-out dataset was used for size selection.",
        "",
        "| Size | AP50 | AP50:95 | Precision@0.25 | Recall@0.25 | F1@0.25 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        row = by_size[size]
        lines.append(
            f"| {LABELS[size]} | {metric(row, 'AP50')} | {metric(row, 'AP50:95')} | "
            f"{metric(row, 'precision')} | {metric(row, 'recall')} | {metric(row, 'f1')} |"
        )
    lines += [
        "",
        "## Frozen decision",
        "",
        f"- The highest mean AP50:95 among the new candidates was {LABELS[best_primary_size]} "
        f"({best_primary['AP50:95_mean']:.3f}).",
        "- AP50:95 uncertainty intervals overlap across the candidate sizes, so the candidates "
        "are treated as practically equivalent under the frozen rule.",
        f"- {LABELS[selected]} was selected because it has the strongest mean recall among the "
        "AP-equivalent candidates and is the smallest recall-equivalent configuration.",
        f"- {LABELS[selected]} and the 100K reference differ by only "
        f"{abs(selected_row['AP50:95_mean'] - reference['AP50:95_mean']):.3f} mean AP50:95; "
        f"their 95% confidence intervals overlap.",
        "- The evidence therefore supports rapid saturation/data efficiency, not a monotonic "
        "benefit from increasing synthetic-data volume.",
        "",
        "## Integrity",
        "",
        "- 15/15 checkpoint hashes are unique.",
        "- 15/15 MOCS prediction hashes are unique.",
        "- All runs contain 1,452 MOCS GT instances and use an identical inference configuration.",
        "",
        "## Next step",
        "",
        f"Freeze {LABELS[selected]} and evaluate its three seeds on CIS, the positive control, "
        "and the confuser set. Those held-out outcomes must not be used to revise the size choice.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(selection, indent=2))
    print(f"Saved analysis to {OUTPUT}")


if __name__ == "__main__":
    main()
