#!/usr/bin/env python3
"""Aggregate and visualize the frozen Random-5K YOLO-World evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
FOUNDATIONAL = Path("/home/ssaffari/Data/foundational")
EVAL = FOUNDATIONAL / "adaptation/evaluation_selected_5k_multiseed"
BASELINE_DIR = (
    FOUNDATIONAL
    / "adaptation/evaluation_selected_multiseed_seededloader/zeroshot_common_config"
)
SIZE_RESULTS = REPO / "results/size_selection_multiseed/per_seed_results.csv"
OUT = REPO / "results/final_selected_5k"

CHECKPOINTS = {
    0: FOUNDATIONAL
    / "adaptation/runs_scale_curated_vs_random/"
    "yoloworld_scale_random_5k_score_head_seed0/weights/best.pt",
    1: FOUNDATIONAL
    / "adaptation/runs_random_size_multiseed/"
    "yoloworld_scale_random_5k_score_head_seed1_seededloader/weights/best.pt",
    2: FOUNDATIONAL
    / "adaptation/runs_random_size_multiseed/"
    "yoloworld_scale_random_5k_score_head_seed2_seededloader/weights/best.pt",
}

POSITIVE_DATASETS = ("mocs_dev", "cis_test", "positive_control")
METRICS = ("AP50", "AP50:95", "precision", "recall", "f1")
DISPLAY = {
    "mocs_dev": "MOCS (development)",
    "cis_test": "CIS (held out)",
    "positive_control": "Positive control (held out)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline() -> dict:
    return json.loads((BASELINE_DIR / "summary.json").read_text())["datasets"]


def load_mocs() -> dict[int, dict]:
    rows = list(csv.DictReader(SIZE_RESULTS.open()))
    return {
        int(row["seed"]): {metric: float(row[metric]) for metric in METRICS}
        for row in rows
        if row["size"] == "5K"
    }


def load_adapted() -> tuple[list[dict], dict]:
    mocs = load_mocs()
    per_seed = []
    raw_summaries = {}
    for seed in range(3):
        summary_path = EVAL / f"selected_random_5k_seed{seed}/summary.json"
        summary = json.loads(summary_path.read_text())
        raw_summaries[seed] = summary
        datasets = {"mocs_dev": mocs[seed], **summary["datasets"]}
        for dataset in POSITIVE_DATASETS:
            per_seed.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    **{metric: float(datasets[dataset][metric]) for metric in METRICS},
                }
            )
        confuser = datasets["confuser"]["ALL"]
        per_seed.append(
            {
                "seed": seed,
                "dataset": "confuser",
                "image_fpr": float(confuser["image_fpr"]),
                "fp_per_100_images": float(confuser["fp_per_100_images"]),
            }
        )
    return per_seed, raw_summaries


def summarize(per_seed: list[dict]) -> list[dict]:
    rows = []
    for dataset in (*POSITIVE_DATASETS, "confuser"):
        subset = [row for row in per_seed if row["dataset"] == dataset]
        keys = METRICS if dataset != "confuser" else ("image_fpr", "fp_per_100_images")
        output = {"dataset": dataset, "n_seeds": len(subset)}
        for key in keys:
            values = [row[key] for row in subset]
            output[f"{key}_mean"] = mean(values)
            output[f"{key}_std"] = stdev(values)
        rows.append(output)
    return rows


def write_csvs(per_seed: list[dict], summary: list[dict], baseline: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed_fields = ["seed", "dataset", *METRICS, "image_fpr", "fp_per_100_images"]
    with (OUT / "per_seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_seed)

    summary_fields = sorted(
        {key for row in summary for key in row},
        key=lambda key: (key not in ("dataset", "n_seeds"), key),
    )
    with (OUT / "mean_std_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    comparison = []
    for row in summary:
        dataset = row["dataset"]
        if dataset == "confuser":
            base = baseline[dataset]["ALL"]
            keys = ("image_fpr", "fp_per_100_images")
        else:
            base = baseline[dataset]
            keys = METRICS
        for key in keys:
            comparison.append(
                {
                    "dataset": dataset,
                    "metric": key,
                    "no_adaptation": base[key],
                    "random_5k_mean": row[f"{key}_mean"],
                    "random_5k_std": row[f"{key}_std"],
                    "delta": row[f"{key}_mean"] - base[key],
                }
            )
    with (OUT / "baseline_vs_random5k.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)


def plot(summary: list[dict], baseline: dict) -> None:
    summary_by_dataset = {row["dataset"]: row for row in summary}
    colors = {"baseline": "#6B7280", "adapted": "#0F9D8A"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    metric_specs = [
        ("AP50", "AP50", (0, 1)),
        ("AP50:95", "AP50:95", (0, 0.8)),
        ("precision", "Precision @ 0.25", (0, 1)),
        ("recall", "Recall @ 0.25", (0, 1)),
        ("f1", "F1 @ 0.25", (0, 1)),
    ]
    x = np.arange(len(POSITIVE_DATASETS))
    width = 0.34
    for ax, (metric, title, ylim) in zip(axes.ravel()[:5], metric_specs):
        base_values = [baseline[dataset][metric] for dataset in POSITIVE_DATASETS]
        adapted_values = [summary_by_dataset[dataset][f"{metric}_mean"] for dataset in POSITIVE_DATASETS]
        adapted_error = [summary_by_dataset[dataset][f"{metric}_std"] for dataset in POSITIVE_DATASETS]
        bars_base = ax.bar(x - width / 2, base_values, width, color=colors["baseline"], label="No adaptation")
        bars_adapted = ax.bar(
            x + width / 2,
            adapted_values,
            width,
            yerr=adapted_error,
            capsize=4,
            color=colors["adapted"],
            label="Random-5K (mean ± SD)",
        )
        ax.set_title(title, weight="bold", fontsize=13)
        ax.set_xticks(x, [DISPLAY[d].replace(" (", "\n(") for d in POSITIVE_DATASETS], fontsize=9)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bars in (bars_base, bars_adapted):
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        if metric == "AP50":
            ax.legend(loc="lower right", fontsize=9)

    ax = axes.ravel()[5]
    conf = summary_by_dataset["confuser"]
    base_fpr = baseline["confuser"]["ALL"]["image_fpr"]
    values = [base_fpr, conf["image_fpr_mean"]]
    errors = [0, conf["image_fpr_std"]]
    bars = ax.bar(
        [0, 1],
        values,
        yerr=errors,
        capsize=5,
        color=[colors["baseline"], colors["adapted"]],
        width=0.58,
    )
    ax.set_title("Confuser image-level false-positive rate @ 0.25", weight="bold", fontsize=13)
    ax.set_xticks([0, 1], ["No adaptation", "Random-5K\n(mean ± SD)"])
    ax.set_ylim(0, max(0.72, conf["image_fpr_mean"] + conf["image_fpr_std"] + 0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.text(
        0.5,
        0.97,
        "Lower is better",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#9B1C1C",
        weight="bold",
    )

    fig.suptitle(
        "Frozen Random-5K YOLO-World adaptation: sensitivity gain and specificity cost",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Prompt: construction excavator  •  Fixed operating threshold: 0.25  •  "
        "Adapted values: mean ± sample SD across 3 independent seeds",
        ha="center",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"final_random5k_vs_no_adaptation.{suffix}", dpi=240 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_integrity(raw_summaries: dict) -> None:
    record = {
        "selection": "Random-5K frozen on MOCS before held-out evaluation",
        "checkpoints": {},
        "prediction_files": {},
    }
    for seed, checkpoint in CHECKPOINTS.items():
        record["checkpoints"][f"seed{seed}"] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        }
        seed_dir = EVAL / f"selected_random_5k_seed{seed}"
        record["prediction_files"][f"seed{seed}"] = {
            dataset: {
                "path": str(seed_dir / f"{dataset}_predictions.json"),
                "sha256": sha256(seed_dir / f"{dataset}_predictions.json"),
            }
            for dataset in ("cis_test", "positive_control", "confuser")
        }
        config = raw_summaries[seed]
        record["prediction_files"][f"seed{seed}"]["locked_config"] = {
            key: config[key]
            for key in (
                "prompt",
                "prediction_floor",
                "operating_point",
                "image_size",
                "nms_iou",
                "max_detections",
                "class_agnostic_nms",
            )
        }
    record["checks"] = {
        "unique_checkpoint_hashes": len({v["sha256"] for v in record["checkpoints"].values()}),
        "unique_prediction_hashes_per_dataset": {
            dataset: len(
                {
                    record["prediction_files"][f"seed{seed}"][dataset]["sha256"]
                    for seed in range(3)
                }
            )
            for dataset in ("cis_test", "positive_control", "confuser")
        },
        "locked_configs_identical": len(
            {
                json.dumps(record["prediction_files"][f"seed{seed}"]["locked_config"], sort_keys=True)
                for seed in range(3)
            }
        )
        == 1,
    }
    (OUT / "HASH_AND_CONFIG_VERIFICATION.json").write_text(json.dumps(record, indent=2) + "\n")


def write_report(summary: list[dict], baseline: dict) -> None:
    lookup = {row["dataset"]: row for row in summary}
    lines = [
        "# Frozen Random-5K YOLO-World result",
        "",
        "Random-5K was selected using MOCS only, then frozen. CIS, the independent positive control, and the confuser set were evaluated once without changing the prompt, inference settings, or operating threshold. Adapted values are mean ± sample standard deviation across seeds 0, 1, and 2.",
        "",
        "![Final Random-5K result](final_random5k_vs_no_adaptation.png)",
        "",
        "| Dataset | Model | AP50 | AP50:95 | P@0.25 | R@0.25 | F1@0.25 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in POSITIVE_DATASETS:
        base = baseline[dataset]
        row = lookup[dataset]
        lines.append(
            f"| {DISPLAY[dataset]} | No adaptation | {base['AP50']:.3f} | {base['AP50:95']:.3f} | "
            f"{base['precision']:.3f} | {base['recall']:.3f} | {base['f1']:.3f} |"
        )
        lines.append(
            f"| {DISPLAY[dataset]} | Random-5K | "
            + " | ".join(f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}" for metric in METRICS)
            + " |"
        )
    conf = lookup["confuser"]
    base_conf = baseline["confuser"]["ALL"]
    lines += [
        "",
        "## Confuser specificity",
        "",
        "| Model | Image FPR @ 0.25 | False predictions / 100 images |",
        "|---|---:|---:|",
        f"| No adaptation | {base_conf['image_fpr']:.3f} | {base_conf['fp_per_100_images']:.1f} |",
        f"| Random-5K | {conf['image_fpr_mean']:.3f} ± {conf['image_fpr_std']:.3f} | {conf['fp_per_100_images_mean']:.1f} ± {conf['fp_per_100_images_std']:.1f} |",
        "",
        "## Interpretation",
        "",
        "Positive-only Excavision adaptation substantially increases recall at the fixed operating point on all three positive-image sources. Ranking-based AP is not uniformly improved: MOCS improves, CIS AP50 decreases while AP50:95 is approximately unchanged, and positive-control AP changes only slightly. The confuser false-positive rate increases markedly. The supported claim is therefore a targeted sensitivity/confidence benefit, not universal detection improvement or a sharper excavator/non-excavator boundary.",
        "",
        "The result also shows rapid data saturation: 5K images are AP-equivalent to larger synthetic subsets on MOCS while providing the best mean recall among the equivalent candidates. More positive data alone does not solve specificity; realistic hard negatives are required for that objective.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    baseline = load_baseline()
    per_seed, raw_summaries = load_adapted()
    summary = summarize(per_seed)
    write_csvs(per_seed, summary, baseline)
    plot(summary, baseline)
    write_integrity(raw_summaries)
    write_report(summary, baseline)
    print(OUT)


if __name__ == "__main__":
    main()
