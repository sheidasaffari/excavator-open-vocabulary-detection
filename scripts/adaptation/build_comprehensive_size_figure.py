#!/usr/bin/env python3
"""Build one comprehensive publication figure for the MOCS size study."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/size_selection_multiseed"
INPUT = RESULTS / "mean_std_ci_results.csv"
PNG = RESULTS / "mocs_multiseed_subset_size_comprehensive.png"
PDF = RESULTS / "mocs_multiseed_subset_size_comprehensive.pdf"

SIZES = [5_000, 10_000, 20_000, 50_000, 100_000]
LABELS = ["5K", "10K", "20K", "50K", "100K"]
SELECTED = 5_000
REFERENCE = 100_000
BASELINE = {
    "AP50": 0.7449448558585554,
    "AP50:95": 0.5543160118371251,
    "precision": 0.9563636363636364,
    "recall": 0.3622589531680441,
    "f1": 0.5254745254745254,
}


def main() -> None:
    with INPUT.open() as handle:
        rows = {int(row["n_images"]): row for row in csv.DictReader(handle)}
    if set(rows) != set(SIZES):
        raise RuntimeError(f"Expected sizes {SIZES}, found {sorted(rows)}")

    colors = {
        size: "#2A9D8F" if size == SELECTED else "#E9C46A" if size == REFERENCE else "#457B9D"
        for size in SIZES
    }
    x = np.asarray(SIZES, dtype=float)
    panels = [
        ("AP50", "AP50", (0.73, 0.80)),
        ("AP50:95", "AP50:95", (0.54, 0.60)),
        ("precision", "Precision @ 0.25", (0.72, 1.00)),
        ("recall", "Recall @ 0.25", (0.32, 0.72)),
        ("f1", "F1 @ 0.25", (0.48, 0.78)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), constrained_layout=True)
    for ax, (metric, title, ylim) in zip(axes.flat[:5], panels):
        means = np.asarray([float(rows[size][f"{metric}_mean"]) for size in SIZES])
        stds = np.asarray([float(rows[size][f"{metric}_std"]) for size in SIZES])
        ax.plot(x, means, color="#264653", linewidth=2, zorder=1)
        for xi, mean, std, size in zip(x, means, stds, SIZES):
            color = colors[size]
            ax.errorbar(
                xi, mean, yerr=std, fmt="o", color=color, ecolor=color,
                markersize=8, capsize=4, linewidth=1.8, zorder=2,
            )
            ax.annotate(
                f"{mean:.3f}", (xi, mean), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8,
            )
        ax.axhline(
            BASELINE[metric], color="#6C757D", linestyle="--", linewidth=1.5,
            label=f"No adaptation ({BASELINE[metric]:.3f})",
        )
        ax.set_xscale("log")
        ax.set_xticks(x, LABELS)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Random Excavision training images")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes.flat[5]
    for size in SIZES:
        row = rows[size]
        ap = float(row["AP50:95_mean"])
        recall = float(row["recall_mean"])
        ap_std = float(row["AP50:95_std"])
        recall_std = float(row["recall_std"])
        ax.errorbar(
            ap, recall, xerr=ap_std, yerr=recall_std, fmt="o",
            color=colors[size], ecolor=colors[size], markersize=9, capsize=4,
        )
        ax.annotate(
            LABELS[SIZES.index(size)], (ap, recall), xytext=(6, 6),
            textcoords="offset points", fontweight="bold", fontsize=9,
        )
    ax.axvline(BASELINE["AP50:95"], color="#6C757D", linestyle="--", linewidth=1.3)
    ax.axhline(BASELINE["recall"], color="#6C757D", linestyle="--", linewidth=1.3)
    ax.set_title("AP–recall trade-off", fontweight="bold")
    ax.set_xlabel("Mean AP50:95")
    ax.set_ylabel("Mean recall @ 0.25")
    ax.grid(color="#E1E1E1", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "YOLO-World adaptation with random Excavision subsets on MOCS",
        fontsize=16, fontweight="bold",
    )
    fig.text(
        0.5, -0.012,
        "Mean ± SD across 3 independent seeds  •  Green: frozen 5K selection  •  Gold: 100K reference  •  Dashed: no adaptation",
        ha="center", fontsize=9.5,
    )
    fig.savefig(PNG, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
