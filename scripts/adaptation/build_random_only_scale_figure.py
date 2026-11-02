#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "adaptation/evaluation_scale_mocs_locked_prompt"
OUT = ROOT / "results/scale_study_mocs_locked_prompt"
OUT.mkdir(parents=True, exist_ok=True)

sizes = [5_000, 20_000, 100_000, 500_000]
labels = ["5K", "20K", "100K", "500K"]
zero = json.loads((EVAL / "zeroshot_mocs_locked_prompt_explicit/summary.json").read_text())["datasets"]["mocs_dev"]
random_metrics = []
for label in labels:
    payload = json.loads((EVAL / f"scale_random_{label.lower()}_mocs_locked_prompt/summary.json").read_text())
    random_metrics.append(payload["datasets"]["mocs_dev"])

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
panels = [
    ("AP50", zero["AP50"], "AP50", (0.73, 0.78)),
    ("AP50:95", zero["AP50:95"], "AP50:95", (0.545, 0.595)),
]
for ax, (key, baseline, ylabel, ylim) in zip(axes, panels):
    values = [m[key] for m in random_metrics]
    ax.plot(sizes, values, color="#E67E22", marker="s", linewidth=2.4, markersize=7.5, label="Random synthetic subsets")
    ax.axhline(baseline, color="#4D4D4D", linestyle="--", linewidth=1.8, label=f"No adaptation ({baseline:.3f})")
    for x, y in zip(sizes, values):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color="#C85E0A")
    ax.set_xscale("log")
    ax.set_xticks(sizes, labels)
    ax.set_xlabel("Random synthetic training images")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="best")

fig.suptitle("Effect of random Excavision supervision on YOLO-World", fontsize=13, fontweight="bold")
fig.text(0.5, -0.018, 'MOCS development set   |   prompt: "construction excavator"   |   seed: 0', ha="center", fontsize=8.5, color="#444444")
png = OUT / "mocs_random_vs_no_adaptation_ap50_ap5095.png"
pdf = OUT / "mocs_random_vs_no_adaptation_ap50_ap5095.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)
print(png)
print(pdf)
