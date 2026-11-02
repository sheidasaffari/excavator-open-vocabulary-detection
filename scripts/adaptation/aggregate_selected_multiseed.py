#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--eval-root", type=Path, default=ROOT / "adaptation/evaluation_selected_multiseed")
parser.add_argument("--output-root", type=Path, default=ROOT / "results/selected_multiseed")
args = parser.parse_args()
EVAL = args.eval_root
OUT = args.output_root
OUT.mkdir(parents=True, exist_ok=True)

families = ("curated", "random")
seeds = (0, 1, 2)
positive_datasets = ("mocs_dev", "cis_test", "positive_control")
metrics = ("AP50", "AP50:95", "precision", "recall", "f1")

per_seed = []
for family in families:
    for seed in seeds:
        p = EVAL / f"selected_{family}_100k_seed{seed}" / "summary.json"
        payload = json.loads(p.read_text())
        for dataset in positive_datasets:
            m = payload["datasets"][dataset]
            row = {"family": family, "seed": seed, "dataset": dataset}
            row.update({k: m[k] for k in metrics})
            per_seed.append(row)
        c = payload["datasets"]["confuser"]["ALL"]
        per_seed.append(
            {
                "family": family,
                "seed": seed,
                "dataset": "confuser",
                "image_fpr": c["image_fpr"],
                "fp_per_100_images": c["fp_per_100_images"],
            }
        )

with (OUT / "per_seed_results.csv").open("w", newline="") as f:
    fields = ["family", "seed", "dataset", *metrics, "image_fpr", "fp_per_100_images"]
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(per_seed)

summary_rows = []
for family in families:
    for dataset in (*positive_datasets, "confuser"):
        subset = [r for r in per_seed if r["family"] == family and r["dataset"] == dataset]
        keys = metrics if dataset != "confuser" else ("image_fpr", "fp_per_100_images")
        row = {"family": family, "dataset": dataset, "n_seeds": 3}
        for key in keys:
            values = [float(r[key]) for r in subset]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_std"] = stdev(values)
        summary_rows.append(row)

all_fields = sorted({k for row in summary_rows for k in row}, key=lambda x: (x not in ("family", "dataset", "n_seeds"), x))
with (OUT / "mean_std_results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_fields)
    w.writeheader()
    w.writerows(summary_rows)

md = """# Selected 100K curated and random configurations across three seeds

Configurations were selected and frozen on MOCS before held-out evaluation. Values are mean ± sample standard deviation across seeds 0, 1, and 2. The prompt is `construction excavator`; the prediction floor is 0.001; the operating threshold is 0.25.

## Positive datasets

| Family | Dataset | AP50 | AP50:95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---|---:|---:|---:|---:|---:|
"""
for row in summary_rows:
    if row["dataset"] == "confuser":
        continue
    def fmt(key):
        return f"{row[key + '_mean']:.3f} ± {row[key + '_std']:.3f}"
    md += f"| {row['family'].capitalize()} | {row['dataset']} | {fmt('AP50')} | {fmt('AP50:95')} | {fmt('precision')} | {fmt('recall')} | {fmt('f1')} |\n"

md += """
## Corrected confuser set

| Family | Image FPR | False predictions per 100 images |
|---|---:|---:|
"""
for row in summary_rows:
    if row["dataset"] != "confuser":
        continue
    md += (
        f"| {row['family'].capitalize()} | {row['image_fpr_mean']:.3f} ± {row['image_fpr_std']:.3f} | "
        f"{row['fp_per_100_images_mean']:.1f} ± {row['fp_per_100_images_std']:.1f} |\n"
    )

(OUT / "REPORT.md").write_text(md)
print(OUT / "REPORT.md")
