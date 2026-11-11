#!/usr/bin/env python3
"""Paper-oriented Phase-1 sensitivity/specificity analysis."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import average_precision_score, roc_auc_score


ACID_MANIFEST = Path("/home/ssaffari/Data/real/original/acid/acid_negative_exact_dedup_v1/manifests/negative_dev10.csv")
RECALL_TARGETS = (0.50, 0.60, 0.70, 0.80)
FPR_TARGETS = (0.05, 0.10, 0.20)
FIXED_THRESHOLD = 0.25
BOOTSTRAPS = 2000
RNG_SEED = 20261111
RUNS = Path("/home/ssaffari/Data/foundational/journal_study/runs_phase1")


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-12)


def ranked_matches(preds, gt_by_stem, iou_thr=0.5):
    rows = []
    per_image = {}
    for stem, gt in gt_by_stem.items():
        p = preds.get(stem, {"boxes": [], "scores": []})
        order = np.argsort(-np.asarray(p["scores"], dtype=float)) if p["scores"] else []
        used = set()
        image_rows = []
        for j in order:
            box, score = p["boxes"][int(j)], float(p["scores"][int(j)])
            candidates = [(iou(box, g), k) for k, g in enumerate(gt) if k not in used]
            best_iou, best_k = max(candidates, default=(0.0, -1))
            is_tp = best_iou >= iou_thr
            if is_tp:
                used.add(best_k)
            row = (score, int(is_tp), stem)
            rows.append(row)
            image_rows.append(row)
        per_image[stem] = image_rows
    rows.sort(key=lambda x: -x[0])
    return rows, per_image


def counts_at(rows, threshold, n_gt):
    chosen = [r for r in rows if r[0] >= threshold]
    tp = sum(r[1] for r in chosen)
    fp = len(chosen) - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def threshold_for_recall(rows, n_gt, target):
    need = math.ceil(target * n_gt)
    tp = 0
    for score, is_tp, _ in rows:
        tp += is_tp
        if tp >= need:
            return score
    return None


def recall_at_threshold(rows, n_gt, threshold):
    return sum(is_tp for score, is_tp, _ in rows if score >= threshold) / n_gt


def threshold_for_fpr(max_scores, target):
    # Lowest observed decision threshold whose empirical image-FPR is <= target.
    candidates = sorted(set(max_scores.tolist() + [float(np.nextafter(x, np.inf)) for x in max_scores] + [0.0, 1.0]))
    valid = [t for t in candidates if float(np.mean(max_scores >= t)) <= target]
    return min(valid) if valid else 1.0


def bootstrap_ci(values_num, values_den=None):
    rng = np.random.default_rng(RNG_SEED)
    values_num = np.asarray(values_num, dtype=float)
    if values_den is None:
        values_den = np.ones_like(values_num)
    else:
        values_den = np.asarray(values_den, dtype=float)
    n = len(values_num)
    sims = np.empty(BOOTSTRAPS)
    for i in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        sims[i] = values_num[idx].sum() / max(values_den[idx].sum(), 1e-12)
    return np.quantile(sims, [0.025, 0.975]).tolist()


def coco_ap(eval_json, preds, mocs_stems):
    payload = json.loads(Path(eval_json).read_text())
    images = [x for x in payload["images"] if x["source_dataset"] == "mocs"]
    ids = {Path(x["file_name"]).stem: x["id"] for x in images}
    anns = [x for x in payload["annotations"] if x["image_id"] in set(ids.values())]
    gt = {"images": images, "annotations": anns, "categories": payload["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        detections = []
        for stem in mocs_stems:
            for box, score in zip(preds.get(stem, {}).get("boxes", []), preds.get(stem, {}).get("scores", [])):
                x1, y1, x2, y2 = box
                detections.append({"image_id": ids[stem], "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
        dt = coco.loadRes(detections)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = list(ids.values())
        # Keep the canonical COCO AP definition (maxDets=100). Low-floor
        # proposal coverage is reported separately as oracle_recall_iou50.
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"ap50_95": float(ev.stats[0]), "ap50": float(ev.stats[1]), "ap75": float(ev.stats[2])}


def load_acid_classes():
    out = {}
    with ACID_MANIFEST.open() as f:
        for row in csv.DictReader(f):
            out[Path(row["filename"]).stem] = row["class_names"].split(";")
    return out


def load_predictions(paths):
    families = {}
    for family, path in paths.items():
        content = json.loads(Path(path).read_text())
        families[family] = content["models"]
    return families


def evaluate_model(family, model, entry, eval_json, gt_by_stem, mocs_stems, acid_stems, acid_classes):
    preds = entry["predictions"]
    n_gt = sum(len(gt_by_stem[s]) for s in mocs_stems)
    rows, per_image = ranked_matches(preds, gt_by_stem)
    tp, fp, fn, precision, recall, f1 = counts_at(rows, FIXED_THRESHOLD, n_gt)
    aps = coco_ap(eval_json, preds, mocs_stems)
    max_scores = np.asarray([max(preds.get(s, {}).get("scores", []) or [0.0]) for s in acid_stems])
    positive_match_scores = []
    for stem in mocs_stems:
        best = 0.0
        for box, score in zip(preds.get(stem, {}).get("boxes", []), preds.get(stem, {}).get("scores", [])):
            if max((iou(box, g) for g in gt_by_stem[stem]), default=0.0) >= 0.5:
                best = max(best, float(score))
        positive_match_scores.append(best)
    separation_y = np.r_[np.ones(len(positive_match_scores)), np.zeros(len(max_scores))]
    separation_s = np.r_[positive_match_scores, max_scores]
    acid_hit = max_scores >= FIXED_THRESHOLD
    image_tp, image_gt = [], []
    for stem in mocs_stems:
        chosen = [r for r in per_image[stem] if r[0] >= FIXED_THRESHOLD]
        image_tp.append(sum(r[1] for r in chosen))
        image_gt.append(len(gt_by_stem[stem]))
    r_ci = bootstrap_ci(image_tp, image_gt)
    fpr_ci = bootstrap_ci(acid_hit.astype(float))
    result = {
        "family": family, "model": model, **aps,
        "scope_label_valid": not (family == "Grounding-DINO-T" and model in ("n2_visionfusion", "n2_full")),
        "ap_gap": aps["ap50"] - aps["ap50_95"],
        "tp_025": tp, "fp_025": fp, "fn_025": fn,
        "precision_025": precision, "recall_025": recall, "f1_025": f1,
        "recall_025_ci_low": r_ci[0], "recall_025_ci_high": r_ci[1],
        "oracle_recall_iou50": counts_at(rows, 0.001, n_gt)[4],
        "acid_image_fpr_025": float(acid_hit.mean()),
        "acid_image_fpr_025_ci_low": fpr_ci[0], "acid_image_fpr_025_ci_high": fpr_ci[1],
        "acid_fp_per_100_images_025": 100.0 * sum(sum(score >= FIXED_THRESHOLD for score in preds.get(s, {}).get("scores", [])) for s in acid_stems) / len(acid_stems),
        "acid_max_score_mean": float(max_scores.mean()), "acid_max_score_median": float(np.median(max_scores)),
        "acid_max_score_p95": float(np.quantile(max_scores, 0.95)),
        "localized_positive_vs_negative_auroc": float(roc_auc_score(separation_y, separation_s)),
        "localized_positive_vs_negative_auprc": float(average_precision_score(separation_y, separation_s)),
    }
    for target in RECALL_TARGETS:
        threshold = threshold_for_recall(rows, n_gt, target)
        key = f"at_recall_{int(target*100):02d}"
        result[f"threshold_{key}"] = threshold
        result[f"acid_fpr_{key}"] = float(np.mean(max_scores >= threshold)) if threshold is not None else None
    for target in FPR_TARGETS:
        threshold = threshold_for_fpr(max_scores, target)
        key = f"at_fpr_{int(target*100):02d}"
        result[f"threshold_{key}"] = threshold
        result[f"mocs_recall_{key}"] = recall_at_threshold(rows, n_gt, threshold)
    class_rows = []
    all_classes = sorted({c for cs in acid_classes.values() for c in cs})
    for cls in all_classes:
        stems = [s for s in acid_stems if cls in acid_classes.get(s, [])]
        t70 = result.get("threshold_at_recall_70")
        class_rows.append({"family": family, "model": model, "class": cls, "n_images": len(stems),
                           "image_fpr_025": float(np.mean([max(preds.get(s, {}).get("scores", []) or [0.0]) >= FIXED_THRESHOLD for s in stems])),
                           "image_fpr_at_recall70": float(np.mean([max(preds.get(s, {}).get("scores", []) or [0.0]) >= t70 for s in stems])) if t70 is not None else None})
    # Compact operating curve (up to about 2,000 score-change points); exact
    # matched endpoints above are computed before this plotting downsample.
    curve = []
    cumulative_tp = cumulative_fp = 0
    pos = 0
    stride = max(1, len(rows) // 2000)
    next_mark = 0
    while pos < len(rows):
        score = rows[pos][0]
        end = pos
        while end < len(rows) and rows[end][0] == score:
            cumulative_tp += rows[end][1]; cumulative_fp += 1 - rows[end][1]
            end += 1
        if end >= next_mark or end == len(rows):
            curve.append({"family": family, "model": model, "threshold": score,
                          "mocs_recall": cumulative_tp/n_gt,
                          "mocs_precision": cumulative_tp/max(cumulative_tp+cumulative_fp, 1),
                          "acid_image_fpr": float(np.mean(max_scores >= score))})
            next_mark = end + stride
        pos = end
    return result, class_rows, curve


def add_contrasts(df):
    rows = []
    for family, group in df.groupby("family"):
        idx = group.set_index("model")
        default = "alignment" if family == "YOLOE-v8s" else "fusion"
        pairs = [(f"n1_{default}", f"n0_{default}", "N1-N0"),
                 (f"n2_{default}", f"n0_{default}", "N2-N0"),
                 (f"n2_{default}", f"n1_{default}", "N2-N1")]
        for a, b, name in pairs:
            if a in idx.index and b in idx.index:
                rows.append({"family": family, "contrast": name,
                             "delta_ap50_95": idx.loc[a,"ap50_95"]-idx.loc[b,"ap50_95"],
                             "delta_recall_025": idx.loc[a,"recall_025"]-idx.loc[b,"recall_025"],
                             "delta_acid_fpr_025": idx.loc[a,"acid_image_fpr_025"]-idx.loc[b,"acid_image_fpr_025"],
                             "delta_acid_fpr_at_recall70": idx.loc[a,"acid_fpr_at_recall_70"]-idx.loc[b,"acid_fpr_at_recall_70"]})
    return pd.DataFrame(rows)


def freeze_recommendation(df):
    recommendations = []
    representation_models = {
        "YOLOE-v8s": ["n0_alignment", "n1_alignment", "n2_alignment"],
        "Grounding-DINO-T": ["n0_fusion", "n1_fusion", "n2_fusion"],
    }
    representation_order = {"n0":0, "n1":1, "n2":2}
    for family, names in representation_models.items():
        g = df[(df.family == family) & df.model.isin(names)].copy()
        best_ap = g.ap50_95.max()
        eligible = g[(g.ap50_95 >= best_ap - 0.02) & g.acid_fpr_at_recall_70.notna()].copy()
        best_spec = eligible.acid_fpr_at_recall_70.min()
        eligible = eligible[eligible.acid_fpr_at_recall_70 <= best_spec + 0.05]
        winner = min(eligible.model, key=lambda m:representation_order[m.split("_")[0]]) if len(eligible) else None
        recommendations.append({"decision":"negative_representation", "family":family, "best_ap50_95":best_ap,
                                "frozen_configuration":winner,
                                "reason":"within 0.02 AP50:95 of best and within 5 percentage points of best FPR at 0.70 recall; simplest retained representation" if winner else "no eligible representation"})
    scopes = {
        "YOLOE-v8s": ["n2_terminal", "n2_alignment", "n2_panhead", "n2_full"],
        "Grounding-DINO-T": ["n2_head", "n2_fusion", "n2_visionfusion", "n2_full"],
    }
    complexity = {m:i for family in scopes.values() for i,m in enumerate(family)}
    for family, names in scopes.items():
        if family == "Grounding-DINO-T":
            recommendations.append({"decision":"adaptation_scope", "family": family, "best_ap50_95": None, "frozen_configuration": None,
                                    "reason": "not frozen: intended vision+fusion and full configurations inherited the base backbone/language freeze and duplicated fusion"})
            continue
        g = df[(df.family == family) & (df.model.isin(names))].copy()
        best_ap = g.ap50_95.max()
        eligible = g[g.ap50_95 >= best_ap - 0.02].copy()
        eligible = eligible[eligible.acid_fpr_at_recall_70.notna()]
        if len(eligible):
            best_spec = eligible.acid_fpr_at_recall_70.min()
            eligible = eligible[eligible.acid_fpr_at_recall_70 <= best_spec + 0.05]
        if len(eligible):
            winner = min(eligible.model, key=lambda m: complexity[m])
            reason = "eligible under locked AP and matched-recall specificity guardrails; narrowest retained scope"
        else:
            winner, reason = None, "no scope satisfied the locked freeze rule"
        recommendations.append({"decision":"adaptation_scope", "family": family, "best_ap50_95": best_ap, "frozen_configuration": winner, "reason": reason})
    return recommendations


def make_figures(summary, curves, class_df, out):
    colors = {"base":"#222222", "n0":"#4C78A8", "n1":"#F58518", "n2":"#54A24B"}
    for family in summary.family.unique():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        fam_curves = curves[curves.family == family]
        for model, g in fam_curves.groupby("model"):
            if model not in ("base", "n0_alignment", "n1_alignment", "n2_alignment", "n0_fusion", "n1_fusion", "n2_fusion"):
                continue
            key = "base" if model == "base" else model.split("_")[0]
            gg = g.sort_values("mocs_recall")
            axes[0].plot(gg.mocs_recall, gg.acid_image_fpr, label=model, color=colors[key], linewidth=2)
        axes[0].set(xlabel="MOCS excavator recall", ylabel="ACID negative image-FPR", xlim=(0,1), ylim=(0,1), title="Sensitivity–specificity trade-off")
        axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)
        scopes = summary[(summary.family == family) & summary.model.str.startswith("n2_") & summary.scope_label_valid].sort_values("ap50_95")
        x = np.arange(len(scopes))
        axes[1].bar(x-.18, scopes.ap50_95, .36, label="MOCS AP50:95", color="#4C78A8")
        axes[1].bar(x+.18, 1-scopes.acid_fpr_at_recall_70, .36, label="Specificity at 0.70 recall", color="#E45756")
        axes[1].set_xticks(x, [m.replace("n2_","") for m in scopes.model], rotation=20)
        axes[1].set_ylim(0,1); axes[1].set_title("Adaptation-locus screen"); axes[1].grid(axis="y", alpha=.25); axes[1].legend(fontsize=8)
        fig.suptitle(family); fig.tight_layout()
        fig.savefig(out / f"{family.lower().replace('-','_')}_phase1_tradeoffs.png", dpi=220, bbox_inches="tight")
        fig.savefig(out / f"{family.lower().replace('-','_')}_phase1_tradeoffs.pdf", bbox_inches="tight")
        plt.close(fig)
    # Confuser heat map.
    valid_classes = class_df[~((class_df.family == "Grounding-DINO-T") & class_df.model.isin(["n2_visionfusion","n2_full"]))]
    pivot = valid_classes.pivot_table(index=["family","model"], columns="class", values="image_fpr_at_recall70")
    fig, ax = plt.subplots(figsize=(13, max(5, .42*len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot)), [f"{a} | {b}" for a,b in pivot.index])
    ax.set_title("Per-confuser image false-positive rate at matched 0.70 MOCS recall")
    fig.colorbar(im, ax=ax, label="Image-FPR"); fig.tight_layout()
    fig.savefig(out / "per_confuser_fpr_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "per_confuser_fpr_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def training_curves(out):
    rows = []
    yroot = RUNS / "yoloe_v8s"
    for run in sorted(yroot.glob("*seed0")):
        if run.name.startswith("smoke_") or not (run/"results.csv").is_file():
            continue
        frame = pd.read_csv(run/"results.csv"); frame.columns = [x.strip() for x in frame.columns]
        for _, r in frame.iterrows():
            rows.append({"family":"YOLOE-v8s", "run":run.name.replace("_seed0",""), "epoch":int(r["epoch"]),
                         "mocs_ap50_95":float(r["metrics/mAP50-95(B)"]), "mocs_ap50":float(r["metrics/mAP50(B)"])})
    groot = RUNS / "mm_grounding_dino_t"
    pattern = re.compile(r"Epoch\(val\) \[(\d+)\]\[1010/1010\].*?coco/bbox_mAP: ([0-9.]+).*?coco/bbox_mAP_50: ([0-9.]+)")
    for run in sorted(groot.glob("*seed0")):
        if run.name.startswith("smoke_"):
            continue
        logs = list(run.glob("*/*.log"))
        if not logs:
            continue
        for epoch, ap, ap50 in pattern.findall(logs[0].read_text(errors="replace")):
            rows.append({"family":"Grounding-DINO-T", "run":run.name.replace("_seed0",""), "epoch":int(epoch),
                         "mocs_ap50_95":float(ap), "mocs_ap50":float(ap50)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out/"phase1_training_curves.csv", index=False)
    summary = []
    for (family,run), g in frame.groupby(["family","run"]):
        best = g.loc[g.mocs_ap50_95.idxmax()]
        final = g.sort_values("epoch").iloc[-1]
        summary.append({"family":family,"run":run,"best_epoch":int(best.epoch),"best_mocs_ap50_95":best.mocs_ap50_95,
                        "best_mocs_ap50":best.mocs_ap50,"final_mocs_ap50_95":final.mocs_ap50_95,
                        "best_to_final_drop":best.mocs_ap50_95-final.mocs_ap50_95})
    pd.DataFrame(summary).to_csv(out/"phase1_training_summary.csv", index=False)
    fig, axes = plt.subplots(1,2,figsize=(13,4.8),sharey=False)
    for ax, family in zip(axes,["YOLOE-v8s","Grounding-DINO-T"]):
        for run,g in frame[frame.family==family].groupby("run"):
            ax.plot(g.epoch,g.mocs_ap50_95,marker="o",markersize=2,label=run)
        ax.set(title=family,xlabel="Epoch",ylabel="MOCS validation AP50:95"); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2)
    fig.tight_layout(); fig.savefig(out/"phase1_learning_curves.png",dpi=220,bbox_inches="tight"); fig.savefig(out/"phase1_learning_curves.pdf",bbox_inches="tight"); plt.close(fig)
    return frame


def parameter_table(out):
    rows = [
        {"family":"YOLOE-v8s","scope":"alignment","total_parameters":14763350,"updated_parameters":3252742,"updated_fraction":0.22032,"status":"valid"},
        {"family":"YOLOE-v8s","scope":"terminal","total_parameters":11139470,"updated_parameters":3870,"updated_fraction":0.000347,"status":"valid; fixed-vocabulary head"},
        {"family":"YOLOE-v8s","scope":"panhead","total_parameters":14763350,"updated_parameters":7832534,"updated_fraction":0.53055,"status":"valid"},
        {"family":"YOLOE-v8s","scope":"full","total_parameters":14763350,"updated_parameters":12912246,"updated_fraction":0.87462,"status":"valid detector scope"},
        {"family":"Grounding-DINO-T","scope":"head","total_parameters":172914717,"updated_parameters":1127715,"updated_fraction":0.006522,"status":"valid"},
        {"family":"Grounding-DINO-T","scope":"fusion","total_parameters":172914717,"updated_parameters":36502563,"updated_fraction":0.211102,"status":"valid"},
        {"family":"Grounding-DINO-T","scope":"visionfusion (actual)","total_parameters":172914717,"updated_parameters":36502563,"updated_fraction":0.211102,"status":"invalid label: duplicated fusion due inherited config"},
        {"family":"Grounding-DINO-T","scope":"full (actual)","total_parameters":172914717,"updated_parameters":36502563,"updated_fraction":0.211102,"status":"invalid label: duplicated fusion due inherited config"},
        {"family":"Grounding-DINO-T","scope":"visionfusion (intended)","total_parameters":172914717,"updated_parameters":64023069,"updated_fraction":0.370258,"status":"not run"},
        {"family":"Grounding-DINO-T","scope":"full (intended)","total_parameters":172914717,"updated_parameters":172914717,"updated_fraction":1.0,"status":"not run"},
    ]
    pd.DataFrame(rows).to_csv(out/"phase1_parameter_scopes.csv",index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--yoloe", type=Path, required=True)
    p.add_argument("--gdino", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.eval_json.read_text())
    image_by_stem = {Path(x["file_name"]).stem:x for x in data["images"]}
    gt_by_stem = {s:[] for s,x in image_by_stem.items() if x["source_dataset"] == "mocs"}
    id_to_stem = {x["id"]:Path(x["file_name"]).stem for x in data["images"]}
    for ann in data["annotations"]:
        gt_by_stem[id_to_stem[ann["image_id"]]].append(xywh_to_xyxy(ann["bbox"]))
    mocs_stems = list(gt_by_stem)
    acid_stems = [s for s,x in image_by_stem.items() if x["source_dataset"] == "acid_dev_negative"]
    acid_classes = load_acid_classes()
    families = load_predictions({"YOLOE-v8s":args.yoloe, "Grounding-DINO-T":args.gdino})
    summaries, classes, curves = [], [], []
    for family, models in families.items():
        for model, entry in models.items():
            print(f"Analyzing {family} {model}", flush=True)
            a,b,c = evaluate_model(family, model, entry, args.eval_json, gt_by_stem, mocs_stems, acid_stems, acid_classes)
            summaries.append(a); classes.extend(b); curves.extend(c)
    summary = pd.DataFrame(summaries).sort_values(["family","model"])
    class_df, curve_df = pd.DataFrame(classes), pd.DataFrame(curves)
    contrasts = add_contrasts(summary)
    freeze = freeze_recommendation(summary)
    summary.to_csv(args.output/"phase1_model_summary.csv", index=False)
    class_df.to_csv(args.output/"phase1_per_confuser.csv", index=False)
    curve_df.to_csv(args.output/"phase1_operating_curves.csv", index=False)
    contrasts.to_csv(args.output/"phase1_prespecified_contrasts.csv", index=False)
    (args.output/"phase1_freeze_recommendation.json").write_text(json.dumps(freeze, indent=2)+"\n")
    make_figures(summary, curve_df, class_df, args.output)
    training_curves(args.output)
    parameter_table(args.output)
    receipt = {"analysis_protocol":"locked before ACID-development inference",
               "bootstrap_replicates":BOOTSTRAPS,"bootstrap_seed":RNG_SEED,
               "inputs":{str(p.resolve()):sha256(p) for p in (args.eval_json,args.yoloe,args.gdino)},
               "held_out_not_used":["CIS","frozen positive control","frozen external confuser","controlled occlusion"]}
    (args.output/"phase1_analysis_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(summary.to_string(index=False)); print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
