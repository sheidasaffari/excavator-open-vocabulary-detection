#!/usr/bin/env python3
"""Generate and launch an audited MM-Grounding-DINO-T Phase-1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


DATA_ROOT = Path("/home/ssaffari/Data/foundational/journal_study/data_phase1_v2")
PROJECT = Path("/home/ssaffari/Data/foundational/journal_study/runs_phase1/mm_grounding_dino_t")
PRETRAINED = Path("/home/ssaffari/.cache/torch/hub/checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth")
MOCS_JSON = Path("/home/ssaffari/.codex/.chatgpt-projects/g-p-6944423877d08191ac21a1cee9e7dc1d/grounding_dino_excavision_kickstart/data_5k/mocs_dev.json")
BASE = "/home/ssaffari/miniconda3/envs/mmdet/lib/python3.10/site-packages/mmdet/.mim/configs/mm_grounding_dino/grounding_dino_swin-t_finetune_8xb4_20e_cat.py"
TRAIN_TOOL = "/home/ssaffari/miniconda3/envs/mmdet/lib/python3.10/site-packages/mmdet/.mim/tools/train.py"
PYTHON = "/home/ssaffari/miniconda3/envs/mmdet/bin/python"
CLASSES = ("excavator", "backhoe_loader", "cement_truck", "compactor", "dozer", "dump_truck", "grader", "mobile_crane", "tower_crane", "wheel_loader")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_text(train_json: Path, explicit: bool, scope: str, seed: int, epochs: int) -> str:
    train_classes = CLASSES if explicit else CLASSES[:1]
    num_classes = len(train_classes)
    freeze = {
        "head": ["backbone", "language_model", "neck", "encoder", "decoder", "query_embedding", "level_embed", "memory_trans_fc", "memory_trans_norm", "positional_encoding"],
        "fusion": ["backbone", "language_model"],
        "visionfusion": ["language_model"],
        "full": [],
    }[scope]
    custom = {"absolute_pos_embed": "dict(decay_mult=0.)"}
    custom.update({name: "dict(lr_mult=0.0)" for name in freeze})
    custom_text = ",\n            ".join(repr(k) + ": " + v for k, v in custom.items())
    return f'''_base_ = {BASE!r}

class_name = {train_classes!r}
num_classes = {num_classes}
metainfo = dict(classes=class_name)
val_metainfo = dict(classes=('excavator',))
load_from = {str(PRETRAINED)!r}

model = dict(
    backbone=dict(with_cp=False),
    encoder=dict(num_cp=0),
    bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    dataset=dict(data_root='', ann_file={str(train_json)!r}, data_prefix=dict(img=''), metainfo=metainfo))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    dataset=dict(data_root='', ann_file={str(MOCS_JSON)!r}, data_prefix=dict(img=''), metainfo=val_metainfo))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file={str(MOCS_JSON)!r})
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    # Replace, rather than recursively merge with, the base config's frozen
    # backbone/language keys. Without _delete_, visionfusion and full silently
    # collapse to the official fusion scope.
    paramwise_cfg=dict(_delete_=True, custom_keys={{
            {custom_text}
    }}))
train_cfg = dict(max_epochs={epochs}, val_interval=1)
randomness = dict(seed={seed}, deterministic=False)
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["n0", "n1", "n2"], required=True)
    parser.add_argument("--scope", choices=["head", "fusion", "visionfusion", "full"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    variants = {"n0": "n0_positive_only", "n1": "n1_background_negative", "n2": "n2_explicit_d3"}
    variant = variants[args.representation]
    train_json = DATA_ROOT / "coco" / f"{variant}_train.json"
    if not train_json.is_file():
        raise FileNotFoundError(train_json)
    name = f"{args.representation}_{args.scope}_seed{args.seed}"
    if args.smoke:
        name = "smoke_" + name
    work_dir = PROJECT / name
    work_dir.mkdir(parents=True, exist_ok=False)
    epochs = 1 if args.smoke else args.epochs
    config = work_dir / "locked_config.py"
    config.write_text(config_text(train_json, args.representation == "n2", args.scope, args.seed, epochs))
    receipt = {
        "model_family": "MM-Grounding-DINO-T", "representation": args.representation,
        "scope": args.scope, "seed": args.seed, "epochs": epochs,
        "pretrained": str(PRETRAINED), "pretrained_sha256": sha256(PRETRAINED),
        "train_json": str(train_json), "train_json_sha256": sha256(train_json),
        "mocs_json": str(MOCS_JSON), "mocs_json_sha256": sha256(MOCS_JSON),
        "selection": "MOCS AP50:95 with ACID-dev specificity guardrail; development only",
        "held_out_not_used": ["CIS", "positive control", "external confuser set", "controlled occlusion"],
    }
    (work_dir / "locked_input.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run([PYTHON, TRAIN_TOOL, str(config), "--work-dir", str(work_dir)], env=env, check=True)


if __name__ == "__main__":
    main()
