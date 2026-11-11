#!/usr/bin/env python3
"""YOLOE-v8-S Phase-1 representation/adaptation-scope trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOEPETrainer, YOLOETrainerFromScratch


CHECKPOINT = Path("/home/ssaffari/Data/foundational/yoloe/checkpoints/yoloe-v8s-detect-from-seg.pt")
DATA_ROOT = Path("/home/ssaffari/Data/foundational/journal_study/data_phase1_v2")
PROJECT = Path("/home/ssaffari/Data/foundational/journal_study/runs_phase1/yoloe_v8s")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_only(model, prefixes: tuple[str, ...]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def set_predicate(model, predicate) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = bool(predicate(name))


class AuditMixin:
    scope = "unknown"

    def apply_scope(self, model):
        raise NotImplementedError

    def build_optimizer(self, model, *args, **kwargs):
        self.apply_scope(model)
        optimizer = super().build_optimizer(model, *args, **kwargs)
        rows, trainable, total = [], 0, 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
                rows.append({"name": name, "parameters": count})
        if not rows:
            raise RuntimeError("No trainable parameters")
        (Path(self.save_dir) / "trainable_parameters.json").write_text(json.dumps({
            "scope": self.scope, "trainable_parameters": trainable,
            "total_parameters": total, "trainable_fraction": trainable / total, "tensors": rows,
        }, indent=2) + "\n")
        return optimizer


class TerminalHead(AuditMixin, YOLOEPETrainer):
    scope = "terminal_head_fixed_vocabulary"

    def apply_scope(self, model):
        set_only(model, ("model.22.cv3.0.2", "model.22.cv3.1.2", "model.22.cv3.2.2"))


class Alignment(AuditMixin, YOLOETrainerFromScratch):
    scope = "open_prompt_alignment"

    def apply_scope(self, model):
        set_only(model, ("model.22.reprta", "model.22.cv3", "model.22.cv4"))


class PanHead(AuditMixin, YOLOETrainerFromScratch):
    scope = "open_prompt_pan_head"

    def apply_scope(self, model):
        def selected(name: str) -> bool:
            if name.startswith("model.22.savpe"):
                return False
            try:
                return int(name.split(".")[1]) >= 10
            except (IndexError, ValueError):
                return False
        set_predicate(model, selected)


class Full(AuditMixin, YOLOETrainerFromScratch):
    scope = "open_prompt_full_detector"

    def apply_scope(self, model):
        set_predicate(model, lambda name: not name.startswith("model.22.savpe"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=["n0", "n1", "n2"], required=True)
    parser.add_argument("--scope", choices=["terminal", "alignment", "panhead", "full"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    variants = {"n0": "n0_positive_only", "n1": "n1_background_negative", "n2": "n2_explicit_d3"}
    variant = variants[args.representation]
    data_yaml = DATA_ROOT / "configs" / f"{variant}.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    name = f"{args.representation}_{args.scope}_seed{args.seed}"
    if args.smoke:
        name = "smoke_" + name
    run_dir = PROJECT / name
    run_dir.mkdir(parents=True, exist_ok=False)
    locked_yaml = run_dir / "locked_data.yaml"
    locked_yaml.write_text(data_yaml.read_text())
    train_source = run_dir / "locked_train_source.yaml"
    train_source.write_text(data_yaml.read_text())
    val_source = run_dir / "locked_val_source.yaml"
    val_source.write_text(data_yaml.read_text())
    open_data = {"train": {"yolo_data": [str(train_source)]}, "val": {"yolo_data": [str(val_source)]}}
    trainers = {"terminal": TerminalHead, "alignment": Alignment, "panhead": PanHead, "full": Full}
    lrs = {"terminal": 1e-3, "alignment": 1e-4, "panhead": 5e-5, "full": 1e-5}
    audit = {
        "model_family": "YOLOE-v8-S detection derivative (not YOLOE-26)",
        "representation": args.representation, "scope": args.scope, "seed": args.seed,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "data_yaml": str(data_yaml), "data_yaml_sha256": sha256(data_yaml),
        "epochs": 1 if args.smoke else args.epochs, "batch": args.batch,
        "selection": "MOCS AP50:95 with ACID-dev specificity guardrail; development only",
        "held_out_not_used": ["CIS", "positive control", "external confuser set", "controlled occlusion"],
    }
    (run_dir / "locked_input.json").write_text(json.dumps(audit, indent=2) + "\n")
    model = YOLOE(str(CHECKPOINT), task="detect")
    training_data = str(locked_yaml) if args.scope == "terminal" else open_data
    model.train(
        trainer=trainers[args.scope], data=training_data, project=str(PROJECT), name=name, exist_ok=True,
        seed=args.seed, deterministic=False, epochs=1 if args.smoke else args.epochs,
        patience=10, batch=args.batch, imgsz=640, device=args.device, workers=args.workers,
        cache=False, optimizer="AdamW", lr0=lrs[args.scope], lrf=0.1, cos_lr=True,
        warmup_epochs=1.0, close_mosaic=0, mosaic=0.0, auto_augment=None, erasing=0.0,
        hsv_h=0.005, hsv_s=0.2, hsv_v=0.15, scale=0.2, translate=0.05,
        fliplr=0.5, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        bgr=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, amp=True,
        plots=False, save=True, save_period=1, verbose=True,
    )


if __name__ == "__main__":
    main()
