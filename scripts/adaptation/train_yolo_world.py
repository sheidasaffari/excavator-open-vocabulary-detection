#!/usr/bin/env python3
"""Train a locked YOLO-World adaptation arm on an audited image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import distributed
from ultralytics import YOLOWorld
from ultralytics.data.build import ContiguousDistributedSampler, InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import train as detect_train_module
from ultralytics.models.yolo.world.train import WorldTrainer


ROOT = Path(__file__).resolve().parents[2]

ARMS = {
    "score_head": {
        "freeze": list(range(22)),
        "freeze_bbox": True,
        "lr0": 1e-3,
    },
    "fusion_head": {
        "freeze": [*range(12), 13, 14, 16, 17, 19, 20],
        "freeze_bbox": True,
        "lr0": 5e-4,
    },
    "full": {
        "freeze": [],
        "freeze_bbox": False,
        "lr0": 1e-4,
    },
}


def seed_ultralytics_loader(seed: int) -> None:
    """Patch this process's loader factory so CLI seeds affect batches/workers."""

    def build_dataloader_seeded(
        dataset,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        pin_memory: bool = True,
    ) -> InfiniteDataLoader:
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), workers)
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + rank + 1_000_003 * seed)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0 and pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=drop_last and len(dataset) % batch != 0,
        )

    # DetectionTrainer imported the factory into its own module namespace.
    detect_train_module.build_dataloader = build_dataloader_seeded


class SelectiveWorldTrainer(WorldTrainer):
    """WorldTrainer that can freeze WorldDetect localization exactly."""

    def build_optimizer(self, model, *args, **kwargs):
        if getattr(self, "freeze_bbox_branch", False):
            for name, parameter in model.named_parameters():
                if ".model.22.cv2." in f".{name}" or ".model.22.dfl." in f".{name}":
                    parameter.requires_grad = False
        optimizer = super().build_optimizer(model, *args, **kwargs)
        self._write_trainable_audit(model)
        return optimizer

    def _write_trainable_audit(self, model) -> None:
        rows = []
        trainable = total = 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
                rows.append({"name": name, "parameters": count})
        payload = {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
            "tensors": rows,
        }
        (Path(self.save_dir) / "trainable_parameters.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(model: Path, val_images: Path, train_list: Path, expected_images: int) -> list[str]:
    if not model.is_file() or not val_images.is_dir() or not train_list.is_file():
        raise FileNotFoundError("Model, MOCS validation directory, or train list is missing")
    lines = [line.strip() for line in train_list.read_text().splitlines() if line.strip()]
    if len(lines) != expected_images or len(set(lines)) != expected_images:
        raise RuntimeError(
            f"Expected {expected_images:,} unique paths, found {len(lines):,} "
            f"({len(set(lines)):,} unique)"
        )
    missing = [path for path in lines if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing):,} training images are missing; first: {missing[0]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_ultralytics_loader(args.seed)
    train_lines = validate_inputs(args.model, args.val_images, args.train_list, args.expected_images)

    arm = ARMS[args.arm]
    name = args.name or f"yoloworld_a10_20k_{args.arm}_seed0"
    if args.smoke:
        name = f"smoke_{name}"

    audit_dir = args.project / name
    audit_dir.mkdir(parents=True, exist_ok=False)
    # Ultralytics' loader uses a hard-coded generator seed in this installed
    # version, so changing only train(seed=...) produces identical trajectories.
    # Preserve dataset membership but deterministically permute its order to
    # guarantee genuinely distinct stochastic runs for nonzero seeds.
    effective_train_lines = list(train_lines)
    if args.seed != 0:
        random.Random(args.seed).shuffle(effective_train_lines)
    effective_train_list = audit_dir / "locked_train_manifest.txt"
    effective_train_list.write_text("\n".join(effective_train_lines) + "\n")

    data_yaml = audit_dir / "locked_data.yaml"
    data_yaml.write_text(
        f"path: {ROOT}\n"
        f"train: {effective_train_list}\n"
        f"val: {args.val_images}\n\n"
        "names:\n  0: excavator\n"
    )
    (audit_dir / "locked_input.json").write_text(
        json.dumps(
            {
                "train_list": str(args.train_list),
                "train_list_sha256": sha256(args.train_list),
                "effective_train_list": str(effective_train_list),
                "effective_train_list_sha256": sha256(effective_train_list),
                "order_seed": args.seed,
                "n_images": len(train_lines),
                "seed": args.seed,
                "model": str(args.model),
                "model_sha256": sha256(args.model),
                "arm": args.arm,
                "freeze": arm["freeze"],
                "freeze_bbox_branch": arm["freeze_bbox"],
            },
            indent=2,
        )
        + "\n"
    )

    class ArmTrainer(SelectiveWorldTrainer):
        freeze_bbox_branch = arm["freeze_bbox"]

    model = YOLOWorld(str(args.model))
    model.train(
        trainer=ArmTrainer,
        data=str(data_yaml),
        project=str(args.project),
        name=name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        epochs=1 if args.smoke else args.epochs,
        patience=10 if args.smoke else 12,
        batch=args.batch,
        imgsz=640,
        device=args.device,
        workers=args.workers,
        cache=False,
        optimizer="AdamW",
        lr0=arm["lr0"],
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=1.0,
        freeze=arm["freeze"],
        close_mosaic=0,
        mosaic=0.0,
        auto_augment=None,
        erasing=0.0,
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.15,
        scale=0.2,
        translate=0.05,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        bgr=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        amp=True,
        plots=False,
        save=True,
        save_period=5,
        verbose=True,
    )


if __name__ == "__main__":
    main()
