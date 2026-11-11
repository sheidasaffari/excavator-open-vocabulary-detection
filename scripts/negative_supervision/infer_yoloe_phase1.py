#!/usr/bin/env python3
"""Common low-floor inference for all Phase-1 YOLOE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ultralytics import YOLOE


BASE = Path("/home/ssaffari/Data/foundational/yoloe/checkpoints/yoloe-v8s-detect-from-seg.pt")
RUNS = Path("/home/ssaffari/Data/foundational/journal_study/runs_phase1/yoloe_v8s")
MODELS = {
    "base": (BASE, False),
    "n0_alignment": (RUNS / "n0_alignment_seed0/weights/best.pt", False),
    "n1_alignment": (RUNS / "n1_alignment_seed0/weights/best.pt", False),
    "n2_alignment": (RUNS / "n2_alignment_seed0/weights/best.pt", False),
    "n2_terminal": (RUNS / "n2_terminal_seed0/weights/best.pt", True),
    "n2_panhead": (RUNS / "n2_panhead_seed0/weights/best.pt", False),
    "n2_full": (RUNS / "n2_full_seed0/weights/best.pt", False),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    args = p.parse_args()
    dataset = json.loads(args.eval_json.read_text())
    images = [Path(x["file_name"]) for x in dataset["images"]]
    source = {Path(x["file_name"]).stem: x["source_dataset"] for x in dataset["images"]}
    output = {"metadata": {"eval_json": str(args.eval_json.resolve()), "floor": 0.001, "prompt": "excavator"}, "models": {}}
    for model_name in args.models:
        checkpoint, fixed = MODELS[model_name]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLOE(str(checkpoint), task="detect")
        if not fixed:
            model.set_classes(["excavator"])
        raw = {}
        for start in range(0, len(images), args.batch):
            chunk = images[start:start + args.batch]
            results = model.predict(
                [str(x) for x in chunk], conf=0.001, imgsz=640, max_det=300,
                iou=0.70, device=args.device, batch=args.batch, verbose=False,
            )
            for result in results:
                stem = Path(result.path).stem
                keep = result.boxes.cls == 0 if fixed else result.boxes.conf >= 0
                raw[stem] = {
                    "dataset": source[stem],
                    "boxes": result.boxes.xyxy[keep].detach().cpu().tolist(),
                    "scores": result.boxes.conf[keep].detach().cpu().tolist(),
                }
            print(f"{model_name}: {min(start + args.batch, len(images))}/{len(images)}", flush=True)
        output["models"][model_name] = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            "fixed_vocabulary_class0": fixed, "predictions": raw,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
