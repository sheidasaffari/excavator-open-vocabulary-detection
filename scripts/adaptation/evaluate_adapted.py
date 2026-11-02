#!/usr/bin/env python3
"""Evaluate an adapted YOLO-World checkpoint with the locked study settings."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from ultralytics import YOLOWorld

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/evaluation"))
from common_eval import evaluate, load_yolo_gt  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

def image_paths(directory):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

def infer(model, images, args):
    raw = {}
    for start in range(0, len(images), args.batch):
        chunk = images[start:start + args.batch]
        results = model.predict(source=[str(p) for p in chunk], conf=args.prediction_floor,
            imgsz=args.image_size, iou=args.nms_iou, max_det=args.max_detections,
            agnostic_nms=False, device=args.device, batch=args.batch, verbose=False)
        for result in results:
            raw[Path(result.path).stem] = {
                "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                "scores": result.boxes.conf.cpu().numpy().tolist()}
    return raw

def flatten(raw):
    return [(stem, score, box) for stem, item in raw.items()
            for box, score in zip(item["boxes"], item["scores"])]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--prompt", default="construction excavator")
    p.add_argument("--prediction-floor", type=float, default=0.001)
    p.add_argument("--operating-threshold", type=float, default=0.25)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--nms-iou", type=float, default=0.70)
    p.add_argument("--max-detections", type=int, default=300)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    images = image_paths(args.images)
    model = YOLOWorld(str(args.checkpoint))
    model.set_classes([args.prompt])
    raw = infer(model, images, args)
    gt = load_yolo_gt(str(args.labels), str(args.images), [p.stem for p in images])
    metrics = evaluate(flatten(raw), {s: item["boxes"] for s, item in gt.items()},
                       score_thr=args.operating_threshold)
    payload = {"dataset": args.dataset_name, "checkpoint": str(args.checkpoint),
        "prompt": args.prompt, "prediction_floor": args.prediction_floor,
        "operating_threshold": args.operating_threshold, "image_size": args.image_size,
        "nms_iou": args.nms_iou, "max_detections": args.max_detections,
        "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
