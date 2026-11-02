#!/usr/bin/env python3
"""Run YOLO-World (zero-shot) on a folder of images with one text prompt
(a single class name, no trailing period needed for YOLO-World/CLIP-style
usage). Saves raw predictions (box, score) per image as JSON."""
import argparse
import json
import os

from ultralytics import YOLOWorld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--prompt", required=True, help='e.g. "excavator" (no trailing period)')
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.001, help="low threshold; filtering done at eval time")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLOWorld(args.model)
    model.set_classes([args.prompt])

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.images_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))

    results = {}
    for i, stem in enumerate(stems):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(args.images_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        preds = model.predict(img_path, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy().tolist()
        scores = preds.boxes.conf.cpu().numpy().tolist()
        results[stem] = {"boxes": boxes, "scores": scores}

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(stems)}] done")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} images' predictions to {args.out_json}")


if __name__ == "__main__":
    main()
