#!/usr/bin/env python3
"""Run Grounding DINO (zero-shot) on a folder of images with one text prompt.
Saves raw predictions (box, score) per image as JSON. Memory-conscious:
processes one image at a time, no batching, periodic cache clearing."""
import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--prompt", required=True, help='e.g. "excavator."')
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--box_thr", type=float, default=0.1)
    ap.add_argument("--text_thr", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    model.eval().to(args.device)

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
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=args.box_thr, text_threshold=args.text_thr,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = post["boxes"].cpu().numpy().tolist()
        scores = post["scores"].cpu().numpy().tolist()
        labels = post.get("labels", post.get("text_labels", [""] * len(boxes)))
        results[stem] = {"boxes": boxes, "scores": scores, "labels": list(labels)}

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(stems)}] done")
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} images' predictions to {args.out_json}")


if __name__ == "__main__":
    main()
