#!/usr/bin/env python3
"""Convert MMDet prediction PKLs to the common compact JSON schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mmengine.fileio import load


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--eval-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dataset = json.loads(args.eval_json.read_text())
    by_path = {str(Path(x["file_name"]).resolve()): x["source_dataset"] for x in dataset["images"]}
    output = {"metadata": {"eval_json": str(args.eval_json.resolve()), "floor": 0.0, "prompt": "excavator"}, "models": {}}
    for pkl in sorted(args.input_root.glob("*.pkl")):
        predictions = {}
        for item in load(pkl):
            image_path = str(Path(item["img_path"]).resolve())
            stem = Path(image_path).stem
            instances = item["pred_instances"]
            labels = instances.get("labels")
            keep = labels == 0 if labels is not None else slice(None)
            predictions[stem] = {
                "dataset": by_path[image_path],
                "boxes": instances["bboxes"][keep].cpu().tolist(),
                "scores": instances["scores"][keep].cpu().tolist(),
            }
        output["models"][pkl.stem] = {"prediction_pkl": str(pkl), "prediction_pkl_sha256": sha256(pkl), "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
