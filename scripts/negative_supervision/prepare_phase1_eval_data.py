#!/usr/bin/env python3
"""Build one combined MOCS-positive + ACID-dev-negative COCO inference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MOCS = Path("/home/ssaffari/.codex/.chatgpt-projects/g-p-6944423877d08191ac21a1cee9e7dc1d/grounding_dino_excavision_kickstart/data_5k/mocs_dev.json")
ACID = Path("/home/ssaffari/Data/real/original/acid/acid_negative_exact_dedup_v1/annotations/negative_dev10_background.coco.json")
ACID_IMAGES = Path("/home/ssaffari/Data/real/original/acid/acid_negative_exact_dedup_v1/views/background/dev10/images")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mocs = json.loads(MOCS.read_text())
    acid = json.loads(ACID.read_text())
    images, annotations = [], []
    id_map = {}
    next_image = 1
    for image in mocs["images"]:
        new = dict(image)
        id_map[("mocs", image["id"])] = next_image
        new["id"] = next_image
        new["source_dataset"] = "mocs"
        new["source_image_id"] = str(image["id"])
        images.append(new)
        next_image += 1
    ann_id = 1
    for ann in mocs["annotations"]:
        new = dict(ann)
        new["id"] = ann_id
        new["image_id"] = id_map[("mocs", ann["image_id"])]
        new["category_id"] = 1
        annotations.append(new)
        ann_id += 1
    for image in acid["images"]:
        new = dict(image)
        new["id"] = next_image
        new["file_name"] = str((ACID_IMAGES / image["file_name"]).resolve())
        new["source_dataset"] = "acid_dev_negative"
        new["source_image_id"] = image.get("acid_image_id", Path(image["file_name"]).stem)
        images.append(new)
        next_image += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "excavator", "supercategory": "machine"}],
        "metadata": {
            "mocs_images": len(mocs["images"]), "mocs_boxes": len(mocs["annotations"]),
            "acid_dev_negative_images": len(acid["images"]),
            "forbidden_sources_not_used": ["CIS", "positive_control", "external_confuser", "controlled_occlusion"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
