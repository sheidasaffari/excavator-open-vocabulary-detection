#!/usr/bin/env python3
"""Independent structural validator for Phase-1 data artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    errors, checks = [], {}
    audit = json.loads((root / "AUDIT.json").read_text())
    variants = {
        "n0_positive_only": (1, False),
        "n1_background_negative": (1, False),
        "n2_explicit_d3": (10, True),
    }
    for name, (num_classes, explicit) in variants.items():
        image_dir = root / name / "images/train"
        label_dir = root / name / "labels/train"
        images = sorted(image_dir.iterdir())
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != 10000 or len(labels) != 10000:
            errors.append(f"{name}: expected 10,000 images/labels, got {len(images)}/{len(labels)}")
        broken = [p for p in images if not p.is_file()]
        if broken:
            errors.append(f"{name}: {len(broken)} broken image links")
        invalid_lines = 0
        empty = 0
        class_counts = Counter()
        for label in labels:
            text = label.read_text().strip()
            if not text:
                empty += 1
                continue
            for line in text.splitlines():
                fields = line.split()
                if len(fields) != 5:
                    invalid_lines += 1
                    continue
                cid = int(fields[0])
                coords = [float(v) for v in fields[1:]]
                if cid < 0 or cid >= num_classes or any(v < 0 or v > 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                    invalid_lines += 1
                class_counts[cid] += 1
        if invalid_lines:
            errors.append(f"{name}: {invalid_lines} invalid YOLO label lines")
        manifest = rows(root / "manifests" / f"{name}_exposures.csv")
        if len(manifest) != 10000:
            errors.append(f"{name}: exposure manifest has {len(manifest)} rows")
        stream_counts = Counter(r["stream"] for r in manifest)
        coco = json.loads((root / "coco" / f"{name}_train.json").read_text())
        if len(coco["images"]) != 10000:
            errors.append(f"{name}: COCO image count is {len(coco['images'])}")
        missing_coco = [r["file_name"] for r in coco["images"] if not Path(r["file_name"]).is_file()]
        if missing_coco:
            errors.append(f"{name}: {len(missing_coco)} missing COCO paths")
        if len(coco["categories"]) != num_classes:
            errors.append(f"{name}: expected {num_classes} COCO categories")
        checks[name] = {
            "images": len(images), "labels": len(labels), "empty_labels": empty,
            "stream_counts": dict(stream_counts), "class_box_counts": dict(class_counts),
            "coco_annotations": len(coco["annotations"]), "explicit": explicit,
        }

    n1 = rows(root / "manifests/n1_background_negative_exposures.csv")
    n2 = rows(root / "manifests/n2_explicit_d3_exposures.csv")
    sequence1 = [(r["stream"], r["source_image_id"], r["source_sha256"]) for r in n1]
    sequence2 = [(r["stream"], r["source_image_id"], r["source_sha256"]) for r in n2]
    if sequence1 != sequence2:
        errors.append("N1 and N2 do not use the identical ordered source exposure sequence")

    split_hashes = {}
    for split in ("train70", "dev10", "reserved20"):
        split_rows = rows(root / "manifests" / f"positive_{split}.csv")
        split_hashes[split] = {r["sha256"] for r in split_rows}
    for a, b in (("train70", "dev10"), ("train70", "reserved20"), ("dev10", "reserved20")):
        overlap = split_hashes[a] & split_hashes[b]
        if overlap:
            errors.append(f"Positive split SHA overlap {a}/{b}: {len(overlap)}")
    if {k: len(v) for k, v in split_hashes.items()} != {"train70": 1531, "dev10": 219, "reserved20": 437}:
        errors.append("Unexpected positive split counts")
    if audit["counts"]["negative_train"] != 5405:
        errors.append("Unexpected negative train count in audit")

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "checks": checks}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
