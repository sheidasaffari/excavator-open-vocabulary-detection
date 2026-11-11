#!/usr/bin/env python3
"""Build exact-hash-clean ACID positive splits and matched Phase-1 datasets.

The input exact-hash canonicalization is treated as immutable. This script does
not claim near-duplicate removal. It creates matched N0/N1/N2 training records,
YOLO views/YAMLs, COCO JSON, positive split manifests, and an audit receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


SEED = 20261108
CLASSES = {
    0: "excavator",
    1: "backhoe_loader",
    2: "cement_truck",
    3: "compactor",
    4: "dozer",
    5: "dump_truck",
    6: "grader",
    7: "mobile_crane",
    8: "tower_crane",
    9: "wheel_loader",
}
NAME_TO_ID = {name: cid for cid, name in CLASSES.items()}
CLOSE = (1, 4, 9)
OTHER = (2, 3, 5, 6, 7, 8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_annotations(acid_root: Path) -> dict[str, list[dict]]:
    archive = acid_root / "Object Detection Annotation (VOC format).zip"
    out: dict[str, list[dict]] = {}
    with zipfile.ZipFile(archive) as zf:
        for member in sorted(n for n in zf.namelist() if n.endswith(".xml") and not n.startswith("__MACOSX/")):
            root = ET.fromstring(zf.read(member))
            image_id = Path((root.findtext("filename") or "").strip()).stem
            objects = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                box = obj.find("bndbox")
                objects.append({
                    "class_id": NAME_TO_ID[name],
                    "xmin": float(box.findtext("xmin")),
                    "ymin": float(box.findtext("ymin")),
                    "xmax": float(box.findtext("xmax")),
                    "ymax": float(box.findtext("ymax")),
                })
            out[image_id] = objects
    return out


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def multilabel_split(rows: list[dict], ratios=(0.70, 0.10, 0.20)) -> dict[str, list[dict]]:
    """Deterministic greedy split with exact sizes and class-presence balancing."""
    rng = random.Random(SEED)
    names = ("train70", "dev10", "reserved20")
    raw = [len(rows) * x for x in ratios]
    targets = [int(x) for x in raw]
    for i in sorted(range(3), key=lambda j: raw[j] - targets[j], reverse=True)[: len(rows) - sum(targets)]:
        targets[i] += 1
    totals = Counter(cid for row in rows for cid in row["class_ids_set"])
    shuffled = rows[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda r: sum(1 / totals[c] for c in r["class_ids_set"]), reverse=True)
    splits = [[] for _ in range(3)]
    counts = [Counter() for _ in range(3)]
    for row in shuffled:
        candidates = [i for i in range(3) if len(splits[i]) < targets[i]]
        scored = []
        for i in candidates:
            size_need = (targets[i] - len(splits[i])) / max(1, targets[i])
            class_need = sum(
                (totals[c] * ratios[i] - counts[i][c]) / max(1.0, totals[c] * ratios[i])
                for c in row["class_ids_set"]
            ) / len(row["class_ids_set"])
            scored.append((4 * class_need + size_need + rng.random() * 1e-10, i))
        chosen = max(scored)[1]
        splits[chosen].append(row)
        counts[chosen].update(row["class_ids_set"])
    return {name: split for name, split in zip(names, splits)}


def cycle_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    while len(selected) < n:
        batch = rows[:]
        rng.shuffle(batch)
        selected.extend(batch[: n - len(selected)])
    return selected


def class_balanced_sample(rows: list[dict], class_ids: tuple[int, ...], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools = {cid: [r for r in rows if cid in r["class_ids_set"]] for cid in class_ids}
    if any(not pool for pool in pools.values()):
        raise RuntimeError("Empty class pool in balanced sampler")
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = Counter()
    selected = []
    for i in range(n):
        cid = class_ids[i % len(class_ids)]
        pool = pools[cid]
        selected.append(pool[offsets[cid] % len(pool)])
        offsets[cid] += 1
        if offsets[cid] % len(pool) == 0:
            rng.shuffle(pool)
    return selected


def yolo_text(objects: list[dict], width: int, height: int, target_only: bool) -> str:
    lines = []
    for obj in objects:
        if target_only and obj["class_id"] != 0:
            continue
        xc = (obj["xmin"] + obj["xmax"]) / (2 * width)
        yc = (obj["ymin"] + obj["ymax"]) / (2 * height)
        bw = (obj["xmax"] - obj["xmin"]) / width
        bh = (obj["ymax"] - obj["ymin"]) / height
        lines.append(f"{obj['class_id']} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def coco_payload(records: list[dict], explicit: bool) -> dict:
    images, annotations = [], []
    ann_id = 1
    for image_index, record in enumerate(records, start=1):
        images.append({
            "id": image_index,
            "file_name": str(record["final_image"]),
            "width": record["width"],
            "height": record["height"],
            "source_image_id": record["image_id"],
            "stream": record["stream"],
        })
        objects = record["objects"] if explicit else [o for o in record["objects"] if o["class_id"] == 0]
        if record["stream"].startswith("negative") and not explicit:
            objects = []
        for obj in objects:
            w, h = obj["xmax"] - obj["xmin"], obj["ymax"] - obj["ymin"]
            annotations.append({
                "id": ann_id, "image_id": image_index, "category_id": obj["class_id"] + 1,
                "bbox": [obj["xmin"], obj["ymin"], w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    category_ids = range(10) if explicit else range(1)
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid + 1, "name": CLASSES[cid]} for cid in category_ids],
    }


def materialize_variant(root: Path, final_root: Path, name: str, exposures: list[tuple[str, dict]], explicit: bool) -> list[dict]:
    image_dir = root / name / "images" / "train"
    label_dir = root / name / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    records = []
    for index, (stream, row) in enumerate(exposures):
        suffix = row["image_path"].suffix.lower()
        alias = f"{index:05d}_{stream}_{row['image_id']}{suffix}"
        image_target = image_dir / alias
        os.symlink(row["image_path"], image_target)
        objects = row["objects"]
        target_only = not explicit
        label = "" if stream.startswith("negative") and not explicit else yolo_text(objects, row["width"], row["height"], target_only)
        (label_dir / f"{Path(alias).stem}.txt").write_text(label, encoding="utf-8")
        record = dict(row)
        record.update(
            stream=stream,
            alias=alias,
            materialized_image=image_target,
            final_image=final_root / name / "images" / "train" / alias,
        )
        records.append(record)
    return records


def yaml_text(train_images: Path, val_images: Path, explicit: bool) -> str:
    names = CLASSES if explicit else {0: CLASSES[0]}
    lines = ["path: /", f"train: {train_images}", f"val: {val_images}", "", "names:"]
    lines.extend(f"  {cid}: {name}" for cid, name in names.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acid-root", type=Path, default=Path("/home/ssaffari/Data/real/original/acid"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acid_root = args.acid_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    dedup = acid_root / "acid_negative_exact_dedup_v1"
    canonical = load_csv(dedup / "manifests/canonical_images_all_classes.csv")
    annotations = parse_annotations(acid_root)
    rows = []
    for source in canonical:
        image_id = source["image_id"]
        image_path = acid_root / "ACID Images" / source["filename"]
        class_ids = {int(x) for x in source["class_ids"].split(";")}
        rows.append({
            "image_id": image_id,
            "filename": source["filename"],
            "image_path": image_path,
            "sha256": source["sha256"],
            "width": int(source["width"]),
            "height": int(source["height"]),
            "class_ids_set": class_ids,
            "objects": annotations[image_id],
        })
    positives = [r for r in rows if 0 in r["class_ids_set"]]
    negatives = [r for r in rows if 0 not in r["class_ids_set"]]
    positive_splits = multilabel_split(positives)
    negative_train_ids = {r["image_id"] for r in load_csv(dedup / "manifests/negative_train70.csv")}
    negative_train = [r for r in negatives if r["image_id"] in negative_train_ids]
    if len(positive_splits["train70"]) + len(positive_splits["dev10"]) + len(positive_splits["reserved20"]) != 2187:
        raise RuntimeError("Unexpected canonical positive total")
    if len(negative_train) != 5405:
        raise RuntimeError("Unexpected negative train total")

    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    try:
        manifest_root = staging / "manifests"
        for split_name, split_rows in positive_splits.items():
            serialized = [{
                "image_id": r["image_id"], "filename": r["filename"], "sha256": r["sha256"],
                "width": r["width"], "height": r["height"],
                "class_ids": ";".join(map(str, sorted(r["class_ids_set"]))),
                "source_image_path": str(r["image_path"]), "split": split_name,
            } for r in sorted(split_rows, key=lambda x: x["image_id"])]
            write_csv(manifest_root / f"positive_{split_name}.csv", serialized)

        pos5k = cycle_sample(positive_splits["train70"], 5000, SEED + 1)
        pos10k = cycle_sample(positive_splits["train70"], 10000, SEED + 2)
        close2500 = class_balanced_sample(negative_train, CLOSE, 2500, SEED + 3)
        other2500 = class_balanced_sample(negative_train, OTHER, 2500, SEED + 4)
        neg5k = [("negative_close", r) for r in close2500] + [("negative_other", r) for r in other2500]
        random.Random(SEED + 5).shuffle(neg5k)
        variants = {
            "n0_positive_only": ([("positive", r) for r in pos10k], False),
            "n1_background_negative": ([("positive", r) for r in pos5k] + neg5k, False),
            "n2_explicit_d3": ([("positive", r) for r in pos5k] + neg5k, True),
        }
        mocs_images = Path("/home/ssaffari/Data/real/mocs/selected/images")
        audit_variants = {}
        for name, (exposures, explicit) in variants.items():
            random.Random(SEED + 6).shuffle(exposures)
            materialized = materialize_variant(staging, output, name, exposures, explicit)
            config_dir = staging / "configs"
            config_dir.mkdir(exist_ok=True)
            yml = config_dir / f"{name}.yaml"
            yml.write_text(yaml_text(output / name / "images/train", mocs_images, explicit), encoding="utf-8")
            coco_dir = staging / "coco"
            coco_dir.mkdir(exist_ok=True)
            coco_path = coco_dir / f"{name}_train.json"
            coco_path.write_text(json.dumps(coco_payload(materialized, explicit), separators=(",", ":")) + "\n")
            exposure_rows = [{
                "record_index": i, "stream": r["stream"], "alias": r["alias"],
                "source_image_id": r["image_id"], "source_sha256": r["sha256"],
                "class_ids": ";".join(map(str, sorted(r["class_ids_set"]))),
            } for i, r in enumerate(materialized)]
            exposure_manifest = manifest_root / f"{name}_exposures.csv"
            write_csv(exposure_manifest, exposure_rows)
            audit_variants[name] = {
                "records": len(materialized), "unique_source_images": len({r['image_id'] for r in materialized}),
                "stream_counts": dict(Counter(r["stream"] for r in materialized)),
                "explicit_multiclass": explicit,
                "yaml": str(output / "configs" / yml.name),
                "coco": str(output / "coco" / coco_path.name),
            }

        source_files = {
            "canonical_manifest": dedup / "manifests/canonical_images_all_classes.csv",
            "negative_train_manifest": dedup / "manifests/negative_train70.csv",
            "annotation_zip": acid_root / "Object Detection Annotation (VOC format).zip",
        }
        audit = {
            "seed": SEED,
            "duplicate_policy": "SHA-256 exact-file groups only; no near-duplicate claim.",
            "counts": {
                "canonical_positive": len(positives), "canonical_negative": len(negatives),
                "positive_splits": {k: len(v) for k, v in positive_splits.items()},
                "negative_train": len(negative_train),
            },
            "variants": audit_variants,
            "classes": CLASSES,
            "source_sha256": {name: sha256(path) for name, path in source_files.items()},
            "held_out_not_used": ["CIS", "positive_control", "external_confuser_set", "controlled_occlusion_test"],
        }
        (staging / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
        (staging / "README.md").write_text(
            "# Phase-1 ACID training data\n\n"
            "Matched N0/N1/N2 records for representation and adaptation-locus screening. "
            "All ACID test labels are excluded from paper testing; MOCS is development and CIS remains untouched. "
            "Exact duplicates are controlled by SHA-256 only. See `AUDIT.json`.\n"
        )
        shutil.copy2(Path(__file__).resolve(), staging / "build_phase1_data.py")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(audit, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
