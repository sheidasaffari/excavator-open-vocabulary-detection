#!/usr/bin/env python3
"""Create a deterministic, traceable subset of an Excavision JSONL manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--view", nargs="+")
    parser.add_argument("--sample-id", nargs="+")
    parser.add_argument("--require-depth", action="store_true")
    parser.add_argument("--min-instances", type=int)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--status", nargs="+", default=["valid"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_depth(row: dict) -> bool:
    modalities = row.get("modalities", {})
    if "dense_depth" in modalities:
        return bool(modalities["dense_depth"])
    return bool(row.get("depth") or row.get("paths", {}).get("depth"))


def keep(row: dict, args: argparse.Namespace) -> bool:
    source = row.get("source_collection")
    if args.source and source not in args.source:
        return False
    if args.view and row.get("view_type", "unknown") not in args.view:
        return False
    if args.sample_id and row.get("sample_id") not in set(args.sample_id):
        return False
    if args.require_depth and not has_depth(row):
        return False
    count = int(row.get("instance_count", 0))
    if args.min_instances is not None and count < args.min_instances:
        return False
    if args.max_instances is not None and count > args.max_instances:
        return False
    if args.status and row.get("validation_status") not in args.status:
        return False
    return True


def main() -> None:
    args = parse_args()
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row:
                raise ValueError(f"Missing sample_id on line {line_number}")
            if keep(row, args):
                rows.append(row)

    rows.sort(key=lambda row: row["sample_id"])
    if args.limit is not None and len(rows) > args.limit:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.limit)
        rows.sort(key=lambda row: row["sample_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    recipe = {
        "subset_schema": "excavision-subset/1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_manifest": str(args.manifest.resolve()),
        "parent_manifest_sha256": sha256(args.manifest),
        "output": str(args.output.resolve()),
        "selected_count": len(rows),
        "filters": {
            "source": args.source,
            "view": args.view,
            "sample_id": args.sample_id,
            "require_depth": args.require_depth,
            "min_instances": args.min_instances,
            "max_instances": args.max_instances,
            "status": args.status,
            "limit": args.limit,
            "seed": args.seed,
        },
    }
    recipe_path = args.output.parent / "selection.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {len(rows)} rows -> {args.output}")
    print(f"Recipe -> {recipe_path}")


if __name__ == "__main__":
    main()
