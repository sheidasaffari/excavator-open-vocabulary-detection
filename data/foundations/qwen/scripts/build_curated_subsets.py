#!/usr/bin/env python3
"""Map historical curated image lists back to master-manifest raw records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


COLOR_STEM = re.compile(
    r"^(?P<generator>Ex\d+)_(?P<scene>.+_s\d+)_(?P<stem>Excavator_\d+_cam\d+_frame\d+)$"
)
UNION_PREFIX = re.compile(r"^r[123]__shard-\d+__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-release", required=True, type=Path)
    parser.add_argument("--curated-root", required=True, type=Path)
    parser.add_argument("--colored-root", required=True, type=Path)
    parser.add_argument("--colored-green-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 10000, 20000, 50000, 100000])
    parser.add_argument("--curation-name", default="curated_v4_rank_neutral")
    return parser.parse_args()


def size_label(size: int) -> str:
    if size % 1000:
        raise ValueError(f"size must be an exact multiple of 1000: {size}")
    return f"{size // 1000}k"


def raw_colored_location(flat_stem: str, colored_root: Path,
                         green_root: Path) -> tuple[str, Path, str]:
    match = COLOR_STEM.match(flat_stem)
    if not match:
        raise ValueError(f"Unrecognized flattened colored stem: {flat_stem}")
    generator, scene, raw_stem = (
        match.group("generator"), match.group("scene"), match.group("stem")
    )
    candidates = [
        ("colored", colored_root / generator / scene / f"{raw_stem}.jpg"),
        ("colored_green", green_root / generator / scene / f"{raw_stem}.jpg"),
    ]
    found = [(source, path) for source, path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one raw colored image for {flat_stem}; found {found}")
    source, path = found[0]
    return source, path, f"{source}__{generator}__{scene}__{raw_stem}"


def master_indexes(parquet_path: Path) -> tuple[set[str], dict[str, list[str]]]:
    table = pq.read_table(parquet_path, columns=["sample_id", "source_stem"])
    ids = set(table.column("sample_id").to_pylist())
    by_stem: dict[str, list[str]] = {}
    for sample_id, stem in zip(
        table.column("sample_id").to_pylist(), table.column("source_stem").to_pylist()
    ):
        by_stem.setdefault(stem, []).append(sample_id)
    return ids, by_stem


def map_curated_path(line: str, master_ids: set[str], by_stem: dict[str, list[str]],
                     colored_root: Path, green_root: Path) -> dict:
    curated_path = Path(line)
    if "/color-project/" in line:
        source, raw_image, sample_id = raw_colored_location(
            curated_path.stem, colored_root, green_root
        )
        mapping_method = "flattened_colored_name_to_generator_scene_frame"
    else:
        raw_stem = UNION_PREFIX.sub("", curated_path.stem)
        matches = by_stem.get(raw_stem, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one master row for union stem {raw_stem}; found {matches}"
            )
        sample_id = matches[0]
        source = sample_id.split("__", 1)[0]
        raw_image = None
        mapping_method = "union_name_to_unique_master_source_stem"
    if sample_id not in master_ids:
        raise RuntimeError(f"Mapped sample absent from master manifest: {sample_id}")
    return {
        "sample_id": sample_id,
        "historical_curated_image_path": line,
        "resolved_raw_colored_image_path": str(raw_image) if raw_image else None,
        "mapping_method": mapping_method,
        "mapped_source": source,
    }


def ranked_unique_pool(csv_path: Path, kind: str, master_ids: set[str],
                       by_stem: dict[str, list[str]], colored_root: Path,
                       green_root: Path, limit: int | None = None) -> tuple[list[dict], int]:
    selected: list[dict] = []
    seen: set[str] = set()
    duplicate_aliases = 0
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if kind == "excavision" and str(row.get("bbox_gate_pass", "")) not in {"1", "1.0"}:
                continue
            mapped = map_curated_path(
                row["image_path"], master_ids, by_stem, colored_root, green_root
            )
            if mapped["sample_id"] in seen:
                duplicate_aliases += 1
                continue
            seen.add(mapped["sample_id"])
            mapped["source_ranking"] = kind
            mapped["source_rank"] = int(float(row["rank"]))
            selected.append(mapped)
            if limit is not None and len(selected) >= limit:
                break
    return selected, duplicate_aliases


def historical_manifest_audit(path: Path, master_ids: set[str],
                              by_stem: dict[str, list[str]], colored_root: Path,
                              green_root: Path) -> dict:
    mapped = [
        map_curated_path(line, master_ids, by_stem, colored_root, green_root)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    counts = Counter(item["sample_id"] for item in mapped)
    return {
        "rows": len(mapped),
        "unique_canonical_sample_ids": len(counts),
        "duplicate_alias_rows": sum(count - 1 for count in counts.values()),
        "duplicate_sample_ids": sum(count > 1 for count in counts.values()),
    }


def load_required_rows(master_jsonl: Path, required: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with master_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sample_id"] in required:
                rows[row["sample_id"]] = row
    missing = sorted(required - rows.keys())
    if missing:
        raise RuntimeError(f"Master JSONL is missing {len(missing)} requested IDs: {missing[:10]}")
    return rows


def validate_raw_paths(row: dict, roots: dict[str, Path]) -> list[str]:
    root = roots[row["source_root_alias"]]
    missing = []
    for key in ("image", "label_2d", "label_3d", "occlusion"):
        relative = row["paths"].get(key)
        if not relative or not (root / relative).is_file():
            missing.append(key)
    depth = row["paths"].get("depth")
    if row["modalities"]["dense_depth"] and (not depth or not (root / depth).is_file()):
        missing.append("depth")
    return missing


def main() -> None:
    args = parse_args()
    sizes = sorted(set(args.sizes))
    release_path = args.master_release / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    master_jsonl = args.master_release / release["artifacts"]["master_manifest"]
    master_parquet = args.master_release / release["artifacts"]["parquet"]
    master_ids, by_stem = master_indexes(master_parquet)

    colored_root = args.colored_root.resolve()
    green_root = args.colored_green_root.resolve()
    colored_ranking = args.curated_root / "colored_ranked_rank_bboxonly.csv"
    excavision_ranking = args.curated_root / "excavision_ranked_rank_neutral.csv"
    colored_pool, colored_duplicate_aliases = ranked_unique_pool(
        colored_ranking, "colored", master_ids, by_stem, colored_root, green_root
    )
    max_size = max(sizes)
    max_colored = min(round(max_size * 0.25), len(colored_pool))
    max_excavision = max_size - max_colored
    excavision_pool, excavision_duplicate_aliases = ranked_unique_pool(
        excavision_ranking, "excavision", master_ids, by_stem,
        colored_root, green_root, limit=max_excavision,
    )
    if len(excavision_pool) < max_excavision:
        raise RuntimeError(
            f"Only {len(excavision_pool)} unique ranked Excavision samples; need {max_excavision}"
        )
    ranking_hashes = {
        "excavision": sha256(excavision_ranking),
        "colored": sha256(colored_ranking),
    }

    mapped_by_size: dict[int, list[dict]] = {}
    historical_audits = {}
    for size in sizes:
        source_list = args.curated_root / "manifests" / size_label(size) / "train.txt"
        if not source_list.is_file():
            raise FileNotFoundError(source_list)
        n_colored = min(round(size * 0.25), len(colored_pool))
        n_excavision = size - n_colored
        mapped = excavision_pool[:n_excavision] + colored_pool[:n_colored]
        for subset_order, item in enumerate(mapped, 1):
            item = dict(item)
            item["subset_order"] = subset_order
            mapped[subset_order - 1] = item
        if len({item["sample_id"] for item in mapped}) != size:
            raise RuntimeError(f"Canonical deduplication failed for {size_label(size)}")
        mapped_by_size[size] = mapped
        historical_audits[size] = historical_manifest_audit(
            source_list, master_ids, by_stem, colored_root, green_root
        )

    previous: set[str] = set()
    nesting = {}
    for size in sizes:
        current = {item["sample_id"] for item in mapped_by_size[size]}
        nesting[str(size)] = {
            "contains_previous_subset": previous <= current,
            "previous_subset_size": len(previous),
        }
        if not previous <= current:
            raise RuntimeError(f"Curated {size_label(size)} is not nested over the previous size")
        previous = current

    master_rows = load_required_rows(master_jsonl, previous)
    roots = {alias: Path(path) for alias, path in release["source_roots"].items() if path}
    validation_missing = {}
    for sample_id, row in master_rows.items():
        missing = validate_raw_paths(row, roots)
        if missing:
            validation_missing[sample_id] = missing
    if validation_missing:
        examples = list(validation_missing.items())[:10]
        raise RuntimeError(f"Raw-path validation failed for {len(validation_missing)} rows: {examples}")

    args.output.mkdir(parents=True, exist_ok=True)
    index = {
        "subset_family": args.curation_name,
        "parent_release_id": release["release_id"],
        "parent_manifest_sha256": release["artifacts"]["master_manifest_sha256"],
        "sizes": [],
        "nesting": nesting,
        "ranking_sources": {
            "excavision": str(excavision_ranking.resolve()),
            "colored": str(colored_ranking.resolve()),
            "excavision_duplicate_aliases_skipped_before_max_size": excavision_duplicate_aliases,
            "colored_duplicate_aliases_skipped": colored_duplicate_aliases,
        },
    }
    for size in sizes:
        label = size_label(size)
        destination = args.output / label
        destination.mkdir(parents=True, exist_ok=True)
        source_list = args.curated_root / "manifests" / label / "train.txt"
        source_counts = Counter()
        view_counts = Counter()
        depth_count = 0
        manifest_temp = destination / "manifest.jsonl.tmp"
        ids_temp = destination / "sample_ids.txt.tmp"
        with manifest_temp.open("w", encoding="utf-8") as manifest, ids_temp.open("w", encoding="utf-8") as ids:
            for item in mapped_by_size[size]:
                row = dict(master_rows[item["sample_id"]])
                row["subset_provenance"] = {
                    "subset_family": args.curation_name,
                    "subset_size": size,
                    "subset_order": item["subset_order"],
                    "source_ranking": item["source_ranking"],
                    "source_rank": item["source_rank"],
                    "historical_curated_image_path": item["historical_curated_image_path"],
                    "mapping_method": item["mapping_method"],
                }
                manifest.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                ids.write(row["sample_id"] + "\n")
                source_counts[row["source_collection"]] += 1
                view_counts[row["view_type"]] += 1
                depth_count += int(row["modalities"]["dense_depth"])
        os.replace(manifest_temp, destination / "manifest.jsonl")
        os.replace(ids_temp, destination / "sample_ids.txt")
        selection = {
            "subset_schema": "excavision-curated-subset/1.0.0",
            "subset_id": f"{args.curation_name}-{label}",
            "subset_family": args.curation_name,
            "selected_count": size,
            "nested": True,
            "parent_release_id": release["release_id"],
            "parent_manifest": str(master_jsonl),
            "parent_manifest_sha256": release["artifacts"]["master_manifest_sha256"],
            "historical_curated_manifest": str(source_list.resolve()),
            "historical_curated_manifest_sha256": sha256(source_list),
            "historical_manifest_audit": historical_audits[size],
            "canonical_duplicate_policy": (
                "Deduplicate by master sample_id and backfill from the next-ranked unique sample."
            ),
            "backfilled_unique_samples": historical_audits[size]["duplicate_alias_rows"],
            "ranking_sources": {
                "excavision": str(excavision_ranking.resolve()),
                "excavision_sha256": ranking_hashes["excavision"],
                "colored": str(colored_ranking.resolve()),
                "colored_sha256": ranking_hashes["colored"],
            },
            "mapping_policy": {
                "union": "strip optional round/shard prefix, then require one master source_stem match",
                "colored": "decode flattened generator/scene/frame name and require one raw colored root",
            },
            "source_counts": dict(sorted(source_counts.items())),
            "view_type_counts": dict(sorted(view_counts.items())),
            "dense_depth_count": depth_count,
            "raw_path_validation": "passed",
            "manifest_sha256": sha256(destination / "manifest.jsonl"),
            "sample_ids_sha256": sha256(destination / "sample_ids.txt"),
        }
        dump(destination / "selection.json", selection)
        index["sizes"].append({
            "label": label,
            "count": size,
            "directory": label,
            "source_counts": selection["source_counts"],
            "dense_depth_count": depth_count,
            "manifest_sha256": selection["manifest_sha256"],
        })
        print(f"{label}: {size:,} rows; sources={dict(source_counts)}; depth={depth_count:,}")
    dump(args.output / "index.json", index)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
