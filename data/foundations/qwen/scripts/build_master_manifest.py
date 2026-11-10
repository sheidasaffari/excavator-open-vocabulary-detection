#!/usr/bin/env python3
"""Build the lightweight, traceable Excavision master manifest.

The builder reads metadata prefixes from label JSON files, never image pixels,
never copies media, and does not expand 2D/3D landmark arrays. Output is
resumable at the source-collection partition level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


SCHEMA_VERSION = "excavision-master-manifest/1.0.0"
BUILDER_VERSION = "1.0.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
STEM_RE = re.compile(
    r"^(?P<prefix>.+)_Excavator_(?P<asset>\d+)_cam(?P<camera>\d+)_frame(?P<frame>\d+)$"
)
SHORT_STEM_RE = re.compile(
    r"^Excavator_(?P<asset>\d+)_cam(?P<camera>\d+)_frame(?P<frame>\d+)$"
)
DEPTH_IMAGE_RE = re.compile(r"^(?P<stem>.+)_depth\d+\.(?:png|tif|tiff)$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path | None, root: Path) -> str | None:
    return path.relative_to(root).as_posix() if path is not None else None


def read_object_prefix(path: Path, stop_key: str, max_bytes: int = 262_144) -> dict:
    """Parse top-level JSON fields preceding a large array-valued key."""
    marker = f'"{stop_key}"'
    with path.open("r", encoding="utf-8") as stream:
        text = stream.read(max_bytes)
    index = text.find(marker)
    if index < 0:
        raise ValueError(f"key {stop_key!r} not found in first {max_bytes} bytes")
    prefix = text[:index].rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1]
    return json.loads(prefix + "}")


def parse_stem(stem: str) -> dict:
    match = STEM_RE.match(stem) or SHORT_STEM_RE.match(stem)
    if not match:
        return {"camera_id": None, "frame_id": None, "asset_id": None}
    groups = match.groupdict()
    return {
        "camera_id": int(groups["camera"]),
        "frame_id": int(groups["frame"]),
        "asset_id": int(groups["asset"]),
    }


def normalize_view(value: object) -> str:
    view = str(value or "").strip().lower()
    if view in {"ground", "ground-level", "ground_level"}:
        return "ground"
    if view == "aerial":
        return "aerial"
    if view == "mixed":
        return "mixed"
    return "unknown"


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (entry for entry in directory.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def depth_maps(scene: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    directory = scene / "DepthMap"
    images: dict[str, Path] = {}
    metadata: dict[str, Path] = {}
    if not directory.is_dir():
        return images, metadata
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = DEPTH_IMAGE_RE.match(path.name)
        if match:
            images.setdefault(match.group("stem"), path)
        elif path.name.endswith("_depth.json"):
            metadata[path.name[: -len("_depth.json")]] = path
    return images, metadata


def label_only_2d(directory: Path, image_stems: set[str]) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.is_file() and path.name.endswith("_2d.json"):
            stem = path.name[: -len("_2d.json")]
            if stem not in image_stems:
                yield path


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    source_collection: str
    source_root_alias: str
    source_root: Path
    source_stem: str
    generation_group_id: str
    scene_id: str | None
    source_split: str | None
    image: Path
    label_2d: Path
    label_3d: Path
    occlusion: Path
    depth: Path | None
    depth_metadata: Path | None
    generator_id: str | None = None


class Adapter:
    source_collection: str
    root_alias: str

    def __init__(self, root: Path):
        self.root = root.resolve()

    def candidates(self) -> Iterator[Candidate]:
        raise NotImplementedError

    def label_only(self) -> Iterator[dict]:
        return iter(())


class ProcessedAdapter(Adapter):
    source_collection = "excavision_processed"
    root_alias = "excavision_processed"

    def candidates(self) -> Iterator[Candidate]:
        image_root = self.root / "images-jpeg"
        label_root = self.root / "labels"
        for split in ("train", "val", "test"):
            split_dir = image_root / split
            if not split_dir.is_dir():
                continue
            for shard in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                for image in image_files(shard):
                    stem = image.stem
                    timestamp = stem[:19] if re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem) else stem
                    yield Candidate(
                        sample_id=f"processed__{stem}",
                        source_collection=self.source_collection,
                        source_root_alias=self.root_alias,
                        source_root=self.root,
                        source_stem=stem,
                        generation_group_id=f"processed__{timestamp}",
                        scene_id=None,
                        source_split=split,
                        image=image,
                        label_2d=label_root / "raw_2d" / split / shard.name / f"{stem}_2d.json",
                        label_3d=label_root / "raw_3d" / split / shard.name / f"{stem}_3d.json",
                        occlusion=label_root / "raw_occ" / split / shard.name / f"{stem}_occlusion.json",
                        depth=None,
                        depth_metadata=None,
                    )

    def label_only(self) -> Iterator[dict]:
        image_root = self.root / "images-jpeg"
        label_root = self.root / "labels" / "raw_2d"
        for split in ("train", "val", "test"):
            split_dir = label_root / split
            if not split_dir.is_dir():
                continue
            for shard in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
                images = {path.stem for path in image_files(image_root / split / shard.name)}
                for label in label_only_2d(shard, images):
                    yield {
                        "source_collection": self.source_collection,
                        "reason": "label_2d_without_image",
                        "path": relative(label, self.root),
                    }


class SceneAdapter(Adapter):
    round_number: int | None = None

    def scenes(self) -> Iterator[tuple[Path, str | None]]:
        raise NotImplementedError

    def candidate_for(self, scene: Path, generator_id: str | None, image: Path,
                      depths: dict[str, Path], depth_json: dict[str, Path]) -> Candidate:
        stem = image.stem
        scene_name = scene.name
        depth_key = stem
        scene_prefix = scene_name + "_"
        if stem.startswith(scene_prefix):
            # Newgen RGB/label stems include the capture timestamp, while files
            # inside DepthMap use only Excavator_<n>_cam<n>_frame<n>.
            depth_key = stem[len(scene_prefix):]
        prefix = self.source_collection
        sample_id = f"{prefix}__{stem}"
        group_id = f"{prefix}__{scene_name}"
        if generator_id:
            sample_id = f"colored__{generator_id}__{scene_name}__{stem}"
            group_id = f"colored__{generator_id}__{scene_name}"
        return Candidate(
            sample_id=sample_id,
            source_collection=self.source_collection,
            source_root_alias=self.root_alias,
            source_root=self.root,
            source_stem=stem,
            generation_group_id=group_id,
            scene_id=scene_name,
            source_split=None,
            image=image,
            label_2d=scene / f"{stem}_2d.json",
            label_3d=scene / f"{stem}_3d.json",
            occlusion=scene / f"{stem}_occlusion.json",
            depth=depths.get(depth_key) or depths.get(stem),
            depth_metadata=depth_json.get(depth_key) or depth_json.get(stem),
            generator_id=generator_id,
        )

    def candidates(self) -> Iterator[Candidate]:
        for scene, generator_id in self.scenes():
            depths, depth_json = depth_maps(scene)
            for image in image_files(scene):
                yield self.candidate_for(scene, generator_id, image, depths, depth_json)

    def label_only(self) -> Iterator[dict]:
        for scene, generator_id in self.scenes():
            images = {path.stem for path in image_files(scene)}
            for label in label_only_2d(scene, images):
                yield {
                    "source_collection": self.source_collection,
                    "reason": "label_2d_without_image",
                    "path": relative(label, self.root),
                    "generator_id": generator_id,
                    "scene_id": scene.name,
                }


class NewgenAdapter(SceneAdapter):
    root_alias = "newgen_targeted"

    def __init__(self, root: Path, round_number: int):
        super().__init__(root)
        self.round_number = round_number
        self.source_collection = f"newgen_r{round_number}"

    def scenes(self) -> Iterator[tuple[Path, str | None]]:
        round_dir = self.root / f"round{self.round_number}"
        if not round_dir.is_dir():
            return
        for scene in sorted((p for p in round_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            yield scene, None


class ColoredAdapter(SceneAdapter):
    source_collection = "colored"
    root_alias = "colored_output"

    def scenes(self) -> Iterator[tuple[Path, str | None]]:
        if not self.root.is_dir():
            return
        generators = sorted(
            (p for p in self.root.iterdir() if p.is_dir() and re.match(r"^Ex\d+$", p.name)),
            key=lambda path: (int(path.name[2:]), path.name),
        )
        for generator in generators:
            for scene in sorted((p for p in generator.iterdir() if p.is_dir()), key=lambda p: p.name):
                yield scene, generator.name


def build_row(candidate: Candidate, release_id: str) -> tuple[dict | None, dict | None]:
    required = {
        "image": candidate.image,
        "label_2d": candidate.label_2d,
        "label_3d": candidate.label_3d,
        "occlusion": candidate.occlusion,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return None, {
            "sample_id": candidate.sample_id,
            "source_collection": candidate.source_collection,
            "reason": "missing_required_files",
            "missing": missing,
            "paths": {name: relative(path, candidate.source_root) for name, path in required.items()},
        }

    flags: list[str] = []
    try:
        header_2d = read_object_prefix(candidate.label_2d, "annotations")
        image_meta = (header_2d.get("images") or [{}])[0]
    except Exception as error:
        image_meta = {}
        flags.append(f"label_2d_header_parse_error:{type(error).__name__}")
    try:
        header_3d = read_object_prefix(candidate.label_3d, "frames")
    except Exception as error:
        header_3d = {}
        flags.append(f"label_3d_header_parse_error:{type(error).__name__}")

    visible_2d = image_meta.get("visible_excavators")
    visible_3d = header_3d.get("num_excavators")
    instance_count = visible_2d if isinstance(visible_2d, int) else visible_3d
    if not isinstance(instance_count, int):
        instance_count = 0
        flags.append("instance_count_unavailable")
    if isinstance(visible_2d, int) and isinstance(visible_3d, int) and visible_2d != visible_3d:
        flags.append("instance_count_2d_3d_mismatch")

    parsed = parse_stem(candidate.source_stem)
    camera = header_3d.get("camera_info") if isinstance(header_3d.get("camera_info"), dict) else {}
    raw_view = header_3d.get("view_type")
    depth_exists = candidate.depth is not None and candidate.depth.is_file()
    depth_meta_exists = candidate.depth_metadata is not None and candidate.depth_metadata.is_file()
    if depth_exists != depth_meta_exists:
        flags.append("depth_image_metadata_pair_incomplete")

    row = {
        "sample_id": candidate.sample_id,
        "schema_version": SCHEMA_VERSION,
        "manifest_release": release_id,
        "source_collection": candidate.source_collection,
        "source_stem": candidate.source_stem,
        "source_root_alias": candidate.source_root_alias,
        "generation_group_id": candidate.generation_group_id,
        "scene_id": candidate.scene_id,
        "camera_id": parsed["camera_id"],
        "frame_id": parsed["frame_id"],
        "paths": {
            "image": relative(candidate.image, candidate.source_root),
            "label_2d": relative(candidate.label_2d, candidate.source_root),
            "label_3d": relative(candidate.label_3d, candidate.source_root),
            "occlusion": relative(candidate.occlusion, candidate.source_root),
            "depth": relative(candidate.depth, candidate.source_root),
            "depth_metadata": relative(candidate.depth_metadata, candidate.source_root),
            "canonical_annotation": None,
        },
        "modalities": {
            "rgb": True,
            "labels_2d": True,
            "labels_3d": True,
            "occlusion": True,
            "camera": bool(camera),
            "dense_depth": depth_exists,
        },
        "instance_count": instance_count,
        "view_type": normalize_view(raw_view),
        "selection_features": {
            "source_split": candidate.source_split,
            "generator_id": candidate.generator_id,
            "excavator_asset_id": parsed["asset_id"],
            "image_width": image_meta.get("width"),
            "image_height": image_meta.get("height"),
            "raw_view_type": raw_view,
            "camera_height": header_3d.get("camera_height"),
            "camera_lens_mm": camera.get("lens"),
            "pc_name": header_3d.get("pc_name"),
            "hdri_name": header_3d.get("hdri_name"),
            "occlusion_population_scope": (
                "all_48_landmarks_including_off_frame"
                if candidate.source_collection == "excavision_processed"
                else "in_frame_landmarks_only"
            ),
            "depth_resolution": "resolved_actual_path" if depth_exists else "unavailable",
        },
        "split_group_id": candidate.generation_group_id,
        "split": "unassigned",
        "validation_status": "warning" if flags else "valid",
        "validation_flags": flags,
    }
    return row, None


class ParquetWriter:
    """Write a compact, flat selection table without retaining all rows."""

    def __init__(self, path: Path, batch_size: int = 10_000):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("pyarrow is required for Parquet output; use --no-parquet to skip") from error
        self.path = path
        self.batch_size = batch_size
        self.buffer: list[dict] = []
        self.pa = pa
        self.schema = pa.schema([
            ("sample_id", pa.string()), ("manifest_release", pa.string()),
            ("source_collection", pa.string()), ("source_root_alias", pa.string()),
            ("source_stem", pa.string()), ("generation_group_id", pa.string()),
            ("scene_id", pa.string()), ("camera_id", pa.int64()),
            ("frame_id", pa.int64()), ("image_path", pa.string()),
            ("label_2d_path", pa.string()), ("label_3d_path", pa.string()),
            ("occlusion_path", pa.string()), ("depth_path", pa.string()),
            ("depth_metadata_path", pa.string()), ("instance_count", pa.int64()),
            ("view_type", pa.string()), ("dense_depth", pa.bool_()),
            ("camera_metadata", pa.bool_()), ("source_split", pa.string()),
            ("generator_id", pa.string()), ("excavator_asset_id", pa.int64()),
            ("image_width", pa.int64()), ("image_height", pa.int64()),
            ("camera_height", pa.float64()), ("camera_lens_mm", pa.float64()),
            ("pc_name", pa.string()), ("hdri_name", pa.string()),
            ("split_group_id", pa.string()), ("split", pa.string()),
            ("validation_status", pa.string()),
        ])
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")

    @staticmethod
    def flatten(row: dict) -> dict:
        paths = row["paths"]
        modalities = row["modalities"]
        features = row["selection_features"]
        return {
            "sample_id": row["sample_id"],
            "manifest_release": row["manifest_release"],
            "source_collection": row["source_collection"],
            "source_root_alias": row["source_root_alias"],
            "source_stem": row["source_stem"],
            "generation_group_id": row["generation_group_id"],
            "scene_id": row["scene_id"],
            "camera_id": row["camera_id"],
            "frame_id": row["frame_id"],
            "image_path": paths["image"],
            "label_2d_path": paths["label_2d"],
            "label_3d_path": paths["label_3d"],
            "occlusion_path": paths["occlusion"],
            "depth_path": paths["depth"],
            "depth_metadata_path": paths["depth_metadata"],
            "instance_count": row["instance_count"],
            "view_type": row["view_type"],
            "dense_depth": modalities["dense_depth"],
            "camera_metadata": modalities["camera"],
            "source_split": features["source_split"],
            "generator_id": features["generator_id"],
            "excavator_asset_id": features["excavator_asset_id"],
            "image_width": features["image_width"],
            "image_height": features["image_height"],
            "camera_height": features["camera_height"],
            "camera_lens_mm": features["camera_lens_mm"],
            "pc_name": features["pc_name"],
            "hdri_name": features["hdri_name"],
            "split_group_id": row["split_group_id"],
            "split": row["split"],
            "validation_status": row["validation_status"],
        }

    def add(self, row: dict) -> None:
        self.buffer.append(self.flatten(row))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = self.pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def summary_template(adapter: Adapter) -> dict:
    return {
        "source_collection": adapter.source_collection,
        "source_root_alias": adapter.root_alias,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "complete_rows": 0,
        "valid_rows": 0,
        "warning_rows": 0,
        "invalid_candidates": 0,
        "label_only_records": 0,
        "view_type_counts": Counter(),
        "instance_count_counts": Counter(),
        "dense_depth_rows": 0,
        "validation_flag_counts": Counter(),
    }


def finalize_summary(summary: dict) -> dict:
    result = dict(summary)
    for key in ("view_type_counts", "instance_count_counts", "validation_flag_counts"):
        result[key] = dict(sorted(result[key].items(), key=lambda item: str(item[0])))
    result["completed_at_utc"] = utc_now()
    return result


def build_partition(adapter: Adapter, release_id: str, partitions: Path,
                    resume: bool, limit: int | None) -> tuple[Path, dict]:
    partition_dir = partitions / f"source_collection={adapter.source_collection}"
    manifest = partition_dir / "manifest.jsonl"
    invalid_path = partition_dir / "invalid_samples.jsonl"
    report = partition_dir / "summary.json"
    if resume and manifest.is_file() and report.is_file():
        return manifest, json.loads(report.read_text(encoding="utf-8"))

    partition_dir.mkdir(parents=True, exist_ok=True)
    temp = manifest.with_suffix(".jsonl.tmp")
    invalid_temp = invalid_path.with_suffix(".jsonl.tmp")
    summary = summary_template(adapter)
    with temp.open("w", encoding="utf-8") as output, invalid_temp.open("w", encoding="utf-8") as invalid:
        for index, candidate in enumerate(adapter.candidates()):
            if limit is not None and index >= limit:
                break
            row, problem = build_row(candidate, release_id)
            if problem:
                invalid.write(json.dumps(problem, sort_keys=True) + "\n")
                summary["invalid_candidates"] += 1
                continue
            output.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            summary["complete_rows"] += 1
            summary[f"{row['validation_status']}_rows"] += 1
            summary["view_type_counts"][row["view_type"]] += 1
            summary["instance_count_counts"][str(row["instance_count"])] += 1
            summary["dense_depth_rows"] += int(row["modalities"]["dense_depth"])
            summary["validation_flag_counts"].update(row["validation_flags"])
            if summary["complete_rows"] % 25_000 == 0:
                print(f"[{adapter.source_collection}] {summary['complete_rows']:,} complete rows", flush=True)

        if limit is None:
            for problem in adapter.label_only():
                invalid.write(json.dumps(problem, sort_keys=True) + "\n")
                summary["label_only_records"] += 1
    os.replace(temp, manifest)
    os.replace(invalid_temp, invalid_path)
    final = finalize_summary(summary)
    json_dump(report, final)
    return manifest, final


def merge_partitions(manifests: list[Path], output: Path, write_parquet: bool) -> dict:
    master = output / "master_manifest.jsonl"
    master_temp = master.with_suffix(".jsonl.tmp")
    parquet_temp = output / "master_samples.parquet.tmp"
    parquet_final = output / "master_samples.parquet"
    if parquet_temp.exists():
        parquet_temp.unlink()
    parquet = ParquetWriter(parquet_temp) if write_parquet else None
    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    count = 0
    with master_temp.open("w", encoding="utf-8") as combined:
        for manifest in manifests:
            with manifest.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = row["sample_id"]
                    if sample_id in sample_ids:
                        duplicate_ids.append(sample_id)
                    else:
                        sample_ids.add(sample_id)
                    combined.write(line)
                    if parquet:
                        parquet.add(row)
                    count += 1
    if parquet:
        parquet.close()
    if duplicate_ids:
        master_temp.unlink(missing_ok=True)
        parquet_temp.unlink(missing_ok=True)
        raise RuntimeError(f"Duplicate sample IDs found; first examples: {duplicate_ids[:10]}")
    os.replace(master_temp, master)
    if parquet:
        os.replace(parquet_temp, parquet_final)
    return {
        "rows": count,
        "unique_sample_ids": len(sample_ids),
        "master_manifest": master.name,
        "master_manifest_sha256": sha256(master),
        "master_manifest_bytes": master.stat().st_size,
        "parquet": parquet_final.name if parquet else None,
        "parquet_sha256": sha256(parquet_final) if parquet else None,
        "parquet_bytes": parquet_final.stat().st_size if parquet else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--newgen-root", type=Path, required=True)
    parser.add_argument("--colored-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--sources", nargs="+",
        choices=["excavision_processed", "newgen_r1", "newgen_r2", "newgen_r3", "colored"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rebuild-sources", nargs="+", default=[],
        choices=["excavision_processed", "newgen_r1", "newgen_r2", "newgen_r3", "colored"],
        help="Rebuild these partitions even when --resume finds completed output.",
    )
    parser.add_argument("--limit-per-source", type=int)
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.sources or [
        "excavision_processed", "newgen_r1", "newgen_r2", "newgen_r3", "colored"
    ])
    adapters: list[Adapter] = [
        ProcessedAdapter(args.processed_root),
        NewgenAdapter(args.newgen_root, 1),
        NewgenAdapter(args.newgen_root, 2),
        NewgenAdapter(args.newgen_root, 3),
        ColoredAdapter(args.colored_root),
    ]
    adapters = [adapter for adapter in adapters if adapter.source_collection in selected]
    rebuild_sources = set(args.rebuild_sources)
    for adapter in adapters:
        if not adapter.root.is_dir():
            raise FileNotFoundError(f"Missing root for {adapter.source_collection}: {adapter.root}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    partitions = output / "partitions"
    partitions.mkdir(exist_ok=True)
    invalid_path = output / "invalid_samples.jsonl"

    release = {
        "release_schema": "excavision-manifest-release/1.0.0",
        "release_id": args.release_id,
        "builder_version": BUILDER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "metadata_level": "lightweight_label_headers_no_image_decode_no_landmark_expansion",
        "source_roots": {
            "excavision_processed": str(args.processed_root.resolve()),
            "newgen_targeted": str(args.newgen_root.resolve()),
            "colored_output": str(args.colored_root.resolve()),
        },
        "selected_sources": [adapter.source_collection for adapter in adapters],
        "limit_per_source": args.limit_per_source,
        "partitions": {},
        "artifacts": {},
    }
    json_dump(output / "release.json", release)

    manifests: list[Path] = []
    for adapter in adapters:
        print(f"Indexing {adapter.source_collection} from {adapter.root}", flush=True)
        manifest, summary = build_partition(
            adapter, args.release_id, partitions,
            args.resume and adapter.source_collection not in rebuild_sources,
            args.limit_per_source,
        )
        manifests.append(manifest)
        release["partitions"][adapter.source_collection] = summary
        json_dump(output / "release.json", release)

    release["artifacts"] = merge_partitions(manifests, output, not args.no_parquet)
    invalid_temp = invalid_path.with_suffix(".jsonl.tmp")
    with invalid_temp.open("w", encoding="utf-8") as combined_invalid:
        for adapter in adapters:
            partition_invalid = partitions / f"source_collection={adapter.source_collection}" / "invalid_samples.jsonl"
            if partition_invalid.is_file():
                with partition_invalid.open("r", encoding="utf-8") as stream:
                    shutil.copyfileobj(stream, combined_invalid)
    os.replace(invalid_temp, invalid_path)
    release["invalid_records"] = sum(1 for line in invalid_path.open(encoding="utf-8") if line.strip())
    release["completed_at_utc"] = utc_now()
    json_dump(output / "release.json", release)
    json_dump(output / "validation_report.json", {
        "release_id": args.release_id,
        "rows": release["artifacts"]["rows"],
        "unique_sample_ids": release["artifacts"]["unique_sample_ids"],
        "duplicate_sample_ids": 0,
        "invalid_records": release["invalid_records"],
        "partitions": release["partitions"],
        "status": "valid" if release["artifacts"]["rows"] else "invalid",
    })
    print(json.dumps({
        "output": str(output),
        "rows": release["artifacts"]["rows"],
        "invalid_records": release["invalid_records"],
        "manifest_sha256": release["artifacts"]["master_manifest_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed source partitions can be reused with --resume.", file=sys.stderr)
        raise
