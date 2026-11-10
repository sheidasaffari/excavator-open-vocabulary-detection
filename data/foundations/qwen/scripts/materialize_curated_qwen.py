#!/usr/bin/env python3
"""Materialize canonical annotations and five-level Qwen conversations.

The largest requested nested subset is canonicalized once into JSONL shards.
Each selected subset receives copy-free Qwen conversation views referencing the
original media and the shared canonical store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable


LEVELS = {
    1: "detection_count_grounding",
    2: "visibility_occlusion",
    3: "articulated_pose",
    4: "camera_depth_geometry",
    5: "answerability_abstention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-release", type=Path, required=True)
    parser.add_argument("--subset-family", type=Path, required=True)
    parser.add_argument("--keypoint-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=["5k", "10k", "20k", "50k", "100k"])
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Smoke-test limit per requested subset.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def chunks(values: list[dict], size: int) -> Iterable[tuple[int, list[dict]]]:
    for start in range(0, len(values), size):
        yield start // size, values[start:start + size]


def vector_elevation(a: list[float], b: list[float]) -> float:
    delta = [b[i] - a[i] for i in range(3)]
    return round(math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1]))), 3)


def interior_angle(a: list[float], b: list[float], c: list[float]) -> float:
    u = [a[i] - b[i] for i in range(3)]
    v = [c[i] - b[i] for i in range(3)]
    denominator = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(x * x for x in v))
    if denominator == 0:
        raise ValueError("zero-length articulation vector")
    cosine = max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(3)) / denominator))
    return round(math.degrees(math.acos(cosine)), 3)


def normalized_bbox(box: list[float], width: int, height: int) -> list[float]:
    return [round(box[0] / width, 6), round(box[1] / height, 6),
            round(box[2] / width, 6), round(box[3] / height, 6)]


def canonical_component_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def visibility_state(x: float, y: float, flag: int) -> tuple[bool, str]:
    if flag == 2:
        return True, "fully_visible"
    if flag == 1:
        return True, "self_occluded"
    if x == 0 and y == 0:
        return False, "out_of_frame_ray_state_unavailable"
    return True, "external_or_unresolved_occlusion"


def absolute_paths(row: dict, roots: dict[str, Path]) -> dict[str, Path | None]:
    root = roots[row["source_root_alias"]]
    return {
        key: (root / relative) if relative else None
        for key, relative in row["paths"].items()
    }


def make_canonical(row: dict, roots: dict[str, Path], registry: list[dict],
                   parent_sha: str) -> dict:
    paths = absolute_paths(row, roots)
    raw_2d = json_file(paths["label_2d"])
    raw_3d = json_file(paths["label_3d"])
    raw_occ = json_file(paths["occlusion"])
    image_record = raw_2d["images"][0]
    width, height = int(image_record["width"]), int(image_record["height"])
    annotations = {int(item["instance_id"]): item for item in raw_2d["annotations"]}
    frames = {int(item["frame_id"]): item for item in raw_3d["frames"]}
    occlusions = {int(item["excavator_id"]): item for item in raw_occ["excavators"]}
    instance_ids = sorted(annotations)
    errors: list[str] = []
    if set(instance_ids) != set(frames) or set(instance_ids) != set(occlusions):
        errors.append("instance_id_set_mismatch_across_2d_3d_occlusion")
    if len(registry) != 48:
        errors.append("keypoint_registry_not_48_landmarks")

    canonical_instances = []
    for instance_id in instance_ids:
        annotation = annotations[instance_id]
        frame = frames.get(instance_id)
        occ = occlusions.get(instance_id)
        if frame is None or occ is None:
            continue
        triplets = [annotation["keypoints"][i:i + 3] for i in range(0, len(annotation["keypoints"]), 3)]
        if len(triplets) != len(registry):
            errors.append(f"instance_{instance_id}_keypoint_count_{len(triplets)}")
            continue
        world = frame["bone_location_3d_world"]
        camera = frame["bone_location_3d_camera"]
        root_xyz = world[registry[0]["raw_name"]]
        landmarks = []
        visibility_counts: Counter[str] = Counter()
        depths = []
        for spec, (x, y, flag) in zip(registry, triplets):
            raw_name = spec["raw_name"]
            inside, state = visibility_state(x, y, int(flag))
            visibility_counts[state] += 1
            camera_xyz = camera[raw_name]
            optical_depth = round(-float(camera_xyz[2]), 6)
            depths.append(optical_depth)
            world_xyz = world[raw_name]
            landmarks.append({
                **spec,
                "image_xy": [x, y] if inside else None,
                "inside_image": inside,
                "raw_visibility_flag": int(flag),
                "visibility_state": state,
                "world_xyz": world_xyz,
                "camera_xyz": camera_xyz,
                "optical_depth": optical_depth,
                "root_relative_xyz": [round(world_xyz[i] - root_xyz[i], 6) for i in range(3)],
            })
        point = lambda name: world[name]
        articulation = {
            "boom_elevation_deg": vector_elevation(point("Arm2-To-Body"), point("Arm2")),
            "stick_elevation_deg": vector_elevation(point("Arm2"), point("Bone.002-A")),
            "bucket_orientation_deg": vector_elevation(point("Bone.002-A"), point("Bucket-003")),
            "boom_stick_interior_angle_deg": interior_angle(
                point("Arm2-To-Body"), point("Arm2"), point("Bone.002-A")
            ),
            "stick_bucket_profile_angle_deg": interior_angle(
                point("Arm2"), point("Bone.002-A"), point("Bucket-003")
            ),
        }
        components = {
            canonical_component_name(name): values
            for name, values in occ["component_occlusion"].items()
        }
        canonical_instances.append({
            "instance_id": instance_id,
            "category": "excavator",
            "bbox_xywh_pixels": annotation["bbox"],
            "bbox_xywh_normalized": normalized_bbox(annotation["bbox"], width, height),
            "bbox_3d_world_aabb": frame.get("bbox_3d", occ.get("bounding_box_3d")),
            "landmarks": landmarks,
            "visibility_summary_canonical": {
                state: visibility_counts[state] for state in (
                    "fully_visible", "self_occluded", "external_or_unresolved_occlusion",
                    "out_of_frame_ray_state_unavailable"
                )
            },
            "visibility_summary_raw": {
                "visibility_ratio_image": annotation.get("visibility_ratio_image"),
                "visibility_ratio_ray_casting": annotation.get("visibility_ratio_ray_casting"),
                "passed_threshold": annotation.get("passed_threshold"),
            },
            "raw_nearest_surface_component_visibility": components,
            "raw_component_visibility_population": {
                "scope": row["selection_features"]["occlusion_population_scope"],
                "landmark_count": sum(int(values["total"]) for values in components.values()),
            },
            "derived_articulation": articulation,
            "derived_geometry": {
                "landmark_optical_depth_min": round(min(depths), 6),
                "landmark_optical_depth_max": round(max(depths), 6),
                "landmark_optical_depth_mean": round(sum(depths) / len(depths), 6),
            },
            "answerability": {
                "directly_visible_landmarks": visibility_counts["fully_visible"],
                "self_occluded_landmarks": visibility_counts["self_occluded"],
                "external_or_unresolved_landmarks": visibility_counts["external_or_unresolved_occlusion"],
                "out_of_frame_landmarks": visibility_counts["out_of_frame_ray_state_unavailable"],
                "exact_metric_geometry_from_rgb_only": False,
            },
        })

    camera_info = raw_3d["camera_info"]
    lens, sensor_width = float(camera_info["lens"]), float(camera_info["sensor_width"])
    focal = round(lens / sensor_width * width, 6)
    depth_metadata = json_file(paths["depth_metadata"]) if paths.get("depth_metadata") else None
    return {
        "schema": {"name": "excavision_canonical", "version": "1.0.0"},
        "sample_id": row["sample_id"],
        "source_collection": row["source_collection"],
        "source_round": row["selection_features"].get("source_split"),
        "source_capture": row["scene_id"],
        "image": {
            "path": str(paths["image"]), "width": width, "height": height,
            "format": paths["image"].suffix.lstrip(".").upper(),
        },
        "depth": {
            "path": str(paths["depth"]) if paths.get("depth") else None,
            "metadata_path": str(paths["depth_metadata"]) if paths.get("depth_metadata") else None,
            "metadata": depth_metadata,
        },
        "scene": {
            "pc_name": raw_3d.get("pc_name"), "hdri_name": raw_3d.get("hdri_name"),
            "view_type": row["view_type"], "raw_view_type": raw_3d.get("view_type"),
            "camera_height": raw_3d.get("camera_height"),
            "num_excavators": raw_3d.get("num_excavators", len(canonical_instances)),
        },
        "camera": {
            **camera_info,
            "derived_intrinsics_assuming_horizontal_sensor_fit_square_pixels": {
                "fx": focal, "fy": focal, "cx": width / 2, "cy": height / 2,
                "assumption_warning": "Verify Blender sensor_fit and pixel aspect before metric reprojection use.",
            },
        },
        "instances": canonical_instances,
        "provenance": {
            "parent_release": row["manifest_release"], "parent_manifest_sha256": parent_sha,
            "source_root_alias": row["source_root_alias"],
            "raw_paths": {key: str(path) if path else None for key, path in paths.items()},
            "curation_provenance": {
                key: row.get("subset_provenance", {}).get(key)
                for key in ("subset_family", "source_ranking", "source_rank",
                            "historical_curated_image_path", "mapping_method")
            },
        },
        "raw_image_record": image_record,
        "validation": {"status": "valid" if not errors else "invalid", "errors": errors},
    }


def select_level5_landmark(sample_id: str, landmarks: list[dict]) -> dict:
    digest = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
    visible = [item for item in landmarks if item["visibility_state"] == "fully_visible"]
    hidden = [item for item in landmarks if item["visibility_state"] != "fully_visible"]
    candidates = hidden if digest % 2 and hidden else visible if visible else landmarks
    return candidates[(digest // 2) % len(candidates)]


def make_conversations(annotation: dict, canonical_ref: dict) -> list[dict]:
    sample_id = annotation["sample_id"]
    rgb = annotation["image"]["path"]
    instances = annotation["instances"]
    primary = instances[0]
    instance_id = primary["instance_id"]
    base = {
        "sample_id": sample_id,
        "images": [rgb],
        "canonical_annotation": canonical_ref,
        "messages": [],
    }
    outputs = []

    def add(level: int, question: str, answer: dict, images: list[str] | None = None) -> None:
        row = dict(base)
        row.update({
            "id": f"{sample_id}__L{level}", "experiment_level": level,
            "task": LEVELS[level], "images": images or [rgb],
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": compact(answer)},
            ],
        })
        outputs.append(row)

    add(1, "<image>\nHow many excavator instances are annotated, and what are their pixel "
           "bounding boxes in [x,y,width,height] format?", {
        "count": len(instances),
        "instances": [{"instance_id": item["instance_id"],
                       "bbox_xywh": item["bbox_xywh_pixels"]} for item in instances],
    })
    add(2, f"<image>\nFor excavator instance {instance_id}, summarize the 48 landmark "
           "visibility states.", primary["visibility_summary_canonical"])
    add(3, f"<image>\nEstimate the articulated pose of excavator instance {instance_id}. "
           "Return boom elevation, stick elevation, bucket orientation, boom-stick interior "
           "angle, and stick-bucket profile angle in degrees.", primary["derived_articulation"])

    depth_path = annotation["depth"]["path"]
    if depth_path:
        metadata = annotation["depth"]["metadata"] or {}
        range_text = ""
        if metadata.get("from_min") is not None and metadata.get("from_max") is not None:
            range_text = (f" The depth encoding metadata spans {metadata['from_min']} to "
                          f"{metadata['from_max']} {metadata.get('units', 'scene units')}.")
        add(4, f"<image>\n<image>\nThe first image is RGB and the second is its aligned depth "
               f"map.{range_text} Report the annotated landmark optical-depth range and mean "
               f"for excavator instance {instance_id}. Is exact metric depth observable from "
               "RGB alone?", {
            **primary["derived_geometry"], "answerable_with_provided_modalities": True,
            "observable_from_rgb_alone": False, "depth_map_provided": True,
        }, [rgb, depth_path])
    else:
        add(4, f"<image>\nOnly an RGB image is provided. Can the exact metric landmark "
               f"optical-depth range and mean for excavator instance {instance_id} be determined "
               "reliably? Do not invent metric values.", {
            "answerable_with_provided_modalities": False, "observable_from_rgb_alone": False,
            "reason": "Exact metric depth is not identifiable from this RGB image alone.",
        })

    landmark = select_level5_landmark(sample_id, primary["landmarks"])
    state = landmark["visibility_state"]
    direct = state == "fully_visible"
    add(5, f"<image>\nIs the {landmark['semantic_name']} landmark of excavator instance "
           f"{instance_id} directly observable in the RGB image? Answer yes or no using the "
           "visibility evidence; do not infer a hidden point as visible.", {
        "answer": "yes" if direct else "no", "visibility_state": state,
        "reason": ("The landmark is directly visible." if direct else
                   "The landmark is not directly visible in the RGB evidence."),
    })
    return outputs


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def process_one(payload: tuple[dict, dict[str, Path], list[dict], str, dict]) -> tuple[dict, list[dict]]:
    row, roots, registry, parent_sha, canonical_ref = payload
    annotation = make_canonical(row, roots, registry, parent_sha)
    if annotation["validation"]["status"] != "valid":
        raise RuntimeError(f"Invalid canonical record {row['sample_id']}: {annotation['validation']}")
    return annotation, make_conversations(annotation, canonical_ref)


def main() -> None:
    args = parse_args()
    release = json_file(args.master_release / "release.json")
    roots = {key: Path(value) for key, value in release["source_roots"].items() if value}
    registry_document = json_file(args.keypoint_registry)
    registry = registry_document["landmarks"]
    subset_rows = {
        label: load_manifest(args.subset_family / label / "manifest.jsonl")
        for label in args.subsets
    }
    if args.limit:
        subset_rows = {label: values[:args.limit] for label, values in subset_rows.items()}
    largest = max(args.subsets, key=lambda label: len(subset_rows[label]))
    rows = subset_rows[largest]
    selected_ids = {label: {row["sample_id"] for row in values}
                    for label, values in subset_rows.items()}
    if any(not ids <= selected_ids[largest] for ids in selected_ids.values()):
        raise RuntimeError("Requested subsets are not nested inside the largest subset")

    canonical_dir = args.output / "canonical"
    base_dir = args.output / "qwen_base"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    canonical_index = []
    for shard_number, shard_rows in chunks(rows, args.shard_size):
        shard_name = f"part-{shard_number:05d}.jsonl"
        canonical_path = canonical_dir / shard_name
        base_path = base_dir / shard_name
        if canonical_path.exists() and base_path.exists() and not args.overwrite:
            print(f"reuse {shard_name}", flush=True)
        else:
            payloads = []
            for line_number, row in enumerate(shard_rows, 1):
                ref = {"path": str(canonical_path), "line": line_number,
                       "sample_id": row["sample_id"]}
                payloads.append((row, roots, registry, release["artifacts"]["master_manifest_sha256"], ref))
            canonical_lines, qwen_lines = [], []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for annotation, conversations in executor.map(process_one, payloads):
                    canonical_lines.append(compact(annotation))
                    qwen_lines.extend(compact(item) for item in conversations)
            atomic_write(canonical_path, "\n".join(canonical_lines) + "\n")
            atomic_write(base_path, "\n".join(qwen_lines) + "\n")
            print(f"built {shard_name}: {len(canonical_lines)} annotations, "
                  f"{len(qwen_lines)} conversations", flush=True)
        canonical_index.append({
            "shard": shard_name, "annotation_count": len(shard_rows),
            "first_sample_id": shard_rows[0]["sample_id"],
            "last_sample_id": shard_rows[-1]["sample_id"],
            "sha256": sha256(canonical_path),
        })

    subset_reports = []
    base_paths = sorted(base_dir.glob("part-*.jsonl"))
    for label in args.subsets:
        destination = args.output / "subsets" / label
        destination.mkdir(parents=True, exist_ok=True)
        all_file = destination / "qwen_sft_all_5_levels.jsonl"
        level_files = {level: destination / f"level_{level}.jsonl" for level in LEVELS}
        temporary_all = all_file.with_suffix(".jsonl.tmp")
        temporary_levels = {level: path.with_suffix(".jsonl.tmp")
                            for level, path in level_files.items()}
        counts = Counter()
        with temporary_all.open("w", encoding="utf-8") as all_stream:
            level_streams = {level: path.open("w", encoding="utf-8")
                             for level, path in temporary_levels.items()}
            try:
                for base_path in base_paths:
                    with base_path.open("r", encoding="utf-8") as stream:
                        for line in stream:
                            item = json.loads(line)
                            if item["sample_id"] not in selected_ids[label]:
                                continue
                            all_stream.write(line)
                            level_streams[item["experiment_level"]].write(line)
                            counts[item["experiment_level"]] += 1
            finally:
                for stream in level_streams.values():
                    stream.close()
        os.replace(temporary_all, all_file)
        for level, path in level_files.items():
            os.replace(temporary_levels[level], path)
        expected = len(subset_rows[label])
        if any(counts[level] != expected for level in LEVELS):
            raise RuntimeError(f"Conversation count mismatch for {label}: {counts}")
        report = {
            "subset": label, "sample_count": expected,
            "conversation_count": sum(counts.values()),
            "level_counts": {str(level): counts[level] for level in LEVELS},
            "all_levels_path": str(all_file), "all_levels_sha256": sha256(all_file),
            "raw_media_copied": False,
        }
        atomic_write(destination / "materialization.json", json.dumps(report, indent=2) + "\n")
        subset_reports.append(report)
        print(f"{label}: {expected:,} samples, {sum(counts.values()):,} conversations", flush=True)

    summary = {
        "schema": "excavision-qwen-materialization/1.0.0",
        "parent_release_id": release["release_id"],
        "parent_manifest_sha256": release["artifacts"]["master_manifest_sha256"],
        "subset_family": str(args.subset_family),
        "keypoint_registry": str(args.keypoint_registry.resolve()),
        "keypoint_registry_sha256": sha256(args.keypoint_registry),
        "canonical_annotation_count": len(rows),
        "canonical_shard_size": args.shard_size,
        "canonical_shards": canonical_index,
        "subsets": subset_reports,
        "media_policy": "Reference original absolute RGB/depth paths; do not copy media.",
    }
    atomic_write(args.output / "materialization.json", json.dumps(summary, indent=2) + "\n")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
