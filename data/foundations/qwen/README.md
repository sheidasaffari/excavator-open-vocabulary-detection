# Excavision foundation-data manifests for Qwen

This directory defines the versioned, traceable interface between the three
Excavision image collections and Qwen2.5-VL experiments. It intentionally does
not store the image pool in Git. Images and raw labels remain in their source
collections; manifests select them by stable ID and record their provenance.

## Current status

- A validated 10-image pilot spans the processed corpus, Newgen targeted rounds
  1–3, and the colored corpus.
- All ten pilot samples pass cross-file structural validation.
- The pilot produces 50 examples: ten examples for each of five experiment
  levels.
- Six pilot samples have dense depth; four exercise RGB-only abstention.
- The audited full pool contains approximately 1,032,309 complete image/label
  samples: 675,152 processed, 347,202 Newgen, and 9,955 colored.

The full-pool counts are an inventory snapshot, not a committed train split.
They must be reproduced by the full indexing run and written into its release
report before that release is treated as authoritative.

## Files

- `schemas/excavision_keypoints_v1.json`: fixed 48-landmark registry, including
  raw Blender bone names, semantic aliases, stable components, and visibility
  conventions.
- `schemas/master_manifest_v1.schema.json`: JSON Schema for one image-centered
  master-manifest row.
- `MASTER_MANIFEST.md`: identity, provenance, partitioning, versioning, and split
  policy for the full dataset.
- `examples/pilot_manifest_10.jsonl`: the validated pilot's ten lightweight
  manifest rows. These refer to files in the generated pilot bundle and do not
  copy the underlying media into Git.
- `examples/pilot_validation_report.json`: pilot validation summary.
- `scripts/select_subset.py`: deterministic selection from any JSONL master
  manifest, with a sidecar recipe and parent-manifest checksum.
- `scripts/build_master_manifest.py`: resumable full-pool indexer with adapters
  for processed, Newgen rounds 1–3, and colored data. It reads only label
  headers, never decodes/copies image pixels, and emits JSONL plus Parquet.
- `FULL_RELEASE_REPORT.md`: counts, exclusion reconciliation, artifact sizes,
  and SHA-256 values for the completed `excavision-master-v1` release.

## Five experiment levels

1. Detection, counting, and grounding.
2. Landmark/component visibility and occlusion.
3. Articulated boom/stick/bucket pose.
4. Camera, depth, and metric geometry.
5. Answerability, uncertainty, and abstention.

## Visibility policy

- `v=2`: in frame and directly visible on the expected component.
- `v=1`: in frame and self-occluded.
- `v=0` with nonzero coordinates: in frame and externally occluded, or the
  expected surface could not be resolved.
- `[0,0,0]`: outside the image; individual ray state is unavailable in the 2D
  record. Legacy processed aggregates may still count that landmark, while
  Newgen and colored aggregates omit it.

The canonical representation must keep `inside_image`, `image_xy`,
`raw_visibility_flag`, and `visibility_state` separate.

## Selecting a smaller LoRA subset

The master manifest is an index, not an instruction to train on every image.
A subset is another immutable manifest whose rows retain the original
`sample_id`. For example:

```bash
python data/foundations/qwen/scripts/select_subset.py \
  --manifest /path/to/master_manifest.jsonl \
  --output /path/to/subsets/lora_depth_balanced/sample_ids.jsonl \
  --source newgen_r1 newgen_r2 newgen_r3 colored \
  --require-depth \
  --max-instances 3 \
  --limit 20000 \
  --seed 20261109
```

The script writes the selected rows plus `selection.json`, which records the
exact filters, random seed, UTC creation time, parent path, and parent SHA-256.
No images are duplicated. Canonical annotations and Qwen conversations can be
materialized only for the chosen IDs.

## Recommended full run

1. Build a lightweight, immutable master index for all complete samples.
2. Validate paths, modality availability, label alignment, ID uniqueness, and
   schema conformance.
3. Partition by source collection and generation group for scalable reads.
4. Freeze group-aware train/validation/test partitions before LoRA selection.
5. Create several named subset manifests (for example 5k, 20k, 100k) using
   explicit selection recipes.
6. Materialize expensive canonical landmark records and Qwen SFT examples only
   for selected subsets.

This design supports a small high-diversity LoRA set without losing the ability
to revisit the complete pool later.

## Building the full lightweight index

```bash
python data/foundations/qwen/scripts/build_master_manifest.py \
  --processed-root /home/ssaffari/Data/excavision/processed \
  --newgen-root /home/ssaffari/Data/newgen/output/targeted \
  --colored-root /home/ssaffari/Data/color-project/output-colored \
  --output /home/ssaffari/Data/foundations/qwen/releases/excavision-master-v1 \
  --release-id excavision-master-v1 \
  --resume
```

The release contains `master_manifest.jsonl`, a compact
`master_samples.parquet`, source-partition JSONL files, `release.json`,
`validation_report.json`, and `invalid_samples.jsonl`. A source partition is
published atomically only after it finishes; `--resume` reuses completed
partitions after interruption.

To correct or refresh selected adapters without rescanning the rest, combine
`--resume` with `--rebuild-sources`, followed by one or more source names.
