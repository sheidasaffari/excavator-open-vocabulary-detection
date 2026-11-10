# Master manifest specification

## Design goals

The manifest must be image-centered, source-preserving, immutable per release,
selectable without copying media, and safe against filename collisions. It is
the catalog of what exists. Experiment subsets and model-ready conversations
are derived artifacts.

## Stable identity

`sample_id` is globally unique and source-scoped. It must never depend on the
row number or a mutable absolute root. The current convention is:

```text
processed__<source-stem>
newgen_r<round>__<source-stem>
colored__Ex<generator>__<capture-or-scene>__<source-stem>
```

The row also stores the original source stem, collection, group identifiers,
and relative source paths. A release may add derived fields, but it must not
reassign an existing ID to different media.

## Required row layers

Each row has four logical layers:

1. Identity: `sample_id`, schema version, manifest release, source collection.
2. Provenance: source root alias, relative image/label paths, generation group,
   raw metadata version, and optional checksums.
3. Availability: whether 2D, 3D, occlusion, camera, and dense depth data exist.
4. Selection features: instance count, view type, visibility/occlusion summary,
   truncation, pose bins, camera bins, depth availability, and validation state.

The compact pilot rows predate some optional full-index fields. The full index
should populate all fields that can be obtained without loading image tensors.

## Storage layout

The release should contain:

```text
releases/<release_id>/
  master_manifest.jsonl
  master_samples.parquet
  release.json
  validation_report.json
  partitions/source_collection=<name>/...
  subsets/<subset_name>/
    sample_ids.jsonl
    selection.json
    summary.json
```

JSONL is the transparent interchange format. Parquet is the scalable analysis
format and should be partitioned by source collection (and, where useful,
generation round). Both representations describe the same rows and release.

High-volume landmark data should be stored separately from the lightweight
sample table, keyed by `sample_id` and `instance_id`:

- `instances`: boxes, object pose, stable instance identity.
- `landmarks`: 48 ordered 2D/3D landmarks and visibility state.
- `component_visibility`: raw nearest-surface aggregation plus stable semantic
  component aggregation.

This prevents a simple image selection query from reading tens of millions of
landmark records.

## Subsets are immutable views

A subset contains selected master rows or IDs, never renamed samples. Every
subset recipe records:

- parent release ID and SHA-256;
- exact filter and balancing configuration;
- random seed;
- selection code version;
- creation time and selected count;
- group-aware split assignment.

Regenerating the same recipe against the same parent must return the same IDs.
If the parent changes, create a new subset version.

## Split policy and leakage control

Random image-level splitting is not acceptable for synthetic sequences because
near-duplicate frames can cross partitions. Split first by a generation group
such as capture/session/scene/camera sequence, then select samples within the
assigned groups. The held-out evaluation set should remain fixed across LoRA
subset sizes.

## LoRA selection features

A smaller subset may be better than the complete pool because the pool contains
heavy redundancy. Selection should preserve coverage across:

- source collection and Newgen round;
- aerial/ground view and camera bins;
- one/multiple excavators and instance-count bins;
- visible, self-occluded, externally occluded, and truncated cases;
- boom/stick/bucket pose bins;
- depth present/absent and answerability classes;
- generation groups, backgrounds, lighting, and difficulty.

Recommended comparisons are fixed-budget 5k/20k/100k subsets, a random baseline
at each budget, and the same held-out group-aware evaluation set. This tests
whether curated diversity is more useful than raw volume.

## Source-specific facts that must remain traceable

- Processed legacy occlusion aggregates cover all 48 landmarks, including
  off-frame points.
- Newgen and colored aggregates cover in-frame landmarks only.
- Dense depth exists in Newgen and colored sources; processed `.npy` files are
  not to be silently interpreted as depth.
- Some dense-depth metadata declares an incorrect filename. The canonical row
  must retain the declared value and separately record the resolved real path.
- Raw bone names remain preserved even when semantic aliases are used.
- `root_relative_xyz` means world position minus landmark 0, not scale
  normalization.

