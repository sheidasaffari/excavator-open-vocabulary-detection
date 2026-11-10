# Excavision master manifest v1 release report

Release built and validated on 2026-11-10 with builder version `1.0.0`.
The generated data artifacts are stored outside Git under
`/home/ssaffari/Data/foundations/qwen/releases/excavision-master-v1`.

## Included samples

| Source collection | Complete rows | Dense depth | Label-only exclusions |
|---|---:|---:|---:|
| Processed | 675,152 | 0 | 308 |
| Newgen round 1 | 276,996 | 276,996 | 1 |
| Newgen round 2 | 49,945 | 49,945 | 2 |
| Newgen round 3 | 20,261 | 20,261 | 0 |
| Colored | 9,955 | 9,955 | 1 |
| **Total** | **1,032,309** | **357,157** | **312** |

All 1,032,309 included rows have unique `sample_id` values and valid required
paths. The 312 exclusions are all `label_2d_without_image`; they remain listed
in `invalid_samples.jsonl` and are not silently discarded.

## Artifact verification

- `master_manifest.jsonl`: 1,032,309 lines, 1,641,876,952 bytes.
- JSONL SHA-256: `43ca45a4dbb8dfedcd33ac338ef98ee7c301096959ddf085524db54a7cfd82bb`.
- `master_samples.parquet`: 1,032,309 rows, 104 row groups, 31 columns,
  50,810,612 bytes.
- Parquet SHA-256: `34f5adcd43018d3ce0212d69e6656bff6f810cbc0cc37171c5638595fa6b21dc`.
- Full-pool subset smoke test: deterministic selection of 100 depth-enabled
  samples with at most three excavators succeeded.

The JSONL is the complete nested interchange representation. The Parquet file
is the compact, flat selection table intended for filtering and experiment
design.

