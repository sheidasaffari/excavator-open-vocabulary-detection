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
| Colored green | 12,886 | 12,886 | 0 |
| **Total** | **1,045,195** | **370,043** | **312** |

All 1,045,195 included rows have unique `sample_id` values and valid required
paths. The 312 exclusions are all `label_2d_without_image`; they remain listed
in `invalid_samples.jsonl` and are not silently discarded.

## Artifact verification

- `master_manifest.jsonl`: 1,045,195 lines, 1,663,207,830 bytes.
- JSONL SHA-256: `5ce1e731d580eb61ad92eabc47e8cc0fc8d432b62d21f9ca07463b0cff277539`.
- `master_samples.parquet`: 1,045,195 rows, 105 row groups, 31 columns,
  51,193,282 bytes.
- Parquet SHA-256: `68855af0952af27b58e99188962794c6d57423b4230632cd98bde87e6d889309`.
- Full-pool subset smoke test: deterministic selection of 100 depth-enabled
  samples with at most three excavators succeeded.

The JSONL is the complete nested interchange representation. The Parquet file
is the compact, flat selection table intended for filtering and experiment
design.
