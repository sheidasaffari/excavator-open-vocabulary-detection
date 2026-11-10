# Canonical curated subset family

The historical `curated_v4_rank_neutral` ranking was mapped to the Excavision
master manifest and reconstructed at five unique-image budgets. No image or
label files were copied.

| Subset | Unique rows | Dense depth | Historical duplicate aliases replaced |
|---|---:|---:|---:|
| 5K | 5,000 | 3,030 | 1 |
| 10K | 10,000 | 5,962 | 2 |
| 20K | 20,000 | 11,561 | 12 |
| 50K | 50,000 | 27,392 | 46 |
| 100K | 100,000 | 45,781 | 172 |

The historical duplicates were Newgen samples represented by both a normal
union filename and a round/shard-prefixed alias. Both names resolve to the same
raw image and master `sample_id`. Each duplicate was removed and replaced by
the next-ranked unique sample, preserving the requested budget.

The subsets are strictly nested:

```text
5K ⊂ 10K ⊂ 20K ⊂ 50K ⊂ 100K
```

Every manifest row contains portable raw RGB, 2D, 3D, occlusion, and optional
depth paths through a master root alias. It also records the original curated
image path, source ranking, source rank, and subset order. `selection.json`
stores the master checksum, historical-list checksum, ranking checksums, source
composition, depth count, duplicate audit, and resulting manifest checksum.

Generated artifacts are stored outside Git at:

```text
/home/ssaffari/Data/foundations/qwen/releases/excavision-master-v1/
  subsets/curated_v4_rank_neutral/{5k,10k,20k,50k,100k}/
```
