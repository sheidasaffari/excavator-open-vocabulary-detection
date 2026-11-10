# Curated canonical and Qwen materialization

The nested `curated_v4_rank_neutral` family was materialized for the five
selected budgets. Canonical annotations are stored once for the largest 100K
pool in 100 deterministic JSONL shards. Each subset has copy-free Qwen
conversation views referencing original RGB/depth media and the shared
canonical record.

| Subset | Samples | Conversations | Per level | Depth-enabled Level 4 |
|---|---:|---:|---:|---:|
| 5K | 5,000 | 25,000 | 5,000 | 3,030 |
| 10K | 10,000 | 50,000 | 10,000 | 5,962 |
| 20K | 20,000 | 100,000 | 20,000 | 11,561 |
| 50K | 50,000 | 250,000 | 50,000 | 27,392 |
| 100K | 100,000 | 500,000 | 100,000 | 45,781 |

The five levels are detection/count/grounding, landmark visibility and
occlusion, articulated pose, camera/depth/metric geometry, and
answerability/abstention.

## Output

```text
/home/ssaffari/Data/foundations/qwen/releases/excavision-master-v1/
  materialized/curated_v4_rank_neutral_qwen_v1/
    materialization.json
    canonical/part-00000.jsonl ... part-00099.jsonl
    qwen_base/part-00000.jsonl ... part-00099.jsonl
    subsets/{5k,10k,20k,50k,100k}/
      level_1.jsonl ... level_5.jsonl
      qwen_sft_all_5_levels.jsonl
      materialization.json
```

The materialized release occupies approximately 5.8 GiB. No RGB, depth, or
source-label files were copied.

## Full validation

- 100,000 canonical records and 100,000 unique canonical `sample_id` values.
- Zero invalid canonical records.
- Zero missing referenced RGB/depth files.
- Exact subset membership at every experiment level.
- Unique conversation ID and sample membership within every level.
- Combined conversation count equals five times the subset sample count.
- Depth-enabled Level-4 counts exactly match each curated subset manifest.
- RGB-only Level-4 examples require abstention and contain no invented metric
  target.

Before training, assign a fixed group-aware train/validation/test split using
`split_group_id`; do not randomly split individual images from the same
synthetic capture across train and evaluation.
