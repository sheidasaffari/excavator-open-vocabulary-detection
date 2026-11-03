# Multi-seed random-subset selection on MOCS

All checkpoints were evaluated with the locked prompt `construction excavator`, prediction floor 0.001, NMS IoU 0.70, image size 640, and operating threshold 0.25.
MOCS is the development set. No held-out dataset was used for size selection.

| Size | AP50 | AP50:95 | Precision@0.25 | Recall@0.25 | F1@0.25 |
|---:|---:|---:|---:|---:|---:|
| 5K | 0.762 ± 0.005 | 0.577 ± 0.006 | 0.868 ± 0.034 | 0.620 ± 0.043 | 0.722 ± 0.018 |
| 10K | 0.761 ± 0.009 | 0.576 ± 0.005 | 0.852 ± 0.102 | 0.612 ± 0.082 | 0.705 ± 0.014 |
| 20K | 0.767 ± 0.015 | 0.579 ± 0.015 | 0.912 ± 0.039 | 0.554 ± 0.050 | 0.687 ± 0.026 |
| 50K | 0.764 ± 0.004 | 0.576 ± 0.003 | 0.920 ± 0.029 | 0.529 ± 0.081 | 0.669 ± 0.061 |
| 100K | 0.768 ± 0.002 | 0.578 ± 0.007 | 0.946 ± 0.022 | 0.477 ± 0.084 | 0.630 ± 0.068 |

## Frozen decision

- The highest mean AP50:95 among the new candidates was 20K (0.579).
- AP50:95 uncertainty intervals overlap across the candidate sizes, so the candidates are treated as practically equivalent under the frozen rule.
- 5K was selected because it has the strongest mean recall among the AP-equivalent candidates and is the smallest recall-equivalent configuration.
- 5K and the 100K reference differ by only 0.002 mean AP50:95; their 95% confidence intervals overlap.
- The evidence therefore supports rapid saturation/data efficiency, not a monotonic benefit from increasing synthetic-data volume.

## Integrity

- 15/15 checkpoint hashes are unique.
- 15/15 MOCS prediction hashes are unique.
- All runs contain 1,452 MOCS GT instances and use an identical inference configuration.

## Next step

Freeze 5K and evaluate its three seeds on CIS, the positive control, and the confuser set. Those held-out outcomes must not be used to revise the size choice.
