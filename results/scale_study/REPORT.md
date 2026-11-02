# YOLO-World scale study on MOCS with the locked prompt

All adapted checkpoints were evaluated with prompt `construction excavator`, prediction floor 0.001, image size 640, NMS IoU 0.70, maximum 300 detections, and operating threshold 0.25. MOCS is the development set. No held-out set was used.

| Condition | Size | AP50 | AP50:95 | P@0.25 | R@0.25 | F1@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | none | 0.745 | 0.554 | 0.956 | 0.362 | 0.525 |
| Curated | 5K | 0.715 | 0.546 | 0.936 | 0.530 | 0.676 |
| Random | 5K | 0.767 | 0.581 | 0.906 | 0.576 | 0.704 |
| Curated | 20K | 0.771 | 0.570 | 0.960 | 0.464 | 0.626 |
| Random | 20K | 0.752 | 0.562 | 0.867 | 0.608 | 0.715 |
| Curated | 100K | 0.769 | 0.581 | 0.981 | 0.214 | 0.352 |
| Random | 100K | 0.770 | 0.587 | 0.951 | 0.467 | 0.626 |
| Curated | 500K | 0.770 | 0.577 | 0.948 | 0.491 | 0.647 |
| Random | 500K | 0.758 | 0.573 | 0.949 | 0.401 | 0.564 |

## Interpretation

- Random 100K has the highest AP50:95 (0.587), followed by random 5K and curated 100K (both approximately 0.581).
- Random 20K has the best fixed-threshold recall (0.608) and F1 (0.715).
- Curated selection is better than random at 20K and 500K in AP50:95, but worse at 5K and 100K. The alpha-0.5 curation policy is therefore not consistently superior for YOLO-World adaptation under the locked deployment prompt.
- Every configuration except curated 5K exceeds zero-shot AP50:95. Every configuration except curated 100K exceeds zero-shot recall at 0.25.
- Curated 100K combines good ranking AP with very low recall at 0.25 and very high precision, indicating a score-calibration shift rather than loss of box-ranking quality.
- Results are seed-0 development results. Configuration confirmation requires additional training seeds before held-out testing.
