# Qualitative failure and recovery cases

This directory contains publication-ready composites derived from the frozen
evaluation artifacts. It intentionally contains no standalone source images.

## Figures

- `grounding_dino_confusion_examples`: five negative/confuser images that Grounding
  DINO incorrectly detects as excavators at the locked 0.25 operating threshold.
  The examples cover backhoe-loader, bulldozer, wheel-loader, detached attachment,
  and crane confusions. These images have negative image-level ground truth, so no
  ground-truth bounding box exists.
- `yolo_world_failure_examples`: five MOCS excavator instances illustrating
  near-threshold suppression, severe under-scoring of a valid box, localization
  failure, a multi-instance proposal miss, and a small/distant proposal failure.
- `yolo_world_recovered_after_adaptation`: five cases in which no adaptation misses
  at the locked threshold and all three Random-100K adapted seeds recover a valid
  IoU≥0.50 detection. The displayed adapted box is from seed 0; all seed scores are
  printed in the corresponding panel.
- `yolo_world_selected_random5k_recoveries`: six additional cases from MOCS, CIS,
  and the independent positive control in which no adaptation misses at the locked
  threshold and all three frozen Random-5K seeds recover a valid IoU≥0.50
  detection. This is the qualitative figure for the final selected model; the
  earlier Random-100K examples are retained as an archived qualitative artifact.

PNG files are intended for repository display. Vector PDF versions are supplied for
paper preparation. `failure_case_manifest.csv` records the original image IDs,
datasets, failure labels, scores, and IoUs;
`selected_random5k_recovery_manifest.csv` records the new frozen-model examples.

## Reproduction

Run `scripts/visualization/build_failure_case_figures.py` with the private
foundational-results root and MOCS root. The example IDs and selection criteria are
frozen in that script. Raw images, annotations, and prediction files are not
distributed by this repository.

The Random-5K recovery figure is regenerated with
`scripts/visualization/build_selected_5k_recovery_examples.py`.
