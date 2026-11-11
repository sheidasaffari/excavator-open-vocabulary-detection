# Negative-supervision Phase-1 scripts

These are the exact study scripts used to prepare, train, infer, and analyze the
Phase-1 negative-supervision screen. They are published for methodological
transparency; private datasets, checkpoints, and raw predictions are not included.

## Workflow

1. `build_phase1_data.py` creates matched N0, N1, and N2 training views.
2. `validate_phase1_data.py` audits counts, labels, image existence, and split
   integrity.
3. `run_phase1_screening_2gpu.sh` launches the one-seed two-family screen.
4. `train_yoloe_phase1.py` and `train_gdino_phase1.py` create locked run receipts
   and train each requested arm.
5. `prepare_phase1_eval_data.py` constructs the combined MOCS-positive and
   ACID-development-negative evaluation index.
6. `infer_yoloe_phase1.py` and `run_gdino_phase1_inference.py` retain low-floor
   predictions from every checkpoint.
7. `convert_gdino_predictions.py` converts MMDetection outputs into the common
   schema.
8. `analyze_phase1.py` computes COCO AP, fixed and matched operating points,
   proposal recall, per-confuser FPR, bootstrap intervals, parameter-efficiency
   tables, learning curves, and the locked development decision.

The scripts preserve the original local path constants as part of the experimental
receipt. To run them elsewhere, replace those constants with paths to your private
data and upstream model installations. Do not use CIS or the frozen confirmatory
sets during development selection.

## Grounding DINO configuration correction

`train_gdino_phase1.py` includes `_delete_=True` when overriding MMEngine's
`paramwise_cfg`. This explicitly replaces the inherited optimizer freeze map. The
correction prevents intended vision+fusion or full runs from silently collapsing
to the official fusion scope.
