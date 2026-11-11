# Phase-1 negative-supervision results

This folder contains aggregate, development-only results for the YOLOE-v8-S and
MM-Grounding-DINO-T negative-representation and adaptation-scope screen.

The experiment used MOCS for positive development evaluation and a disjoint ACID
development-negative split for specificity. CIS and all frozen confirmatory sets
remained untouched. No private image, raw prediction, or checkpoint is distributed.

See the [complete explanation](../../docs/negative_supervision_phase1.md) before
interpreting the tables.

## Files

- `model_summary.csv`: AP, fixed-threshold metrics, proposal recall, score
  separation, bootstrap intervals, and matched operating points for all 14 models.
- `prespecified_contrasts.csv`: N1−N0, N2−N0, and N2−N1 comparisons.
- `per_confuser_results.csv`: class-specific image-FPR at threshold 0.25 and at
  matched 0.70 MOCS recall.
- `training_summary.csv`: best epoch, best validation AP, final AP, and overfitting
  gap.
- `parameter_scopes.csv`: updated parameter counts and the validity status of each
  adaptation scope.
- `development_decisions.json`: output of the locked development selection rule.
- `analysis_receipt.json`: input hashes and analysis settings.
- `*_tradeoffs.png`: family-specific sensitivity/specificity and scope figures.
- `per_confuser_matched_recall_heatmap.png`: class-specific specificity comparison.
- `learning_curves.png`: development learning histories.

## Configuration disclosure

The Grounding DINO runs originally named `n2_visionfusion` and `n2_full` inherited
the same optimizer freeze map as `n2_fusion`. They are retained in the machine table
with `scope_label_valid=false`, but excluded from the adaptation-scope figure and
from model selection. They must not be interpreted as broader-scope ablations.
