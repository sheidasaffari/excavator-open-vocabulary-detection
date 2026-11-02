# Minimal error decomposition before adaptation

## Scope

This diagnostic uses only frozen predictions and previously locked evaluation data.
It does not change prompts, thresholds, annotations, image inclusion, or model weights.
MOCS is development evidence; the positive-control and corrected confuser benchmarks
remain confirmatory. The analyses are intended to identify the minimum appropriate
adaptation mechanism, not to select a new deployment threshold.

## YOLO-World: confidence dominates the missed detections

Each GT excavator was assigned to one mutually exclusive raw-oracle category using
the low-floor prediction files:

- **success:** a prediction with score >= 0.25 reaches IoU >= 0.50;
- **confidence failure:** an IoU >= 0.50 prediction exists, but only below 0.25;
- **localization failure:** best IoU is in [0.30, 0.50);
- **proposal failure:** best IoU is below 0.30.

A score-descending, one-to-one greedy sensitivity analysis produced nearly identical
results; assignment conflicts affected at most 0.3% of MOCS GT and none of the
positive-control GT.

| Dataset | Prompt | Success | Confidence | Localization | Proposal |
|---|---|---:|---:|---:|---:|
| MOCS dev | `excavator` | 23.1% | **65.7%** | 2.5% | 8.6% |
| MOCS dev | `construction excavator` | 38.2% | **51.0%** | 2.8% | 8.1% |
| Positive control | `excavator` | 10.2% | **78.1%** | 6.6% | 5.1% |
| Positive control | `construction excavator` | 24.8% | **65.0%** | 6.6% | 3.6% |

The selected prompt barely changes IoU >= 0.50 coverage:

- MOCS: 88.8% to 89.1%;
- positive control: 88.3% to 89.8%.

It mainly raises the scores of already localized boxes. Therefore YOLO-World's low
recall at 0.25 is primarily a **text/cross-modal scoring problem**, not missing visual
proposals. A smaller geometric failure remains (approximately 10% localization plus
proposal failure on MOCS and positive control).

## Grounding DINO: object/category boundary dominates confuser detections

The corrected confuser set has no confuser bounding boxes, so overlap was not inferred
automatically. All 86/110 images firing at the locked score threshold of 0.25 were
manually reviewed in deterministic boxed contact sheets. Primary labels were mutually
exclusive; duplicate/fragmented detections were tracked separately.

| Primary failure | Images | Share of firing images | Median top score |
|---|---:|---:|---:|
| Whole-machine/object category confusion | **80** | **93.0%** | 0.627 |
| Isolated part/attachment confusion | 6 | 7.0% | 0.554 |
| Clear context-only trigger | 0 | 0.0% | — |

Additional observations:

- all six isolated-attachment images fired, with top scores 0.382–0.932;
- 17/86 firing images (19.8%) showed duplicate/fragmented detections of the same false target;
- 7/86 (8.1%) contained multiple distinct false targets;
- cluttered-site and ordinary-negative failures were visually object-centered, not
  empty-background boxes.

The minimal conclusion is that Grounding DINO's fixed-threshold confuser problem is
primarily an overly broad **object/category boundary**. This audit does not prove that
context is irrelevant; paired insert/remove scenes would be needed for that causal
claim.

## Adaptation decision supported by the decomposition

The first pilot should prioritize:

1. **YOLO-World:** text/cross-modal fusion or score-calibration adaptation, while
   measuring whether already valid boxes move above threshold without degrading
   confuser specificity.
2. **Grounding DINO:** whole-machine hard negatives, especially loaders, backhoes,
   dozers, trucks, and cranes.
3. **Grounding DINO:** isolated-component counterfactuals to address part/whole
   confusion.
4. **Both models:** retain positive excavator examples and include a frozen-model
   retention/distillation arm.
5. Treat context-only negatives and localization-specific changes as secondary pilot
   controls rather than the central intervention.

## Guardrails

- Do not tune the adaptation recipe on the frozen positive-control or confuser sets.
- Use MOCS and a new synthetic negative/counterfactual validation set for pilot
  selection.
- Evaluate the locked pilot on ACID, CIS, positive control, and confusers only after
  the recipe is frozen.
- The Grounding DINO taxonomy is a single-reviewer, image-level diagnostic because no
  confuser GT boxes exist.

## Supporting artifacts

- `yolo/report.md`, `yolo/summary.csv`, `yolo/per_gt_decomposition.csv`,
  `yolo/provenance.json`, `yolo/yolo_error_decomposition.py`
- `gd/REPORT.md`, `gd/manual_audit_decisions.csv`, `gd/audit_sample_manifest.csv`,
  `gd/audit_sheets/`, `gd/build_audit_sheets.py`, `gd/finalize_manual_audit.py`
