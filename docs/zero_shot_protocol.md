# Protocol (locked before running final evaluation)

Step 1+2 of the foundation-model research plan: benchmark Grounding DINO
and YOLO-World (both zero-shot, no Excavision-specific training) on three
independent real-world test sources.

## Datasets

| dataset | role | n images | GT format |
|---|---|---|---|
| **MOCS** (`data/mocs/`) | **development** (prompt/threshold calibration) | 1010 | YOLO txt, single class `excavator` |
| **ACID** (`data/acid/`) | held-out test | 605 | YOLO txt, single class `excavator` |
| **CIS** (`data/cis/`) | held-out test | 2000 (reference 1000 + test 1000, combined into one 2000-image held-out set -- the reference/test split there was from an earlier, unrelated DINOv2-similarity-ranking exercise, not a train/test split for this protocol) | YOLO txt, single class `excavator` |

**Decision**: MOCS chosen as the development set (reasonably sized, well-known
public benchmark). ACID and CIS are the two **untouched** held-out test sets.
No prompt or threshold decision is ever made by looking at ACID/CIS results.

## Models (both zero-shot, as-shipped, no fine-tuning)

- **Grounding DINO** (`IDEA-Research/grounding-dino-base`, via HuggingFace `transformers`)
- **YOLO-World** (`yolov8s-worldv2.pt`, via `ultralytics`)

## Prompt protocol

Three prompts, predefined before looking at any result:
1. `excavator.` (canonical)
2. `hydraulic excavator.`
3. `construction excavator.`

(Period-separated phrasing follows Grounding DINO's own recommended
convention; the same three semantic concepts are used for YOLO-World,
without forcing identical punctuation, since its official usage doesn't
require it.)

**Selection rule**: for each model independently, evaluate all 3 prompts on
the **development set (MOCS) only**. Select the prompt maximizing
AP50:95 on MOCS; break ties using MOCS recall. Lock that choice. Apply it
unchanged to ACID and CIS. Never select a different prompt per final test
dataset.

## Reporting

1. **Main table**: canonical prompt (`excavator.`), same for both models, all 3 datasets -- the direct apples-to-apples comparison.
2. **Deployment table**: development-selected prompt per model, all 3 datasets.
3. **Prompt-sensitivity table**: all 3 prompts x both models, MOCS (dev) results only, plus Δ_prompt = max(AP) - min(AP) and mean/std across prompts.

## Metrics

- AP50, AP50:95 (threshold-independent)
- Precision, Recall, F1 at a fixed confidence threshold of 0.25 (operating-point metrics; threshold chosen a priori, not tuned per dataset)
- Single class (`excavator`), IoU matching via standard COCO-style greedy matching

## Known limitations of this pass (flagged, not solved here)

- All 3 datasets are single-class (`excavator` only) -- there are **no
  negative/confuser images** (crane, loader, bulldozer, backhoe, no-equipment
  scenes) in any of them, so this pass cannot measure false-positive rate
  against confusable categories or "fires on anything construction-related"
  failure modes. That requires a separate negative-image set (flagged as
  future work, not part of this run).
- The causal text-vs-vision diagnosis (Step 2's deeper intervention design:
  frozen baseline / text-only adaptation / vision-only adaptation / oracle-region
  recognition-vs-localization probes) is **not** part of this pass -- this
  pass only produces the baseline numbers and prompt-sensitivity evidence
  that the diagnosis will build on.
