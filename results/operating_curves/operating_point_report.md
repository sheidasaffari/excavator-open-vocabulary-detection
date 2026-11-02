# Operating-point analysis

Frozen evaluation inputs: **127 positive images / 137 GT boxes** and **110 corrected confuser images**. All curves use IoU ≥ 0.50 for a true positive. No prompt, annotation, inclusion rule, or model weight was changed.

The threshold sweep is descriptive and was not used to retune the locked deployment threshold. The matched operating points below are post-hoc diagnostic summaries, not newly selected deployment settings.

## Locked threshold 0.25

| Configuration | Recall | Positive-only precision | Combined precision | Confuser image FPR | Confuser FP/image |
|---|---:|---:|---:|---:|---:|
| Grounding DINO · excavator. | 0.847 | 0.614 | 0.359 | 0.782 | 1.218 |
| YOLO-World · excavator | 0.102 | 0.560 | 0.378 | 0.109 | 0.109 |
| YOLO-World · construction excavator | 0.248 | 0.723 | 0.540 | 0.145 | 0.145 |

## Maximum recall subject to a confuser-FPR cap

| Configuration | FPR cap | Threshold | Recall | Observed FPR | Confuser FP/image |
|---|---:|---:|---:|---:|---:|
| Grounding DINO · excavator. | 0.05 | 0.922 | 0.058 | 0.045 | 0.045 |
| YOLO-World · excavator | 0.05 | 0.376 | 0.044 | 0.036 | 0.036 |
| YOLO-World · construction excavator | 0.05 | 0.583 | 0.029 | 0.036 | 0.036 |
| Grounding DINO · excavator. | 0.10 | 0.889 | 0.139 | 0.100 | 0.100 |
| YOLO-World · excavator | 0.10 | 0.265 | 0.080 | 0.100 | 0.100 |
| YOLO-World · construction excavator | 0.10 | 0.416 | 0.109 | 0.100 | 0.100 |
| Grounding DINO · excavator. | 0.15 | 0.881 | 0.153 | 0.145 | 0.145 |
| YOLO-World · excavator | 0.15 | 0.161 | 0.292 | 0.136 | 0.145 |
| YOLO-World · construction excavator | 0.15 | 0.227 | 0.292 | 0.145 | 0.145 |
| Grounding DINO · excavator. | 0.25 | 0.805 | 0.328 | 0.245 | 0.245 |
| YOLO-World · excavator | 0.25 | 0.085 | 0.423 | 0.236 | 0.255 |
| YOLO-World · construction excavator | 0.25 | 0.150 | 0.431 | 0.245 | 0.255 |
| Grounding DINO · excavator. | 0.50 | 0.489 | 0.657 | 0.500 | 0.518 |
| YOLO-World · excavator | 0.50 | 0.018 | 0.686 | 0.500 | 0.627 |
| YOLO-World · construction excavator | 0.50 | 0.037 | 0.650 | 0.491 | 0.518 |

## Maximum recall subject to a confuser-FPPI cap

| Configuration | FP/image cap | Threshold | Recall | Confuser FPR | Observed FP/image |
|---|---:|---:|---:|---:|---:|
| Grounding DINO · excavator. | 0.05 | 0.922 | 0.058 | 0.045 | 0.045 |
| YOLO-World · excavator | 0.05 | 0.376 | 0.044 | 0.036 | 0.036 |
| YOLO-World · construction excavator | 0.05 | 0.583 | 0.029 | 0.036 | 0.036 |
| Grounding DINO · excavator. | 0.10 | 0.889 | 0.139 | 0.100 | 0.100 |
| YOLO-World · excavator | 0.10 | 0.265 | 0.080 | 0.100 | 0.100 |
| YOLO-World · construction excavator | 0.10 | 0.416 | 0.109 | 0.100 | 0.100 |
| Grounding DINO · excavator. | 0.15 | 0.881 | 0.153 | 0.145 | 0.145 |
| YOLO-World · excavator | 0.15 | 0.161 | 0.292 | 0.136 | 0.145 |
| YOLO-World · construction excavator | 0.15 | 0.227 | 0.292 | 0.145 | 0.145 |
| Grounding DINO · excavator. | 0.25 | 0.805 | 0.328 | 0.245 | 0.245 |
| YOLO-World · excavator | 0.25 | 0.086 | 0.416 | 0.227 | 0.245 |
| YOLO-World · construction excavator | 0.25 | 0.163 | 0.409 | 0.227 | 0.236 |
| Grounding DINO · excavator. | 0.50 | 0.509 | 0.620 | 0.491 | 0.500 |
| YOLO-World · excavator | 0.50 | 0.025 | 0.635 | 0.418 | 0.491 |
| YOLO-World · construction excavator | 0.50 | 0.047 | 0.635 | 0.436 | 0.464 |
| Grounding DINO · excavator. | 1.00 | 0.298 | 0.832 | 0.727 | 1.000 |
| YOLO-World · excavator | 1.00 | 0.010 | 0.759 | 0.627 | 1.000 |
| YOLO-World · construction excavator | 1.00 | 0.013 | 0.766 | 0.655 | 1.000 |

## Minimum confuser FPR while meeting a recall target

| Configuration | Recall target | Threshold | Observed recall | Confuser FPR | Confuser FP/image |
|---|---:|---:|---:|---:|---:|
| Grounding DINO · excavator. | 0.25 | 0.833 | 0.255 | 0.200 | 0.200 |
| YOLO-World · excavator | 0.25 | 0.168 | 0.255 | 0.136 | 0.145 |
| YOLO-World · construction excavator | 0.25 | 0.245 | 0.255 | 0.145 | 0.145 |
| Grounding DINO · excavator. | 0.50 | 0.644 | 0.504 | 0.364 | 0.364 |
| YOLO-World · excavator | 0.50 | 0.055 | 0.504 | 0.273 | 0.291 |
| YOLO-World · construction excavator | 0.50 | 0.100 | 0.504 | 0.273 | 0.291 |
| Grounding DINO · excavator. | 0.60 | 0.535 | 0.606 | 0.473 | 0.482 |
| YOLO-World · excavator | 0.60 | 0.029 | 0.606 | 0.373 | 0.427 |
| YOLO-World · construction excavator | 0.60 | 0.058 | 0.606 | 0.400 | 0.427 |
| Grounding DINO · excavator. | 0.75 | 0.414 | 0.752 | 0.600 | 0.636 |
| YOLO-World · excavator | 0.75 | 0.011 | 0.752 | 0.609 | 0.918 |
| YOLO-World · construction excavator | 0.75 | 0.018 | 0.752 | 0.627 | 0.882 |
| Grounding DINO · excavator. | 0.80 | 0.378 | 0.803 | 0.655 | 0.745 |
| YOLO-World · excavator | 0.80 | 0.006 | 0.803 | 0.682 | 1.318 |
| YOLO-World · construction excavator | 0.80 | 0.007 | 0.803 | 0.755 | 1.355 |
| Grounding DINO · excavator. | 0.85 | 0.197 | 0.854 | 0.845 | 1.536 |
| YOLO-World · excavator | 0.85 | 0.003 | 0.854 | 0.791 | 2.091 |
| YOLO-World · construction excavator | 0.85 | 0.002 | 0.854 | 0.836 | 2.391 |

## Interpretation guardrails

- The positive-control and confuser sets are confirmatory, not threshold-selection data.
- Curves show achievable post-hoc trade-offs on these frozen sets; they do not authorize choosing a new threshold from them.
- Raw confidence scores are not calibrated across architectures, so comparisons should use matched recall/FPR points rather than equal thresholds alone.
- Grounding DINO curves are valid down to 0.01 after rerunning the corrected confuser set at that floor; YOLO-World curves extend to their saved 0.001 floor.
- The confuser benchmark is curated and relatively small; category-level population claims require additional confirmation.
