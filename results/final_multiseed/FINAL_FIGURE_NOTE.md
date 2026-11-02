# Final figure note

The figure compares the fixed zero-shot YOLO-World model with selective score-head adaptation on the random 100K Excavision subset. All evaluations use the same prompt (`construction excavator`), prediction floor (0.001), operating threshold (0.25), image size, and NMS configuration.

## Quantitative reading

- **MOCS:** ΔAP50 +0.023, ΔAP50:95 +0.024, and Δrecall +0.115.
- **CIS:** ΔAP50 -0.026, ΔAP50:95 -0.002, and Δrecall +0.102.
- **Positive control:** ΔAP50 +0.021, ΔAP50:95 -0.005, and Δrecall +0.119.
- **Confusers:** image FPR changes from 0.164 to 0.173 ± 0.081.

## Defensible interpretation

Positive-only synthetic adaptation consistently raises recall at the fixed operating point across all three real-image sources. Ranking-based AP is not uniformly improved: the clearest gains occur on MOCS, CIS AP50 decreases, and AP50:95 on CIS and the positive-control set is approximately unchanged or slightly lower. Confuser FPR is not reduced and is seed-sensitive. Accordingly, the experiment supports a targeted confidence/recall adaptation claim, not a claim of universal detection or category-boundary improvement.
