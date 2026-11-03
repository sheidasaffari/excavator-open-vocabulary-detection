#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="/home/ssaffari/miniconda3/envs/yolo/bin/python"
EVALUATOR="/home/ssaffari/Data/foundational/adaptation/evaluate_adapted.py"
OLD_RUNS="/home/ssaffari/Data/foundational/adaptation/runs_scale_curated_vs_random"
NEW_RUNS="/home/ssaffari/Data/foundational/adaptation/runs_random_size_multiseed"
RUNS_100K="/home/ssaffari/Data/foundational/adaptation/runs_selected_multiseed_seededloader"
OUTPUT="/home/ssaffari/Data/foundational/adaptation/evaluation_random_size_multiseed_mocs"
LOGS="${OUTPUT}/logs"

if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite existing output: ${OUTPUT}" >&2
    exit 1
fi
mkdir -p "${LOGS}"

checkpoint_for() {
    local size="$1" seed="$2"
    case "${size}:${seed}" in
        5k:0|20k:0|100k:0)
            echo "${OLD_RUNS}/yoloworld_scale_random_${size}_score_head_seed0/weights/best.pt"
            ;;
        100k:1|100k:2)
            echo "${RUNS_100K}/yoloworld_selected_random_100k_score_head_seed${seed}_seededloader/weights/best.pt"
            ;;
        *)
            echo "${NEW_RUNS}/yoloworld_scale_random_${size}_score_head_seed${seed}_seededloader/weights/best.pt"
            ;;
    esac
}

evaluate_one() {
    local size="$1" seed="$2" gpu="$3"
    local checkpoint arm
    checkpoint="$(checkpoint_for "${size}" "${seed}")"
    arm="random_${size}_seed${seed}"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 1; }
    "${PYTHON}" "${EVALUATOR}" \
        --checkpoint "${checkpoint}" \
        --arm "${arm}" \
        --device "${gpu}" \
        --batch 64 \
        --datasets mocs_dev \
        --output-root "${OUTPUT}" \
        2>&1 | tee "${LOGS}/${arm}.log"
}

gpu0_queue() {
    evaluate_one 5k 0 0
    evaluate_one 5k 2 0
    evaluate_one 10k 1 0
    evaluate_one 20k 0 0
    evaluate_one 20k 2 0
    evaluate_one 50k 1 0
    evaluate_one 100k 0 0
    evaluate_one 100k 2 0
}

gpu1_queue() {
    evaluate_one 5k 1 1
    evaluate_one 10k 0 1
    evaluate_one 10k 2 1
    evaluate_one 20k 1 1
    evaluate_one 50k 0 1
    evaluate_one 50k 2 1
    evaluate_one 100k 1 1
}

gpu0_queue & PID0=$!
gpu1_queue & PID1=$!

status=0
wait "${PID0}" || status=$?
wait "${PID1}" || status=$?
exit "${status}"
