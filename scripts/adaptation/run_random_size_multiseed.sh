#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/ssaffari/miniconda3/envs/yolo/bin/python"
TRAIN="/home/ssaffari/Data/foundational/adaptation/train_yolo_world.py"
MANIFESTS="/home/ssaffari/Data/AUTCONS-II/cvpr/outputs/manifests/random_scale"
RUNS="/home/ssaffari/Data/foundational/adaptation/runs_random_size_multiseed"
LOGS="${RUNS}/logs"

mkdir -p "${RUNS}" "${LOGS}"

# Freeze the selection rule before training or evaluating the new configurations.
cat > "${RUNS}/FROZEN_SELECTION_RULE.md" <<'EOF'
# Frozen subset-size selection rule

Candidate sizes: 5K, 10K, 20K, and 50K random Excavision images.

Each size must have three valid and independently seeded runs.

Selection is performed only on the MOCS development set:

1. Primary metric: mean AP50:95 across the three seeds.
2. A configuration is considered practically equivalent to the best when:
   - its mean AP50:95 is within 0.005 of the best mean, or
   - its 95% confidence interval overlaps that of the best configuration.
3. Among practically equivalent configurations, prefer higher mean recall at
   the locked 0.25 operating threshold.
4. If mean recall differs by no more than 0.01, prefer the smaller training set
   as the more data-efficient configuration.
5. CIS, positive control, and confuser results must not be used for subset-size
   selection. They are evaluated only after one size is frozen.
6. All runs use the same model, score-head adaptation, prompt, optimizer,
   batch size, image size, augmentations, epoch limit, and checkpoint policy.
EOF

# Record the exact source manifests before training.
sha256sum \
  "${MANIFESTS}/5k/train.txt" \
  "${MANIFESTS}/10k/train.txt" \
  "${MANIFESTS}/20k/train.txt" \
  "${MANIFESTS}/50k/train.txt" \
  > "${RUNS}/input_manifest_sha256.txt"

# Confirm the existing seed-0 checkpoints that will be reused.
test -f \
  "/home/ssaffari/Data/foundational/adaptation/runs_scale_curated_vs_random/yoloworld_scale_random_5k_score_head_seed0/weights/best.pt"

test -f \
  "/home/ssaffari/Data/foundational/adaptation/runs_scale_curated_vs_random/yoloworld_scale_random_20k_score_head_seed0/weights/best.pt"

train_run() {
    local size_label="$1"
    local expected_images="$2"
    local seed="$3"
    local gpu="$4"

    local manifest="${MANIFESTS}/${size_label}/train.txt"
    local name="yoloworld_scale_random_${size_label}_score_head_seed${seed}_seededloader"
    local log="${LOGS}/${name}.log"

    test -f "${manifest}"

    if [[ -e "${RUNS}/${name}" ]]; then
        echo "ERROR: output already exists: ${RUNS}/${name}" >&2
        return 1
    fi

    echo "Starting ${name} on GPU ${gpu}"

    "${PYTHON}" "${TRAIN}" \
      --arm score_head \
      --device "${gpu}" \
      --epochs 30 \
      --batch 64 \
      --workers 8 \
      --train-list "${manifest}" \
      --expected-images "${expected_images}" \
      --seed "${seed}" \
      --project "${RUNS}" \
      --name "${name}" \
      2>&1 | tee "${log}"

    echo "Completed ${name}"
}

# Work is balanced by the total number of training examples processed.
gpu0_queue() {
    train_run 50k 50000 0 0
    train_run 10k 10000 0 0
    train_run 50k 50000 2 0
    train_run 10k 10000 1 0
}

gpu1_queue() {
    train_run 50k 50000 1 1
    train_run 20k 20000 1 1
    train_run 20k 20000 2 1
    train_run 10k 10000 2 1
    train_run 5k 5000 1 1
    train_run 5k 5000 2 1
}

gpu0_queue &
PID_GPU0=$!

gpu1_queue &
PID_GPU1=$!

set +e
wait "${PID_GPU0}"
STATUS_GPU0=$?
wait "${PID_GPU1}"
STATUS_GPU1=$?
set -e

if [[ "${STATUS_GPU0}" -ne 0 || "${STATUS_GPU1}" -ne 0 ]]; then
    echo "At least one GPU queue failed."
    echo "GPU 0 status: ${STATUS_GPU0}"
    echo "GPU 1 status: ${STATUS_GPU1}"
    exit 1
fi

echo "All 10 new training runs completed successfully."
echo "Outputs: ${RUNS}"
