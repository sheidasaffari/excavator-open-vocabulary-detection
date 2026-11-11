#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ssaffari/.codex/.chatgpt-projects/g-p-6944423877d08191ac21a1cee9e7dc1d/foundation_journal_study"
OUT="/home/ssaffari/Data/foundational/journal_study"
LOGS="$OUT/logs_phase1"
STATUS="$OUT/status_phase1"
YOLO_PY="/home/ssaffari/miniconda3/envs/yolo/bin/python"
MMDET_PY="/home/ssaffari/miniconda3/envs/mmdet/bin/python"
mkdir -p "$LOGS" "$STATUS"

if [[ ! -f "$OUT/data_phase1_v2/AUDIT.json" ]]; then
  echo "Missing prepared data: $OUT/data_phase1_v2/AUDIT.json" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 2 ]]; then
  echo "Two visible GPUs are required" >&2
  exit 2
fi

pair_commands() {
  local pair="$1"
  local env0="$2"
  local env1="$3"
  mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [[ "${free_mib[0]}" -lt 24000 || "${free_mib[1]}" -lt 24000 ]]; then
    echo "Insufficient free VRAM before $pair: GPU0=${free_mib[0]} MiB GPU1=${free_mib[1]} MiB" >&2
    exit 3
  fi
  echo "[$(date -Is)] START $pair" | tee "$STATUS/${pair}.started"
  bash -lc "CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 $env0" >"$LOGS/${pair}_gpu0.log" 2>&1 &
  local pid0=$!
  bash -lc "CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 $env1" >"$LOGS/${pair}_gpu1.log" 2>&1 &
  local pid1=$!
  local rc0=0 rc1=0
  wait "$pid0" || rc0=$?
  wait "$pid1" || rc1=$?
  if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED $pair gpu0=$rc0 gpu1=$rc1" | tee "$STATUS/${pair}.failed"
    exit 1
  fi
  echo "[$(date -Is)] COMPLETE $pair" | tee "$STATUS/${pair}.complete"
}

# One-epoch end-to-end smoke pair. A failure stops the expensive queue.
pair_commands "00_smoke_models" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n2 --scope alignment --device 0 --seed 0 --smoke" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n2 --scope fusion --seed 0 --smoke"

# YOLOE-v8-S: fixed default for N0/N1/N2 is open-prompt alignment.
pair_commands "01_yoloe_rep_n0_n1" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n0 --scope alignment --device 0 --seed 0" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n1 --scope alignment --device 0 --seed 0"

# N2 alignment completes the representation screen; terminal begins locus screen.
pair_commands "02_yoloe_n2_alignment_terminal" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n2 --scope alignment --device 0 --seed 0" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n2 --scope terminal --device 0 --seed 0"

pair_commands "03_yoloe_n2_panhead_full" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n2 --scope panhead --device 0 --seed 0" \
  "$YOLO_PY $ROOT/train_yoloe_phase1.py --representation n2 --scope full --device 0 --seed 0"

# MM-Grounding-DINO-T: fixed default for N0/N1/N2 is official fusion scope.
pair_commands "04_gdino_rep_n0_n1" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n0 --scope fusion --seed 0" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n1 --scope fusion --seed 0"

# N2 fusion completes the representation screen; head begins locus screen.
pair_commands "05_gdino_n2_fusion_head" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n2 --scope fusion --seed 0" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n2 --scope head --seed 0"

pair_commands "06_gdino_n2_visionfusion_full" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n2 --scope visionfusion --seed 0" \
  "$MMDET_PY $ROOT/train_gdino_phase1.py --representation n2 --scope full --seed 0"

echo "[$(date -Is)] PHASE 1 COMPLETE — stop at selection gate" | tee "$STATUS/PHASE1_COMPLETE"
echo "Do not run confirmatory tests yet. Evaluate MOCS and ACID-dev, then freeze representation and scope."
