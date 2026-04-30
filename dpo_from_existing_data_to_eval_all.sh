#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# GLOBAL CONFIG
# =========================
DPO_DATA_ROOT="$(readlink -f "${REPO_ROOT}/dpo_pipeline_outputs")"
SFT_RESULTS_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/sft_3epoch"
DPO_OUT_ROOT="${REPO_ROOT}/dpo_runs_from_existing_data"

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=8192
SEEDS=(42 123 2024)

# =========================
# MODELS
# =========================
MODELS=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Llama-3.3-70B-Instruct"
)

declare -A SFT_MODEL_DIRS=(
  ["Llama-3.1-8B-Instruct"]="${SFT_RESULTS_DIR}/merged_llama3.1_8b_instruct_sft_3ep"
  ["Llama-3.2-3B-Instruct"]="${SFT_RESULTS_DIR}/merged_llama3.2_3b_instruct_sft_3ep"
  ["Llama-3.3-70B-Instruct"]="${SFT_RESULTS_DIR}/merged_void_llama3.3_70b_instruct_sft_3ep"
  ["Qwen2.5-1.5B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_1.5b_instruct_sft_3ep"
  ["Qwen2.5-7B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_7b_instruct_sft_3ep"
  ["Qwen2.5-14B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_14b_instruct_sft_3ep"
)

mkdir -p "${DPO_OUT_ROOT}"
DPO_OUT_ROOT="$(readlink -f "${DPO_OUT_ROOT}")"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi

if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
  echo "ERROR: Unable to detect available GPUs."
  exit 1
fi

TP="${NUM_GPUS}"
TENSOR_PARALLEL_SIZE="${NUM_GPUS}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: Missing ${label}: ${path}"
    exit 1
  fi
}

dataset_loadable() {
  local path="$1"
  python - <<'PY' "${path}"
import sys
from datasets import load_from_disk

load_from_disk(sys.argv[1])
print("OK")
PY
}

require_path "${DPO_DATA_ROOT}" "DPO data root"
require_path "${SFT_RESULTS_DIR}" "SFT results root"
require_path "${FINBEN_TASKS_PATH}" "FinBen tasks path"
require_path "${REPO_ROOT}/train_dpo.py" "DPO training script"
require_path "${REPO_ROOT}/merge_lora.py" "LoRA merge script"

echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "Using TP=${TP} and TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"

# =========================
# MAIN LOOP
# =========================
for MODEL_TAG in "${MODELS[@]}"; do
  SFT_MODEL="${SFT_MODEL_DIRS[${MODEL_TAG}]:-}"
  OUT_TAG="${MODEL_TAG}_epoch3_sftMerged"
  DPO_DATA_DIR="${DPO_DATA_ROOT}/${OUT_TAG}/dpo_data"

  if [[ -z "${SFT_MODEL}" ]]; then
    echo "Skipping ${MODEL_TAG}: no SFT model mapping configured"
    continue
  fi

  if [[ ! -f "${SFT_MODEL}/config.json" ]]; then
    echo "Skipping ${MODEL_TAG}: merged SFT model not found at ${SFT_MODEL}"
    continue
  fi

  if [[ ! -d "${DPO_DATA_DIR}" ]]; then
    echo "Skipping ${MODEL_TAG}: DPO data directory not found at ${DPO_DATA_DIR}"
    continue
  fi

  if ! dataset_loadable "${DPO_DATA_DIR}" >/dev/null 2>&1; then
    echo "Skipping ${MODEL_TAG}: DPO data is not a valid saved dataset at ${DPO_DATA_DIR}"
    continue
  fi

  for SEED in "${SEEDS[@]}"; do
    OUT_ROOT="${DPO_OUT_ROOT}/${OUT_TAG}/seed_${SEED}"
    DPO_RUNS_DIR="${OUT_ROOT}/dpo_runs"
    EVAL_DIR="${OUT_ROOT}/lm_eval_results"

    mkdir -p "${DPO_RUNS_DIR}" "${EVAL_DIR}"

    DPO_RUN_NAME="dpo_${OUT_TAG}_seed${SEED}"
    DPO_OUTPUT_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}"
    MERGED_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}-merged"

    echo "============================================================"
    echo "MODEL      : ${MODEL_TAG}"
    echo "SEED       : ${SEED}"
    echo "SFT_MODEL  : ${SFT_MODEL}"
    echo "DPO_DATA   : ${DPO_DATA_DIR}"
    echo "OUT_ROOT   : ${OUT_ROOT}"
    echo "TP/GPUS    : TP=${TP} NUM_GPUS=${NUM_GPUS}"
    echo "============================================================"

    export PYTHONHASHSEED="${SEED}"

    # =========================
    # 1) Train DPO (LoRA)
    # =========================
    python "${REPO_ROOT}/train_dpo.py" \
      --model_name "${SFT_MODEL}" \
      --train_data_path "${DPO_DATA_DIR}" \
      --valid_data_path "${DPO_DATA_DIR}" \
      --output_dir "${DPO_OUTPUT_DIR}" \
      --num_gpus "${NUM_GPUS}" \
      --seed "${SEED}"

    # =========================
    # 2) Merge DPO adapter
    # =========================
    python "${REPO_ROOT}/merge_lora.py" \
      --base "${SFT_MODEL}" \
      --adapter "${DPO_OUTPUT_DIR}" \
      --out "${MERGED_DIR}" \
      --dtype bf16

    # =========================
    # 3) Eval (lm_eval + vLLM)
    # =========================
    lm_eval --model vllm \
      --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},enforce_eager=True" \
      --tasks PvExtraction_full \
      --num_fewshot 0 \
      --batch_size auto \
      --output_path "${EVAL_DIR}/PvExtraction_full" \
      --log_samples \
      --apply_chat_template \
      --include_path "${FINBEN_TASKS_PATH}"

    echo "DONE: ${MODEL_TAG} seed=${SEED}"
  done
done

echo
echo "All DPO runs finished. Outputs under:"
echo "  ${DPO_OUT_ROOT}"
