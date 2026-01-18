#!/bin/bash
set -euo pipefail

# =========================
# Global settings
# =========================
DATASET_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"
PY_SCRIPT="sft_peft_ddp.py"
FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

MAX_LEN=8192
BATCH_SIZE=1
EPOCHS=3
LR=2e-4

# SINGLE source of truth
TP=2   # = num_gpus = tensor_parallel_size

# Use absolute path to avoid cwd surprises on HPC
PIPELINE_ROOT="$(readlink -f ./runs_pv)"

# =========================
# Models (HF-style, easy to modify)
# =========================
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
)

# =========================
# Grad-accum heuristic
# =========================
guess_grad_accum () {
  case "$1" in
    meta-llama/Llama-3.3-70B-Instruct) echo 4 ;;
    meta-llama/Llama-3.1-8B-Instruct)  echo 8 ;;
    meta-llama/Llama-3.2-3B-Instruct)  echo 8 ;;
    Qwen/Qwen2.5-1.5B-Instruct)         echo 8 ;;
    *)                                  echo 8 ;;
  esac
}

# =========================
# NCCL safety (H100/H200)
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

mkdir -p "${PIPELINE_ROOT}"

# =========================
# Main loop
# =========================
for MODEL in "${MODELS[@]}"; do

  # Skip AWQ for training
  if [[ "${MODEL}" == *"AWQ"* ]]; then
    echo "Skip ${MODEL} (AWQ not suitable for QLoRA training)"
    continue
  fi

  MODEL_TAG="$(basename "${MODEL}")"
  RUN_TAG="epoch${EPOCHS}"

  MODEL_DIR="${PIPELINE_ROOT}/${MODEL_TAG}/${RUN_TAG}"
  ADAPTER_DIR="${MODEL_DIR}/sft_adapter"
  MERGED_DIR="${MODEL_DIR}/merged"
  LOG_DIR="${MODEL_DIR}/logs"

  # IMPORTANT: use a dedicated eval folder like your working script
  EVAL_DIR="${MODEL_DIR}/lm_eval_results"

  mkdir -p "${ADAPTER_DIR}" "${MERGED_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

  # Make them absolute too
  MODEL_DIR="$(readlink -f "${MODEL_DIR}")"
  ADAPTER_DIR="$(readlink -f "${ADAPTER_DIR}")"
  MERGED_DIR="$(readlink -f "${MERGED_DIR}")"
  EVAL_DIR="$(readlink -f "${EVAL_DIR}")"
  LOG_DIR="$(readlink -f "${LOG_DIR}")"

  GA="$(guess_grad_accum "${MODEL}")"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Folder: ${MODEL_DIR}"
  echo "TP=${TP} | GA=${GA} | epochs=${EPOCHS} | lr=${LR}"
  echo "Adapter: ${ADAPTER_DIR}"
  echo "Merged : ${MERGED_DIR}"
  echo "Eval   : ${EVAL_DIR}"
  echo "============================================================"

  # --------------------------------------------------
  # 1) SFT (QLoRA, DDP)   (currently commented out as you had)
  # --------------------------------------------------
  # torchrun --nproc_per_node="${TP}" "${PY_SCRIPT}" \
  #   --dataset_path "${DATASET_PATH}" \
  #   --model_name "${MODEL}" \
  #   --output_dir "${ADAPTER_DIR}" \
  #   --use_qlora --bf16 \
  #   --max_length "${MAX_LEN}" \
  #   --batch_size "${BATCH_SIZE}" \
  #   --grad_accum "${GA}" \
  #   --epochs "${EPOCHS}" \
  #   --lr "${LR}" \
  #   2>&1 | tee "${LOG_DIR}/sft.log"

  # --------------------------------------------------
  # 2) Merge LoRA → full model
  # --------------------------------------------------
  ADAPTER_PATH="${ADAPTER_DIR}/lora_adapter"
  if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    ADAPTER_PATH="${ADAPTER_DIR}"
  fi

  echo "Using adapter path: ${ADAPTER_PATH}"

  python merge_lora.py \
    --base "${MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --out "${MERGED_DIR}" \
    --dtype bf16 \
    2>&1 | tee "${LOG_DIR}/merge.log"

  # --------------------------------------------------
  # 3) Evaluation (lm-eval + vLLM)
  #    MATCH the known-good behavior:
  #    output_path is a PREFIX, not a directory.
  # --------------------------------------------------
  TASK="PvExtraction_full"

  # Ensure we do NOT have a directory that collides with the prefix.
  # This is a common cause of "saved" but nothing where you expect.
  rm -rf "${EVAL_DIR}/${TASK}"

  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TP},gpu_memory_utilization=0.9,max_model_len=${MAXKAX_LEN:-${MAX_LEN}}" \
    --tasks "${TASK}" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${EVAL_DIR}/${TASK}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"

  echo "Eval outputs under: ${EVAL_DIR}"
  echo "Files produced (if any):"
  find "${EVAL_DIR}" -maxdepth 1 -type f -name "*${TASK}*" -print || true

  echo "✔ DONE: ${MODEL_TAG}"
done
