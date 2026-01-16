#!/bin/bash
set -euo pipefail

############################################
# Global paths (adjust only if needed)
############################################
SFT_ROOT="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs"
FINBEN_TASKS="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"
RESULTS_ROOT="results"

TASKS="PvExtraction_full"
DTYPE="bf16"
TP=2
GPU_MEM_UTIL=0.90
MAX_LEN=8096

############################################
# Models to merge + evaluate
############################################
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  #"Qwen/QwQ-32B-AWQ"
  #"meta-llama/Llama-3.1-8B-Instruct"
  #"meta-llama/Llama-3.2-3B-Instruct"
  #"Qwen/Qwen2.5-1.5B-Instruct"
)

############################################
# Helper: make filesystem-safe name
############################################
safe_name () {
  echo "$1" | tr '/:' '__'
}

############################################
# Loop over models
############################################
for BASE_MODEL in "${MODELS[@]}"; do
  NAME="$(safe_name "${BASE_MODEL}")"

  ADAPTER_DIR="${SFT_ROOT}/pv_sft_${NAME}_qlora_ddp2_epoch10/lora_adapter"
  MERGED_DIR="${SFT_ROOT}/pv_sft_${NAME}_merged_epoch10"
  OUT_PATH="${RESULTS_ROOT}/PV_qlora_${NAME}_merged_epoch10"

  echo "============================================================"
  echo "Model       : ${BASE_MODEL}"
  echo "Adapter dir : ${ADAPTER_DIR}"
  echo "Merged dir  : ${MERGED_DIR}"
  echo "Eval output : ${OUT_PATH}"
  echo "============================================================"

  # -------- sanity checks --------
  if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "SKIP: adapter dir not found: ${ADAPTER_DIR}" >&2
    continue
  fi

  if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    echo "SKIP: adapter_config.json missing in ${ADAPTER_DIR}" >&2
    continue
  fi

  mkdir -p "${MERGED_DIR}"

  ############################################
  # 1) Merge LoRA
  ############################################
  echo "[1/2] Merging LoRA → full model"
  python merge_lora.py \
    --base "${BASE_MODEL}" \
    --adapter "${ADAPTER_DIR}" \
    --out "${MERGED_DIR}" \
    --dtype "${DTYPE}"

  ############################################
  # 2) Evaluate merged model with FinBen
  ############################################
  echo "[2/2] Running lm_eval"
  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_LEN}" \
    --tasks "${TASKS}" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${OUT_PATH}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS}"

  echo "DONE: ${BASE_MODEL}"
  echo
done

echo "ALL MODELS FINISHED."
