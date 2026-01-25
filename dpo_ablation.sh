#!/usr/bin/env bash
set -euo pipefail

# =========================
# GLOBAL CONFIG
# =========================

# Fixed SFT merged model (ablation base)
SFT_MODEL="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs_pv/Qwen2.5-1.5B-Instruct/epoch3/merged"
SFT_MODEL="$(readlink -f "${SFT_MODEL}")"

# This is the EXISTING prepared DPO/TAB-DPO training data directory
TAB_DPO_DATA_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs/Qwen2.5-1.5B-Instruct_epoch3_sftMerged/dpo_data"
TAB_DPO_DATA_DIR="$(readlink -f "${TAB_DPO_DATA_DIR}")"

# NEW output root for ablation runs (CHANGE IF YOU WANT)
ABL_OUT_ROOT="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/tab_dpo_ablation_outputs"
ABL_OUT_ROOT="$(readlink -f "${ABL_OUT_ROOT}")"

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

# One knob to rule them all
TP=2
NUM_GPUS="${TP}"
TENSOR_PARALLEL_SIZE="${TP}"

MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90

TASK_NAME="PvExtraction_full"
FEWSHOT=0

mkdir -p "${ABL_OUT_ROOT}"

# =========================
# DEFAULTS (must match tab_dpo_ablation.py)
# =========================
# DEFAULT_POLICY_PROB_THRESHOLD="0.66"
# DEFAULT_ENABLE_CLASS_BALANCE="False"
# DEFAULT_ENABLE_TOKEN_WEIGHTING="True"
# DEFAULT_ENABLE_LENGTH_NORM="True"

# =========================
# SANITY CHECKS
# =========================
if [[ ! -f "${SFT_MODEL}/config.json" ]]; then
  echo "ERROR: merged SFT model not found:"
  echo "  ${SFT_MODEL}"
  exit 1
fi

if [[ ! -d "${TAB_DPO_DATA_DIR}" ]]; then
  echo "ERROR: TAB-DPO training data dir not found:"
  echo "  ${TAB_DPO_DATA_DIR}"
  exit 1
fi

echo "============================================================"
echo "BASE SFT_MODEL      : ${SFT_MODEL}"
echo "TAB_DPO_DATA_DIR    : ${TAB_DPO_DATA_DIR}"
echo "ABL_OUT_ROOT        : ${ABL_OUT_ROOT}"
echo "TP/GPUS             : TP=${TP} NUM_GPUS=${NUM_GPUS}"
echo "============================================================"

# =========================
# ABLATION GRID
# Rule: Only one knob deviates from default per run.
# =========================

# Each entry: "TAG|EXTRA_ARGS..."
# TAG is used in output folder naming.
ABLATIONS=(
  # Baseline = all false (optional but useful)
  #"all_false|"

  # # enable_class_balance ablation (default False -> True)
  "class_balance=True|--enable_class_balance True"

  # # enable_token_weighting ablation (default True -> False)
  "token_weighting=True|--enable_token_weighting True"

  # # enable_length_norm ablation (default True -> False)
  "length_norm=True|--enable_length_norm True"

  # # policy_prob_threshold ablations (default 0.66 -> other values)
  "policy_prob_threshold=0.8|--policy_prob_threshold 0.8"
  "policy_prob_threshold=0.9|--policy_prob_threshold 0.9"
  "policy_prob_threshold=0.99|--policy_prob_threshold 0.99"
)

# =========================
# RUN ABLATIONS
# =========================
for ENTRY in "${ABLATIONS[@]}"; do
  TAG="${ENTRY%%|*}"
  EXTRA_ARGS="${ENTRY#*|}"

  RUN_TAG="Qwen2.5-1.5B_epoch3_merged__${TAG}"
  OUT_ROOT="${ABL_OUT_ROOT}/${RUN_TAG}"

  DPO_RUNS_DIR="${OUT_ROOT}/dpo_runs"
  EVAL_DIR="${OUT_ROOT}/lm_eval_results"
  mkdir -p "${DPO_RUNS_DIR}" "${EVAL_DIR}"

  DPO_OUTPUT_DIR="${DPO_RUNS_DIR}/tab_dpo_${RUN_TAG}"
  MERGED_DIR="${DPO_RUNS_DIR}/tab_dpo_${RUN_TAG}-merged"

  echo
  echo "------------------------------------------------------------"
  echo "RUN_TAG         : ${RUN_TAG}"
  echo "EXTRA_ARGS      : ${EXTRA_ARGS}"
  echo "DPO_OUTPUT_DIR  : ${DPO_OUTPUT_DIR}"
  echo "MERGED_DIR      : ${MERGED_DIR}"
  echo "EVAL_DIR        : ${EVAL_DIR}"
  echo "------------------------------------------------------------"

  # =========================
  # 1) Train TAB-DPO with ablation setting
  # =========================
  python tab_dpo_ablation.py \
    --model_name "${SFT_MODEL}" \
    --train_data_path "${TAB_DPO_DATA_DIR}" \
    --valid_data_path "${TAB_DPO_DATA_DIR}" \
    --output_dir "${DPO_OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    ${EXTRA_ARGS}

  # =========================
  # 2) Merge LoRA adapter
  # =========================
  python merge_lora.py \
    --base "${SFT_MODEL}" \
    --adapter "${DPO_OUTPUT_DIR}" \
    --out "${MERGED_DIR}" \
    --dtype bf16

  # =========================
  # 3) Eval (lm_eval + vLLM)
  # =========================
  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks "${TASK_NAME}" \
    --num_fewshot "${FEWSHOT}" \
    --batch_size auto \
    --output_path "${EVAL_DIR}/${TASK_NAME}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}"

  echo "✔ DONE: ${RUN_TAG}"
done

echo
echo "All TAB-DPO ablation runs finished. Outputs under:"
echo "  ${ABL_OUT_ROOT}"
