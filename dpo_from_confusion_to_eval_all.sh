#!/usr/bin/env bash
set -euo pipefail

# =========================
# GLOBAL CONFIG
# =========================
SFT_RUNS_ROOT="$(readlink -f ./runs_pv)"   # SFT pipeline output root
DPO_OUT_ROOT="$(readlink -f ./dpo_pipeline_outputs)"

DATA_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"
FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

EPOCHS=3   # must match SFT

# One knob to rule them all
TP=1
NUM_GPUS="${TP}"
TENSOR_PARALLEL_SIZE="${TP}"

MAX_TOKENS=8192
TEMPERATURE=0.0
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90

NEG_PER_SAMPLE=1
SEED=42
PRINT_SAMPLES=3

# =========================
# MODELS (same as SFT)
# =========================
MODELS=(
  #"meta-llama/Llama-3.3-70B-Instruct"
  #"meta-llama/Llama-3.1-8B-Instruct"
  #"meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

mkdir -p "${DPO_OUT_ROOT}"

# =========================
# MAIN LOOP
# =========================
for MODEL in "${MODELS[@]}"; do

  MODEL_TAG="$(basename "${MODEL}")"

  # --------------------------------------------------
  # Locate merged SFT model
  # --------------------------------------------------
  SFT_MODEL="${SFT_RUNS_ROOT}/${MODEL_TAG}/epoch${EPOCHS}/merged"
  SFT_MODEL="$(readlink -f "${SFT_MODEL}")"

  if [[ ! -f "${SFT_MODEL}/config.json" ]]; then
    echo "ERROR: merged SFT model not found:"
    echo "  ${SFT_MODEL}"
    echo "Skipping ${MODEL}"
    continue
  fi

  # --------------------------------------------------
  # Output folders (one per SFT model)
  # --------------------------------------------------
  OUT_TAG="${MODEL_TAG}_epoch${EPOCHS}_sftMerged"
  OUT_ROOT="${DPO_OUT_ROOT}/${OUT_TAG}"

  CONF_DIR="${OUT_ROOT}/confusion"
  PRED_DIR="${OUT_ROOT}/pred"
  DPO_DATA_DIR="${OUT_ROOT}/dpo_data"
  DPO_RUNS_DIR="${OUT_ROOT}/dpo_runs"
  EVAL_DIR="${OUT_ROOT}/lm_eval_results"

  mkdir -p "${CONF_DIR}" "${PRED_DIR}" "${DPO_DATA_DIR}" "${DPO_RUNS_DIR}" "${EVAL_DIR}"

  CODE_CONF_CSV="${CONF_DIR}/code_confusion_summary.csv"
  SUBCODE_CONF_CSV="${CONF_DIR}/subcode_confusion_summary.csv"
  PRED_JSONL="${PRED_DIR}/pred_dump.jsonl"

  DPO_RUN_NAME="dpo_${OUT_TAG}"
  DPO_OUTPUT_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}"
  MERGED_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}-merged"

  echo "============================================================"
  echo "MODEL      : ${MODEL}"
  echo "SFT_MODEL  : ${SFT_MODEL}"
  echo "OUT_ROOT   : ${OUT_ROOT}"
  echo "TP/GPUS    : TP=${TP} NUM_GPUS=${NUM_GPUS}"
  echo "============================================================"

  # =========================
  # 1) Infer + confusion
  # =========================
  python infer_vllm_and_confusion.py \
    --model "${SFT_MODEL}" \
    --data  "${DATA_DIR}" \
    --out_code_csv "${CODE_CONF_CSV}" \
    --out_subcode_csv "${SUBCODE_CONF_CSV}" \
    --tp "${TP}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --out_pred_jsonl "${PRED_JSONL}"

  # =========================
  # 2) Prepare DPO data
  # =========================
  python prepare_dpo_data.py \
    --input_dir "${DATA_DIR}" \
    --output_dir "${DPO_DATA_DIR}" \
    --code_confusion_file "${CODE_CONF_CSV}" \
    --subcode_confusion_file "${SUBCODE_CONF_CSV}" \
    --negatives_per_sample "${NEG_PER_SAMPLE}" \
    --seed "${SEED}" \
    --print_samples "${PRINT_SAMPLES}"

  # =========================
  # 3) Train DPO (LoRA)
  # =========================
  python train_dpo.py \
    --model_name "${SFT_MODEL}" \
    --train_data_path "${DPO_DATA_DIR}" \
    --valid_data_path "${DPO_DATA_DIR}" \
    --output_dir "${DPO_OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}"

  # =========================
  # 4) Merge DPO adapter
  # =========================
  python merge_lora.py \
    --base "${SFT_MODEL}" \
    --adapter "${DPO_OUTPUT_DIR}" \
    --out "${MERGED_DIR}" \
    --dtype bf16

  # =========================
  # 5) Eval (lm_eval + vLLM)
  # =========================
  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks PvExtraction_full \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${EVAL_DIR}/PvExtraction_full" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}"

  echo "✔ DONE: ${MODEL_TAG}"
done

echo
echo "All DPO runs finished. Outputs under:"
echo "  ${DPO_OUT_ROOT}"
