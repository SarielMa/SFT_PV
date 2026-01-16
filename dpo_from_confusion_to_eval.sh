#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG (edit if needed)
# =========================
SFT_MODEL="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/runs/pv_sft_Qwen2.5-1.5B-Instruct_merged_epoch10"
DATA_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"
FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

# One knob to rule them all (keep these the same)
TP=1
NUM_GPUS="${TP}"
TENSOR_PARALLEL_SIZE="${TP}"

MAX_TOKENS=8192
TEMPERATURE=0.0
MAX_MODEL_LEN=8096
GPU_MEM_UTIL=0.90

NEG_PER_SAMPLE=1
SEED=42
PRINT_SAMPLES=3

# =========================
# DERIVED OUTPUT FOLDERS
# =========================
SFT_TAG="$(basename "${SFT_MODEL}")"
# Put all outputs under a folder tied to the SFT model name
OUT_ROOT="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs/${SFT_TAG}"

CONF_DIR="${OUT_ROOT}/confusion"
PRED_DIR="${OUT_ROOT}/pred"
DPO_DATA_DIR="${OUT_ROOT}/dpo_data"
DPO_RUNS_DIR="${OUT_ROOT}/dpo_runs"
EVAL_DIR="${OUT_ROOT}/lm_eval_results"

mkdir -p "${CONF_DIR}" "${PRED_DIR}" "${DPO_DATA_DIR}" "${DPO_RUNS_DIR}" "${EVAL_DIR}"

CODE_CONF_CSV="${CONF_DIR}/code_confusion_summary.csv"
SUBCODE_CONF_CSV="${CONF_DIR}/subcode_confusion_summary.csv"
PRED_JSONL="${PRED_DIR}/pred_dump.jsonl"

# Name the DPO run folders based on SFT tag (so you can run multiple SFTs safely)
DPO_RUN_NAME="dpo_${SFT_TAG}"
DPO_OUTPUT_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}"
MERGED_DIR="${DPO_RUNS_DIR}/${DPO_RUN_NAME}-merged"

echo "SFT_MODEL: ${SFT_MODEL}"
echo "OUT_ROOT : ${OUT_ROOT}"
echo "TP/GPUS  : TP=${TP} NUM_GPUS=${NUM_GPUS} TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo

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
# 4) Merge LoRA adapter
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

echo
echo "Done. All outputs are under: ${OUT_ROOT}"
