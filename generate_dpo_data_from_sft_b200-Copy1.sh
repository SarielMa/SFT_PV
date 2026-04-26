#!/usr/bin/env bash

set -euo pipefail

REQUESTED_MODEL="${1:-}"

SFT_RESULTS_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/sft_3epoch"
DPO_PIPELINE_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs"
DATA_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test"

TP=2
MAX_TOKENS=8192
TEMPERATURE=0.0
NEG_PER_SAMPLE=1
SEED=42
PRINT_SAMPLES=3

MODELS=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.3-70B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
)

declare -A SFT_MODEL_DIRS=(
  ["Llama-3.1-8B-Instruct"]="${SFT_RESULTS_DIR}/merged_llama3.1_8b_instruct_sft_3ep"
  ["Llama-3.2-3B-Instruct"]="${SFT_RESULTS_DIR}/merged_llama3.2_3b_instruct_sft_3ep"
  ["Llama-3.3-70B-Instruct"]="${SFT_RESULTS_DIR}/merged_void_llama3.3_70b_instruct_sft_3ep"
  ["Qwen2.5-1.5B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_1.5b_instruct_sft_3ep"
  ["Qwen2.5-7B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_7b_instruct_sft_3ep"
  ["Qwen2.5-14B-Instruct"]="${SFT_RESULTS_DIR}/merged_qwen2.5_14b_instruct_sft_3ep"
)

mkdir -p "${DPO_PIPELINE_DIR}"

for MODEL_ID in "${MODELS[@]}"; do
  if [[ -n "${REQUESTED_MODEL}" && "${MODEL_ID}" != "${REQUESTED_MODEL}" ]]; then
    continue
  fi

  SFT_MODEL="${SFT_MODEL_DIRS[${MODEL_ID}]:-}"
  if [[ -z "${SFT_MODEL}" ]]; then
    echo "Skipping MODEL=${MODEL_ID}: no SFT model mapping configured"
    continue
  fi

  if [[ ! -f "${SFT_MODEL}/config.json" ]]; then
    echo "Skipping MODEL=${MODEL_ID}: merged SFT model not found at ${SFT_MODEL}"
    continue
  fi

  OUT_TAG="${MODEL_ID}_epoch3_sftMerged"
  OUT_ROOT="${DPO_PIPELINE_DIR}/${OUT_TAG}"
  CONF_DIR="${OUT_ROOT}/confusion"
  PRED_DIR="${OUT_ROOT}/pred"
  DPO_DATA_DIR="${OUT_ROOT}/dpo_data"

  mkdir -p "${CONF_DIR}" "${PRED_DIR}" "${DPO_DATA_DIR}"

  CODE_CONF_CSV="${CONF_DIR}/code_confusion_summary.csv"
  SUBCODE_CONF_CSV="${CONF_DIR}/subcode_confusion_summary.csv"
  PRED_JSONL="${PRED_DIR}/pred_dump.jsonl"

  echo "============================================================"
  echo "MODEL      : ${MODEL_ID}"
  echo "SFT_MODEL  : ${SFT_MODEL}"
  echo "OUT_ROOT   : ${OUT_ROOT}"
  echo "TP         : ${TP}"
  echo "============================================================"

  python infer_vllm_and_confusion.py \
    --model "${SFT_MODEL}" \
    --data "${DATA_DIR}" \
    --out_code_csv "${CODE_CONF_CSV}" \
    --out_subcode_csv "${SUBCODE_CONF_CSV}" \
    --tp "${TP}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --out_pred_jsonl "${PRED_JSONL}"

  python prepare_dpo_data.py \
    --input_dir "${DATA_DIR}" \
    --output_dir "${DPO_DATA_DIR}" \
    --code_confusion_file "${CODE_CONF_CSV}" \
    --subcode_confusion_file "${SUBCODE_CONF_CSV}" \
    --negatives_per_sample "${NEG_PER_SAMPLE}" \
    --seed "${SEED}" \
    --print_samples "${PRINT_SAMPLES}"
done

if [[ -n "${REQUESTED_MODEL}" ]]; then
  echo "Finished DPO data generation for: ${REQUESTED_MODEL}"
else
  echo "Finished DPO data generation for all configured SFT models."
fi
