#!/bin/bash

set -e

########################################
# Base configs
########################################

MODEL_NAME="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/sft_results_epoch3/Qwen2.5-1.5B-Instruct/epoch3/merged"
DATA_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs/Qwen2.5-1.5B-Instruct_epoch3_sftMerged/dpo_data"

TRAIN_SCRIPT="train_dpo_ablation.py"
MERGE_SCRIPT="merge_lora.py"

# 🔥 beta sweep
BETAS=(0.80 0.92 0.93)
# BETAS=(0.99)
# eval configs
TENSOR_PARALLEL_SIZE=2
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=8192

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

BASE_OUT_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/po_ablation_runs"

########################################
# Loop over beta
########################################

for BETA in "${BETAS[@]}"; do

  echo "========================================"
  echo "Running BETA=${BETA}"
  echo "========================================"

  RUN_DIR="${BASE_OUT_DIR}/beta_${BETA}"
  MODEL_OUT="${RUN_DIR}/model"
  MERGED_DIR="${RUN_DIR}/merged_model"
  EVAL_DIR="${RUN_DIR}/eval"

  mkdir -p "${MODEL_OUT}"
  mkdir -p "${MERGED_DIR}"
  mkdir -p "${EVAL_DIR}"

  # export CUDA_VISIBLE_DEVICES=0,1

  ########################################
  # Train (LoRA)
  ########################################
  torchrun --nproc_per_node=${TENSOR_PARALLEL_SIZE} ${TRAIN_SCRIPT} \
    --model_name "${MODEL_NAME}" \
    --train_data_path "${DATA_PATH}" \
    --valid_data_path "${DATA_PATH}" \
    --output_dir "${MODEL_OUT}" \
    --num_gpus 1 \
    --system_prompt "You are a helpful assistant." \
    \
    --enable_length_norm False \
    --length_norm_by "tokens" \
    \
    --enable_token_weighting False \
    --token_weight_code 1.1 \
    --token_weight_subcode 1.2 \
    --token_weight_span 1.1 \
    --normalize_by_weight_mass True \
    \
    --enable_class_balance True \
    --class_balance_strategy "effective_num" \
    --class_balance_beta ${BETA} \
    --class_balance_alpha 1.0 \
    --class_balance_combine "mean" \
    --class_balance_max_weight 3.0 \
    --class_balance_use_code False \
    --class_balance_use_subcode True

  ########################################
  # 🔥 Merge LoRA → FULL MODEL
  ########################################
  echo "Merging LoRA model..."

  python ${MERGE_SCRIPT} \
    --base "${MODEL_NAME}" \
    --adapter "${MODEL_OUT}" \
    --out "${MERGED_DIR}" \
    --dtype bf16

  ########################################
  # Eval (vLLM + FinBen)
  ########################################
  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks PvExtraction_full \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${EVAL_DIR}/results" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}"
    # --verbosity DEBUG

done

echo "All beta runs completed!"