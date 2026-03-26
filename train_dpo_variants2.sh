#!/bin/bash

set -e

########################################
# Base configs
########################################

MODEL_NAME="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/sft_results_epoch3/Qwen2.5-1.5B-Instruct/epoch3/merged"
DATA_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs/Qwen2.5-1.5B-Instruct_epoch3_sftMerged/dpo_data"

TRAIN_SCRIPT="train_dpo_variants.py"
MERGE_SCRIPT="merge_lora.py"

# METHODS=("dpo" "ipo" "cal_dpo" "dpop")
METHODS=("cal_dpo" "dpop")

SEEDS=(42 123 2024)

# eval configs
TENSOR_PARALLEL_SIZE=1
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=8192

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

BASE_OUT_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/po_runs"

########################################
# Loop
########################################

for METHOD in "${METHODS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    echo "========================================"
    echo "Running METHOD=${METHOD}, SEED=${SEED}"
    echo "========================================"

    RUN_DIR="${BASE_OUT_DIR}/${METHOD}/seed_${SEED}"
    MODEL_OUT="${RUN_DIR}/model"
    MERGED_DIR="${RUN_DIR}/merged_model"
    EVAL_DIR="${RUN_DIR}/eval"

    mkdir -p "${MODEL_OUT}"
    mkdir -p "${MERGED_DIR}"
    mkdir -p "${EVAL_DIR}"

    ########################################
    # Seed
    ########################################
    export PYTHONHASHSEED=${SEED}
    export CUDA_VISIBLE_DEVICES=0

    ########################################
    # Train (LoRA)
    ########################################
    python ${TRAIN_SCRIPT} \
      --model_name "${MODEL_NAME}" \
      --train_data_path "${DATA_PATH}" \
      --valid_data_path "${DATA_PATH}" \
      --output_dir "${MODEL_OUT}" \
      --method "${METHOD}" \
      --beta 0.5 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2

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
    # Eval (vLLM requires merged model)
    ########################################
    lm_eval --model vllm \
      --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
      --tasks PvExtraction_full \
      --num_fewshot 0 \
      --batch_size auto \
      --output_path "${EVAL_DIR}/PvExtraction_full" \
      --log_samples \
      --apply_chat_template \
      --include_path "${FINBEN_TASKS_PATH}"

  done
done

echo "All runs completed!"