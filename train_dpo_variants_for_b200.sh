#!/bin/bash

set -e

REQUESTED_METHOD="${1:-}"

########################################
# Base configs
########################################

SFT_RESULTS_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/sft_results_epoch3"
DPO_PIPELINE_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/dpo_pipeline_outputs"

MODELS=(
  "Llama-3.1-8B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.3-70B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
)

TRAIN_SCRIPT="train_dpo_variants_with_tdpo_tidpo.py"
MERGE_SCRIPT="merge_lora.py"

ALL_METHODS=("dpo" "ipo" "cal_dpo" "dpop" "tdpo" "ti_dpo")
METHODS=()

if [[ -n "${REQUESTED_METHOD}" ]]; then
  case "${REQUESTED_METHOD}" in
    dpo|ipo|cal_dpo|dpop|tdpo|ti_dpo)
      METHODS=("${REQUESTED_METHOD}")
      ;;
    *)
      echo "Invalid method: ${REQUESTED_METHOD}"
      echo "Valid methods: dpo ipo cal_dpo dpop tdpo ti_dpo"
      exit 1
      ;;
  esac
else
  METHODS=("${ALL_METHODS[@]}")
fi
#METHODS=("dpo")
# SEEDS=(42 123 2024)
SEEDS=(42)

# eval configs
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=8192

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

BASE_OUT_DIR="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/po_variants_runs_april24"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi

if [[ -z "${NUM_GPUS}" || "${NUM_GPUS}" -lt 1 ]]; then
  echo "Unable to detect available GPUs."
  exit 1
fi

TENSOR_PARALLEL_SIZE="${NUM_GPUS}"

echo "Detected NUM_GPUS=${NUM_GPUS}"
echo "Using tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"

########################################
# Loop
########################################

for MODEL_ID in "${MODELS[@]}"; do
  MODEL_NAME="${SFT_RESULTS_DIR}/${MODEL_ID}/epoch3/merged"
  DATA_PATH="${DPO_PIPELINE_DIR}/${MODEL_ID}_epoch3_sftMerged/dpo_data"

  if [[ ! -d "${MODEL_NAME}" ]]; then
    echo "Skipping MODEL=${MODEL_ID}: merged SFT model not found at ${MODEL_NAME}"
    continue
  fi

  if [[ ! -d "${DATA_PATH}" ]]; then
    echo "Skipping MODEL=${MODEL_ID}: DPO data not found at ${DATA_PATH}"
    continue
  fi

  for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do

      echo "========================================"
      echo "Running MODEL=${MODEL_ID}, METHOD=${METHOD}, SEED=${SEED}"
      echo "========================================"

      RUN_DIR="${BASE_OUT_DIR}/${MODEL_ID}/${METHOD}/seed_${SEED}"
      MODEL_OUT="${RUN_DIR}/model"
      MERGED_DIR="${RUN_DIR}/merged_model"
      EVAL_DIR="${RUN_DIR}/eval"
      FINBEN_OUT="${EVAL_DIR}/PvExtraction_full"

      mkdir -p "${MODEL_OUT}"
      mkdir -p "${MERGED_DIR}"
      mkdir -p "${EVAL_DIR}"

      ########################################
      # Seed
      ########################################
      export PYTHONHASHSEED=${SEED}

      ########################################
      # Train (LoRA)
      ########################################
      METHOD_ARGS=()
      if [[ "${METHOD}" == "tdpo" ]]; then
        METHOD_ARGS+=(--beta 0.1)
      else
        METHOD_ARGS+=(--beta 0.5)
      fi

      python ${TRAIN_SCRIPT} \
        --model_name "${MODEL_NAME}" \
        --train_data_path "${DATA_PATH}" \
        --valid_data_path "${DATA_PATH}" \
        --output_dir "${MODEL_OUT}" \
        --num_gpus "${NUM_GPUS}" \
        --method "${METHOD}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        "${METHOD_ARGS[@]}"

      ########################################
      # Merge LoRA -> full model
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
        --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},enforce_eager=True" \
        --tasks PvExtraction_full \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path "${FINBEN_OUT}" \
        --log_samples \
        --apply_chat_template \
        --include_path "${FINBEN_TASKS_PATH}"

    done
  done
done

echo "All runs completed!"
