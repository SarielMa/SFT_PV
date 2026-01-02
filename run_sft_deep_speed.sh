#!/bin/bash
set -euo pipefail

# =========================
# User settings
# =========================
DATASET_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"
PY_SCRIPT="sft_deepspeed.py"   # your clean DS/DDP python script

MAX_LEN_DEFAULT=8192
BATCH_SIZE=1
EPOCHS=3
LR=2e-4

# If you want to override CPU workers for rank0 tokenization:
NUM_PROC="${SLURM_CPUS_PER_TASK:-8}"   # used by sft_deepspeed.py when --num_proc=0

# Models to SFT (QLoRA)
MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/QwQ-32B-AWQ"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

# =========================
# Helper: choose grad_accum
# =========================
guess_grad_accum () {
  local model="$1"
  case "$model" in
    meta-llama/Llama-3.3-70B-Instruct)
      echo 4   # good starting point on 2x H200; increase if you want larger effective batch
      ;;
    meta-llama/Llama-3.1-8B-Instruct)
      echo 8
      ;;
    meta-llama/Llama-3.2-3B-Instruct)
      echo 8
      ;;
    Qwen/Qwen2.5-1.5B-Instruct)
      echo 8
      ;;
    *)
      echo 8
      ;;
  esac
}

# =========================
# Helper: create DS config
# =========================
write_ds_config () {
  local ds_path="$1"
  local ga="$2"
  local micro_bs="$3"

  cat > "${ds_path}" << EOF
{
  "train_micro_batch_size_per_gpu": ${micro_bs},
  "gradient_accumulation_steps": ${ga},
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
EOF
}

# =========================
# Run loop
# =========================
for MODEL in "${MODELS[@]}"; do
  # ---- Skip AWQ models for training ----
  if [[ "${MODEL}" == *"AWQ"* ]]; then
    echo "============================================================"
    echo "SKIP: ${MODEL}"
    echo "Reason: AWQ checkpoint is for inference quantization; not a suitable base for QLoRA training."
    echo "============================================================"
    continue
  fi

  # Output dir name safe for filesystem
  SAFE_NAME="$(echo "${MODEL}" | tr '/:' '__')"
  OUT_DIR="./runs/pv_sft_${SAFE_NAME}_qlora_ds2"

  # ---- Per-model settings ----
  GA="$(guess_grad_accum "${MODEL}")"

  # max_length override for especially large models (optional)
  MAX_LEN="${MAX_LEN_DEFAULT}"
  # On 2x H200, 8192 is plausible for 70B QLoRA; keep default.
  # If you still OOM, set MAX_LEN=4096 here for 70B.

  # Unique DS config file per model (prevents accidental overwrite if parallelizing later)
  DS_CONFIG="ds_zero2_${SAFE_NAME}.json"
  write_ds_config "${DS_CONFIG}" "${GA}" "${BATCH_SIZE}"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Output: ${OUT_DIR}"
  echo "max_length=${MAX_LEN} micro_batch=${BATCH_SIZE} grad_accum=${GA} epochs=${EPOCHS} lr=${LR}"
  echo "DeepSpeed config: ${DS_CONFIG}"
  echo "============================================================"

  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,COLL
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_DEBUG_FILE="nccl_%h_%p.log"
  export NCCL_P2P_LEVEL=NVL
  export NCCL_IB_DISABLE=1


  deepspeed --num_gpus=2 "${PY_SCRIPT}" \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --use_qlora --bf16 --do_eval \
    --max_length "${MAX_LEN}" --batch_size "${BATCH_SIZE}" --grad_accum "${GA}" --epochs "${EPOCHS}" --lr "${LR}" \
    --deepspeed "${DS_CONFIG}" \
    --num_proc "${NUM_PROC}"
done
