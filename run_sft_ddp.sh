#!/bin/bash
set -euo pipefail

# =========================
# User settings
# =========================
DATASET_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"
PY_SCRIPT="sft_peft_ddp.py"   # the torchrun/DDP script (no DeepSpeed)

MAX_LEN_DEFAULT=8192
BATCH_SIZE=1
EPOCHS=10
LR=2e-4

# Models to SFT (QLoRA)
MODELS=(
  #"meta-llama/Llama-3.3-70B-Instruct"
  #"meta-llama/Llama-3.1-8B-Instruct"
  #"meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

# =========================
# Helper: choose grad_accum
# =========================
guess_grad_accum () {
  local model="$1"
  case "$model" in
    meta-llama/Llama-3.3-70B-Instruct)
      echo 4
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
# Optional: NCCL debug (turn off when stable)
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG_FILE="nccl_%h_%p.log"

# =========================
# Run loop
# =========================
for MODEL in "${MODELS[@]}"; do
  # ---- Skip AWQ models for training ----
  if [[ "${MODEL}" == *"AWQ"* ]]; then
    echo "============================================================"
    echo "SKIP: ${MODEL}"
    echo "Reason: AWQ is inference quantization; not a good base for QLoRA/LoRA training."
    echo "============================================================"
    continue
  fi

  SAFE_NAME="$(echo "${MODEL}" | tr '/:' '__')"
  OUT_DIR="./runs/pv_sft_${SAFE_NAME}_qlora_ddp2_epoch10"

  GA="$(guess_grad_accum "${MODEL}")"
  MAX_LEN="${MAX_LEN_DEFAULT}"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Output: ${OUT_DIR}"
  echo "max_length=${MAX_LEN} micro_batch=${BATCH_SIZE} grad_accum=${GA} epochs=${EPOCHS} lr=${LR}"
  echo "Launcher: torchrun --nproc_per_node=2"
  echo "============================================================"

  # NOTE:
  # - --use_qlora enables 4-bit training
  # - --bf16 is recommended on H200
  # - omit --do_eval unless you *know* test split exists and is non-empty (avoids deadlocks)
  torchrun --nproc_per_node=2 "${PY_SCRIPT}" \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --use_qlora --bf16 \
    --max_length "${MAX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GA}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}"

done
