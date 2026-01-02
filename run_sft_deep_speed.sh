# python sft.py \
#   --dataset_path /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/ \
#   --model_name Qwen/Qwen2.5-1.5B-Instruct \
#   --output_dir ./runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora_epoch3 \
#   --use_qlora --bf16 --do_eval \
#   --max_length 8192 --batch_size 1 --grad_accum 16 --epochs 3 --lr 2e-4

# #!/bin/bash
# set -euo pipefail

# # ---- create DeepSpeed config on the fly (no extra file needed) ----
# DS_CONFIG="ds_zero2.json"
# cat > ${DS_CONFIG} << 'EOF'
# {
#   "train_micro_batch_size_per_gpu": 1,
#   "gradient_accumulation_steps": 8,
#   "zero_optimization": {
#     "stage": 2,
#     "overlap_comm": true,
#     "contiguous_gradients": true,
#     "reduce_scatter": true,
#     "allgather_partitions": true
#   },
#   "bf16": { "enabled": true },
#   "fp16": { "enabled": false },
#   "gradient_clipping": 1.0,
#   "steps_per_print": 50,
#   "wall_clock_breakdown": false
# }
# EOF

# # ---- run with DeepSpeed on 2 GPUs ----
# deepspeed --num_gpus=2 sft_deepspeed.py \
#   --dataset_path /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/ \
#   --model_name Qwen/Qwen2.5-1.5B-Instruct \
#   --output_dir ./runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora_epoch3_ds2 \
#   --use_qlora --bf16 --do_eval \
#   --max_length 8192 --batch_size 1 --grad_accum 8 --epochs 3 --lr 2e-4 \
#   --deepspeed ${DS_CONFIG}

#!/bin/bash
set -euo pipefail

# DeepSpeed config created on the fly
DS_CONFIG="ds_zero2.json"

# Dataset
DATASET_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test/"

# Common training settings
MAX_LEN=8192
BATCH_SIZE=1
EPOCHS=3
LR=2e-4

# Models to SFT (QLoRA)
MODELS=(
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/QwQ-32B-AWQ"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
)

# ---- create DeepSpeed config (ZeRO-2) ----
# NOTE: we will overwrite gradient_accumulation_steps per-model below by regenerating this file each run.
write_ds_config () {
  local GA="$1"
  cat > "${DS_CONFIG}" << EOF
{
  "train_micro_batch_size_per_gpu": ${BATCH_SIZE},
  "gradient_accumulation_steps": ${GA},
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "allgather_partitions": true
  },
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
EOF
}

# ---- per-model grad_accum (safe starting points) ----
# With 2 GPUs, global effective batch = BATCH_SIZE * grad_accum * 2
# You can tune these. These are set conservative to avoid OOM at 8192.
guess_grad_accum () {
  local model="$1"

  case "$model" in
    meta-llama/Llama-3.3-70B-Instruct)
      echo 32
      ;;
    Qwen/QwQ-32B-AWQ)
      # AWQ training may be problematic; keep conservative
      echo 16
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

# ---- run loop ----
for MODEL in "${MODELS[@]}"; do
  GA="$(guess_grad_accum "${MODEL}")"
  write_ds_config "${GA}"

  # Output dir name safe for filesystem
  SAFE_NAME="$(echo "${MODEL}" | tr '/:' '__')"
  OUT_DIR="./runs/pv_sft_${SAFE_NAME}_qlora_len${MAX_LEN}_ds2"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Output: ${OUT_DIR}"
  echo "max_length=${MAX_LEN} batch=${BATCH_SIZE} grad_accum=${GA} epochs=${EPOCHS} lr=${LR}"
  echo "============================================================"

  # IMPORTANT NOTE:
  # - If QwQ-32B-AWQ fails to load for training, replace it with a non-AWQ base model (fp16/bf16) for SFT.
  # - You must have edited sft_deepspeed.py to remove device_map="auto" and accept --deepspeed.

  deepspeed --num_gpus=2 sft_deepspeed.py \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --use_qlora --bf16 --do_eval \
    --max_length "${MAX_LEN}" --batch_size "${BATCH_SIZE}" --grad_accum "${GA}" --epochs "${EPOCHS}" --lr "${LR}" \
    --deepspeed "${DS_CONFIG}"
done

