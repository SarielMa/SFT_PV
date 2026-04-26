#!/bin/bash
#SBATCH --job-name=sft_finben
#SBATCH --mail-type=ALL
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_sft_finben_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

module purge
module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finben_b200

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

model_slug() {
  local model_basename="$1"
  model_basename="${model_basename,,}"
  model_basename="${model_basename/llama-/llama}"
  model_basename="${model_basename//-/_}"
  printf '%s' "${model_basename}"
}

check_hf_access() {
  local model_name="$1"
  python - "${model_name}" <<'PY'
import sys
from transformers import AutoConfig

model_name = sys.argv[1]
AutoConfig.from_pretrained(model_name)
print(f"HF access OK: {model_name}")
PY
}

# =========================
# Global settings
# =========================
DATASET_PATH="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/non_test"
PY_SCRIPT="${REPO_ROOT}/sft_peft_ddp.py"
MERGE_SCRIPT="${REPO_ROOT}/merge_lora.py"
FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"

MAX_LEN=8192
BATCH_SIZE=1
EPOCHS=3
LR=2e-4
TP=2

OUTPUT_ROOT="$(readlink -f "${REPO_ROOT}/sft_3epoch")"
PIPELINE_ROOT="$(readlink -f "${OUTPUT_ROOT}/.sft_pipeline_runs")"

# =========================
# Models
# =========================
MODELS=(
  # "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
)

# =========================
# Grad-accum heuristic
# =========================
guess_grad_accum () {
  case "$1" in
    meta-llama/Llama-3.3-70B-Instruct) echo 4 ;;
    meta-llama/Llama-3.1-8B-Instruct)  echo 8 ;;
    meta-llama/Llama-3.2-3B-Instruct)  echo 8 ;;
    Qwen/Qwen2.5-1.5B-Instruct)        echo 8 ;;
    Qwen/Qwen2.5-7B-Instruct)          echo 8 ;;
    Qwen/Qwen2.5-14B-Instruct)         echo 8 ;;
    *)                                 echo 8 ;;
  esac
}

# =========================
# NCCL safety (H100/H200)
# =========================
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}" "${PIPELINE_ROOT}"

echo "Environment check"
which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
which torchrun
which lm_eval
nvidia-smi

require_path "${PY_SCRIPT}" "SFT script"
require_path "${MERGE_SCRIPT}" "merge script"
require_path "${DATASET_PATH}" "dataset"
require_path "${FINBEN_TASKS_PATH}" "FinBen task path"

python - "${DATASET_PATH}" <<'PY'
import sys
from datasets import load_from_disk

dataset_path = sys.argv[1]
ds = load_from_disk(dataset_path)
if hasattr(ds, "keys"):
    print("dataset_splits", list(ds.keys()))
else:
    print("dataset_type", type(ds).__name__)
PY

for MODEL in "${MODELS[@]}"; do
  check_hf_access "${MODEL}"
done

# =========================
# Main loop
# =========================
for MODEL in "${MODELS[@]}"; do
  if [[ "${MODEL}" == *"AWQ"* ]]; then
    echo "Skip ${MODEL} (AWQ not suitable for QLoRA training)"
    continue
  fi

  MODEL_TAG="$(basename "${MODEL}")"
  MODEL_SLUG="$(model_slug "${MODEL_TAG}")"
  RUN_TAG="sft_${EPOCHS}ep"

  WORK_DIR="${PIPELINE_ROOT}/${MODEL_SLUG}/${RUN_TAG}"
  ADAPTER_DIR="${WORK_DIR}/sft_adapter"
  LOG_DIR="${WORK_DIR}/logs"
  EVAL_DIR="${WORK_DIR}/lm_eval_results"
  MERGED_DIR="${OUTPUT_ROOT}/merged_${MODEL_SLUG}_sft_${EPOCHS}ep"

  mkdir -p "${ADAPTER_DIR}" "${LOG_DIR}" "${EVAL_DIR}" "${MERGED_DIR}"

  WORK_DIR="$(readlink -f "${WORK_DIR}")"
  ADAPTER_DIR="$(readlink -f "${ADAPTER_DIR}")"
  LOG_DIR="$(readlink -f "${LOG_DIR}")"
  EVAL_DIR="$(readlink -f "${EVAL_DIR}")"
  MERGED_DIR="$(readlink -f "${MERGED_DIR}")"

  GA="$(guess_grad_accum "${MODEL}")"

  echo "============================================================"
  echo "Model: ${MODEL}"
  echo "Work dir : ${WORK_DIR}"
  echo "Adapter  : ${ADAPTER_DIR}"
  echo "Merged   : ${MERGED_DIR}"
  echo "Eval     : ${EVAL_DIR}"
  echo "TP=${TP} | GA=${GA} | epochs=${EPOCHS} | lr=${LR}"
  echo "============================================================"

  torchrun --nproc_per_node="${TP}" "${PY_SCRIPT}" \
    --dataset_path "${DATASET_PATH}" \
    --model_name "${MODEL}" \
    --output_dir "${ADAPTER_DIR}" \
    --use_qlora --bf16 \
    --max_length "${MAX_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GA}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    2>&1 | tee "${LOG_DIR}/sft.log"

  ADAPTER_PATH="${ADAPTER_DIR}/lora_adapter"
  if [[ ! -f "${ADAPTER_PATH}/adapter_config.json" ]]; then
    ADAPTER_PATH="${ADAPTER_DIR}"
  fi
  require_path "${ADAPTER_PATH}" "adapter output"

  python "${MERGE_SCRIPT}" \
    --base "${MODEL}" \
    --adapter "${ADAPTER_PATH}" \
    --out "${MERGED_DIR}" \
    --dtype bf16 \
    2>&1 | tee "${LOG_DIR}/merge.log"

  TASK="PvExtraction_full"
  rm -rf "${EVAL_DIR}/${TASK}"

  lm_eval --model vllm \
    --model_args "pretrained=${MERGED_DIR},tensor_parallel_size=${TP},gpu_memory_utilization=0.9,max_model_len=${MAX_LEN},enforce_eager=True" \
    --tasks "${TASK}" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${EVAL_DIR}/${TASK}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_${TASK}.log"

  echo "Eval outputs under: ${EVAL_DIR}"
  find "${EVAL_DIR}" -maxdepth 1 -type f -name "*${TASK}*" -print || true
  echo "Completed: ${MODEL_TAG}"
done
