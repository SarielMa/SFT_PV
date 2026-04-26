#!/bin/bash
#SBATCH --job-name=finben_qwen_0shot
#SBATCH --mail-type=ALL
#SBATCH --time=00-10:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_h200
#SBATCH --output=%j_gpu_job.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

module purge
module load StdEnv || true
module load CUDA/12.6.0

export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finben_202604

which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft

FINBEN_TASKS_PATH="/home/lm2445/project_pi_sjf37/lm2445/finben/FinBen/tasks/pv_miner"
OUT_ROOT="/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft/finben_0shot_qwen_outputs"

GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=8192

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
)

mkdir -p "${OUT_ROOT}"

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

for MODEL in "${MODELS[@]}"; do
  MODEL_TAG="$(basename "${MODEL}")"
  MODEL_OUT_DIR="${OUT_ROOT}/${MODEL_TAG}"
  LOG_DIR="${MODEL_OUT_DIR}/logs"

  mkdir -p "${MODEL_OUT_DIR}" "${LOG_DIR}"

  FINBEN_OUT="$(readlink -f "${MODEL_OUT_DIR}/PvExtraction_full")"

  echo "========================================"
  echo "Evaluating MODEL=${MODEL}"
  echo "Output prefix=${FINBEN_OUT}"
  echo "========================================"

  lm_eval --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks PvExtraction_full \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "${FINBEN_OUT}" \
    --log_samples \
    --apply_chat_template \
    --include_path "${FINBEN_TASKS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_PvExtraction_full.log"
done

echo "All evaluations completed."
echo "Outputs under: ${OUT_ROOT}"
