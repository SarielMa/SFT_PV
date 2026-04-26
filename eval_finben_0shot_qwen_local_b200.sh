#!/bin/bash
#SBATCH --job-name=0shot_2qwen
#SBATCH --mail-type=ALL
#SBATCH --time=00-2:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:1
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_po_tdpo_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# Clear inherited conda state from the submission shell before touching modules.
for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

# YCRC's miniconda module runs `conda deactivate` on unload.
# If the submission shell carries a stale miniconda module without a live
# conda shell function, a plain `module purge` can terminate the batch script
# before our own conda init logic runs.
if ! command -v conda >/dev/null 2>&1; then
  conda() { return 0; }
  export -f conda
  _FAKE_CONDA_FOR_PURGE=1
fi

module --force purge || true
if [[ "${_FAKE_CONDA_FOR_PURGE:-0}" == "1" ]]; then
  unset -f conda || true
  unset _FAKE_CONDA_FOR_PURGE
fi

module load StdEnv || true
module load CUDA/12.8.0

export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="/tmp/${USER}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

module load miniconda

if [[ -n "${EBROOTMINICONDA:-}" && -f "${EBROOTMINICONDA}/etc/profile.d/conda.sh" ]]; then
  source "${EBROOTMINICONDA}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
  CONDA_BASE="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "Failed to initialize conda after loading the miniconda module." >&2
  exit 1
fi

conda activate finben_b200

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
