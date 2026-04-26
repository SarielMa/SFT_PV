#!/bin/bash
#SBATCH --job-name=sft_qwen25_7b
#SBATCH --mail-type=ALL
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus=b200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_b200
#SBATCH --output=%j_sft_qwen25_7b_b200.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail
set -x

export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*"; }
trap 'rc=$?; echo "[ERROR] rc=${rc} line=${LINENO} cmd=${BASH_COMMAND}" >&2; exit $rc' ERR

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${REPO_ROOT}/run_sft_from_sft_to_finben_b200_qwen25_7b.sh"

log "SFT submit bootstrap start"
log "hostname=$(hostname)"
log "pwd=$(pwd)"
log "user=${USER:-unknown}"
log "shell=${SHELL:-unknown}"
log "repo_root=${REPO_ROOT}"

for var in CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_PYTHON_EXE CONDA_PKGS_DIRS CONDA_ENVS_PATH _CE_CONDA _CE_M _CONDA_EXE _CONDA_ROOT; do
  unset "${var}" || true
done
unset -f conda 2>/dev/null || true
unset -f __conda_activate 2>/dev/null || true
unset -f __conda_reactivate 2>/dev/null || true
unset -f __conda_hashr 2>/dev/null || true

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
CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
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

cd "${REPO_ROOT}"

log "Environment check"
which nvcc
nvcc --version
which python
python --version
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
which torchrun
which lm_eval
nvidia-smi

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Missing run script: ${RUN_SCRIPT}" >&2
  exit 1
fi

exec bash "${RUN_SCRIPT}" "${REPO_ROOT}"
