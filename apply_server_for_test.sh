#!/bin/bash
#SBATCH --job-name=test_sft_5models
#SBATCH --mail-type=ALL
#SBATCH --time=00-5:00:00
#SBATCH --nodes=1
#SBATCH --gpus=h200:2
#SBATCH --mem=256G
#SBATCH --partition=gpu_h200
#SBATCH --output=%j_gpu_job.txt
#SBATCH --mail-user=linhai.ma@yale.edu

set -euo pipefail

# ---- CUDA toolkit for DeepSpeed ops ----
module purge
module load StdEnv || true
module load CUDA/12.6.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Triton cache off NFS (recommended)
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# ---- Conda (robust) ----
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finben_vllm3

# Debug
which nvcc
nvcc --version
which python
python -c "import torch; print('torch cuda:', torch.version.cuda); print('gpus:', torch.cuda.device_count())"
nvidia-smi

cd /home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/sft
sh merge_and_eval_sft.sh
