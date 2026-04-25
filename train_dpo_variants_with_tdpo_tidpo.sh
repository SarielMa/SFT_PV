#!/bin/bash
#SBATCH --job-name=qwen25_dpo_voice_sft
#SBATCH --partition=gpu_rtx6000
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/qwen25_dpo_voice_sft_%j.out
#SBATCH --error=logs/qwen25_dpo_voice_sft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ganesh.puthiaraju@yale.edu

set -e
set -o pipefail

echo "=========================================="
echo "=== Qwen2.5 DPO Training ==="
echo "=== Job Started at $(date) ==="
echo "=========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
echo ""

mkdir -p logs outputs

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_train

echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
which python
python --version

echo ""
echo "=========================================="
echo "=== CUDA Setup ==="
echo "=========================================="

export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

echo "✓ Using conda CUDA: $CUDA_HOME"

if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | head -n 4
else
    echo "⚠️  nvcc not found in conda environment"
    if module load CUDA/11.8 2>/dev/null; then
        echo "✓ Loaded CUDA/11.8 module"
    elif module load CUDA/12.0 2>/dev/null; then
        echo "✓ Loaded CUDA/12.0 module"
    elif module load CUDA/11.7 2>/dev/null; then
        echo "✓ Loaded CUDA/11.7 module"
    else
        echo "⚠️  No CUDA module found, using system default"
    fi

    if command -v nvcc &> /dev/null; then
        CUDA_BIN_DIR="$(dirname "$(which nvcc)")"
        export CUDA_HOME="$(dirname "$CUDA_BIN_DIR")"
        export CUDA_PATH="$CUDA_HOME"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
        echo "✓ Using system CUDA: $CUDA_HOME"
        nvcc --version | head -n 4
    else
        echo "✗ nvcc still not found after loading CUDA module"
        exit 1
    fi
fi

echo "CUDA_VISIBLE_DEVICES (after env): ${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi -L || true
echo ""

echo "Installing dependencies..."
pip install transformers==4.57.3 datasets==4.4.1 accelerate==1.12.0 -q 2>/dev/null || echo "✓ Core OK"
pip install peft==0.18.0 bitsandbytes==0.48.2 -q 2>/dev/null || echo "✓ PEFT/BnB OK"
pip install numpy scipy scikit-learn pandas tqdm safetensors -q 2>/dev/null || echo "✓ Utils OK"
echo "✓ Dependencies ready"
echo ""

export HF_HOME="/nfs/roberts/scratch/pi_sjf37/gp528/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

echo "✓ HF_HOME: $HF_HOME"
echo "✓ TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo ""

TRAIN_DATA="/home/gp528/qwen_dpo_data/dpo_data"
# The current Python script still requires --valid_data_path even when --eval_strategy no.
# Reuse TRAIN_DATA as a harmless placeholder; it will not be used for evaluation.
VALID_DATA="$TRAIN_DATA"
MODEL_NAME="lm2445/voice_qwen2.5_1.5b_instruct_sft_3ep"
PY_SCRIPT="/nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/train_dpo_variants_with_tdpo_tidpo.py"
OUTPUT_DIR="/nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/outputs/qwen25_voice_sft_dpo_1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

check_data() {
  if [ -d "$1" ]; then
    COUNT=$(find "$1" -type f -name "*.arrow" 2>/dev/null | wc -l)
    echo "✓ Found: $1 ($COUNT arrow files)"
    return 0
  else
    echo "✗ NOT FOUND: $1"
    return 1
  fi
}

check_data "$TRAIN_DATA" || exit 1

if [ ! -f "$PY_SCRIPT" ]; then
  echo "✗ Python script not found: $PY_SCRIPT"
  exit 1
fi

echo ""
echo "Model:  $MODEL_NAME"
echo "Train:  $TRAIN_DATA"
echo "Output: $OUTPUT_DIR"
echo "Script: $PY_SCRIPT"
echo ""

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using master port: $MASTER_PORT"
echo ""

torchrun \
  --nproc_per_node=1 \
  --master_port=$MASTER_PORT \
  "$PY_SCRIPT" \
  --model_name "$MODEL_NAME" \
  --train_data_path "$TRAIN_DATA" \
  --valid_data_path "$VALID_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --method tdpo \
  --beta 0.1 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --eval_strategy no

EXIT_CODE=$?

echo ""
echo "Cleaning up..."
rm -rf "$TRITON_CACHE_DIR" || true
echo "✓ Cleanup complete"

echo ""
echo "=========================================="
echo "=== Job Completed at $(date) ==="
echo "=========================================="
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
