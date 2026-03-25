#!/bin/bash
#SBATCH --job-name=ablation0.99
#SBATCH --partition=gpu
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --output=logs/eppc_tab_dpo_%j.out
#SBATCH --error=logs/eppc_tab_dpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=linhai.ma@yale.edu

set -e
set -o pipefail

echo "=========================================="
echo "=== EPPC TAB-DPO (McCleary) ==="
echo "=== Job Started at $(date) ==="
echo "=========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
echo ""

mkdir -p logs outputs

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

module load miniconda

export BASHRCSOURCED=1
set +u 2>/dev/null || true
source ~/.bashrc
conda activate llm_train
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

# ============================================================================
# CUDA SETUP (same spirit as your working scripts)
# ============================================================================

echo ""
echo "=========================================="
echo "=== CUDA Setup for McCleary ==="
echo "=========================================="

# OPTION 1: Use conda's CUDA (most reliable if nvcc exists in env)
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH

echo "✓ Using conda CUDA: $CUDA_HOME"

# Verify CUDA is accessible
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | head -n 4
else
    echo "⚠️  nvcc not found in conda environment"
    echo "⚠️  Attempting to load system CUDA module..."

    module avail 2>&1 | grep -i cuda || true

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

# ============================================================================
# DEEPSPEED COMPATIBILITY FIXES
# ============================================================================

export DS_BUILD_OPS=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_UTILS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_QUANTIZER=0
export DS_SKIP_CUDA_CHECK=1
echo "✓ DeepSpeed JIT disabled + CUDA checks disabled"

# ============================================================================
# DEPENDENCIES (optional; you can remove if env already has these)
# ============================================================================
echo "Installing dependencies..."
pip install transformers==4.57.3 datasets==4.4.1 accelerate==1.12.0 -q 2>/dev/null || echo "✓ Core OK"
pip install peft==0.18.0 bitsandbytes==0.48.2 -q 2>/dev/null || echo "✓ PEFT/BnB OK"
pip install "deepspeed>=0.15.0" -q 2>/dev/null || echo "✓ DeepSpeed OK"
pip install numpy scipy scikit-learn pandas tqdm safetensors -q 2>/dev/null || echo "✓ Utils OK"
echo "✓ Dependencies ready"
echo ""

# ============================================================================
# ENV VARS
# ============================================================================

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

# NOTE: secrets redacted; keep your existing values or export them before sbatch.
export HF_TOKEN="${HF_TOKEN:-hf_REDACTED}"
export WANDB_API_KEY="${WANDB_API_KEY:-wandb_REDACTED}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

export WANDB_PROJECT="eppc-annotator-llama31"
export WANDB_LOG_MODEL="false"

echo "✓ HF_HOME: $HF_HOME"
echo "✓ TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo ""

# ============================================================================
# DATA PATHS
# ============================================================================

# IMPORTANT: these must be DPO preference datasets saved with columns:
#   prompt, chosen, rejected  (and optionally example_weight)
TRAIN_DATA="/home/gp528/eppc_dataset_local_train/train/resplit/resplit_20260117_014052/train_data"
VALID_DATA="/home/gp528/eppc_dataset_local_train/train/resplit/resplit_20260117_014052/valid_data"
TEST_DATA="/home/gp528/eppc_dataset_local_train/train/resplit/resplit_20260117_014052/valid_data"

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
check_data "$VALID_DATA" || exit 1
check_data "$TEST_DATA" || exit 1
echo ""

# ============================================================================
# TRAINING CONFIG
# ============================================================================

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR="./outputs/eppc_llama31_3b_tab_dpo_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Model:  $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo ""

DS_CONFIG="ds_config_zero2.json"
if [ ! -f "$DS_CONFIG" ]; then
cat > "$DS_CONFIG" << 'EOF'
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "offload_param": { "device": "none" },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOF
fi
echo "✓ DeepSpeed config ready: $DS_CONFIG"

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "Using master port: $MASTER_PORT"
echo ""

# ============================================================================
# RUN TRAINING (TAB-DPO)
# ============================================================================

python \
  /nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/train_dpo_ablation.py \
  --model_name "$MODEL_NAME" \
  --train_data_path "$TRAIN_DATA" \
  --valid_data_path "$VALID_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --num_gpus 1 \
  --system_prompt "You are a helpful assistant." \
  --enable_length_norm False \
  --length_norm_by "tokens" \
  --enable_token_weighting False \
  --token_weight_code 1.1 \
  --token_weight_subcode 1.2 \
  --token_weight_span 1.1 \
  --normalize_by_weight_mass True \
  --enable_class_balance True \
  --class_balance_strategy "effective_num" \
  --class_balance_beta 0.99 \
  --class_balance_alpha 1.0 \
  --class_balance_combine "mean" \
  --class_balance_max_weight 3.0 \
  --class_balance_use_code False \
  --class_balance_use_subcode True

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
