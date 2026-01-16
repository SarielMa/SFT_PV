%%writefile /nfs/roberts/scratch/pi_sjf37/gp528/FinBen/Llama_eightb_train/eppc_dpo_train_token_barrier_fixed.sh
#!/bin/bash
#SBATCH --job-name=eppc_llama31_dpo_mccleary
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/eppc_dpo_%j.out
#SBATCH --error=logs/eppc_dpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ganesh.puthiaraju@yale.edu

# ============================================================================
# EPPC DPO Training - MCCLEARY CLUSTER VERSION
# Token-Weighted Constrained DPO with Adaptive Barrier
# ============================================================================

echo "=========================================="
echo "=== EPPC DPO Training (McCleary) ==="
echo "=== Job Started at $(date) ==="
echo "=========================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS"
echo ""

# ============================================================================-
# SETUP DIRECTORIES
# ============================================================================

echo "Setting up directories..."
mkdir -p logs outputs

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo ""
echo "=========================================="
echo "=== Environment Setup ==="
echo "=========================================="

# Load miniconda
module load miniconda
source ~/.bashrc

# Activate environment
conda activate llm_train
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"

# ============================================================================
# CUDA SETUP FOR MCCLEARY
# ============================================================================

echo ""
echo "=========================================="
echo "=== CUDA Setup for McCleary ==="
echo "=========================================="

# Use conda's CUDA
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
    
    # Try to load CUDA module
    if module load CUDA/11.8 2>/dev/null; then
        echo "✓ Loaded CUDA/11.8 module"
    elif module load CUDA/12.0 2>/dev/null; then
        echo "✓ Loaded CUDA/12.0 module"
    elif module load CUDA/11.7 2>/dev/null; then
        echo "✓ Loaded CUDA/11.7 module"
    else
        echo "⚠️  No CUDA module found, using system default"
    fi
fi

# ============================================================================
# INSTALL/UPDATE DEPENDENCIES
# ============================================================================

echo ""
echo "=========================================="
echo "=== Installing Dependencies ==="
echo "=========================================="

# Install PyTorch with CUDA support
# pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q 2>/dev/null || echo "✓ PyTorch OK"

# # Install core dependencies
# pip install transformers==4.57.3 datasets==4.4.1 accelerate==1.12.0 -q 2>/dev/null || echo "✓ Core OK"
# pip install peft==0.18.0 bitsandbytes==0.48.2 -q 2>/dev/null || echo "✓ PEFT OK"

# # Install monitoring tools (optional for DPO)
# pip install wandb>=0.18.0 tensorboard>=2.17.0 -q 2>/dev/null || echo "⚠ Monitoring optional"

# # Install utilities
# pip install numpy scipy scikit-learn pandas tqdm -q 2>/dev/null || echo "✓ Utils OK"

echo "✓ All dependencies installed"

# ============================================================================
# SET ENVIRONMENT VARIABLES
# ============================================================================

echo ""
echo "=========================================="
echo "=== Environment Variables ==="
echo "=========================================="

# HuggingFace cache (McCleary paths)
export HF_HOME="/nfs/roberts/scratch/pi_sjf37/gp528/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ---- Secrets (DO NOT hardcode) ----
: "${HF_TOKEN:?HF_TOKEN is not set}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Disable Weights & Biases unless explicitly enabled
export WANDB_DISABLED="${WANDB_DISABLED:-true}"


# Training environment
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO

# CRITICAL: Set TRITON_CACHE_DIR to local temporary location
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

# WandB settings (optional - DPO script has report_to="none")
export WANDB_PROJECT="eppc-dpo-llama31"
export WANDB_LOG_MODEL="false"

echo "✓ HF_HOME: $HF_HOME"
echo "✓ TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "✓ CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ============================================================================
# VERIFY DATA PATHS (MCCLEARY)
# ============================================================================

echo ""
echo "=========================================="
echo "=== Data Verification ==="
echo "=========================================="

# DPO dataset paths
TRAIN_DATA="/home/gp528/eppc_dpo_dataset_revised_v1/train"
VALID_DATA="/home/gp528/eppc_dpo_dataset_revised_v1/valid"

# Base model path (can be HuggingFace model ID or local path)
BASE_MODEL="/nfs/roberts/scratch/pi_sjf37/gp528/FinBen/fixed_output_llama31_8b_version_0"

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

# Check if base model is local or HuggingFace
if [ -d "$BASE_MODEL" ]; then
    echo "✓ Local model path found: $BASE_MODEL"
elif [[ "$BASE_MODEL" == *"/"* ]]; then
    echo "✓ HuggingFace model ID detected: $BASE_MODEL"
    echo "  Will download from HuggingFace Hub"
else
    echo "⚠️  Model identifier: $BASE_MODEL"
    echo "  Assuming HuggingFace model ID"
fi

# ============================================================================
# GPU CHECK
# ============================================================================

echo ""
echo "=========================================="
echo "=== GPU Information ==="
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

echo ""
echo "=========================================="
echo "=== DPO Training Configuration ==="
echo "=========================================="

OUTPUT_DIR="./outputs/eppc_dpo_llama31_mccleary_high_weight_2e4_barrier_57_6_fixed_norm_$(date +%Y%m%d_%H%M%S)"

echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""
echo "DPO Settings:"
echo "  - Beta: 0.5"
echo "  - Policy probability threshold: 0.8"
echo "  - SFT barrier weight: 0.5"
echo "  - Weight code/subcode: 3.0x"
echo "  - Weight span: 1.5x"
echo "  - Weight diff tokens: 1.5x"
echo ""
echo "Training Settings:"
echo "  - Epochs: 6"
echo "  - Batch size per GPU: 1"
echo "  - Gradient accumulation: 2"
echo "  - Effective batch size: 1"
echo "  - Learning rate: 1e-5"
echo "  - LoRA rank: 16"
echo "  - Max sequence length: 8192"
echo "  - Precision: BF16"
echo ""

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# UPDATE DPO SCRIPT WITH CORRECT PATHS
# ============================================================================

echo ""
echo "=========================================="
echo "=== Preparing DPO Training Script ==="
echo "=========================================="

# Create a modified version of the DPO script with correct paths
cat > train_dpo_mccleary.py << 'PYTHON_SCRIPT_END'
"""
Token-Weighted Constrained DPO - McCleary Cluster Version
============================================================================
Adapted for McCleary cluster with:
- Configurable paths via command line
- Multi-GPU support option
- Proper SLURM environment integration
- Automatic HuggingFace/local path detection
============================================================================
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class Config:
    model_name: str
    train_data_path: str
    valid_data_path: str
    output_dir: str

    max_length: int = 8192
    max_prompt_length: int = 8192

    # DPO parameters
    beta: float = 0.5
    policy_prob_threshold: float = 0.66
    sft_barrier_weight: float = 0.5

    # Token weighting
    weight_code_value: float = 1.1
    weight_subcode_value: float = 1.2
    weight_span_value: float = 1.1
    weight_diff_tokens: float = 1.2

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm=1e9
    logging_steps: int = 10
    save_steps: int = 50

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# ============================================================================
# MODEL PATH DETECTION
# ============================================================================

def detect_model_source(model_identifier: str) -> Tuple[str, str]:
    """
    Detect if model is local path or HuggingFace model ID.
    
    Returns:
        (source_type, resolved_path)
        source_type: 'local' or 'huggingface'
        resolved_path: absolute path if local, model ID if HuggingFace
    """
    # Check if it's a local path
    path = Path(model_identifier)
    if path.exists() and path.is_dir():
        # Local path exists
        return 'local', str(path.resolve())
    
    # Check if it looks like a HuggingFace model ID (username/model-name)
    if '/' in model_identifier:
        return 'huggingface', model_identifier
    
    # Could be a local relative path that doesn't exist yet (error case)
    # or a simple model name (unlikely but possible)
    if not '/' in model_identifier:
        # Probably an error - single word without slash
        print(f"⚠️  Warning: '{model_identifier}' doesn't look like a valid path or HuggingFace ID")
        print(f"   Assuming HuggingFace model ID")
        return 'huggingface', model_identifier
    
    return 'huggingface', model_identifier


# Value extraction
_CODE_RE = re.compile(r'"Code"\s*:\s*"([^"]+)"')
_SUBCODE_RE = re.compile(r'"Sub-code"\s*:\s*"([^"]+)"|\"Subcode\"\s*:\s*\"([^\"]+)\"')
_SPAN_RE = re.compile(r'"Span"\s*:\s*"([^"]+)"')


def extract_values_from_jsonish(text: str) -> Dict[str, List[str]]:
    codes = _CODE_RE.findall(text)
    subs = []
    for m in _SUBCODE_RE.findall(text):
        subs.append(m[0] if m[0] else m[1])
    spans = _SPAN_RE.findall(text)

    def uniq(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"code": uniq(codes), "subcode": uniq(subs), "span": uniq(spans)}


def _find_subsequence(haystack: List[int], needle: List[int], start: int, end: int) -> Optional[int]:
    if not needle or end - start < len(needle):
        return None
    for i in range(start, end - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return None


class DPODataCollatorTokenWeighted:
    def __init__(self, tokenizer, max_length: int, max_prompt_length: int):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id

        self.response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self._logged_responses = False
        self._logged_chosen_masking = False
        self._logged_rejected_masking = False

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.max_prompt_length:
            return prompt
        ids = ids[-self.max_prompt_length:]
        return self.tok.decode(ids, skip_special_tokens=False)

    def _make_labels(self, input_ids: List[int], offsets: List[Tuple[int, int]], 
                     full_text: str, is_chosen: bool = True) -> List[int]:
        labels = input_ids.copy()
        
        idx = full_text.find(self.response_template)
        if idx == -1:
            should_log = (is_chosen and not self._logged_chosen_masking) or \
                        (not is_chosen and not self._logged_rejected_masking)
            if should_log:
                print(f"⚠️  WARNING: Response template not found in {'CHOSEN' if is_chosen else 'REJECTED'} text!")
                print(f"Text preview: {full_text[:200]}")
                print("Masking all tokens as safety measure.")
            return [-100] * len(labels)
        
        response_start_char = idx + len(self.response_template)
        
        masked_count = 0
        first_response_token_idx = None
        for i, (start, end) in enumerate(offsets):
            if start >= response_start_char:
                if first_response_token_idx is None:
                    first_response_token_idx = i
                break
            labels[i] = -100
            masked_count += 1
        
        should_log = (is_chosen and not self._logged_chosen_masking) or \
                    (not is_chosen and not self._logged_rejected_masking)
        
        if should_log:
            total_tokens = len(labels)
            response_tokens = total_tokens - masked_count
            
            masked_text = full_text[:response_start_char]
            unmasked_text = full_text[response_start_char:]
            
            print(f"\n{'='*80}")
            print(f"PROMPT MASKING DIAGNOSTIC - {'CHOSEN' if is_chosen else 'REJECTED'} RESPONSE")
            print(f"{'='*80}")
            print(f"\n📊 STATISTICS:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Masked (prompt): {masked_count}")
            print(f"  Unmasked (response): {response_tokens}")
            print(f"  Response starts at character: {response_start_char}")
            
            if is_chosen:
                self._logged_chosen_masking = True
            else:
                self._logged_rejected_masking = True
        
        return labels

    def _init_weights(self, labels: List[int]) -> List[float]:
        return [0.0 if lab == -100 else 1.0 for lab in labels]

    def _apply_value_weights_with_offsets(
        self,
        full_text: str,
        offsets: List[Tuple[int,int]],
        labels: List[int],
        weights: List[float],
        assistant_content: str,
        w_code: float,
        w_sub: float,
        w_span: float,
    ) -> None:
        base = full_text.rfind(assistant_content)
        if base == -1:
            return

        vals = extract_values_from_jsonish(assistant_content)
        targets: List[Tuple[int,int,float]] = []

        def add_targets(value_list: List[str], w: float):
            for v in value_list:
                if not v:
                    continue
                start_local = assistant_content.find(v)
                while start_local != -1:
                    s = base + start_local
                    e = s + len(v)
                    targets.append((s, e, w))
                    start_local = assistant_content.find(v, start_local + 1)

        add_targets(vals["code"], w_code)
        add_targets(vals["subcode"], w_sub)
        add_targets(vals["span"], w_span)

        if not targets:
            return

        for i, (a, b) in enumerate(offsets):
            if labels[i] == -100:
                continue
            if a == b == 0:
                continue
            for (s, e, w) in targets:
                if a < e and b > s:
                    weights[i] = max(weights[i], w)

    def _apply_value_weights_fallback_token_search(
        self,
        input_ids: List[int],
        labels: List[int],
        weights: List[float],
        assistant_content: str,
        w_code: float,
        w_sub: float,
        w_span: float,
    ) -> None:
        vals = extract_values_from_jsonish(assistant_content)
        resp_start = next((i for i, lab in enumerate(labels) if lab != -100), len(labels))
        resp_end = len(labels)

        def mark(value_list: List[str], w: float):
            for v in value_list:
                v_ids = self.tok.encode(v, add_special_tokens=False)
                pos = _find_subsequence(input_ids, v_ids, resp_start, resp_end)
                if pos is None:
                    continue
                for j in range(pos, min(pos + len(v_ids), len(weights))):
                    if labels[j] != -100:
                        weights[j] = max(weights[j], w)

        mark(vals["code"], w_code)
        mark(vals["subcode"], w_sub)
        mark(vals["span"], w_span)

    def _apply_diff_weight(
        self,
        chosen_ids: List[int], chosen_labels: List[int], chosen_w: List[float],
        rejected_ids: List[int], rejected_labels: List[int], rejected_w: List[float],
        w_diff: float,
    ) -> None:
        c0 = next((i for i, lab in enumerate(chosen_labels) if lab != -100), None)
        r0 = next((i for i, lab in enumerate(rejected_labels) if lab != -100), None)
        if c0 is None or r0 is None:
            return

        c_len = sum(1 for i in range(c0, len(chosen_labels)) if chosen_labels[i] != -100)
        r_len = sum(1 for i in range(r0, len(rejected_labels)) if rejected_labels[i] != -100)
        L = min(c_len, r_len)

        for k in range(L):
            ci = c0 + k
            ri = r0 + k
            if chosen_labels[ci] == -100 or rejected_labels[ri] == -100:
                continue
            if chosen_ids[ci] != rejected_ids[ri]:
                chosen_w[ci] *= w_diff
                rejected_w[ri] *= w_diff

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [], "chosen_weights": [],
            "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": [], "rejected_weights": [],
        }

        for idx, ex in enumerate(features):
            prompt = self._truncate_prompt(ex["prompt"])
            chosen_resp = ex["chosen"]
            rejected_resp = ex["rejected"]

            if not self._logged_responses:
                print(f"\n{'='*80}")
                print(f"EXAMPLE #{idx + 1} - CHOSEN vs REJECTED RESPONSES")
                print(f"{'='*80}")
                print(f"\n✅ CHOSEN RESPONSE (first 500 chars):")
                print(chosen_resp[:500])
                print(f"\n❌ REJECTED RESPONSE (first 500 chars):")
                print(rejected_resp[:500])
                print(f"{'='*80}\n")
                self._logged_responses = True

            chosen_text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_resp}],
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_text = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_resp}],
                tokenize=False,
                add_generation_prompt=False,
            )

            chosen_enc = self.tok(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            rejected_enc = self.tok(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )

            c_ids = chosen_enc["input_ids"]
            r_ids = rejected_enc["input_ids"]
            c_am = chosen_enc["attention_mask"]
            r_am = rejected_enc["attention_mask"]
            c_offsets = chosen_enc.get("offset_mapping", None)
            r_offsets = rejected_enc.get("offset_mapping", None)

            c_labels = self._make_labels(c_ids, c_offsets, chosen_text, is_chosen=True)
            r_labels = self._make_labels(r_ids, r_offsets, rejected_text, is_chosen=False)

            c_w = self._init_weights(c_labels)
            r_w = self._init_weights(r_labels)

            # Apply value weights
            cfg = CFG  # Access global config
            if c_offsets is not None:
                self._apply_value_weights_with_offsets(
                    chosen_text, c_offsets, c_labels, c_w,
                    chosen_resp, cfg.weight_code_value, cfg.weight_subcode_value, cfg.weight_span_value
                )
            else:
                self._apply_value_weights_fallback_token_search(
                    c_ids, c_labels, c_w,
                    chosen_resp, cfg.weight_code_value, cfg.weight_subcode_value, cfg.weight_span_value
                )

            if r_offsets is not None:
                self._apply_value_weights_with_offsets(
                    rejected_text, r_offsets, r_labels, r_w,
                    rejected_resp, cfg.weight_code_value, cfg.weight_subcode_value, cfg.weight_span_value
                )
            else:
                self._apply_value_weights_fallback_token_search(
                    r_ids, r_labels, r_w,
                    rejected_resp, cfg.weight_code_value, cfg.weight_subcode_value, cfg.weight_span_value
                )

            if cfg.weight_diff_tokens and cfg.weight_diff_tokens > 1.0:
                self._apply_diff_weight(
                    c_ids, c_labels, c_w,
                    r_ids, r_labels, r_w,
                    cfg.weight_diff_tokens
                )

            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_am)
            batch["chosen_labels"].append(c_labels)
            batch["chosen_weights"].append(c_w)

            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_am)
            batch["rejected_labels"].append(r_labels)
            batch["rejected_weights"].append(r_w)

        # Pad
        max_len = max(
            max(len(x) for x in batch["chosen_input_ids"]),
            max(len(x) for x in batch["rejected_input_ids"]),
        )

        def pad_int(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        def pad_float(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        batch["chosen_input_ids"] = pad_int(batch["chosen_input_ids"], self.pad_id)
        batch["rejected_input_ids"] = pad_int(batch["rejected_input_ids"], self.pad_id)
        batch["chosen_attention_mask"] = pad_int(batch["chosen_attention_mask"], 0)
        batch["rejected_attention_mask"] = pad_int(batch["rejected_attention_mask"], 0)
        batch["chosen_labels"] = pad_int(batch["chosen_labels"], -100)
        batch["rejected_labels"] = pad_int(batch["rejected_labels"], -100)
        batch["chosen_weights"] = pad_float(batch["chosen_weights"], 0.0)
        batch["rejected_weights"] = pad_float(batch["rejected_weights"], 0.0)

        out = {}
        for k in batch:
            if "weights" in k:
                out[k] = torch.tensor(batch[k], dtype=torch.float32)
            else:
                out[k] = torch.tensor(batch[k], dtype=torch.long)
        return out


def weighted_sequence_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather = shift_labels.clone()
    gather[gather == -100] = 0
    per_token_logps = torch.gather(log_probs, dim=2, index=gather.unsqueeze(2)).squeeze(2)

    mask = (shift_labels != -100).float()
    w = shift_weights * mask
    
    # ---- weight normalization: mean weight over valid tokens = 1.0 ----
    tok_count = mask.sum(dim=1).clamp(min=1.0)                 # [B]
    w_sum_raw = w.sum(dim=1).clamp(min=1e-6)                  # [B]
    w = w * (tok_count / w_sum_raw).unsqueeze(1)              # [B, T-1]
    w_sum = tok_count                                         # normalized denom
    # -------------------------------------------------------------------
    
    seq_logp = (per_token_logps * w).sum(dim=1)
    seq_logp_avg = seq_logp / w_sum
    return seq_logp_avg, w_sum

class RefAdapterOff:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._has_ctx = hasattr(model, "disable_adapter")

    def __enter__(self):
        if self._has_ctx:
            self._ctx = self.model.disable_adapter()
            return self._ctx.__enter__()
        if hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()
        return None

    def __exit__(self, exc_type, exc, tb):
        if self._has_ctx:
            return self._ctx.__exit__(exc_type, exc, tb)
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()
        return False


class TWCDPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = model.device
    
        c_ids = inputs["chosen_input_ids"].to(device)
        c_am  = inputs["chosen_attention_mask"].to(device)
        c_lab = inputs["chosen_labels"].to(device)
        c_w   = inputs["chosen_weights"].to(device)
    
        r_ids = inputs["rejected_input_ids"].to(device)
        r_am  = inputs["rejected_attention_mask"].to(device)
        r_lab = inputs["rejected_labels"].to(device)
        r_w   = inputs["rejected_weights"].to(device)
    
        # ------------------------------------------------------------------
        # Helper: compute (1) weighted avg sequence logp and (2) per-token logp
        # ------------------------------------------------------------------
        def _seq_avg_logp_and_token_logps(_ids, _am, _lab, _wts):
            outputs = model(input_ids=_ids, attention_mask=_am, use_cache=False)
            logits = outputs.logits
    
            shift_logits  = logits[:, :-1, :].contiguous()
            shift_labels  = _lab[:, 1:].contiguous()
            shift_weights = _wts[:, 1:].contiguous()
    
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gather = shift_labels.clone()
            gather[gather == -100] = 0
    
            # per-token log p(y_t | x, y_<t) for the ground-truth token
            token_logps = torch.gather(log_probs, dim=2, index=gather.unsqueeze(2)).squeeze(2)  # [B, T-1]
    
            token_mask = (shift_labels != -100)  # assistant response tokens only (excludes prompt + padding)
            mask_f = token_mask.float()
            w = shift_weights * mask_f
            
            # ---- weight normalization: mean weight over valid tokens = 1.0 ----
            tok_count = mask_f.sum(dim=1).clamp(min=1.0)              # [B]
            w_sum_raw = w.sum(dim=1).clamp(min=1e-6)                 # [B]
            w = w * (tok_count / w_sum_raw).unsqueeze(1)             # [B, T-1]
            w_sum = tok_count                                        # normalized denom
            # -------------------------------------------------------------------
            
            seq_logp = (token_logps * w).sum(dim=1)
            seq_logp_avg = seq_logp / w_sum
            return seq_logp_avg, token_logps, token_mask, w

    
        # ---------------------------
        # Policy logps (adapter ON)
        # ---------------------------
        pi_c, c_token_logps, c_token_mask, c_token_w = _seq_avg_logp_and_token_logps(c_ids, c_am, c_lab, c_w)
        pi_r, _,            _,            _          = _seq_avg_logp_and_token_logps(r_ids, r_am, r_lab, r_w)
    
        # ------------------------------
        # Reference logps (adapter OFF)
        # ------------------------------
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with RefAdapterOff(model):
                ref_c, _ = weighted_sequence_logp(model, c_ids, c_am, c_lab, c_w)
                ref_r, _ = weighted_sequence_logp(model, r_ids, r_am, r_lab, r_w)
        if was_training:
            model.train()
    
        # ---------------------------
        # DPO loss (unchanged)
        # ---------------------------
        pi_logratio  = pi_c - pi_r
        ref_logratio = ref_c - ref_r
        logits = pi_logratio - ref_logratio
        p_pref = torch.sigmoid(CFG.beta * logits)
    
        dpo_loss = -F.logsigmoid(CFG.beta * logits)  # [B]
    
        # ------------------------------------------------------------------
        # ✅ TOKEN-LEVEL ADAPTIVE BARRIER (sequence barrier removed)
        #
        # Gate per token using token logp:
        #   gate_t = 1[ log p_t < log(threshold) ]
        # Barrier penalty per token is NLL:  -log p_t
        # Average over gated tokens (weighted), then mean over batch.
        # ------------------------------------------------------------------
        # log threshold (more stable than exp)
        log_tau = torch.log(torch.tensor(CFG.policy_prob_threshold, device=device, dtype=c_token_logps.dtype))
    
        # gate only on valid response tokens (prompt/pad already excluded by c_token_mask)
        gate = (c_token_logps < log_tau).float() * c_token_mask.float()  # [B, T-1]
    
        token_nll = -c_token_logps  # [B, T-1]
    
        barrier_num = (token_nll * c_token_w * gate).sum(dim=1)  # [B]
        barrier_den = (c_token_w * gate).sum(dim=1).clamp(min=1.0)  # [B]
        barrier_loss = (barrier_num / barrier_den).mean() * CFG.sft_barrier_weight
    
        # Total
        total = dpo_loss.mean() + barrier_loss
    
        # ---------------------------
        # Logging
        # ---------------------------
        with torch.no_grad():
            # keep your old "prob_chosen" metric as geometric-mean proxy (exp(avg logp))
            policy_chosen_prob_geom = torch.exp(pi_c)
    
            active_tokens = c_token_mask.float().sum().clamp(min=1.0)
            barrier_active_pct = gate.sum() / active_tokens
    
            metrics = {
                "loss": total.detach().float(),
                "loss/dpo": dpo_loss.mean().detach().float(),
                "loss/barrier": barrier_loss.detach().float(),
                "policy/prob_chosen": policy_chosen_prob_geom.mean().detach().float(),
                "barrier/active_pct": barrier_active_pct.detach().float(),
                "pref/p_pref": p_pref.mean().detach().float(),
                "logits/mean": logits.mean().detach().float(),
                "logits/max": logits.max().detach().float(),
                "logits/min": logits.min().detach().float(),
            }
        self.log({k: v.item() for k, v in metrics.items()})
    
        return (total, metrics) if return_outputs else total

    def prediction_step(self, model, inputs, prediction_loss_only: bool, 
                       ignore_keys: Optional[List[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# Global config (will be set in main)
CFG = None


def main():
    global CFG
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()
    
    CFG = Config(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,
    )
    
    os.makedirs(CFG.output_dir, exist_ok=True)

    print("="*80)
    print("FIXED Token-Weighted DPO with Adaptive Barrier")
    print("="*80)
    print(f"✅ Corrected prompt masking using offset mapping")
    print(f"✅ Reduced weight ratio: 3x")
    print(f"✅ Adaptive Barrier: Activates NLL when policy_prob < {CFG.policy_prob_threshold}")
    print(f"✅ Beta: {CFG.beta}")
    print(f"✅ Barrier weight: {CFG.sft_barrier_weight}")
    print(f"✅ Auto-detect HuggingFace/Local model paths")
    print("="*80)

    # ============================================================================
    # DETECT MODEL SOURCE
    # ============================================================================
    
    print("\n" + "="*80)
    print("MODEL SOURCE DETECTION")
    print("="*80)
    
    source_type, resolved_path = detect_model_source(CFG.model_name)
    
    if source_type == 'local':
        print(f"✓ Detected LOCAL model")
        print(f"  Path: {resolved_path}")
        model_path = resolved_path
    else:
        print(f"✓ Detected HUGGINGFACE model")
        print(f"  Model ID: {resolved_path}")
        print(f"  Will download from HuggingFace Hub")
        
        # Check if HF_TOKEN is available
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        if hf_token:
            print(f"  ✓ HuggingFace token found (length: {len(hf_token)})")
        else:
            print(f"  ⚠️  No HuggingFace token found")
            print(f"     Public models will work, private models may fail")
        
        model_path = resolved_path
    
    print("="*80)

    # ============================================================================
    # LOAD TOKENIZER
    # ============================================================================

    print("\nLoading tokenizer...")
    
    # Get HF token if available
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=hf_token,  # Pass token for private models
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    
    print(f"✓ Tokenizer loaded from: {model_path}")

    # ============================================================================
    # LOAD DATASETS
    # ============================================================================

    print("\nLoading dataset...")
    train_ds = load_from_disk(CFG.train_data_path)
    valid_ds = load_from_disk(CFG.valid_data_path)
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    needed = {"prompt", "chosen", "rejected"}
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        cols = set(ds.column_names)
        missing = needed - cols
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    # ============================================================================
    # LOAD MODEL
    # ============================================================================

    print("\nLoading base model...")
    
    # Determine device map based on number of GPUs
    if args.num_gpus > 1:
        device_map = "auto"
        print(f"Using multi-GPU setup with {args.num_gpus} GPUs")
    else:
        device_map = {"": 0}
        print("Using single GPU setup")
    
    print(f"Loading from: {model_path}")
    if source_type == 'huggingface':
        print(f"  This may take a while for first-time download...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache=False,
        token=hf_token,  # Pass token for private models
    )
    
    print(f"✓ Model loaded successfully from {source_type} source")
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ============================================================================
    # APPLY LORA
    # ============================================================================

    print("\nApplying LoRA...")
    lora = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ============================================================================
    # CREATE DATA COLLATOR
    # ============================================================================

    collator = DPODataCollatorTokenWeighted(tok, CFG.max_length, CFG.max_prompt_length)

    # ============================================================================
    # TRAINING ARGUMENTS
    # ============================================================================

    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        max_grad_norm=CFG.max_grad_norm,
        bf16=True,
        logging_steps=CFG.logging_steps,
        save_steps=CFG.save_steps,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=CFG.save_steps,
    )

    # ============================================================================
    # CREATE TRAINER
    # ============================================================================

    trainer = TWCDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    # ============================================================================
    # TRAIN
    # ============================================================================

    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {CFG.output_dir}...")
    trainer.save_model(CFG.output_dir)
    
    # Save model info
    model_info = {
        "base_model": CFG.model_name,
        "base_model_source": source_type,
        "base_model_resolved_path": model_path,
        "training_completed": True,
    }
    
    with open(os.path.join(CFG.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel source: {source_type.upper()}")
    print(f"Base model: {CFG.model_name}")
    print(f"Output saved to: {CFG.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
PYTHON_SCRIPT_END

echo "✓ DPO training script created: train_dpo_mccleary.py"

# ============================================================================
# RUN TRAINING
# ============================================================================

echo ""
echo "=========================================="
echo "=== Starting DPO Training ==="
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Run with Python (single or multi-GPU based on allocation)
python train_dpo_mccleary.py \
    --model_name "$BASE_MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --valid_data_path "$VALID_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_gpus 2

EXIT_CODE=$?

# ============================================================================
# CLEANUP
# ============================================================================

echo ""
echo "Cleaning up temporary files..."
rm -rf "$TRITON_CACHE_DIR"
echo "✓ Cleanup complete"

# ============================================================================
# COMPLETION STATUS
# ============================================================================

echo ""
echo "=========================================="
echo "=== Job Completed at $(date) ==="
echo "=========================================="
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS"
    echo ""
    echo "Results: $OUTPUT_DIR"
    
    if [ -f "$OUTPUT_DIR/all_results.json" ]; then
        echo ""
        echo "Final metrics:"
        cat "$OUTPUT_DIR/all_results.json"
    fi
    
    if [ -f "$OUTPUT_DIR/model_info.json" ]; then
        echo ""
        echo "Model info:"
        cat "$OUTPUT_DIR/model_info.json"
    fi
else
    echo ""
    echo "❌ FAILED"
    echo "Check logs: logs/eppc_dpo_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
