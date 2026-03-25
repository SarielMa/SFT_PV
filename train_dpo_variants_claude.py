#!/usr/bin/env python3
"""
DPO Variants Training Script

Implements 5 different Direct Preference Optimization techniques:
1. DPO - Standard Direct Preference Optimization
2. SimPO - Simple Preference Optimization (reference-free, length-normalized)
3. IPO - Identity Preference Optimization (squared loss)
4. Cal-DPO - Calibrated DPO (with calibration regularization)
5. DPO-P - DPO-Positive (prevents likelihood decrease)

Key simplifications from TAB DPO:
- No token-level weighting
- No length normalization (except SimPO which has it built-in)
- No class imbalance handling
- No adaptive barrier loss
- Clean, minimal implementation focusing on loss function differences
"""

import os
import json
import argparse
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class Config:
    model_name: str
    train_data_path: str
    valid_data_path: str
    output_dir: str
    
    # DPO technique selection
    dpo_variant: str = "dpo"  # Options: dpo, simpo, ipo, caldpo, dpop
    
    max_length: int = 8192
    max_prompt_length: int = 8192
    
    # DPO parameters (used by all variants as applicable)
    beta: float = 0.1
    
    # SimPO specific
    simpo_gamma: float = 1.0  # Target reward margin
    
    # IPO specific
    ipo_tau: float = 0.1  # Regularization parameter (1/(2*tau) is the target margin)
    
    # Cal-DPO specific
    caldpo_alpha: float = 1.0  # Calibration weight
    
    # DPO-P specific
    dpop_lambda: float = 1.0  # Penalty weight for likelihood decrease
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 50
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# ============================================================================
# Model path detection
# ============================================================================
def detect_model_source(model_identifier: str) -> Tuple[str, str]:
    """Detect if model is local or from HuggingFace Hub"""
    path = Path(model_identifier)
    if path.exists() and path.is_dir():
        return "local", str(path.resolve())
    return "huggingface", model_identifier


# ============================================================================
# Data Collator
# ============================================================================
class DPODataCollator:
    """
    Simple data collator for DPO variants.
    Builds chosen and rejected sequences, masks prompts.
    """
    
    def __init__(self, tokenizer, cfg: Config, system_prompt: str):
        self.tok = tokenizer
        self.cfg = cfg
        self.system_prompt = system_prompt
        
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id
        
        self._logged_example = False
    
    def _truncate_prompt(self, prompt: str) -> str:
        """Truncate prompt to max_prompt_length"""
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.cfg.max_prompt_length:
            return prompt
        ids = ids[-self.cfg.max_prompt_length:]
        return self.tok.decode(ids, skip_special_tokens=False)
    
    def _build_full_text(self, prompt: str, answer: str) -> str:
        """Build complete conversation text with system, user, and assistant"""
        return self.tok.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    
    def _make_labels(self, input_ids: List[int], answer_ids: List[int]) -> List[int]:
        """
        Create labels by masking everything except the answer tokens.
        Simple subsequence search to find where answer starts.
        """
        labels = [-100] * len(input_ids)
        
        # Find answer tokens in the full sequence
        if not answer_ids:
            return labels
        
        # Search for answer_ids as subsequence
        for i in range(len(input_ids) - len(answer_ids) + 1):
            if input_ids[i:i+len(answer_ids)] == answer_ids:
                # Found answer, unmask these positions
                for j in range(i, i+len(answer_ids)):
                    labels[j] = input_ids[j]
                break
        
        return labels
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of examples"""
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }
        
        for idx, ex in enumerate(features):
            prompt = self._truncate_prompt(ex["prompt"])
            chosen_resp = ex["chosen"]
            rejected_resp = ex["rejected"]
            
            # Log first example
            if not self._logged_example:
                print(f"\n{'='*80}")
                print(f"EXAMPLE #{idx + 1} - CHOSEN vs REJECTED")
                print(f"{'='*80}")
                print(f"Prompt: {prompt[:200]}...")
                print(f"\n✅ CHOSEN: {chosen_resp[:200]}...")
                print(f"\n❌ REJECTED: {rejected_resp[:200]}...")
                print(f"{'='*80}\n")
                self._logged_example = True
            
            # Build full texts
            chosen_text = self._build_full_text(prompt, chosen_resp)
            rejected_text = self._build_full_text(prompt, rejected_resp)
            
            # Tokenize
            chosen_enc = self.tok(
                chosen_text,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                add_special_tokens=True,
            )
            rejected_enc = self.tok(
                rejected_text,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                add_special_tokens=True,
            )
            
            # Get answer token IDs for label creation
            chosen_answer_ids = self.tok.encode(chosen_resp, add_special_tokens=False)
            rejected_answer_ids = self.tok.encode(rejected_resp, add_special_tokens=False)
            
            # Create labels
            chosen_labels = self._make_labels(chosen_enc["input_ids"], chosen_answer_ids)
            rejected_labels = self._make_labels(rejected_enc["input_ids"], rejected_answer_ids)
            
            # Add to batch
            batch["chosen_input_ids"].append(chosen_enc["input_ids"])
            batch["chosen_attention_mask"].append(chosen_enc["attention_mask"])
            batch["chosen_labels"].append(chosen_labels)
            batch["rejected_input_ids"].append(rejected_enc["input_ids"])
            batch["rejected_attention_mask"].append(rejected_enc["attention_mask"])
            batch["rejected_labels"].append(rejected_labels)
        
        # Pad to common length
        max_len = max(
            max(len(x) for x in batch["chosen_input_ids"]),
            max(len(x) for x in batch["rejected_input_ids"]),
        )
        
        def pad_int(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]
        
        batch["chosen_input_ids"] = pad_int(batch["chosen_input_ids"], self.pad_id)
        batch["rejected_input_ids"] = pad_int(batch["rejected_input_ids"], self.pad_id)
        batch["chosen_attention_mask"] = pad_int(batch["chosen_attention_mask"], 0)
        batch["rejected_attention_mask"] = pad_int(batch["rejected_attention_mask"], 0)
        batch["chosen_labels"] = pad_int(batch["chosen_labels"], -100)
        batch["rejected_labels"] = pad_int(batch["rejected_labels"], -100)
        
        # Convert to tensors
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ============================================================================
# Helper functions for computing log probabilities
# ============================================================================
def get_batch_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities for a batch of sequences.
    
    Args:
        model: The model to use
        input_ids: Input token IDs [B, T]
        attention_mask: Attention mask [B, T]
        labels: Labels with -100 for masked positions [B, T]
        average_log_prob: If True, return average log prob (for SimPO)
    
    Returns:
        logps: Log probabilities per sequence [B]
        token_counts: Number of non-masked tokens per sequence [B]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute per-token log probabilities using cross-entropy
    per_token_logps = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none',
        ignore_index=-100
    ).view(shift_labels.size())  # [B, T-1]
    
    # Mask out ignored tokens
    mask = (shift_labels != -100).float()
    token_counts = mask.sum(dim=1).clamp(min=1.0)  # [B]
    
    # Sum log probs per sequence
    logps_sum = (per_token_logps * mask).sum(dim=1)  # [B]
    
    if average_log_prob:
        # For SimPO: average log probability
        logps = logps_sum / token_counts
    else:
        # For DPO/IPO/Cal-DPO/DPO-P: sum of log probabilities
        logps = logps_sum
    
    return logps, token_counts


class RefModelContextManager:
    """Context manager to temporarily disable LoRA adapters for reference model"""
    
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


# ============================================================================
# DPO Variants Trainer
# ============================================================================
class DPOVariantsTrainer(Trainer):
    """
    Trainer supporting multiple DPO variants:
    - dpo: Standard DPO
    - simpo: SimPO (reference-free, length-normalized)
    - ipo: IPO (squared loss)
    - caldpo: Cal-DPO (calibrated)
    - dpop: DPO-Positive
    """
    
    def __init__(self, *args, cfg: Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        
        # Validate variant
        valid_variants = ["dpo", "simpo", "ipo", "caldpo", "dpop"]
        if self.cfg.dpo_variant not in valid_variants:
            raise ValueError(f"Invalid dpo_variant: {self.cfg.dpo_variant}. "
                           f"Must be one of {valid_variants}")
        
        print(f"\n{'='*80}")
        print(f"Training with {self.cfg.dpo_variant.upper()}")
        print(f"{'='*80}\n")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss based on selected DPO variant"""
        
        # Get inputs
        c_ids = inputs["chosen_input_ids"]
        c_am = inputs["chosen_attention_mask"]
        c_lab = inputs["chosen_labels"]
        
        r_ids = inputs["rejected_input_ids"]
        r_am = inputs["rejected_attention_mask"]
        r_lab = inputs["rejected_labels"]
        
        # Compute policy logps based on variant
        if self.cfg.dpo_variant == "simpo":
            # SimPO uses average log prob (length-normalized)
            pi_c, c_counts = get_batch_logps(model, c_ids, c_am, c_lab, average_log_prob=True)
            pi_r, r_counts = get_batch_logps(model, r_ids, r_am, r_lab, average_log_prob=True)
            ref_c = ref_r = None  # SimPO doesn't use reference model
        else:
            # Other variants use sum of log probs
            pi_c, c_counts = get_batch_logps(model, c_ids, c_am, c_lab, average_log_prob=False)
            pi_r, r_counts = get_batch_logps(model, r_ids, r_am, r_lab, average_log_prob=False)
            
            # Compute reference model logps (disable LoRA)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                with RefModelContextManager(model):
                    ref_c, _ = get_batch_logps(model, c_ids, c_am, c_lab, average_log_prob=False)
                    ref_r, _ = get_batch_logps(model, r_ids, r_am, r_lab, average_log_prob=False)
            if was_training:
                model.train()
        
        # Compute loss based on variant
        if self.cfg.dpo_variant == "dpo":
            loss = self._compute_dpo_loss(pi_c, pi_r, ref_c, ref_r)
        elif self.cfg.dpo_variant == "simpo":
            loss = self._compute_simpo_loss(pi_c, pi_r)
        elif self.cfg.dpo_variant == "ipo":
            loss = self._compute_ipo_loss(pi_c, pi_r, ref_c, ref_r)
        elif self.cfg.dpo_variant == "caldpo":
            loss = self._compute_caldpo_loss(pi_c, pi_r, ref_c, ref_r)
        elif self.cfg.dpo_variant == "dpop":
            loss = self._compute_dpop_loss(pi_c, pi_r, ref_c, ref_r)
        else:
            raise ValueError(f"Unknown variant: {self.cfg.dpo_variant}")
        
        # Logging
        with torch.no_grad():
            # Compute metrics for logging
            if self.cfg.dpo_variant == "simpo":
                # For SimPO, rewards are already length-normalized
                logits = self.cfg.beta * (pi_c - pi_r)
            else:
                # For others, compute ratio-based logits
                logits = self.cfg.beta * ((pi_c - ref_c) - (pi_r - ref_r))
            
            p_chosen = torch.sigmoid(logits).mean()
            
            metrics = {
                "loss": loss.detach().float(),
                "logits_mean": logits.mean().detach().float(),
                "logits_max": logits.max().detach().float(),
                "logits_min": logits.min().detach().float(),
                "p_chosen": p_chosen.detach().float(),
            }
        
        self.log({k: v.item() for k, v in metrics.items()})
        
        return (loss, metrics) if return_outputs else loss
    
    def _compute_dpo_loss(
        self,
        pi_c: torch.Tensor,
        pi_r: torch.Tensor,
        ref_c: torch.Tensor,
        ref_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard DPO loss.
        L = -E[log(sigmoid(β * (log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)))))]
        """
        pi_logratios = pi_c - pi_r
        ref_logratios = ref_c - ref_r
        logits = self.cfg.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()
        return loss
    
    def _compute_simpo_loss(
        self,
        pi_c: torch.Tensor,
        pi_r: torch.Tensor
    ) -> torch.Tensor:
        """
        SimPO loss (reference-free, length-normalized).
        L = -E[log(sigmoid(β/|y_w| * log(π_θ(y_w)) - β/|y_l| * log(π_θ(y_l)) - γ))]
        
        Note: pi_c and pi_r are already average log probs (length-normalized)
        """
        logits = self.cfg.beta * (pi_c - pi_r) - self.cfg.simpo_gamma
        loss = -F.logsigmoid(logits).mean()
        return loss
    
    def _compute_ipo_loss(
        self,
        pi_c: torch.Tensor,
        pi_r: torch.Tensor,
        ref_c: torch.Tensor,
        ref_r: torch.Tensor
    ) -> torch.Tensor:
        """
        IPO loss (squared loss with target margin).
        L = E[(log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l)) - 1/(2τ))^2]
        """
        pi_logratios = pi_c - pi_r
        ref_logratios = ref_c - ref_r
        logits = pi_logratios - ref_logratios
        target_margin = 1.0 / (2.0 * self.cfg.ipo_tau)
        loss = ((logits - target_margin) ** 2).mean()
        return loss
    
    def _compute_caldpo_loss(
        self,
        pi_c: torch.Tensor,
        pi_r: torch.Tensor,
        ref_c: torch.Tensor,
        ref_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Cal-DPO loss (DPO + calibration regularization).
        L = DPO_loss + α * E[(log(π_θ(y_w)/π_ref(y_w)))^2]
        
        The calibration term keeps the policy close to the reference on chosen responses.
        """
        # Standard DPO loss
        pi_logratios = pi_c - pi_r
        ref_logratios = ref_c - ref_r
        logits = self.cfg.beta * (pi_logratios - ref_logratios)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        # Calibration term: penalize deviation from reference on chosen
        pi_c_logratio = pi_c - ref_c
        calibration_loss = (pi_c_logratio ** 2).mean()
        
        # Combined loss
        loss = dpo_loss + self.cfg.caldpo_alpha * calibration_loss
        return loss
    
    def _compute_dpop_loss(
        self,
        pi_c: torch.Tensor,
        pi_r: torch.Tensor,
        ref_c: torch.Tensor,
        ref_r: torch.Tensor
    ) -> torch.Tensor:
        """
        DPO-Positive loss (DPO + penalty for likelihood decrease).
        L = DPO_loss + λ * E[max(0, 1 - exp(log(π_θ(y_w)/π_ref(y_w))))]
        
        The penalty term prevents the chosen response likelihood from decreasing.
        """
        # Standard DPO loss
        pi_logratios = pi_c - pi_r
        ref_logratios = ref_c - ref_r
        logits = self.cfg.beta * (pi_logratios - ref_logratios)
        dpo_loss = -F.logsigmoid(logits).mean()
        
        # Penalty term: penalize when π_θ(y_w)/π_ref(y_w) < 1
        # That is, when log ratio < 0
        pi_c_logratio = pi_c - ref_c
        ratio = torch.exp(pi_c_logratio)  # π_θ(y_w) / π_ref(y_w)
        penalty = torch.clamp(1.0 - ratio, min=0.0)
        penalty_loss = penalty.mean()
        
        # Combined loss
        loss = dpo_loss + self.cfg.dpop_lambda * penalty_loss
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, 
                       ignore_keys: Optional[List[str]] = None):
        """Prediction step for evaluation"""
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train DPO variants")
    parser.add_argument("--model_name", type=str, required=True, 
                       help="Model name or path")
    parser.add_argument("--train_data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--valid_data_path", type=str, required=True,
                       help="Path to validation data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--dpo_variant", type=str, default="dpo",
                       choices=["dpo", "simpo", "ipo", "caldpo", "dpop"],
                       help="DPO variant to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--system_prompt", type=str, 
                       default="You are a helpful assistant.",
                       help="System prompt for chat template")
    
    # Variant-specific parameters
    parser.add_argument("--beta", type=float, default=0.1,
                       help="Beta parameter for all variants")
    parser.add_argument("--simpo_gamma", type=float, default=1.0,
                       help="SimPO target reward margin")
    parser.add_argument("--ipo_tau", type=float, default=0.1,
                       help="IPO regularization parameter")
    parser.add_argument("--caldpo_alpha", type=float, default=1.0,
                       help="Cal-DPO calibration weight")
    parser.add_argument("--dpop_lambda", type=float, default=1.0,
                       help="DPO-P penalty weight")
    
    args = parser.parse_args()
    
    # Create config
    cfg = Config(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,
        dpo_variant=args.dpo_variant,
        beta=args.beta,
        simpo_gamma=args.simpo_gamma,
        ipo_tau=args.ipo_tau,
        caldpo_alpha=args.caldpo_alpha,
        dpop_lambda=args.dpop_lambda,
    )
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"DPO VARIANT: {cfg.dpo_variant.upper()}")
    print("=" * 80)
    
    # Detect model source
    source_type, model_path = detect_model_source(cfg.model_name)
    print(f"Model source: {source_type} | Model path/ID: {model_path}")
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=hf_token,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    print("✓ Tokenizer loaded")
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds = load_from_disk(cfg.train_data_path)
    valid_ds = load_from_disk(cfg.valid_data_path)
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")
    
    # Verify dataset columns
    needed = {"prompt", "chosen", "rejected"}
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        cols = set(ds.column_names)
        missing = needed - cols
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")
    
    # Load model
    print("\nLoading model...")
    device_map = "auto" if args.num_gpus > 1 else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        use_cache=False,
        token=hf_token,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print("✓ Model loaded")
    
    # Apply LoRA
    print("\nApplying LoRA...")
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    
    # Create data collator
    print("\nCreating data collator...")
    collator = DPODataCollator(tok, cfg=cfg, system_prompt=args.system_prompt)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        bf16=True,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
    )
    
    # Create trainer
    trainer = DPOVariantsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tok,
        cfg=cfg,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    
    # Save model info
    model_info = {
        "base_model": cfg.model_name,
        "base_model_source": source_type,
        "base_model_resolved_path": model_path,
        "dpo_variant": cfg.dpo_variant,
        "training_completed": True,
    }
    with open(os.path.join(cfg.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE - {cfg.dpo_variant.upper()}")
    print("=" * 80)
    print(f"Output saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
