"""
Train 5 preference-optimization variants (NO TAB-DPO, NO token weighting, NO class balancing):

Methods implemented (per papers):
  1) dpo        : Direct Preference Optimization (plain)               (Eq. DPO)      [Rafailov et al. 2023]
  2) simpo      : SimPO, reference-free reward + length norm + margin  (Eq. 6)        [SimPO OpenReview]
  3) ipo        : IPO, squared loss on DPO-style log-ratio gap          (Alg.1 / Eq.17)[Gheshlaghi Azar et al. 2024]
  4) cal_dpo    : Cal-DPO = preference loss + calibration squares       (Eq. 10)       [Cal-DPO arXiv 2412.14516]
  5) dpop       : DPO-Positive (Smaug) adds positive-preservation term  (Eq. 3)        [Smaug arXiv 2402.13228]


Example:
  python train_dpo_variants.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --train_data_path /path/to/train_ds \
    --valid_data_path /path/to/valid_ds \
    --output_dir /path/to/out \
    --method dpo \
    --beta 0.1

Method-specific args:
  - dpo     : --beta
  - simpo   : --beta, --simpo_gamma
  - ipo     : --beta (uses target = 1/(2*beta))
  - cal_dpo : --beta (cal targets +/- 1/(2*beta))
  - dpop    : --beta, --dpop_lambda
"""

import os
import re
import json
import math
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    model_name: str
    train_data_path: str
    valid_data_path: str
    output_dir: str

    # sequence lengths
    max_length: int = 8192
    max_prompt_length: int = 8192

    # method selection
    method: str = "dpo"  # dpo|simpo|ipo|cal_dpo|dpop

    # common hyperparams
    beta: float = 0.5

    # SimPO-specific
    simpo_gamma: float = 0.5

    # DPOP-specific
    dpop_lambda: float = 1.0

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1e9
    logging_steps: int = 10
    save_steps: int = 50
    eval_strategy: str = "no"  # "no"|"steps" etc, depending on transformers version

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
    path = Path(model_identifier)
    if path.exists() and path.is_dir():
        return "local", str(path.resolve())
    return "huggingface", model_identifier


def _find_subsequence(haystack: List[int], needle: List[int], start: int, end: int) -> Optional[int]:
    if not needle or end - start < len(needle):
        return None
    for i in range(start, end - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


# ============================================================================
# Data collator (NO weights; masks everything before the answer)
# ============================================================================
class DPOPairCollator:
    """
    Builds two sequences per example:
      chosen_text   = chat(system,user,assistant=chosen)
      rejected_text = chat(system,user,assistant=rejected)

    Then creates labels that supervise ONLY the answer portion (mask prompt tokens as -100).
    """

    def __init__(self, tokenizer, cfg: Config, system_prompt: str):
        self.tok = tokenizer
        self.cfg = cfg
        self.system_prompt = system_prompt

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id

        # Preserve answer tokens if truncation is needed
        self.tok.truncation_side = "left"
        self.tok.padding_side = "right"

        self._logged_one = False
        self._logged_mask_fallback = False

        if not getattr(self.tok, "is_fast", False):
            print("⚠️  WARNING: tokenizer is not fast; offset_mapping unavailable. "
                  "Will use token-subsequence fallback for masking (less robust).")

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.cfg.max_prompt_length:
            return prompt
        ids = ids[-self.cfg.max_prompt_length:]
        return self.tok.decode(ids, skip_special_tokens=False)

    def _build_full_text(self, prompt: str, answer: str) -> str:
        # Prefer chat template when available
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        # Fallback plain format
        return f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant: {answer}"

    def _make_labels_from_answer_offsets(
        self,
        full_text: str,
        input_ids: List[int],
        offsets: List[Tuple[int, int]],
        answer: str,
    ) -> List[int]:
        labels = list(input_ids)

        idx = full_text.rfind(answer)
        if idx == -1:
            return [-100] * len(labels)
        resp_start_char = idx

        for i, (s, e) in enumerate(offsets):
            # Special tokens often have (0,0)
            if s < resp_start_char:
                labels[i] = -100
        return labels

    def _make_labels_from_answer_token_fallback(self, input_ids: List[int], answer: str) -> List[int]:
        labels = list(input_ids)
        answer_ids = self.tok.encode(answer, add_special_tokens=False)
        if not answer_ids:
            return [-100] * len(labels)

        pos = _find_subsequence(input_ids, answer_ids, 0, len(input_ids))
        if pos is None:
            return [-100] * len(labels)

        for i in range(pos):
            labels[i] = -100

        if not self._logged_mask_fallback:
            print("⚠️  Using token-subsequence fallback masking (no offset_mapping). "
                  "Use a fast tokenizer for best masking.")
            self._logged_mask_fallback = True

        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [],
            "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": [],
        }

        for idx, ex in enumerate(features):
            prompt = self._truncate_prompt(ex["prompt"])
            chosen_resp = ex["chosen"]
            rejected_resp = ex["rejected"]

            if not self._logged_one:
                print("\n" + "=" * 80)
                print("DPO VARIANTS: Example CHOSEN/REJECTED preview")
                print("=" * 80)
                print("CHOSEN (first 300 chars):\n", chosen_resp[:300])
                print("\nREJECTED (first 300 chars):\n", rejected_resp[:300])
                print("=" * 80 + "\n")
                self._logged_one = True

            chosen_text = self._build_full_text(prompt, chosen_resp)
            rejected_text = self._build_full_text(prompt, rejected_resp)

            chosen_enc = self.tok(
                chosen_text,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                add_special_tokens=True,
                return_offsets_mapping=getattr(self.tok, "is_fast", False),
            )
            rejected_enc = self.tok(
                rejected_text,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                add_special_tokens=True,
                return_offsets_mapping=getattr(self.tok, "is_fast", False),
            )

            c_ids = chosen_enc["input_ids"]
            r_ids = rejected_enc["input_ids"]
            c_am = chosen_enc["attention_mask"]
            r_am = rejected_enc["attention_mask"]
            c_offsets = chosen_enc.get("offset_mapping", None)
            r_offsets = rejected_enc.get("offset_mapping", None)

            if c_offsets is not None:
                c_labels = self._make_labels_from_answer_offsets(chosen_text, c_ids, c_offsets, chosen_resp)
            else:
                c_labels = self._make_labels_from_answer_token_fallback(c_ids, chosen_resp)

            if r_offsets is not None:
                r_labels = self._make_labels_from_answer_offsets(rejected_text, r_ids, r_offsets, rejected_resp)
            else:
                r_labels = self._make_labels_from_answer_token_fallback(r_ids, rejected_resp)

            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_am)
            batch["chosen_labels"].append(c_labels)

            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_am)
            batch["rejected_labels"].append(r_labels)

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

        out: Dict[str, torch.Tensor] = {
            "chosen_input_ids": torch.tensor(batch["chosen_input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(batch["chosen_attention_mask"], dtype=torch.long),
            "chosen_labels": torch.tensor(batch["chosen_labels"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(batch["rejected_input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(batch["rejected_attention_mask"], dtype=torch.long),
            "rejected_labels": torch.tensor(batch["rejected_labels"], dtype=torch.long),
        }
        return out


# ============================================================================
# Logp helper (fused CE, returns sum logp and token count on supervised tokens)
# ============================================================================
def sequence_logp_sum_and_len(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      sum_logp: [B] sum of token log-probs over supervised (labels != -100) tokens
      tok_len : [B] number of supervised tokens (>=1)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B,T,V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())  # [B,T-1]

    token_mask = (shift_labels != -100).float()
    tok_len = token_mask.sum(dim=1).clamp(min=1.0)

    sum_logp = (-(per_token_nll) * token_mask).sum(dim=1)  # [B]
    return sum_logp, tok_len


# ============================================================================
# Adapter-off context for PEFT reference policy
# ============================================================================
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


# ============================================================================
# Trainer implementing 5 losses
# ============================================================================
class DPOVariantTrainer(Trainer):
    def __init__(self, *args, cfg: Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def _policy_logps(self, model, c_ids, c_am, c_lab, r_ids, r_am, r_lab):
        pi_c_sum, pi_c_len = sequence_logp_sum_and_len(model, c_ids, c_am, c_lab)
        pi_r_sum, pi_r_len = sequence_logp_sum_and_len(model, r_ids, r_am, r_lab)
        return pi_c_sum, pi_c_len, pi_r_sum, pi_r_len

    def _ref_logps(self, model, c_ids, c_am, c_lab, r_ids, r_am, r_lab):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with RefAdapterOff(model):
                ref_c_sum, ref_c_len = sequence_logp_sum_and_len(model, c_ids, c_am, c_lab)
                ref_r_sum, ref_r_len = sequence_logp_sum_and_len(model, r_ids, r_am, r_lab)
        if was_training:
            model.train()
        return ref_c_sum, ref_c_len, ref_r_sum, ref_r_len

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        c_ids = inputs["chosen_input_ids"]
        c_am  = inputs["chosen_attention_mask"]
        c_lab = inputs["chosen_labels"]

        r_ids = inputs["rejected_input_ids"]
        r_am  = inputs["rejected_attention_mask"]
        r_lab = inputs["rejected_labels"]

        method = self.cfg.method.lower()
        beta = float(self.cfg.beta)

        # policy logps (sum + len)
        pi_c_sum, pi_c_len, pi_r_sum, pi_r_len = self._policy_logps(
            model, c_ids, c_am, c_lab, r_ids, r_am, r_lab
        )

        # reference logps when needed
        need_ref = method in ("dpo", "ipo", "cal_dpo", "dpop")
        if need_ref:
            ref_c_sum, ref_c_len, ref_r_sum, ref_r_len = self._ref_logps(
                model, c_ids, c_am, c_lab, r_ids, r_am, r_lab
            )
        else:
            ref_c_sum = ref_r_sum = None

        # helper: DPO-style log-ratio gap h = (logπc-logπr) - (logπref_c-logπref_r)
        if need_ref:
            h = (pi_c_sum - pi_r_sum) - (ref_c_sum - ref_r_sum)
        else:
            h = (pi_c_sum - pi_r_sum)  # reference-free (SimPO uses a different construction)

        # -------------------------
        # Losses (per papers)
        # -------------------------
        if method == "dpo":
            # L = -log σ( beta * h )
            logits = beta * h
            loss_vec = -F.logsigmoid(logits)

        elif method == "simpo":
            # SimPO Eq.(6): -log σ( beta/|yw| logπ(yw|x) - beta/|yl| logπ(yl|x) - gamma )
            gamma = float(self.cfg.simpo_gamma)
            r_w = beta * (pi_c_sum / pi_c_len)
            r_l = beta * (pi_r_sum / pi_r_len)
            logits = (r_w - r_l) - gamma
            loss_vec = -F.logsigmoid(logits)

        elif method == "ipo":
            # IPO (sampled) square loss: (h - 1/(2*beta))^2   (constant factors omitted)
            target = 1.0 / (2.0 * max(beta, 1e-8))
            loss_vec = (h - target) ** 2

        elif method == "cal_dpo":
            # Cal-DPO Eq.(10):
            # preference loss: -log σ( (logπ/πref)_w - (logπ/πref)_l )
            # plus calibration squares pushing:
            #   (logπ/πref)_w ->  +1/(2β)
            #   (logπ/πref)_l ->  -1/(2β)
            logratio_w = (pi_c_sum - ref_c_sum)  # log πθ(yw|x) / πref(yw|x)
            logratio_l = (pi_r_sum - ref_r_sum)  # log πθ(yl|x) / πref(yl|x)

            pref_logits = (logratio_w - logratio_l)  # NOTE: no beta in Eq.(10)
            pref_loss = -F.logsigmoid(pref_logits)

            t = 1.0 / (2.0 * max(beta, 1e-8))
            cal_loss = (logratio_w - t) ** 2 + (logratio_l + t) ** 2

            loss_vec = pref_loss + cal_loss

        elif method == "dpop":
            # DPO-Positive (Smaug) Eq.(3):
            # -log σ( beta * ( (logπ/πref)_w - (logπ/πref)_l - λ * max(0, log(πref/πθ)_w ) ) )
            # where log(πref/πθ)_w = -log(πθ/πref)_w = -logratio_w
            lam = float(self.cfg.dpop_lambda)
            logratio_w = (pi_c_sum - ref_c_sum)
            logratio_l = (pi_r_sum - ref_r_sum)

            penalty = torch.clamp(-logratio_w, min=0.0)  # max(0, log πref/πθ)
            logits = beta * ((logratio_w - logratio_l) - lam * penalty)
            loss_vec = -F.logsigmoid(logits)

        else:
            raise ValueError(f"Unknown --method {self.cfg.method}. Choose from: dpo|simpo|ipo|cal_dpo|dpop")

        loss = loss_vec.mean()

        # -------------------------
        # Logging
        # -------------------------
        with torch.no_grad():
            metrics = {
                "loss": loss.detach().float(),
                "method": 0.0,  # placeholder, string not supported in HF scalar logs
                "pi/delta": (pi_c_sum - pi_r_sum).mean().detach().float(),
            }
            if need_ref:
                metrics["ref/delta"] = (ref_c_sum - ref_r_sum).mean().detach().float()
                metrics["h/mean"] = h.mean().detach().float()
            if method == "simpo":
                metrics["simpo/gamma"] = torch.tensor(self.cfg.simpo_gamma, device=loss.device).float()
            if method == "dpop":
                metrics["dpop/lambda"] = torch.tensor(self.cfg.dpop_lambda, device=loss.device).float()

        self.log({k: float(v.item()) for k, v in metrics.items() if torch.is_tensor(v)})

        return (loss, metrics) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


def str2bool(v):
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
    parser = argparse.ArgumentParser()

    # required paths
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # hardware
    parser.add_argument("--num_gpus", type=int, default=1)

    # chat system prompt
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")

    # method selection
    parser.add_argument("--method", type=str, default="dpo",
                        choices=["dpo", "simpo", "ipo", "cal_dpo", "dpop"])

    # common hyperparams
    parser.add_argument("--beta", type=float, default=0.5)

    # SimPO
    parser.add_argument("--simpo_gamma", type=float, default=0.5)

    # DPOP
    parser.add_argument("--dpop_lambda", type=float, default=1.0)

    # lengths
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=8192)

    # training hparams
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1e9)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="no")  # "no" or "steps"

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    cfg = Config(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,
        method=args.method,
        beta=args.beta,
        simpo_gamma=args.simpo_gamma,
        dpop_lambda=args.dpop_lambda,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"TRAINING DPO VARIANT: {cfg.method.upper()}")
    print("=" * 80)

    source_type, model_path = detect_model_source(cfg.model_name)
    print(f"Model source: {source_type} | Model path/ID: {model_path}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=hf_token,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    # truncation_side is set in collator (left) to preserve answers if needed
    print("✓ Tokenizer loaded")

    print("\nLoading datasets...")
    train_ds = load_from_disk(cfg.train_data_path)
    valid_ds = load_from_disk(cfg.valid_data_path)
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")

    needed = {"prompt", "chosen", "rejected"}
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        cols = set(ds.column_names)
        missing = needed - cols
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

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

    print("\nCreating data collator...")
    collator = DPOPairCollator(tok, cfg=cfg, system_prompt=args.system_prompt)

    # TrainingArguments compatibility: some versions use evaluation_strategy, others use eval_strategy.
    ta_kwargs = dict(
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
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in sig:
        ta_kwargs["eval_strategy"] = cfg.eval_strategy
        if cfg.eval_strategy == "steps":
            ta_kwargs["eval_steps"] = cfg.save_steps
    else:
        ta_kwargs["evaluation_strategy"] = cfg.eval_strategy
        if cfg.eval_strategy == "steps":
            ta_kwargs["eval_steps"] = cfg.save_steps

    training_args = TrainingArguments(**ta_kwargs)

    trainer = DPOVariantTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds if cfg.eval_strategy != "no" else None,
        data_collator=collator,
        tokenizer=tok,
        cfg=cfg,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)

    model_info = {
        "base_model": cfg.model_name,
        "base_model_source": source_type,
        "base_model_resolved_path": model_path,
        "method": cfg.method,
        "beta": cfg.beta,
        "simpo_gamma": cfg.simpo_gamma if cfg.method == "simpo" else None,
        "dpop_lambda": cfg.dpop_lambda if cfg.method == "dpop" else None,
        "training_completed": True,
    }
    with open(os.path.join(cfg.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Output saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
