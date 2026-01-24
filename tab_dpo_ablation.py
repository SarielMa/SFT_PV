#!/usr/bin/env python3
"""
Token-weighted DPO training (chosen/rejected) with:
- Robust prompt masking (mask everything before the *actual answer text*, not a hardcoded template)
- Token-level value weighting (Code/Sub-code/Span) + diff-token upweight
- Adaptive token-level barrier loss
- Memory-safe per-token logp via fused cross-entropy (avoids [B,T,V] log_softmax materialization)

Key fixes vs your pasted version:
1) Collator was using undefined names: tokenizer/system_prompt/example. Now uses self.tok / local ex vars.
2) Removed the old response_template masking path (it was still present and would keep warning).
3) make_labels_from_answer now handles missing offsets (fallback token search).
4) weighted_sequence_logp also uses fused CE (so eval at step 50 won't blow up due to log_softmax).
5) compute_loss no longer forces .to(model.device) (breaks with device_map="auto"); assumes Trainer prepared device.
6) TrainingArguments uses evaluation_strategy (not eval_strategy) for broad compatibility.
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
import math
from collections import Counter

# ----------------------------
# Config
# ----------------------------
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

    # Token weighting (toggleable; defaults preserve your current behavior)
    enable_token_weighting: bool = True
    token_weight_code: float = 1.1
    token_weight_subcode: float = 1.2
    token_weight_span: float = 1.1
    weight_diff_tokens: float = 1.2
    normalize_by_weight_mass: bool = True  # keeps avg token weight ~1.0 (your current behavior)

    # Length normalization (toggleable; default preserves current behavior)
    enable_length_norm: bool = True
    length_norm_by: str = "tokens"   # "tokens" | "weight_mass"

    # Class balancing (example reweighting; default OFF)
    enable_class_balance: bool = True
    class_balance_strategy: str = "effective_num"   # effective_num|inv_freq|inv_sqrt
    class_balance_beta: float = 0.99
    class_balance_alpha: float = 1.0
    class_balance_combine: str = "mean"             # max|geom|mean
    class_balance_max_weight: float = 3.0
    class_balance_use_code: bool = True
    class_balance_use_subcode: bool = True

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1e9
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
    path = Path(model_identifier)
    if path.exists() and path.is_dir():
        return "local", str(path.resolve())
    return "huggingface", model_identifier


# ============================================================================
# "JSON-ish" value extraction for weighting
# ============================================================================
_CODE_RE = re.compile(r'"Code"\s*:\s*"([^"]+)"')
_SUBCODE_RE = re.compile(r'"Sub-code"\s*:\s*"([^"]+)"|\"Subcode\"\s*:\s*\"([^\"]+)\"')
_SPAN_RE = re.compile(r'"Span"\s*:\s*"([^"]+)"')


def extract_values_from_jsonish(text: str) -> Dict[str, List[str]]:
    codes = _CODE_RE.findall(text)
    subs: List[str] = []
    for m in _SUBCODE_RE.findall(text):
        subs.append(m[0] if m[0] else m[1])
    spans = _SPAN_RE.findall(text)

    def uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {"code": uniq(codes), "subcode": uniq(subs), "span": uniq(spans)}

# =============================================================================
# CLASS IMBALANCE: EXAMPLE REWEIGHTING (DPO)
# =============================================================================

def compute_class_weights(
    counts: Dict[str, int],
    strategy: str,
    beta: float = 0.999,
    alpha: float = 1.0,
) -> Dict[str, float]:
    if not counts:
        return {}
    total = float(sum(counts.values()))
    out = {}
    for k, n in counts.items():
        n = max(int(n), 1)
        if strategy == "effective_num":
            w = (1.0 - beta) / (1.0 - (beta ** n))
        elif strategy == "inv_freq":
            w = (total / n) ** alpha
        elif strategy == "inv_sqrt":
            w = (total / n) ** (0.5 * alpha)
        else:
            raise ValueError(f"Unknown class_balance_strategy: {strategy}")
        out[k] = float(w)

    # normalize mean weight = 1
    mean_w = sum(out.values()) / max(len(out), 1)
    if mean_w > 0:
        out = {k: v / mean_w for k, v in out.items()}
    return out


def add_example_weights_dpo(train_ds, cfg: Config):
    """
    Adds column: example_weight (float).
    Uses the CHOSEN response to extract Code/Sub-code strings (like your SFT pipeline).
    """
    if not cfg.enable_class_balance:
        return train_ds.map(lambda ex: {"example_weight": 1.0}, desc="Setting example_weight=1.0")

    code_ctr = Counter()
    sub_ctr = Counter()

    # Count classes
    for ex in train_ds:
        vals = extract_values_from_jsonish(ex.get("chosen", ""))
        if cfg.class_balance_use_code:
            for c in vals.get("code", []):
                code_ctr[c] += 1
        if cfg.class_balance_use_subcode:
            for s in vals.get("subcode", []):
                sub_ctr[s] += 1

    code_w = compute_class_weights(dict(code_ctr), cfg.class_balance_strategy, cfg.class_balance_beta, cfg.class_balance_alpha) if cfg.class_balance_use_code else {}
    sub_w  = compute_class_weights(dict(sub_ctr),  cfg.class_balance_strategy, cfg.class_balance_beta, cfg.class_balance_alpha) if cfg.class_balance_use_subcode else {}

    def _combine(a: float, b: float) -> float:
        if cfg.class_balance_combine == "max":
            return max(a, b)
        if cfg.class_balance_combine == "geom":
            return math.sqrt(max(a, 1e-8) * max(b, 1e-8))
        if cfg.class_balance_combine == "mean":
            return 0.5 * (a + b)
        raise ValueError(f"Unknown class_balance_combine: {cfg.class_balance_combine}")

    def _one(ex):
        vals = extract_values_from_jsonish(ex.get("chosen", ""))

        cw = 1.0
        sw = 1.0
        if cfg.class_balance_use_code:
            codes = vals.get("code", [])
            if codes:
                cw = max(code_w.get(c, 1.0) for c in codes)

        if cfg.class_balance_use_subcode:
            subs = vals.get("subcode", [])
            if subs:
                sw = max(sub_w.get(s, 1.0) for s in subs)

        ew = float(min(_combine(cw, sw), cfg.class_balance_max_weight))
        return {"example_weight": ew}

    train_ds = train_ds.map(_one, desc="Computing example weights (class balance)")

    # Normalize mean example weight to 1.0 (then clamp)
    ws = [float(x) for x in train_ds["example_weight"]]
    mean_w = sum(ws) / max(len(ws), 1)
    if mean_w > 0:
        def _norm(ex):
            ew = float(ex["example_weight"]) / float(mean_w)
            ew = float(min(ew, cfg.class_balance_max_weight))
            return {"example_weight": ew}
        train_ds = train_ds.map(_norm, desc="Normalizing example weights (mean=1.0)")

    ws = [float(x) for x in train_ds["example_weight"]]
    print("\n" + "=" * 80)
    print("CLASS BALANCE SUMMARY (DPO)")
    print("=" * 80)
    print(f"Strategy: {cfg.class_balance_strategy} | beta={cfg.class_balance_beta} | alpha={cfg.class_balance_alpha} | combine={cfg.class_balance_combine} | max_w={cfg.class_balance_max_weight}")
    print(f"Unique Code classes: {len(code_w)} | Unique Subcode classes: {len(sub_w)}")
    print(f"Example weight: min={min(ws):.3f} mean={sum(ws)/len(ws):.3f} max={max(ws):.3f}")
    print("=" * 80)

    return train_ds


def _find_subsequence(haystack: List[int], needle: List[int], start: int, end: int) -> Optional[int]:
    if not needle or end - start < len(needle):
        return None
    for i in range(start, end - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


# ============================================================================
# Data collator
# ============================================================================
class DPODataCollatorTokenWeighted:
    """
    Builds two sequences per example:
      chosen_text  = chat(system,user,assistant=chosen)
      rejected_text= chat(system,user,assistant=rejected)

    Masks all tokens BEFORE the answer substring in each full_text.
    """

    def __init__(
        self,
        tokenizer,
        cfg: Config,
        system_prompt: str,
    ):
        self.tok = tokenizer
        self.cfg = cfg
        self.system_prompt = system_prompt

        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id

        self._logged_responses = False
        self._logged_mask_fallback = False

        if not getattr(self.tok, "is_fast", False):
            # We can still run without offsets, but offsets are strongly preferred.
            print("⚠️  WARNING: tokenizer is not a fast tokenizer; offset_mapping not available. "
                  "Will use token-subsequence fallback for masking, which is less robust.")

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.cfg.max_prompt_length:
            return prompt
        ids = ids[-self.cfg.max_prompt_length:]
        return self.tok.decode(ids, skip_special_tokens=False)

    def _build_full_text(self, prompt: str, answer: str) -> str:
        # Explicit assistant content is crucial: add_generation_prompt=True is NOT enough.
        return self.tok.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

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

    def _make_labels_from_answer_token_fallback(
        self,
        input_ids: List[int],
        answer: str,
    ) -> List[int]:
        """
        Fallback if offset mappings are unavailable.
        We find the answer token IDs inside the full sequence, and mask everything before it.
        """
        labels = list(input_ids)

        # Find answer token ids and locate in the full sequence
        answer_ids = self.tok.encode(answer, add_special_tokens=False)
        if not answer_ids:
            return [-100] * len(labels)

        # Find earliest match
        pos = _find_subsequence(input_ids, answer_ids, 0, len(input_ids))
        if pos is None:
            return [-100] * len(labels)

        for i in range(pos):
            labels[i] = -100

        if not self._logged_mask_fallback:
            print("⚠️  Using token-subsequence fallback masking (no offset_mapping). "
                  "Consider using a fast tokenizer for more reliable masking.")
            self._logged_mask_fallback = True

        return labels

    def _init_weights(self, labels: List[int]) -> List[float]:
        return [0.0 if lab == -100 else 1.0 for lab in labels]

    def _apply_value_weights_with_offsets(
        self,
        full_text: str,
        offsets: List[Tuple[int, int]],
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
        targets: List[Tuple[int, int, float]] = []

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
            "example_weight": [],
        }

        for idx, ex in enumerate(features):
            prompt = self._truncate_prompt(ex["prompt"])
            chosen_resp = ex["chosen"]
            rejected_resp = ex["rejected"]
            batch["example_weight"].append(float(ex.get("example_weight", 1.0)))

            if not self._logged_responses:
                print(f"\n{'='*80}")
                print(f"EXAMPLE #{idx + 1} - CHOSEN vs REJECTED RESPONSES")
                print(f"{'='*80}")
                print(f"\n✅ CHOSEN RESPONSE (first 300 chars):\n{chosen_resp[:300]}")
                print(f"\n❌ REJECTED RESPONSE (first 300 chars):\n{rejected_resp[:300]}")
                print(f"{'='*80}\n")
                self._logged_responses = True

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

            c_w = self._init_weights(c_labels)
            r_w = self._init_weights(r_labels)

            # Apply value weights (Code/Sub-code/Span)
            if self.cfg.enable_token_weighting:
                # Apply value weights (Code/Sub-code/Span)
                if c_offsets is not None:
                    self._apply_value_weights_with_offsets(
                        chosen_text, c_offsets, c_labels, c_w,
                        chosen_resp, self.cfg.token_weight_code, self.cfg.token_weight_subcode, self.cfg.token_weight_span
                    )
                else:
                    self._apply_value_weights_fallback_token_search(
                        c_ids, c_labels, c_w,
                        chosen_resp, self.cfg.token_weight_code, self.cfg.token_weight_subcode, self.cfg.token_weight_span
                    )

                if r_offsets is not None:
                    self._apply_value_weights_with_offsets(
                        rejected_text, r_offsets, r_labels, r_w,
                        rejected_resp, self.cfg.token_weight_code, self.cfg.token_weight_subcode, self.cfg.token_weight_span
                    )
                else:
                    self._apply_value_weights_fallback_token_search(
                        r_ids, r_labels, r_w,
                        rejected_resp, self.cfg.token_weight_code, self.cfg.token_weight_subcode, self.cfg.token_weight_span
                    )

                # Upweight differing tokens
                if self.cfg.weight_diff_tokens and self.cfg.weight_diff_tokens > 1.0:
                    self._apply_diff_weight(
                        c_ids, c_labels, c_w,
                        r_ids, r_labels, r_w,
                        self.cfg.weight_diff_tokens
                    )


            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_am)
            batch["chosen_labels"].append(c_labels)
            batch["chosen_weights"].append(c_w)

            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_am)
            batch["rejected_labels"].append(r_labels)
            batch["rejected_weights"].append(r_w)

        # Pad to common max_len across chosen/rejected
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

        out: Dict[str, torch.Tensor] = {}
        for k in batch:
            if ("weights" in k) or (k == "example_weight"):
                out[k] = torch.tensor(batch[k], dtype=torch.float32)
            else:
                out[k] = torch.tensor(batch[k], dtype=torch.long)
        return out

# ============================================================================
# Logp helpers (memory safe: CE instead of log_softmax)
# ============================================================================
def weighted_sequence_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      seq_logp: [B]  (avg if enable_length_norm else sum), using token weights if enabled
      denom:    [B]  (token count or weight mass depending on cfg)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B,T,V]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())

    per_token_logps = -per_token_nll  # [B,T-1]
    mask = (shift_labels != -100).float()
    tok_count = mask.sum(dim=1).clamp(min=1.0)

    if cfg.enable_token_weighting:
        w = shift_weights * mask
    else:
        w = mask  # all 1s on valid tokens

    # optional normalization: keep mean weight over valid tokens ≈ 1.0 (your current behavior)
    if cfg.normalize_by_weight_mass:
        w_sum = w.sum(dim=1).clamp(min=1e-6)
        w = w * (tok_count / w_sum).unsqueeze(1)

    sum_logp = (per_token_logps * w).sum(dim=1)

    if cfg.enable_length_norm:
        if cfg.length_norm_by == "weight_mass":
            denom = w.sum(dim=1).clamp(min=1e-6)
        else:
            denom = tok_count
        seq_logp = sum_logp / denom
        return seq_logp, denom
    else:
        denom = torch.ones_like(tok_count)
        return sum_logp, denom


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
# Trainer
# ============================================================================
class TWCDPOTrainer(Trainer):
    def __init__(self, *args, cfg: Config, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # NOTE: do NOT force inputs to model.device (breaks with device_map="auto")
        c_ids = inputs["chosen_input_ids"]
        c_am = inputs["chosen_attention_mask"]
        c_lab = inputs["chosen_labels"]
        c_w = inputs["chosen_weights"]

        r_ids = inputs["rejected_input_ids"]
        r_am = inputs["rejected_attention_mask"]
        r_lab = inputs["rejected_labels"]
        r_w = inputs["rejected_weights"]
        
        exw = inputs.get("example_weight", None)
        if (not self.cfg.enable_class_balance) or (exw is None):
            exw = torch.ones(c_ids.size(0), device=c_ids.device, dtype=torch.float32)
        else:
            exw = exw.to(c_ids.device).float().view(-1)

        # ---- DEBUG: number of supervised tokens in chosen response ----
        with torch.no_grad():
            valid_tokens = (c_lab[:, 1:] != -100).sum(dim=1)
            print("chosen_valid_tokens:", valid_tokens.tolist())


        device = c_ids.device

        def _seq_avg_logp_and_token_logps(_ids, _am, _lab, _wts):
            outputs = model(input_ids=_ids, attention_mask=_am, use_cache=False)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = _lab[:, 1:].contiguous()
            shift_weights = _wts[:, 1:].contiguous()

            per_token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.size())

            token_logps = -per_token_nll
            token_mask = (shift_labels != -100)
            mask_f = token_mask.float()

            w = shift_weights * mask_f

            # Token count (valid supervised tokens)
            tok_count = mask_f.sum(dim=1).clamp(min=1.0)
            
            # Optional normalization: keep mean weight over valid tokens = 1.0
            # Only do this if cfg.normalize_by_weight_mass is True
            if self.cfg.normalize_by_weight_mass:
                w_sum_raw = w.sum(dim=1).clamp(min=1e-6)
                w = w * (tok_count / w_sum_raw).unsqueeze(1)
            
            # For the barrier, we keep denom as token count (original behavior)
            w_sum = tok_count


            seq_logp = (token_logps * w).sum(dim=1)
            seq_logp_avg = seq_logp / w_sum
            return seq_logp_avg, token_logps, token_mask, w

        # Policy logps (adapter ON) — use same toggle-aware seq logp as ref
        pi_c, _ = weighted_sequence_logp(model, c_ids, c_am, c_lab, c_w, cfg=self.cfg)
        pi_r, _ = weighted_sequence_logp(model, r_ids, r_am, r_lab, r_w, cfg=self.cfg)
        
        # Keep per-token info for barrier exactly as before (do NOT change barrier behavior)
        _, c_token_logps, c_token_mask, c_token_w = _seq_avg_logp_and_token_logps(c_ids, c_am, c_lab, c_w)

        # Reference logps (adapter OFF)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with RefAdapterOff(model):
                ref_c, _ = weighted_sequence_logp(model, c_ids, c_am, c_lab, c_w, cfg=self.cfg)
                ref_r, _ = weighted_sequence_logp(model, r_ids, r_am, r_lab, r_w, cfg=self.cfg)
        if was_training:
            model.train()

        # DPO logits and loss
        pi_logratio = pi_c - pi_r
        ref_logratio = ref_c - ref_r
        dpo_logits = pi_logratio - ref_logratio
        p_pref = torch.sigmoid(self.cfg.beta * dpo_logits)
        dpo_loss = -F.logsigmoid(self.cfg.beta * dpo_logits)  # [B]
        dpo_loss_mean = (dpo_loss * exw).sum() / exw.sum().clamp(min=1e-8)

        # Adaptive token-level barrier
        print (f" self.cfg.policy_prob_threshold is {self.cfg.policy_prob_threshold}")
        log_tau = torch.log(torch.tensor(self.cfg.policy_prob_threshold, device=device, dtype=c_token_logps.dtype))
        gate = (c_token_logps < log_tau).float() * c_token_mask.float()  # [B,T-1]
        token_nll = -c_token_logps

        barrier_num = (token_nll * c_token_w * gate).sum(dim=1)  # [B]
        
        # Denominator follows length_norm_by:
        # - "tokens": count of active gated tokens
        # - "weight_mass": total weight mass over active gated tokens
        if self.cfg.enable_length_norm and self.cfg.length_norm_by == "weight_mass":
            barrier_den = (c_token_w * gate).sum(dim=1).clamp(min=1e-6)  # [B]
        else:
            barrier_den = gate.sum(dim=1).clamp(min=1.0)                 # [B]
        
        barrier_per_ex = (barrier_num / barrier_den) * self.cfg.sft_barrier_weight  # [B]

        barrier_loss = (barrier_per_ex * exw).sum() / exw.sum().clamp(min=1e-8)

        total = dpo_loss_mean + barrier_loss

        # Logging
        with torch.no_grad():
            policy_chosen_prob_geom = torch.exp(pi_c)
            active_tokens = c_token_mask.float().sum().clamp(min=1.0)
            barrier_active_pct = gate.sum() / active_tokens

            metrics = {
                "loss": total.detach().float(),
                "loss/dpo": dpo_loss_mean.detach().float(),
                "loss/barrier": barrier_loss.detach().float(),
                "policy/prob_chosen": policy_chosen_prob_geom.mean().detach().float(),
                "barrier/active_pct": barrier_active_pct.detach().float(),
                "pref/p_pref": p_pref.mean().detach().float(),
                "logits/mean": dpo_logits.mean().detach().float(),
                "logits/max": dpo_logits.max().detach().float(),
                "logits/min": dpo_logits.min().detach().float(),
            }
        self.log({k: v.item() for k, v in metrics.items()})

        return (total, metrics) if return_outputs else total

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
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--valid_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    # ----- toggles (match SFT flag names) -----
    parser.add_argument("--enable_length_norm", type=str2bool, default=False)
    parser.add_argument("--length_norm_by", type=str, default="tokens", choices=["tokens", "weight_mass"])

    parser.add_argument("--enable_token_weighting", type=str2bool, default=False)
    parser.add_argument("--token_weight_code", type=float, default=1.1)
    parser.add_argument("--token_weight_subcode", type=float, default=1.2)
    parser.add_argument("--token_weight_span", type=float, default=1.1)
    parser.add_argument("--normalize_by_weight_mass", type=str2bool, default=True)

    parser.add_argument("--enable_class_balance", type=str2bool, default=False)
    parser.add_argument("--class_balance_strategy", type=str, default="effective_num", choices=["effective_num", "inv_freq", "inv_sqrt"])
    parser.add_argument("--class_balance_beta", type=float, default=0.99)
    parser.add_argument("--class_balance_alpha", type=float, default=1.0)
    parser.add_argument("--class_balance_combine", type=str, default="mean", choices=["max", "geom", "mean"])
    parser.add_argument("--class_balance_max_weight", type=float, default=3.0)
    parser.add_argument("--class_balance_use_code", type=str2bool, default=True)
    parser.add_argument("--class_balance_use_subcode", type=str2bool, default=True)

    parser.add_argument("--policy_prob_threshold", type=float, default=0.66)
    
    args = parser.parse_args()

    cfg = Config(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        output_dir=args.output_dir,

        # toggles
        enable_length_norm=args.enable_length_norm,
        length_norm_by=args.length_norm_by,

        enable_token_weighting=args.enable_token_weighting,
        token_weight_code=args.token_weight_code,
        token_weight_subcode=args.token_weight_subcode,
        token_weight_span=args.token_weight_span,
        normalize_by_weight_mass=args.normalize_by_weight_mass,

        enable_class_balance=args.enable_class_balance,
        class_balance_strategy=args.class_balance_strategy,
        class_balance_beta=args.class_balance_beta,
        class_balance_alpha=args.class_balance_alpha,
        class_balance_combine=args.class_balance_combine,
        class_balance_max_weight=args.class_balance_max_weight,
        class_balance_use_code=args.class_balance_use_code,
        class_balance_use_subcode=args.class_balance_use_subcode,
        policy_prob_threshold=args.policy_prob_threshold,
        
    )

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 80)
    print("Token-Weighted DPO with Adaptive Barrier (FIXED)")
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
    print("✓ Tokenizer loaded")

    print("\nLoading datasets...")
    train_ds = load_from_disk(cfg.train_data_path)
    valid_ds = load_from_disk(cfg.valid_data_path)
    print(f"Train: {len(train_ds)} samples | Valid: {len(valid_ds)} samples")
    # Add example_weight column for class balancing (train only)
    train_ds = add_example_weights_dpo(train_ds, cfg)
    # For eval, keep neutral weights
    valid_ds = valid_ds.map(lambda ex: {"example_weight": 1.0}, desc="Setting valid example_weight=1.0")

    needed = {"prompt", "chosen", "rejected"}
    for name, ds in [("train", train_ds), ("valid", valid_ds)]:
        cols = set(ds.column_names)
        missing = needed - cols
        if missing:
            raise ValueError(f"{name} dataset missing columns: {missing}")

    print("\nLoading model...")
    # IMPORTANT: device_map="auto" is okay, but then we must not manually .to(model.device) in compute_loss.
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
    collator = DPODataCollatorTokenWeighted(tok, cfg=cfg, system_prompt=args.system_prompt)

    # training_args = TrainingArguments(
    #     output_dir=cfg.output_dir,
    #     num_train_epochs=cfg.num_train_epochs,
    #     per_device_train_batch_size=cfg.per_device_train_batch_size,
    #     gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    #     learning_rate=cfg.learning_rate,
    #     warmup_ratio=cfg.warmup_ratio,
    #     max_grad_norm=cfg.max_grad_norm,
    #     bf16=True,
    #     logging_steps=cfg.logging_steps,
    #     save_steps=cfg.save_steps,
    #     report_to="none",
    #     remove_unused_columns=False,
    #     gradient_checkpointing=True,
    #     gradient_checkpointing_kwargs={"use_reentrant": False},
    #     evaluation_strategy="steps",
    #     eval_steps=cfg.save_steps,
    # )
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
    
        # ✅ correct for 4.51.3
        #eval_strategy="steps",
        #eval_steps=cfg.save_steps,
        eval_strategy = "no",
    )



    trainer = TWCDPOTrainer(
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

    model_info = {
        "base_model": cfg.model_name,
        "base_model_source": source_type,
        "base_model_resolved_path": model_path,
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