import os
import json
import argparse
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================================
# Config
# ============================================================================

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
    method: str = "dpo"  # dpo|simpo|ipo|cal_dpo|dpop|tdpo|ti_dpo

    # common hyperparams
    beta: float = 0.5

    # SimPO-specific
    simpo_gamma: float = 0.5

    # DPOP-specific
    dpop_lambda: float = 1.0

    # TDPO-specific
    tdpo_alpha: float = 0.5
    tdpo_variant: str = "tdpo2"  # tdpo1|tdpo2

    # TI-DPO-specific
    tidpo_lambda_importance: float = 0.5
    tidpo_prior_sigma_div: float = 8.0
    tidpo_gamma: float = 0.1
    tidpo_alpha_triplet: float = 0.0
    tidpo_use_tdpo_base: bool = False
    tidpo_enable_gradient_attribution: bool = True
    tidpo_anchor_top_k: int = 50
    tidpo_anchor_top_p: float = 0.95
    tidpo_anchor_temperature: float = 0.8
    tidpo_anchor_max_new_tokens: int = 64

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
# Model/path helpers
# ============================================================================

def detect_model_source(model_identifier: str) -> Tuple[str, str]:
    if os.path.isdir(model_identifier):
        return "local", os.path.abspath(model_identifier)
    return "huggingface", model_identifier


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

class DPOPairCollator:
    """
    Builds:
      prompt_text   = chat(system, user, assistant generation prompt)
      chosen_text   = chat(system, user, assistant=chosen)
      rejected_text = chat(system, user, assistant=rejected)

    Labels supervise ONLY the answer region (prompt tokens masked as -100).
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
            print(
                "WARNING: tokenizer is not fast; offset_mapping unavailable. "
                "Will use token-subsequence fallback for masking."
            )

    def _truncate_prompt(self, prompt: str) -> str:
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.cfg.max_prompt_length:
            return prompt
        ids = ids[-self.cfg.max_prompt_length:]
        return self.tok.decode(ids, skip_special_tokens=False)

    def _build_prompt_text(self, prompt: str) -> str:
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"

    def _build_full_text(self, prompt: str, answer: str) -> str:
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
        for i, (s, _) in enumerate(offsets):
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
            print("WARNING: using token-subsequence fallback masking.")
            self._logged_mask_fallback = True
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            "prompt_input_ids": [], "prompt_attention_mask": [],
            "chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [],
            "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": [],
        }

        for ex in features:
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

            prompt_text = self._build_prompt_text(prompt)
            chosen_text = self._build_full_text(prompt, chosen_resp)
            rejected_text = self._build_full_text(prompt, rejected_resp)

            prompt_enc = self.tok(
                prompt_text,
                truncation=True,
                max_length=self.cfg.max_length,
                padding=False,
                add_special_tokens=True,
            )
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

            batch["prompt_input_ids"].append(prompt_enc["input_ids"])
            batch["prompt_attention_mask"].append(prompt_enc["attention_mask"])

            batch["chosen_input_ids"].append(c_ids)
            batch["chosen_attention_mask"].append(c_am)
            batch["chosen_labels"].append(c_labels)

            batch["rejected_input_ids"].append(r_ids)
            batch["rejected_attention_mask"].append(r_am)
            batch["rejected_labels"].append(r_labels)

        max_len = 0
        for k, v in batch.items():
            if k.endswith("input_ids") or k.endswith("attention_mask") or k.endswith("labels"):
                max_len = max(max_len, max(len(x) for x in v))

        def pad_int(seqs, pad_val):
            return [s + [pad_val] * (max_len - len(s)) for s in seqs]

        out: Dict[str, torch.Tensor] = {
            "prompt_input_ids": torch.tensor(pad_int(batch["prompt_input_ids"], self.pad_id), dtype=torch.long),
            "prompt_attention_mask": torch.tensor(pad_int(batch["prompt_attention_mask"], 0), dtype=torch.long),
            "chosen_input_ids": torch.tensor(pad_int(batch["chosen_input_ids"], self.pad_id), dtype=torch.long),
            "chosen_attention_mask": torch.tensor(pad_int(batch["chosen_attention_mask"], 0), dtype=torch.long),
            "chosen_labels": torch.tensor(pad_int(batch["chosen_labels"], -100), dtype=torch.long),
            "rejected_input_ids": torch.tensor(pad_int(batch["rejected_input_ids"], self.pad_id), dtype=torch.long),
            "rejected_attention_mask": torch.tensor(pad_int(batch["rejected_attention_mask"], 0), dtype=torch.long),
            "rejected_labels": torch.tensor(pad_int(batch["rejected_labels"], -100), dtype=torch.long),
        }
        return out


# ============================================================================
# Log-prob helpers
# ============================================================================

def sequence_logp_sum_and_len(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    per_token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())

    token_mask = (shift_labels != -100).float()
    tok_len = token_mask.sum(dim=1).clamp(min=1.0)
    sum_logp = (-(per_token_nll) * token_mask).sum(dim=1)
    return sum_logp, tok_len


def log_ratio_kl_and_policy_logp_from_logits(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        denom = loss_mask.sum(-1).clamp_min(1)
        return (
            (logps_margin * loss_mask).sum(-1) / denom,
            (per_position_kl * loss_mask).sum(-1) / denom,
            (per_token_logps * loss_mask).sum(-1) / denom,
        )
    return (
        (logps_margin * loss_mask).sum(-1),
        (per_position_kl * loss_mask).sum(-1),
        (per_token_logps * loss_mask).sum(-1),
    )


def weighted_dpo_margin_and_policy_logp_from_logits(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    weight_matrix: torch.Tensor,
    average_log_prob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape
    assert weight_matrix.shape == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    weight_matrix = weight_matrix[:, 1:]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_logps = reference_logits.log_softmax(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    logps_margin = per_token_logps - per_reference_token_logps

    weighted_logps_margin = logps_margin * weight_matrix

    if average_log_prob:
        denom = (loss_mask.to(weighted_logps_margin.dtype) * weight_matrix).sum(-1).clamp_min(1e-8)
        return (
            (weighted_logps_margin * loss_mask).sum(-1) / denom,
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1),
        )
    return (
        (weighted_logps_margin * loss_mask).sum(-1),
        (per_token_logps * loss_mask).sum(-1),
    )


def weighted_tdpo_stats_from_logits(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    weight_matrix: torch.Tensor,
    average_log_prob: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape
    assert weight_matrix.shape == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    weight_matrix = weight_matrix[:, 1:]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    logps_margin = per_token_logps - per_reference_token_logps

    weighted_margin = logps_margin * weight_matrix

    if average_log_prob:
        denom = (loss_mask.to(weighted_margin.dtype) * weight_matrix).sum(-1).clamp_min(1e-8)
        return (
            (weighted_margin * loss_mask).sum(-1) / denom,
            (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1),
            (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1),
        )
    return (
        (weighted_margin * loss_mask).sum(-1),
        (per_position_kl * loss_mask).sum(-1),
        (per_token_logps * loss_mask).sum(-1),
    )


# ============================================================================
# PEFT reference-policy context
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
# Token-level helpers for TI-DPO
# ============================================================================

def compute_gradient_attribution_from_ids(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
) -> torch.FloatTensor:
    """
    TI-DPO paper style gradient attribution:
      - target = max logit at the last valid position
      - gradient wrt input embeddings
      - token importance = L1 norm of that gradient vector
    """
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)

    original_training = model.training
    model.eval()
    try:
        embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        lengths = attention_mask.to(torch.long).sum(dim=1)
        last_pos = torch.clamp(lengths - 1, min=0)
        batch_idx = torch.arange(input_ids.shape[0], device=device)
        last_logits = logits[batch_idx, last_pos, :]
        target = last_logits.max(dim=-1).values
        grads = torch.autograd.grad(
            outputs=target.sum(),
            inputs=embeddings,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        importances = grads.abs().sum(dim=-1)
        importances = importances * attention_mask.to(importances.dtype)
        return importances.detach()
    finally:
        model.train(original_training)


def left_pad_batch(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    pad_id: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    device = input_ids.device
    bsz, seqlen = input_ids.shape
    lengths = attention_mask.to(torch.long).sum(dim=1)
    out_ids = torch.full_like(input_ids, pad_id)
    out_am = torch.zeros_like(attention_mask)
    for i in range(bsz):
        l = int(lengths[i].item())
        if l <= 0:
            continue
        valid_ids = input_ids[i, :l]
        out_ids[i, seqlen - l:] = valid_ids
        out_am[i, seqlen - l:] = 1
    return out_ids.to(device), out_am.to(device)


def pad_to_length_2d(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    if x.shape[1] >= target_len:
        return x
    pad = torch.full((x.shape[0], target_len - x.shape[1]), pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


# ============================================================================
# Trainer implementing the 7 losses
# ============================================================================

class DPOVariantTrainer(Trainer):
    def __init__(self, *args, cfg: Config, tokenizer=None, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.cfg = cfg
        self.tok = tokenizer

    # -------------------------
    # Sequence-level helpers
    # -------------------------
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

    # -------------------------
    # Token-level helpers
    # -------------------------
    def _forward_logits(self, model, input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.to(torch.float32)

    def _ref_logits(self, model, input_ids, attention_mask):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with RefAdapterOff(model):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.to(torch.float32)
        if was_training:
            model.train()
        return logits

    def _compute_token_importance_weights(
        self,
        model: torch.nn.Module,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        weight_matrix = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

        lam = float(max(0.0, min(1.0, self.cfg.tidpo_lambda_importance)))
        prior_sigma_div = float(max(1.0, self.cfg.tidpo_prior_sigma_div))

        if self.cfg.tidpo_enable_gradient_attribution:
            with torch.enable_grad():
                importances = compute_gradient_attribution_from_ids(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            importances = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)

        for i in range(batch_size):
            valid_mask = attention_mask[i].to(torch.bool)
            if labels is not None:
                valid_mask = valid_mask & (labels[i] != -100)

            valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            if valid_idx.numel() <= 1:
                weight_matrix[i].zero_()
                continue

            scores = importances[i][valid_idx].to(torch.float32).clamp_min(0.0)
            if scores.sum() > 0:
                norm_scores = scores / scores.sum()
            else:
                norm_scores = None

            n = int(valid_idx.numel())
            pos = torch.arange(n, device=device, dtype=torch.float32)
            center = (n - 1) / 2.0
            sigma = max(1.0, n / prior_sigma_div)
            prior = torch.exp(-0.5 * ((pos - center) / sigma) ** 2)
            prior = prior / prior.sum().clamp_min(1e-8)

            if norm_scores is not None:
                mixed = lam * norm_scores + (1.0 - lam) * prior
            else:
                mixed = prior
            mixed = mixed / mixed.sum().clamp_min(1e-8)

            # keep expected scale comparable to the unweighted token sum
            mixed = mixed * float(n)

            weight_matrix[i].zero_()
            weight_matrix[i][valid_idx] = mixed

        return weight_matrix

    def _log_ratio_sequence_from_logits(
        self,
        logits: torch.Tensor,
        ref_logits: torch.Tensor,
        input_ids: torch.LongTensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)

        seq_ids = input_ids[:, 1:].clone()
        seq_ids = torch.clamp(seq_ids, 0, logits.shape[-1] - 1)

        logp_t = torch.gather(logp[:, :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)
        refp_t = torch.gather(ref_logp[:, :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)

        log_ratio = logp_t - refp_t
        mask = token_mask[:, 1:].to(torch.bool)
        log_ratio = log_ratio * mask.to(log_ratio.dtype)
        return log_ratio, mask

    def _generate_anchor_outputs(
        self,
        model: torch.nn.Module,
        prompt_input_ids: torch.LongTensor,
        prompt_attention_mask: torch.LongTensor,
    ) -> torch.LongTensor:
        pad_id = int(self.tok.pad_token_id)
        prompt_input_ids, prompt_attention_mask = left_pad_batch(
            prompt_input_ids, prompt_attention_mask, pad_id
        )

        max_prompt_len = int(prompt_attention_mask.to(torch.long).sum(dim=1).max().item())
        remaining = max(1, int(self.cfg.max_length) - max_prompt_len)
        max_new_tokens = min(int(self.cfg.tidpo_anchor_max_new_tokens), remaining)

        with torch.no_grad():
            anchor_outputs = model.generate(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                do_sample=True,
                top_k=int(self.cfg.tidpo_anchor_top_k),
                top_p=float(self.cfg.tidpo_anchor_top_p),
                temperature=float(self.cfg.tidpo_anchor_temperature),
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_id,
                eos_token_id=self.tok.eos_token_id,
            )

        # Right-pad back to max_length for downstream masking/alignment.
        if anchor_outputs.shape[1] < self.cfg.max_length:
            pad = torch.full(
                (anchor_outputs.shape[0], self.cfg.max_length - anchor_outputs.shape[1]),
                pad_id,
                device=anchor_outputs.device,
                dtype=anchor_outputs.dtype,
            )
            anchor_outputs = torch.cat([anchor_outputs, pad], dim=1)
        else:
            anchor_outputs = anchor_outputs[:, : self.cfg.max_length]
        return anchor_outputs

    def _compute_triplet_loss(
        self,
        model: torch.nn.Module,
        c_ids: torch.LongTensor,
        c_lab: torch.LongTensor,
        r_ids: torch.LongTensor,
        r_lab: torch.LongTensor,
        prompt_ids: torch.LongTensor,
        prompt_am: torch.LongTensor,
    ) -> torch.Tensor:
        if float(self.cfg.tidpo_alpha_triplet) <= 0.0:
            return torch.tensor(0.0, device=c_ids.device, dtype=torch.float32)

        anchor_ids = self._generate_anchor_outputs(model, prompt_ids, prompt_am)
        anchor_am = (anchor_ids != int(self.tok.pad_token_id)).long()
        anchor_logits = self._forward_logits(model, anchor_ids, anchor_am)
        anchor_ref_logits = self._ref_logits(model, anchor_ids, anchor_am)

        c_logits = self._forward_logits(model, c_ids, (c_ids != int(self.tok.pad_token_id)).long())
        c_ref_logits = self._ref_logits(model, c_ids, (c_ids != int(self.tok.pad_token_id)).long())

        r_logits = self._forward_logits(model, r_ids, (r_ids != int(self.tok.pad_token_id)).long())
        r_ref_logits = self._ref_logits(model, r_ids, (r_ids != int(self.tok.pad_token_id)).long())

        pos_mask_full = (c_lab != -100)
        neg_mask_full = (r_lab != -100)

        prompt_lens = prompt_am.to(torch.long).sum(dim=1)
        anchor_mask_full = torch.zeros_like(anchor_ids, dtype=torch.bool)
        for i in range(anchor_ids.shape[0]):
            pl = int(prompt_lens[i].item())
            anchor_mask_full[i, pl:] = anchor_ids[i, pl:] != int(self.tok.pad_token_id)

        d_anchor, m_anchor = self._log_ratio_sequence_from_logits(anchor_logits, anchor_ref_logits, anchor_ids, anchor_mask_full)
        d_pos, m_pos = self._log_ratio_sequence_from_logits(c_logits, c_ref_logits, c_ids, pos_mask_full)
        d_neg, m_neg = self._log_ratio_sequence_from_logits(r_logits, r_ref_logits, r_ids, neg_mask_full)

        max_len = max(d_anchor.shape[1], d_pos.shape[1], d_neg.shape[1])
        d_anchor = pad_to_length_2d(d_anchor, max_len, 0.0)
        d_pos = pad_to_length_2d(d_pos, max_len, 0.0)
        d_neg = pad_to_length_2d(d_neg, max_len, 0.0)
        m_anchor = pad_to_length_2d(m_anchor.to(torch.float32), max_len, 0.0).to(torch.bool)
        m_pos = pad_to_length_2d(m_pos.to(torch.float32), max_len, 0.0).to(torch.bool)
        m_neg = pad_to_length_2d(m_neg.to(torch.float32), max_len, 0.0).to(torch.bool)

        diff_pos = d_anchor - d_pos
        diff_neg = d_anchor - d_neg

        mask_pos = (m_anchor & m_pos).to(diff_pos.dtype)
        mask_neg = (m_anchor & m_neg).to(diff_neg.dtype)

        dist_pos = torch.sum((diff_pos ** 2) * mask_pos, dim=-1)
        dist_neg = torch.sum((diff_neg ** 2) * mask_neg, dim=-1)

        margin = float(self.cfg.tidpo_alpha_triplet)
        triplet_loss = F.relu(dist_pos - dist_neg + margin).mean()
        return triplet_loss

    # -------------------------
    # Main loss
    # -------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        c_ids = inputs["chosen_input_ids"]
        c_am = inputs["chosen_attention_mask"]
        c_lab = inputs["chosen_labels"]

        r_ids = inputs["rejected_input_ids"]
        r_am = inputs["rejected_attention_mask"]
        r_lab = inputs["rejected_labels"]

        prompt_ids = inputs["prompt_input_ids"]
        prompt_am = inputs["prompt_attention_mask"]

        method = self.cfg.method.lower()
        beta = float(self.cfg.beta)

        # ------------------------------------------------------------------
        # Original five sequence-level methods
        # ------------------------------------------------------------------
        if method in ("dpo", "simpo", "ipo", "cal_dpo", "dpop"):
            pi_c_sum, pi_c_len, pi_r_sum, pi_r_len = self._policy_logps(
                model, c_ids, c_am, c_lab, r_ids, r_am, r_lab
            )

            need_ref = method in ("dpo", "ipo", "cal_dpo", "dpop")
            if need_ref:
                ref_c_sum, ref_c_len, ref_r_sum, ref_r_len = self._ref_logps(
                    model, c_ids, c_am, c_lab, r_ids, r_am, r_lab
                )
            else:
                ref_c_sum = ref_r_sum = None

            if need_ref:
                h = (pi_c_sum - pi_r_sum) - (ref_c_sum - ref_r_sum)
            else:
                h = (pi_c_sum - pi_r_sum)

            if method == "dpo":
                logits = beta * h
                loss_vec = -F.logsigmoid(logits)

            elif method == "simpo":
                gamma = float(self.cfg.simpo_gamma)
                r_w = beta * (pi_c_sum / pi_c_len)
                r_l = beta * (pi_r_sum / pi_r_len)
                logits = (r_w - r_l) - gamma
                loss_vec = -F.logsigmoid(logits)

            elif method == "ipo":
                target = 1.0 / (2.0 * max(beta, 1e-8))
                loss_vec = (h - target) ** 2

            elif method == "cal_dpo":
                logratio_w = (pi_c_sum - ref_c_sum)
                logratio_l = (pi_r_sum - ref_r_sum)
                pref_logits = (logratio_w - logratio_l)
                pref_loss = -F.logsigmoid(pref_logits)

                t = 1.0 / (2.0 * max(beta, 1e-8))
                cal_loss = (logratio_w - t) ** 2 + (logratio_l + t) ** 2
                loss_vec = pref_loss + cal_loss

            elif method == "dpop":
                lam = float(self.cfg.dpop_lambda)
                logratio_w = (pi_c_sum - ref_c_sum)
                logratio_l = (pi_r_sum - ref_r_sum)
                penalty = torch.clamp(-logratio_w, min=0.0)
                logits = beta * ((logratio_w - logratio_l) - lam * penalty)
                loss_vec = -F.logsigmoid(logits)

            loss = loss_vec.mean()

            with torch.no_grad():
                metrics = {
                    "loss": loss.detach().float(),
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

        # ------------------------------------------------------------------
        # TDPO
        # ------------------------------------------------------------------
        if method == "tdpo":
            c_logits = self._forward_logits(model, c_ids, c_am)
            r_logits = self._forward_logits(model, r_ids, r_am)
            c_ref_logits = self._ref_logits(model, c_ids, c_am)
            r_ref_logits = self._ref_logits(model, r_ids, r_am)

            c_margin, c_poskl, c_policy_logp = log_ratio_kl_and_policy_logp_from_logits(
                c_logits, c_ref_logits, c_lab, average_log_prob=False
            )
            r_margin, r_poskl, r_policy_logp = log_ratio_kl_and_policy_logp_from_logits(
                r_logits, r_ref_logits, r_lab, average_log_prob=False
            )

            if self.cfg.tdpo_variant.lower() == "tdpo1":
                tdpo_logits = (c_margin - r_margin) - (r_poskl - c_poskl)
            else:
                tdpo_logits = (c_margin - r_margin) - float(self.cfg.tdpo_alpha) * (r_poskl - c_poskl.detach())

            loss_vec = -F.logsigmoid(beta * tdpo_logits)
            loss = loss_vec.mean()

            with torch.no_grad():
                metrics = {
                    "loss": loss.detach().float(),
                    "tdpo/logits_mean": tdpo_logits.mean().detach().float(),
                    "tdpo/chosen_margin": c_margin.mean().detach().float(),
                    "tdpo/rejected_margin": r_margin.mean().detach().float(),
                    "tdpo/chosen_poskl": c_poskl.mean().detach().float(),
                    "tdpo/rejected_poskl": r_poskl.mean().detach().float(),
                    "logps/chosen": c_policy_logp.mean().detach().float(),
                    "logps/rejected": r_policy_logp.mean().detach().float(),
                }

            self.log({k: float(v.item()) for k, v in metrics.items() if torch.is_tensor(v)})
            return (loss, metrics) if return_outputs else loss

        # ------------------------------------------------------------------
        # TI-DPO
        # ------------------------------------------------------------------
        if method == "ti_dpo":
            c_logits = self._forward_logits(model, c_ids, c_am)
            r_logits = self._forward_logits(model, r_ids, r_am)
            c_ref_logits = self._ref_logits(model, c_ids, c_am)
            r_ref_logits = self._ref_logits(model, r_ids, r_am)

            c_weights = self._compute_token_importance_weights(model, c_ids, c_am, labels=c_lab)
            r_weights = self._compute_token_importance_weights(model, r_ids, r_am, labels=r_lab)

            if self.cfg.tidpo_use_tdpo_base:
                c_margin, c_poskl, c_policy_logp = weighted_tdpo_stats_from_logits(
                    c_logits, c_ref_logits, c_lab, c_weights, average_log_prob=False
                )
                r_margin, r_poskl, r_policy_logp = weighted_tdpo_stats_from_logits(
                    r_logits, r_ref_logits, r_lab, r_weights, average_log_prob=False
                )
                base_logits = (c_margin - r_margin) - float(self.cfg.tdpo_alpha) * (r_poskl - c_poskl.detach())
                base_loss_vec = -F.logsigmoid(beta * base_logits)
            else:
                c_margin, c_policy_logp = weighted_dpo_margin_and_policy_logp_from_logits(
                    c_logits, c_ref_logits, c_lab, c_weights, average_log_prob=False
                )
                r_margin, r_policy_logp = weighted_dpo_margin_and_policy_logp_from_logits(
                    r_logits, r_ref_logits, r_lab, r_weights, average_log_prob=False
                )
                base_logits = c_margin - r_margin
                base_loss_vec = -F.logsigmoid(beta * base_logits)
                c_poskl = log_ratio_kl_and_policy_logp_from_logits(c_logits, c_ref_logits, c_lab, average_log_prob=False)[1]
                r_poskl = log_ratio_kl_and_policy_logp_from_logits(r_logits, r_ref_logits, r_lab, average_log_prob=False)[1]

            triplet_loss = self._compute_triplet_loss(
                model, c_ids, c_lab, r_ids, r_lab, prompt_ids, prompt_am
            )
            loss = (base_loss_vec + float(self.cfg.tidpo_gamma) * triplet_loss).mean()

            with torch.no_grad():
                metrics = {
                    "loss": loss.detach().float(),
                    "ti_dpo/base_logits_mean": base_logits.mean().detach().float(),
                    "ti_dpo/chosen_weight_mean": c_weights[c_lab != -100].mean().detach().float()
                        if (c_lab != -100).any() else torch.tensor(0.0, device=loss.device),
                    "ti_dpo/rejected_weight_mean": r_weights[r_lab != -100].mean().detach().float()
                        if (r_lab != -100).any() else torch.tensor(0.0, device=loss.device),
                    "ti_dpo/chosen_margin": c_margin.mean().detach().float(),
                    "ti_dpo/rejected_margin": r_margin.mean().detach().float(),
                    "ti_dpo/chosen_poskl": c_poskl.mean().detach().float(),
                    "ti_dpo/rejected_poskl": r_poskl.mean().detach().float(),
                    "ti_dpo/triplet": triplet_loss.detach().float(),
                    "ti_dpo/gamma": torch.tensor(self.cfg.tidpo_gamma, device=loss.device).float(),
                    "logps/chosen": c_policy_logp.mean().detach().float(),
                    "logps/rejected": r_policy_logp.mean().detach().float(),
                }

            self.log({k: float(v.item()) for k, v in metrics.items() if torch.is_tensor(v)})
            return (loss, metrics) if return_outputs else loss

        raise ValueError(
            f"Unknown --method {self.cfg.method}. "
            f"Choose from: dpo|simpo|ipo|cal_dpo|dpop|tdpo|ti_dpo"
        )

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# ============================================================================
# CLI helpers
# ============================================================================

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
    parser.add_argument(
        "--method",
        type=str,
        default="dpo",
        choices=["dpo", "simpo", "ipo", "cal_dpo", "dpop", "tdpo", "ti_dpo"],
    )

    # common hyperparams
    parser.add_argument("--beta", type=float, default=0.5)

    # SimPO
    parser.add_argument("--simpo_gamma", type=float, default=0.5)

    # DPOP
    parser.add_argument("--dpop_lambda", type=float, default=1.0)

    # TDPO
    parser.add_argument("--tdpo_alpha", type=float, default=0.5)
    parser.add_argument("--tdpo_variant", type=str, default="tdpo2", choices=["tdpo1", "tdpo2"])

    # TI-DPO
    parser.add_argument("--tidpo_lambda_importance", type=float, default=0.5)
    parser.add_argument("--tidpo_prior_sigma_div", type=float, default=8.0)
    parser.add_argument("--tidpo_gamma", type=float, default=0.1)
    parser.add_argument("--tidpo_alpha_triplet", type=float, default=0.0)
    parser.add_argument("--tidpo_use_tdpo_base", type=str2bool, default=False)
    parser.add_argument("--tidpo_enable_gradient_attribution", type=str2bool, default=True)
    parser.add_argument("--tidpo_anchor_top_k", type=int, default=50)
    parser.add_argument("--tidpo_anchor_top_p", type=float, default=0.95)
    parser.add_argument("--tidpo_anchor_temperature", type=float, default=0.8)
    parser.add_argument("--tidpo_anchor_max_new_tokens", type=int, default=64)

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
    parser.add_argument("--eval_strategy", type=str, default="no")

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
        tdpo_alpha=args.tdpo_alpha,
        tdpo_variant=args.tdpo_variant,
        tidpo_lambda_importance=args.tidpo_lambda_importance,
        tidpo_prior_sigma_div=args.tidpo_prior_sigma_div,
        tidpo_gamma=args.tidpo_gamma,
        tidpo_alpha_triplet=args.tidpo_alpha_triplet,
        tidpo_use_tdpo_base=args.tidpo_use_tdpo_base,
        tidpo_enable_gradient_attribution=args.tidpo_enable_gradient_attribution,
        tidpo_anchor_top_k=args.tidpo_anchor_top_k,
        tidpo_anchor_top_p=args.tidpo_anchor_top_p,
        tidpo_anchor_temperature=args.tidpo_anchor_temperature,
        tidpo_anchor_max_new_tokens=args.tidpo_anchor_max_new_tokens,
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
        "tdpo_alpha": cfg.tdpo_alpha if cfg.method in ("tdpo", "ti_dpo") else None,
        "tdpo_variant": cfg.tdpo_variant if cfg.method in ("tdpo", "ti_dpo") else None,
        "tidpo_lambda_importance": cfg.tidpo_lambda_importance if cfg.method == "ti_dpo" else None,
        "tidpo_prior_sigma_div": cfg.tidpo_prior_sigma_div if cfg.method == "ti_dpo" else None,
        "tidpo_gamma": cfg.tidpo_gamma if cfg.method == "ti_dpo" else None,
        "tidpo_alpha_triplet": cfg.tidpo_alpha_triplet if cfg.method == "ti_dpo" else None,
        "tidpo_use_tdpo_base": cfg.tidpo_use_tdpo_base if cfg.method == "ti_dpo" else None,
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
