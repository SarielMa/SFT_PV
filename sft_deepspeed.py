#!/usr/bin/env python3
"""
Clean multi-GPU (DeepSpeed/DDP) SFT with PEFT (QLoRA/LoRA) on an Arrow dataset:
  - query: full instruction + INPUT ... + OUTPUT:
  - answer: JSON string (target)

We build model input as:  query + "\\n" + answer
Loss is computed ONLY on answer tokens (prompt tokens are masked to -100).

Multi-GPU safety (no HF cache races):
  - Rank0 tokenizes ONCE and saves tokenized dataset to disk (+_DONE / +_ERROR flags)
  - Other ranks wait on DONE/ERROR and then load from disk
  - Reuses tokenized dataset if present (skip tokenization)

Hang diagnostics:
  - Adds stage prints
  - Adds file-based sync points after model load and after Trainer init (per-run unique flags)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# IMPORTANT: avoid occasional tokenizer thread deadlocks on shared HPC environments
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------
# Tokenization / Label masking
# ---------------------------
def build_features(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    query = example["query"].rstrip()
    answer = example["answer"].strip()

    prompt_text = query + "\n"
    full_text = prompt_text + answer

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = [-100] * len(input_ids)
    ans_start = min(len(prompt_ids), len(input_ids))
    for i in range(ans_start, len(input_ids)):
        labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SimpleCollator(DataCollatorWithPadding):
    """Pads input_ids/attention_mask/labels to the longest in batch. Pads labels with -100."""

    def __call__(self, features):
        batch = super().__call__([{k: v for k, v in f.items() if k != "labels"} for f in features])

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ---------------------------
# File helpers / barriers
# ---------------------------
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def wait_for_done_or_error(done_path: Path, err_path: Path, timeout_s: int = 3600) -> None:
    t0 = time.time()
    while True:
        if err_path.exists():
            msg = err_path.read_text(errors="ignore").strip()
            raise RuntimeError(f"Rank0 failed. Error: {msg[:2000]}")
        if done_path.exists():
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {done_path} (or {err_path}).")
        time.sleep(2)


def wait_for_flag(flag_path: Path, timeout_s: int = 3600) -> None:
    t0 = time.time()
    while not flag_path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for {flag_path}")
        time.sleep(2)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_peft_out")
    parser.add_argument("--max_length", type=int, default=8192)

    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    # PEFT/QLoRA toggles
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optional eval
    parser.add_argument("--do_eval", action="store_true")

    # Tokenize speed
    parser.add_argument(
        "--num_proc",
        type=int,
        default=0,
        help="Rank0 tokenization workers. 0 => SLURM_CPUS_PER_TASK or 8.",
    )

    args = parser.parse_args()

    # ---- Robust GPU binding for DeepSpeed/DDP ----
    local_rank = args.local_rank
    if local_rank is None or local_rank < 0:
        env_lr = os.environ.get("LOCAL_RANK")
        if env_lr is not None:
            local_rank = int(env_lr)
    if local_rank is not None and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # ---- Rank info ----
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.model_name.replace("/", "_").replace(":", "_")

    # Per-run tag so stale flags from old jobs cannot break sync
    run_tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))

    print(
        f"[rank{rank}] local_rank={local_rank} world_size={world_size} run_tag={run_tag} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    # ---- Load dataset ----
    print(f"[rank{rank}] Loading dataset from {args.dataset_path}", flush=True)
    ds_obj = load_from_disk(args.dataset_path)
    if hasattr(ds_obj, "keys") and "train" in ds_obj:
        train_ds = ds_obj["train"]
        eval_ds = ds_obj.get("test", None)
    else:
        train_ds = ds_obj
        eval_ds = None

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Tokenized dataset dirs ----
    tok_root = out_dir / f"tokenized_{safe_model}_L{args.max_length}"
    tok_train_dir = tok_root / "train"
    tok_eval_dir = tok_root / "eval"
    done_flag = tok_root / "_DONE"
    err_flag = tok_root / "_ERROR"

    if rank == 0:
        tok_root.mkdir(parents=True, exist_ok=True)

    # Choose num_proc for rank0 tokenization
    if args.num_proc and args.num_proc > 0:
        num_proc = args.num_proc
    else:
        num_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    num_proc = max(1, num_proc)

    def _map_fn(ex):
        return build_features(ex, tokenizer=tokenizer, max_length=args.max_length)

    # ---- Rank0 tokenizes only if needed ----
    if rank == 0:
        # Repair partial state: if train exists but _DONE missing, mark it done.
        if tok_train_dir.exists() and not done_flag.exists() and not err_flag.exists():
            atomic_write_text(done_flag, "ok")

        has_good_tok = done_flag.exists() and tok_train_dir.exists()

        if has_good_tok and not err_flag.exists():
            print(f"[rank0] Found existing tokenized dataset at {tok_root}. Skipping tokenization.", flush=True)
        else:
            # Clear flags only when re-tokenizing
            if done_flag.exists():
                done_flag.unlink()
            if err_flag.exists():
                err_flag.unlink()

            try:
                print(f"[rank0] Tokenizing + saving to {tok_root} (num_proc={num_proc})...", flush=True)

                train_tok = train_ds.map(
                    _map_fn,
                    remove_columns=train_ds.column_names,
                    desc="Tokenizing train",
                    num_proc=num_proc,
                    load_from_cache_file=False,
                )
                train_tok.save_to_disk(str(tok_train_dir))

                if eval_ds is not None and args.do_eval:
                    eval_tok = eval_ds.map(
                        _map_fn,
                        remove_columns=eval_ds.column_names,
                        desc="Tokenizing eval",
                        num_proc=num_proc,
                        load_from_cache_file=False,
                    )
                    eval_tok.save_to_disk(str(tok_eval_dir))

                atomic_write_text(done_flag, "ok")
                print("[rank0] Preprocess DONE.", flush=True)

            except Exception as e:
                atomic_write_text(err_flag, repr(e))
                raise

    if is_dist and rank != 0:
        wait_for_done_or_error(done_flag, err_flag, timeout_s=3600)

    # Everyone loads tokenized datasets from disk
    train_tok = load_from_disk(str(tok_train_dir))
    if eval_ds is not None and args.do_eval and tok_eval_dir.exists():
        eval_tok = load_from_disk(str(tok_eval_dir))
    else:
        eval_tok = None

    print(
        f"[rank{rank}] Loaded tokenized datasets. train={len(train_tok)} "
        f"eval={(len(eval_tok) if eval_tok is not None else 0)}",
        flush=True,
    )

    # ---- Model ----
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    quant_config: Optional[BitsAndBytesConfig] = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[rank{rank}] BEFORE model load", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    print(f"[rank{rank}] AFTER model load", flush=True)

    # Sync point: per-run unique flag
    model_loaded_flag = out_dir / f"_{safe_model}_{run_tag}_MODEL_LOADED"
    if is_dist:
        if rank == 0:
            # clear stale then write
            if model_loaded_flag.exists():
                model_loaded_flag.unlink()
            atomic_write_text(model_loaded_flag, "ok")
        else:
            wait_for_flag(model_loaded_flag, timeout_s=3600)
    print(f"[rank{rank}] AFTER model load sync", flush=True)

    # Training-stability settings (important with grad checkpointing + DS)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    if rank == 0:
        model.print_trainable_parameters()

    collator = SimpleCollator(tokenizer=tokenizer, padding=True)

    # ---- Training args ----
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        fp16=(not args.bf16),
        bf16=args.bf16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True, 
        deepspeed=args.deepspeed,
    )

    print(f"[rank{rank}] BEFORE Trainer init", flush=True)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    print(f"[rank{rank}] AFTER Trainer init", flush=True)

    # Sync point: per-run unique flag
    trainer_inited_flag = out_dir / f"_{safe_model}_{run_tag}_TRAINER_INITED"
    if is_dist:
        if rank == 0:
            if trainer_inited_flag.exists():
                trainer_inited_flag.unlink()
            atomic_write_text(trainer_inited_flag, "ok")
        else:
            wait_for_flag(trainer_inited_flag, timeout_s=3600)
    print(f"[rank{rank}] AFTER Trainer init sync", flush=True)

    print(f"[rank{rank}] BEFORE trainer.train()", flush=True)
    trainer.train()
    print(f"[rank{rank}] AFTER trainer.train()", flush=True)

    # ---- Save adapter (rank0 only) ----
    if rank == 0:
        adapter_dir = out_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"\nSaved LoRA adapter to: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
