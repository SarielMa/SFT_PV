#!/usr/bin/env python3
"""
2-GPU (torchrun/DDP) SFT with PEFT (QLoRA/LoRA) on Arrow dataset:
  - input = query + "\\n" + answer
  - loss only on answer tokens (prompt masked to -100)

Run:
  torchrun --nproc_per_node=2 sft_peft_ddp.py ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Avoid occasional tokenizer thread deadlocks on shared HPC
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
    """Pads input_ids/attention_mask/labels to longest in batch. Pads labels with -100."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_peft_out")
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    # torchrun sets these env vars
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load dataset ----
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

    # IMPORTANT for DDP: do NOT use device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map=None,
    )

    # Stability for training
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

    # ---- Tokenize (simple + robust) ----
    def _map_fn(ex):
        return build_features(ex, tokenizer=tokenizer, max_length=args.max_length)

    train_tok = train_ds.map(_map_fn, remove_columns=train_ds.column_names, desc="Tokenizing train")
    if eval_ds is not None and args.do_eval:
        eval_tok = eval_ds.map(_map_fn, remove_columns=eval_ds.column_names, desc="Tokenizing eval")
        if len(eval_tok) == 0:
            eval_tok = None  # avoid DDP eval deadlocks
    else:
        eval_tok = None

    collator = SimpleCollator(tokenizer=tokenizer, padding=True)

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

        # DDP hygiene
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        # safer defaults for DDP
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ---- Save adapter (rank0 only) ----
    if rank == 0:
        adapter_dir = out_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"\nSaved LoRA adapter to: {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
