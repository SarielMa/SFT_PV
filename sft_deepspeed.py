#!/usr/bin/env python3
"""
SFT with PEFT (QLoRA/LoRA) using your Arrow dataset that has:
  - query: full instruction + INPUT ... + OUTPUT:
  - answer: JSON string (target)

We build model input as:  query + "\n" + answer
Loss is computed ONLY on answer tokens (prompt tokens are masked to -100).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

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
import os



# ---------------------------
# Tokenization / Label masking
# ---------------------------
def build_features(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """
    Build input_ids/attention_mask/labels such that:
    - input = query + "\n" + answer
    - labels = -100 for prompt part; labels = answer token ids for answer part
    """
    query = example["query"].rstrip()
    answer = example["answer"].strip()

    # Add a separator between prompt and answer
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

    # labels: mask prompt tokens
    labels = [-100] * len(input_ids)

    # Find where answer starts (prompt length), but be careful with truncation
    ans_start = min(len(prompt_ids), len(input_ids))

    # Supervision only on answer tokens
    for i in range(ans_start, len(input_ids)):
        labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SimpleCollator(DataCollatorWithPadding):
    """
    Pads input_ids/attention_mask/labels to the longest in batch.
    Ensures labels are padded with -100.
    """

    def __call__(self, features):
        # DataCollatorWithPadding pads using tokenizer.pad; but we must pad labels ourselves.
        batch = super().__call__([{k: v for k, v in f.items() if k != "labels"} for f in features])

        # Pad labels
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for f in features:
            lab = f["labels"]
            pad_len = max_len - len(lab)
            padded_labels.append(lab + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset saved by dataset.save_to_disk(). "
                             "Can be either a Dataset (no split) or a DatasetDict with train/test.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base causal LM name/path (e.g., meta-llama/Meta-Llama-3-8B-Instruct).")
    parser.add_argument("--output_dir", type=str, default="./sft_peft_out")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config json")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed local rank")



    # Training hyperparams (reasonable for 32GB + QLoRA)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)

    # PEFT/QLoRA toggles
    parser.add_argument("--use_qlora", action="store_true", help="Enable 4-bit QLoRA (recommended for 32GB).")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (recommended if supported). If off, uses fp16.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optional eval
    parser.add_argument("--do_eval", action="store_true", help="Run eval if test split exists.")
    args = parser.parse_args()
    # if args.local_rank is not None and args.local_rank >= 0:
    #     torch.cuda.set_device(args.local_rank)
    # print(f"[local_rank={args.local_rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # args = parser.parse_args()
    
    # Robust GPU binding for DeepSpeed/DDP
    # DeepSpeed passes --local_rank, and also sets LOCAL_RANK in env in many setups.
    local_rank = args.local_rank
    if local_rank is None or local_rank < 0:
        env_lr = os.environ.get("LOCAL_RANK")
        if env_lr is not None:
            local_rank = int(env_lr)
    
    if local_rank is not None and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    
    print(f"[local_rank={local_rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)


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

    # ---- Model ----
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    quant_config = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # Target modules for Llama-like models; if you use a different architecture, change this list.
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
    model.print_trainable_parameters()

    # ---- Preprocess dataset (tokenize + labels) ----
    def _map_fn(ex):
        return build_features(ex, tokenizer=tokenizer, max_length=args.max_length)

    train_tok = train_ds.map(_map_fn, remove_columns=train_ds.column_names, desc="Tokenizing train")
    if eval_ds is not None and args.do_eval:
        eval_tok = eval_ds.map(_map_fn, remove_columns=eval_ds.column_names, desc="Tokenizing eval")
    else:
        eval_tok = None

    collator = SimpleCollator(tokenizer=tokenizer, padding=True)

    # ---- Training args ----
    # train_args = TrainingArguments(
    #     output_dir=str(out_dir),
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.grad_accum,
    #     num_train_epochs=args.epochs,
    #     learning_rate=args.lr,
    #     logging_steps=args.logging_steps,
    #     save_steps=args.save_steps,
    #     save_total_limit=2,
    #     report_to="none",
    #     optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    #     fp16=(not args.bf16),
    #     bf16=args.bf16,
    #     gradient_checkpointing=True,
    #     evaluation_strategy=("steps" if (eval_tok is not None) else "no"),
    #     eval_steps=args.save_steps if (eval_tok is not None) else None,
    #     load_best_model_at_end=False,
    # )
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        fp16=(not args.bf16),
        bf16=args.bf16,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed

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

    # ---- Save adapter ----
    adapter_dir = out_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter to: {adapter_dir}")


if __name__ == "__main__":
    main()
