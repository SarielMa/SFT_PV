#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model name/path (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (e.g., .../lora_adapter)")
    ap.add_argument("--out", required=True, help="Output dir for merged model")
    ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16","fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto")

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()  # <-- key

    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"Merged model saved to: {args.out}")

if __name__ == "__main__":
    main()
