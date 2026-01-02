#!/usr/bin/env python3
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# PATHS (EDIT THESE)
# ======================
TEST_DATASET_PATH = "/home/lm2445/project_pi_sjf37/lm2445/PV_multiagent/benckmark/PV_benckmark/split_out/test"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # must match training base
LORA_ADAPTER = "./runs/pv_sft_Qwen2.5-1.5B-Instruct_qlora/lora_adapter"

MAX_NEW_TOKENS = 512
DEVICE_MAP = "auto"


def main():
    # ----------------------
    # Load dataset (Arrow)
    # ----------------------
    ds = load_from_disk(TEST_DATASET_PATH)
    print(f"Loaded test dataset with {len(ds)} samples")

    sample = ds[0]   # 👈 pick ONE sample
    prompt = sample["query"]

    print("\n" + "=" * 80)
    print("PROMPT (query)")
    print("=" * 80)
    print(prompt[:2000])  # avoid flooding terminal

    # ----------------------
    # Load model + adapter
    # ----------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map=DEVICE_MAP,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    model.eval()

    # ----------------------
    # Tokenize & generate
    # ----------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,     # deterministic for debugging
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ----------------------
    # Extract model output
    # ----------------------
    model_output = decoded[len(prompt):].strip()

    print("\n" + "=" * 80)
    print("MODEL OUTPUT (raw)")
    print("=" * 80)
    print(model_output)

    # ----------------------
    # Optional: JSON sanity check
    # ----------------------
    print("\n" + "=" * 80)
    print("JSON PARSE CHECK")
    print("=" * 80)
    try:
        parsed = json.loads(model_output)
        print("✓ Valid JSON")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as e:
        print("✗ JSON parse failed")
        print(e)


if __name__ == "__main__":
    main()
