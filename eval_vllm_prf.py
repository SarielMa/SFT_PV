#!/usr/bin/env python3
"""
Evaluate a vLLM model on PV Miner data using pv_utils.py

Dataset format (Arrow):
  - query   : model input
  - answer  : gold JSON output (string or dict)

We run vLLM on query, collect raw outputs, and then call:
  pv_utils.evaluate_eppc_agg(items)
where items = [(gold_answer, pred_text), ...]
This function computes micro P/R/F1 for:
  - code
  - sub-code
  - span  (relaxed match with full containment + jaccard threshold)

It also uses pv_utils.safe_json_loads to parse both gold and pred.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Tuple

from datasets import load_from_disk
from vllm import LLM, SamplingParams

import pv_utils


def batched(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local HF model folder for vLLM")
    ap.add_argument("--data", required=True, help="Arrow dataset folder (load_from_disk)")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")

    # vLLM controls
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # manual chunking (your "batch size")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="Manual prompt batch size. 0 = send all prompts at once.")

    ap.add_argument("--out_json", default=None, help="Optional: save metric dict to json")
    ap.add_argument("--out_pred_jsonl", default=None, help="Optional: dump per-example preds")
    args = ap.parse_args()

    ds = load_from_disk(args.data)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    queries: List[str] = [ex["query"] for ex in ds]
    gold_answers: List[Any] = [ex["answer"] for ex in ds]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, trust_remote_code=True)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ---- vLLM generation (optionally chunked) ----
    pred_texts: List[str] = []
    if args.batch_size and args.batch_size > 0:
        for q_batch in batched(queries, args.batch_size):
            outs = llm.generate(q_batch, sp)
            for out in outs:
                pred_texts.append(out.outputs[0].text if out.outputs else "")
    else:
        outs = llm.generate(queries, sp)
        for out in outs:
            pred_texts.append(out.outputs[0].text if out.outputs else "")

    # ---- Build items for pv_utils ----
    items: List[Tuple[Any, str]] = list(zip(gold_answers, pred_texts))

    # ---- Metrics via your pv_utils ----
    metrics = pv_utils.evaluate_eppc_agg(items)

    print("\n===== PV Miner Evaluation (pv_utils.evaluate_eppc_agg) =====")
    print(f"#Samples: {len(ds)}\n")
    print(f"[Code]     P={metrics['code']['P']:.4f}  R={metrics['code']['R']:.4f}  F1={metrics['code']['f1']:.4f}")
    print(f"[Sub-code] P={metrics['sub-code']['P']:.4f}  R={metrics['sub-code']['R']:.4f}  F1={metrics['sub-code']['f1']:.4f}")
    print(f"[Span]     P={metrics['span']['P']:.4f}  R={metrics['span']['R']:.4f}  F1={metrics['span']['f1']:.4f}\n")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if args.out_pred_jsonl:
        with open(args.out_pred_jsonl, "w", encoding="utf-8") as f:
            for i, (gold, pred) in enumerate(items):
                f.write(json.dumps({
                    "i": i,
                    "gold": pv_utils.safe_json_loads(gold),
                    "pred": pv_utils.safe_json_loads(pred),
                    "raw_pred_text": pred,
                }, ensure_ascii=False) + "\n")

    # ---- (Optional) clean NCCL warning if torch.distributed was initialized ----
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
