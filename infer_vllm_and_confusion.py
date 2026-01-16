#!/usr/bin/env python3
"""
vLLM inference on Arrow dataset with fields:
  - query  : model input prompt (string)
  - answer : ground truth output (string / dict)

We parse both gold and pred using pv_utils.safe_json_loads, then extract sets:
  - Code labels are canonicalized to EXACT strings in pv_utils.Code_set
  - Sub-code labels are canonicalized to EXACT strings in pv_utils.Sub_Code_set

This prevents duplicates like "Clinical Care" vs "Clinical care" vs extra spaces.

Then compute directed multi-label confusion-edge tables (FN->FP pairs):
  T = true set, P = predicted set
  FN = T - P, FP = P - T
  for t in FN, p in FP: count[t,p] += 1

Outputs CSV columns:
  Ground_Truth_*, Predicted_*, Count
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from datasets import load_from_disk
from vllm import LLM, SamplingParams

import pv_utils  # ensure pv_utils.py is in same folder or on PYTHONPATH


# -----------------------------
# Canonicalization helpers
# -----------------------------
def _norm_label(s: str) -> str:
    # collapse whitespace + lowercase for lookup
    return re.sub(r"\s+", " ", s.strip()).lower()


def _build_canon_map(allowed: Set[str]) -> Dict[str, str]:
    """
    Map normalized label -> canonical label (exact string from pv_utils sets).
    """
    m: Dict[str, str] = {}
    for lab in allowed:
        if not isinstance(lab, str):
            continue
        key = _norm_label(lab)
        # If duplicates exist in allowed with same normalized key, last wins.
        # But pv_utils sets should be unique in practice.
        m[key] = lab
    return m


CODE_CANON: Dict[str, str] = _build_canon_map(pv_utils.Code_set)
SUB_CANON: Dict[str, str] = _build_canon_map(pv_utils.Sub_Code_set)


def canonicalize_label(x: Any, canon_map: Dict[str, str]) -> Optional[str]:
    """
    Return canonical label string if x can be mapped to the allowed set,
    else None (drop it).
    """
    if not isinstance(x, str):
        return None
    key = _norm_label(x)
    return canon_map.get(key)


def extract_sets_from_text(text_or_obj: Any) -> Tuple[Set[str], Set[str]]:
    """
    Parse via pv_utils.safe_json_loads and extract canonicalized Code/Sub-code sets.

    Returns:
      (code_set, subcode_set)
    """
    parsed = pv_utils.safe_json_loads(text_or_obj)

    code_set: Set[str] = set()
    subcode_set: Set[str] = set()

    if not isinstance(parsed, dict):
        return code_set, subcode_set

    results = parsed.get("results", [])
    if not isinstance(results, list):
        return code_set, subcode_set

    for item in results:
        if not isinstance(item, dict):
            continue

        c = canonicalize_label(item.get("Code"), CODE_CANON)
        if c is not None:
            code_set.add(c)

        s = canonicalize_label(item.get("Sub-code"), SUB_CANON)
        if s is not None:
            subcode_set.add(s)

    return code_set, subcode_set


# -----------------------------
# Confusion edges
# -----------------------------
def confusion_edges(
    true_sets: List[Set[str]],
    pred_sets: List[Set[str]],
    include_none: bool = False,
    none_token: str = "__NONE__",
) -> pd.DataFrame:
    """
    Directed multi-label confusion edge counts:
      edges t->p for t in (T-P) and p in (P-T)
    """
    ctr = Counter()

    for T, P in zip(true_sets, pred_sets):
        fn = T - P
        fp = P - T

        if fn and fp:
            for t in fn:
                for p in fp:
                    ctr[(t, p)] += 1
        elif include_none:
            if fn and not fp:
                for t in fn:
                    ctr[(t, none_token)] += 1
            elif fp and not fn:
                for p in fp:
                    ctr[(none_token, p)] += 1

    rows = [{"Ground_Truth": k[0], "Predicted": k[1], "Count": v} for k, v in ctr.items()]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["Ground_Truth", "Predicted", "Count"])
    return df.sort_values("Count", ascending=False).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True, help="Local HF model folder for vLLM")
    ap.add_argument("--data", required=True, help="HF dataset folder (load_from_disk)")
    ap.add_argument("--out_code_csv", required=True, help="Output CSV for Code confusion edges")
    ap.add_argument("--out_subcode_csv", required=True, help="Output CSV for Sub-code confusion edges")

    ap.add_argument("--out_pred_jsonl", default=None, help="Optional: save per-example debug dump")
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all")

    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size for vLLM")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--include_none", action="store_true")
    args = ap.parse_args()

    ds = load_from_disk(args.data)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, trust_remote_code=True)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    prompts: List[str] = [ex["query"] for ex in ds]
    gold_answers: List[Any] = [ex["answer"] for ex in ds]

    outs = llm.generate(prompts, sp)

    true_code_sets: List[Set[str]] = []
    pred_code_sets: List[Set[str]] = []
    true_sub_sets: List[Set[str]] = []
    pred_sub_sets: List[Set[str]] = []

    debug_rows = []

    for i, (gold, out) in enumerate(zip(gold_answers, outs)):
        pred_text = out.outputs[0].text if out.outputs else ""

        gold_codes, gold_subs = extract_sets_from_text(gold)
        pred_codes, pred_subs = extract_sets_from_text(pred_text)

        true_code_sets.append(gold_codes)
        pred_code_sets.append(pred_codes)
        true_sub_sets.append(gold_subs)
        pred_sub_sets.append(pred_subs)

        if args.out_pred_jsonl:
            debug_rows.append(
                {
                    "i": i,
                    "true_codes": sorted(list(gold_codes)),
                    "pred_codes": sorted(list(pred_codes)),
                    "true_subcodes": sorted(list(gold_subs)),
                    "pred_subcodes": sorted(list(pred_subs)),
                    "raw_pred_text": pred_text,
                }
            )

    # Code confusion edges
    df_code = confusion_edges(true_code_sets, pred_code_sets, include_none=args.include_none)
    df_code = df_code.rename(columns={"Ground_Truth": "Ground_Truth_Code", "Predicted": "Predicted_Code"})
    df_code.to_csv(args.out_code_csv, index=False)

    # Sub-code confusion edges
    df_sub = confusion_edges(true_sub_sets, pred_sub_sets, include_none=args.include_none)
    df_sub = df_sub.rename(columns={"Ground_Truth": "Ground_Truth_Subcode", "Predicted": "Predicted_Subcode"})
    df_sub.to_csv(args.out_subcode_csv, index=False)

    if args.out_pred_jsonl:
        with open(args.out_pred_jsonl, "w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
