#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

TERMS = ["code", "sub-code", "span"]
METRICS = ["P", "R", "f1"]


def safe_get_metric_block(task_result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(task_result, dict):
        for k in ("evaluate_eppc,none", "evaluate_eppc"):
            if k in task_result and isinstance(task_result[k], dict):
                return task_result[k]
    return task_result


def load_one_results_json(p: Path, task: str, shot: int) -> Optional[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nshot_ok = False
    if isinstance(data.get("n-shot"), dict) and task in data["n-shot"]:
        nshot_ok = (data["n-shot"][task] == shot)
    elif "n-shot" in data and isinstance(data["n-shot"], int):
        nshot_ok = (data["n-shot"] == shot)
    else:
        nshot_ok = True

    if not nshot_ok:
        return None

    results = data.get("results", {})
    if task not in results:
        return None

    block = safe_get_metric_block(results[task])
    if not isinstance(block, dict):
        return None
    for t in TERMS:
        if t not in block:
            return None
    return block


def summarize_trials(model_name: str, trials: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    if not trials:
        return rows

    if len(trials) == 1:
        t0 = trials[0]
        for term in TERMS:
            row = [model_name, term]
            for m in METRICS:
                row.append(round(float(t0[term][m]), 4))
            rows.append(row)
        return rows

    for term in TERMS:
        p_list, r_list, f_list = [], [], []
        for t in trials:
            p_list.append(float(t[term]["P"]))
            r_list.append(float(t[term]["R"]))
            f_list.append(float(t[term]["f1"]))
        row = [model_name, term,
               f"{round(np.mean(p_list), 4)} +- {round(np.std(p_list), 4)}",
               f"{round(np.mean(r_list), 4)} +- {round(np.std(r_list), 4)}",
               f"{round(np.mean(f_list), 4)} +- {round(np.std(f_list), 4)}"]
        rows.append(row)

    return rows


def find_dpo_results_files(dpo_root: Path, model_tag: str, epochs: int, task: str) -> List[Path]:
    """
    New DPO structure (as we standardized):
      dpo_pipeline_outputs/<ModelTag>_epoch<EPOCHS>_sftMerged/lm_eval_results/*PvExtraction_full*results*.json
    """
    out_tag = f"{model_tag}_epoch{epochs}_sftMerged"
    eval_dir = dpo_root / out_tag / "lm_eval_results"
    if not eval_dir.exists():
        return []
    patterns = [
        f"*{task}*results*.json",
        f"*{task}*.json",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted([p for p in eval_dir.glob(pat) if p.is_file() and p.suffix == ".json"]))
    # de-dup
    seen = set()
    uniq = []
    for p in files:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_root", type=str, default="./dpo_pipeline_outputs", help="DPO output root")
    ap.add_argument("--epochs", type=int, default=3, help="epoch number used in SFT (epochX)")
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--task", type=str, default="PvExtraction_full")
    ap.add_argument("--out_csv", type=str, default="DPO_results.csv")
    ap.add_argument("--models", nargs="+", required=True, help='HF ids like meta-llama/Llama-3.2-3B-Instruct')
    args = ap.parse_args()

    root = Path(args.dpo_root).resolve()

    header = ["model", "class type"] + METRICS
    rows: List[List[Any]] = [header]

    for model in args.models:
        model_tag = model.split("/")[-1]
        files = find_dpo_results_files(root, model_tag, args.epochs, args.task)
        if not files:
            print(f"[WARN] No DPO result json found for {model_tag} under {root}/{model_tag}_epoch{args.epochs}_sftMerged/lm_eval_results")
            continue

        trials: List[Dict[str, Any]] = []
        for f in files:
            block = load_one_results_json(f, args.task, args.shot)
            if block is not None:
                trials.append(block)

        if not trials:
            print(f"[WARN] Found json files but none matched task/shot for {model_tag}: {files}")
            continue

        rows.extend(summarize_trials(model_tag, trials))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
