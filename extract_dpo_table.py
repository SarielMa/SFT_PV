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

    # Shot filter (best-effort; tolerate schema differences)
    nshot_ok = True
    if isinstance(data.get("n-shot"), dict) and task in data["n-shot"]:
        nshot_ok = (data["n-shot"][task] == shot)
    elif isinstance(data.get("n-shot"), int):
        nshot_ok = (data["n-shot"] == shot)

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

    # Multi-trial mean +- std (should be rare after filtering/latest-only)
    for term in TERMS:
        p_list = [float(t[term]["P"]) for t in trials]
        r_list = [float(t[term]["R"]) for t in trials]
        f_list = [float(t[term]["f1"]) for t in trials]
        rows.append([
            model_name, term,
            f"{round(np.mean(p_list), 4)} +- {round(np.std(p_list), 4)}",
            f"{round(np.mean(r_list), 4)} +- {round(np.std(r_list), 4)}",
            f"{round(np.mean(f_list), 4)} +- {round(np.std(f_list), 4)}",
        ])
    return rows


def is_jupyter_checkpoint(path: Path) -> bool:
    s = str(path)
    if ".ipynb_checkpoints" in s:
        return True
    if path.name.endswith("-checkpoint.json"):
        return True
    return False


def filter_results_files(files: List[Path]) -> List[Path]:
    kept: List[Path] = []
    for p in files:
        if not p.is_file():
            continue
        if is_jupyter_checkpoint(p):
            continue
        kept.append(p)
    return sorted(set(kept))


def pick_latest(files: List[Path]) -> List[Path]:
    if not files:
        return []
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return [files[-1]]


def find_dpo_results_files(dpo_root: Path, model_tag: str, epochs: int) -> List[Path]:
    """
    Your DPO lm_eval output can mirror the SFT pattern:
      dpo_pipeline_outputs/<tag>/lm_eval_results/PvExtraction_full/<run_id>/results_*.json

    IMPORTANT: results json filename may NOT include task name, so we search for results_*.json recursively.
    """
    out_tag = f"{model_tag}_epoch{epochs}_sftMerged"
    eval_root = dpo_root / out_tag / "lm_eval_results"
    if not eval_root.exists():
        return []

    files: List[Path] = []
    files.extend(eval_root.rglob("results_*.json"))
    files.extend(eval_root.rglob("*results*.json"))  # fallback

    return filter_results_files(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_root", type=str, default="./dpo_pipeline_outputs", help="DPO output root")
    ap.add_argument("--epochs", type=int, default=3, help="epoch number used in SFT (epochX)")
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--task", type=str, default="PvExtraction_full")
    ap.add_argument("--out_csv", type=str, default="ablation_results.csv")
    ap.add_argument("--models", nargs="+", required=True, help='HF ids like meta-llama/Llama-3.2-3B-Instruct')
    ap.add_argument(
        "--latest_only",
        action="store_true",
        help="If set, only use the latest results_*.json per model (recommended).",
    )
    args = ap.parse_args()

    root = Path(args.dpo_root).resolve()

    header = ["model", "class type"] + METRICS
    rows: List[List[Any]] = [header]

    for model in args.models:
        model_tag = model.split("/")[-1]
        files = find_dpo_results_files(root, model_tag, args.epochs)

        if not files:
            print(f"[WARN] No DPO result json found for {model_tag} under "
                  f"{root}/{model_tag}_epoch{args.epochs}_sftMerged/lm_eval_results (searched recursively for results_*.json)")
            continue

        if args.latest_only:
            files = pick_latest(files)

        trials: List[Dict[str, Any]] = []
        for f in files:
            block = load_one_results_json(f, args.task, args.shot)
            if block is not None:
                trials.append(block)

        if not trials:
            print(f"[WARN] Found result jsons but none contain task={args.task} shot={args.shot} for {model_tag}")
            print("       Example files:")
            for jf in files[:5]:
                print(f"       - {jf}")
            continue

        rows.extend(summarize_trials(model_tag, trials))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
