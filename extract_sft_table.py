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
    # lm-eval sometimes nests metrics under "evaluate_eppc,none"
    if isinstance(task_result, dict):
        for k in ("evaluate_eppc,none", "evaluate_eppc"):
            if k in task_result and isinstance(task_result[k], dict):
                return task_result[k]
    return task_result


def load_one_results_json(p: Path, task: str, shot: int) -> Optional[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Shot filter (best-effort)
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
    """
    If there is exactly 1 trial, output plain floats.
    If multiple trials, output mean +- std (kept for robustness).
    """
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

    # Multi-trial mean +- std (shouldn't happen after filtering, but safe)
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
    """
    Filters out Jupyter autosave duplicates like:
      .../.ipynb_checkpoints/results_...-checkpoint.json
    """
    s = str(path)
    if ".ipynb_checkpoints" in s:
        return True
    if path.name.endswith("-checkpoint.json"):
        return True
    return False


def filter_results_files(files: List[Path]) -> List[Path]:
    """
    Remove Jupyter checkpoint duplicates and non-files, then de-dup.
    """
    kept = []
    for p in files:
        if not p.is_file():
            continue
        if is_jupyter_checkpoint(p):
            continue
        kept.append(p)
    return sorted(set(kept))


def pick_latest(files: List[Path]) -> List[Path]:
    """
    Keep only the most recently modified result json (one run).
    This ensures no '+-' formatting when you intend a single run.
    """
    if not files:
        return []
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    return [files[-1]]


def find_lmeval_results_jsons(epoch_dir: Path) -> List[Path]:
    """
    Current structure often looks like:
      epochX/lm_eval_results/PvExtraction_full/<run_id>/results_*.json

    We search recursively under likely roots for results json files,
    then filter out Jupyter checkpoint duplicates.
    """
    candidates: List[Path] = []
    for base in [
        epoch_dir / "lm_eval_results",
        epoch_dir / "eval",
        epoch_dir / "lm_eval_results" / "PvExtraction_full",
    ]:
        if base.exists():
            candidates.append(base)

    files: List[Path] = []
    for base in candidates:
        # main pattern (newer harness)
        files.extend(base.rglob("results_*.json"))
        # fallback if naming differs
        files.extend(base.rglob("*results*.json"))

    files = filter_results_files(files)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="./runs_pv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--task", type=str, default="PvExtraction_full")
    ap.add_argument("--out_csv", type=str, default="PV_SFT.csv")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument(
        "--latest_only",
        action="store_true",
        help="If set, only use the latest results_*.json per model (recommended).",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    rows: List[List[Any]] = [["model", "class type"] + METRICS]

    for model in args.models:
        model_tag = model.split("/")[-1]
        epoch_dir = runs_root / model_tag / f"epoch{args.epochs}"

        if not epoch_dir.exists():
            print(f"[WARN] epoch folder not found: {epoch_dir}")
            continue

        json_files = find_lmeval_results_jsons(epoch_dir)
        if not json_files:
            print(f"[WARN] No results json found under: {epoch_dir}")
            continue

        # Force single-run behavior: keep only the latest file
        if args.latest_only:
            json_files = pick_latest(json_files)

        trials: List[Dict[str, Any]] = []
        for jf in json_files:
            block = load_one_results_json(jf, args.task, args.shot)
            if block is not None:
                trials.append(block)

        if not trials:
            print(f"[WARN] Found results jsons but none contain task={args.task} shot={args.shot} for {model_tag}")
            print("       Example files:")
            for jf in json_files[:5]:
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
