#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

TERMS = ["code", "sub-code", "span"]
METRICS = ["P", "R", "f1"]

# Your stated defaults for ablation
DEFAULTS = {
    "enable_class_balance": False,
    "enable_token_weighting": False,
    "enable_length_norm": False,
    "policy_prob_threshold": 0.66,
}

BOOL_KEYS = {"enable_class_balance", "enable_token_weighting", "enable_length_norm"}
FLOAT_KEYS = {"policy_prob_threshold"}


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


def parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise ValueError(f"Cannot parse bool from: {v}")


def parse_run_tag(run_dir_name: str) -> Tuple[str, str]:
    """
    Expected pattern:
      <something>__<ablation_tag>
    Example:
      Qwen2.5-1.5B_epoch3_merged__token_weighting=False
    If '__' not found, ablation_tag = run_dir_name
    """
    if "__" in run_dir_name:
        left, right = run_dir_name.split("__", 1)
        return run_dir_name, right
    return run_dir_name, run_dir_name

def ablation_tag_to_params(ablation_tag: str) -> Dict[str, Any]:
    """
    Your ablation protocol:

    - Baseline / default / all_false => FFF and threshold=0.66
    - For boolean ablations: only the specified knob becomes True, the other two remain False.
      e.g., class_balance=True => TFF
            token_weighting=True => FTF
            length_norm=True => FFT
    - For threshold ablations: booleans stay FFF, threshold changes.
      e.g., policy_prob_threshold=0.8 => FFF + threshold=0.8
    """
    params = dict(DEFAULTS)
    tag = ablation_tag.strip()

    # Baseline tags
    if tag.lower() in ("baseline", "default", "none", "all_false", "fff"):
        return params

    # If it's not key=value, keep FFF defaults
    if "=" not in tag:
        return params

    k, v = tag.split("=", 1)
    k = k.strip()
    v = v.strip()

    # normalize aliases / historical names
    alias = {
        "class_balance": "enable_class_balance",
        "enable_class_balance": "enable_class_balance",

        "token_weighting": "enable_token_weighting",
        "enable_token_weighting": "enable_token_weighting",

        "length_norm": "enable_length_norm",
        "enable_length_norm": "enable_length_norm",

        "policy_prob_threshold": "policy_prob_threshold",
        "tau": "policy_prob_threshold",
        "threshold": "policy_prob_threshold",
        "policy_threshold": "policy_prob_threshold",
    }
    k = alias.get(k, k)

    # Threshold ablations: keep booleans FFF, only change threshold
    if k == "policy_prob_threshold":
        try:
            params["policy_prob_threshold"] = float(v)
        except ValueError:
            pass
        return params

    # Boolean ablations: only that knob becomes True if v is True, else stay FFF
    if k in ("enable_class_balance", "enable_token_weighting", "enable_length_norm"):
        try:
            is_true = parse_bool(v)
        except Exception:
            is_true = False

        if is_true:
            # enforce "only one True" rule
            params["enable_class_balance"] = False
            params["enable_token_weighting"] = False
            params["enable_length_norm"] = False
            params[k] = True

        return params

    # Unknown key -> keep defaults
    return params


def ablation_tag_to_params_old(ablation_tag: str) -> Dict[str, Any]:
    """
    Convert ablation_tag -> concrete param dict using DEFAULTS.
    Supports:
      - baseline
      - key=value where key in:
          class_balance, token_weighting, length_norm, policy_prob_threshold
      - enable_* keys too
    Examples:
      token_weighting=False
      enable_token_weighting=False
      policy_prob_threshold=0.9
      class_balance=True
    """
    params = dict(DEFAULTS)

    tag = ablation_tag.strip()
    if tag.lower() in ("baseline", "default", "none"):
        return params

    # Allow legacy/other tags without crashing:
    # If it doesn't look like key=value, just return defaults.
    if "=" not in tag:
        return params

    k, v = tag.split("=", 1)
    k = k.strip()
    v = v.strip()

    # normalize aliases
    alias = {
        "class_balance": "enable_class_balance",
        "token_weighting": "enable_token_weighting",
        "length_norm": "enable_length_norm",
        "tau": "policy_prob_threshold",
        "threshold": "policy_prob_threshold",
        "policy_threshold": "policy_prob_threshold",
    }
    k = alias.get(k, k)

    if k in BOOL_KEYS:
        params[k] = parse_bool(v)
        return params
    if k in FLOAT_KEYS:
        params[k] = float(v)
        return params

    # Unknown key: keep defaults (but don't crash)
    return params


def find_ablation_results_files(abl_root: Path, task: str) -> Dict[str, List[Path]]:
    """
    Returns dict:
      run_dir_name -> list[results_json_paths]
    Search pattern:
      <abl_root>/<run_dir>/lm_eval_results/<task>/**/results*.json
    """
    out: Dict[str, List[Path]] = {}
    if not abl_root.exists():
        return out

    for run_dir in sorted([p for p in abl_root.iterdir() if p.is_dir()]):
        eval_root = run_dir / "lm_eval_results" / task
        if not eval_root.exists():
            continue

        files: List[Path] = []
        # lm_eval output varies: results.json or results_*.json etc.
        files.extend(eval_root.rglob("results*.json"))
        files.extend(eval_root.rglob("*results*.json"))

        files = filter_results_files(files)
        if files:
            out[run_dir.name] = files

    return out


def summarize_trials(run_tag: str, ablation_tag: str, params: Dict[str, Any], trials: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    if not trials:
        return rows

    def base_cols(term: str) -> List[Any]:
        return [
            run_tag,
            ablation_tag,
            params["enable_class_balance"],
            params["enable_token_weighting"],
            params["enable_length_norm"],
            params["policy_prob_threshold"],
            term,
        ]

    if len(trials) == 1:
        t0 = trials[0]
        for term in TERMS:
            row = base_cols(term)
            for m in METRICS:
                row.append(round(float(t0[term][m]), 6))
            rows.append(row)
        return rows

    # Multi-trial mean +- std
    for term in TERMS:
        p_list = [float(t[term]["P"]) for t in trials]
        r_list = [float(t[term]["R"]) for t in trials]
        f_list = [float(t[term]["f1"]) for t in trials]
        row = base_cols(term)
        row.extend([
            f"{round(np.mean(p_list), 6)} +- {round(np.std(p_list), 6)}",
            f"{round(np.mean(r_list), 6)} +- {round(np.std(r_list), 6)}",
            f"{round(np.mean(f_list), 6)} +- {round(np.std(f_list), 6)}",
        ])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl_root", type=str, required=True, help="Ablation output root (tab_dpo_ablation_outputs)")
    ap.add_argument("--shot", type=int, default=0)
    ap.add_argument("--task", type=str, default="PvExtraction_full")
    ap.add_argument("--out_csv", type=str, default="tab_dpo_ablation_results.csv")
    ap.add_argument(
        "--latest_only",
        action="store_true",
        help="If set, only use the latest results*.json per ablation run folder (recommended).",
    )
    args = ap.parse_args()

    abl_root = Path(args.abl_root).resolve()

    # header: run info + resolved params + metrics
    header = [
        "run_tag",
        "ablation_tag",
        "enable_class_balance",
        "enable_token_weighting",
        "enable_length_norm",
        "policy_prob_threshold",  
        "class type",
        "P",
        "R",
        "f1",
    ]

    rows: List[List[Any]] = [header]

    run_to_files = find_ablation_results_files(abl_root, args.task)
    if not run_to_files:
        print(f"[WARN] No lm_eval results found under: {abl_root}")
        print(f"       Expected: <run>/lm_eval_results/{args.task}/**/results*.json")
        return

    for run_dir_name, files in run_to_files.items():
        run_tag, ablation_tag = parse_run_tag(run_dir_name)
        params = ablation_tag_to_params(ablation_tag)

        use_files = pick_latest(files) if args.latest_only else files

        trials: List[Dict[str, Any]] = []
        for f in use_files:
            block = load_one_results_json(f, args.task, args.shot)
            if block is not None:
                trials.append(block)

        if not trials:
            print(f"[WARN] Found result jsons but none contain task={args.task} shot={args.shot} for run={run_dir_name}")
            print("       Example files:")
            for jf in use_files[:5]:
                print(f"       - {jf}")
            continue

        rows.extend(summarize_trials(run_tag, ablation_tag, params, trials))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
