#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate LoRA adapters on HumanEval with vLLM.

Supports two evaluation modes matching "LoRA Learns Less and Forgets Less" paper:

1. paper_code_main (multi-sample pass@k):
   - 0-shot, 50 generations per problem
   - temperature=0.2, top_p=0.95
   - pass@1 computed from 50 samples using unbiased estimator

2. greedy_code (legacy):
   - 0-shot, 1 generation per problem
   - temperature=0.0 (greedy)
   - Simple accuracy

Example (paper settings - 50 samples, pass@1):
  python -m lora_spectral_edit.eval_humaneval \
    --base_model_id meta-llama/Llama-2-7b-hf \
    --lora_dir ./my_lora \
    --eval_profile paper_code_main \
    --out_dir ./results

Example (greedy, legacy):
  python -m lora_spectral_edit.eval_humaneval \
    --base_model_id meta-llama/Llama-2-7b-hf \
    --lora_dir ./my_lora \
    --temperature 0.0 \
    --n_generations 1 \
    --out_dir ./results
"""

import os
import json
import math
import shutil
import inspect
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

# vLLM (check at runtime)
HAVE_VLLM = False
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAVE_VLLM = True
except ImportError:
    LLM = None
    SamplingParams = None
    LoRARequest = None

# HumanEval (check at runtime)
HAVE_HUMAN_EVAL = False
try:
    from human_eval.data import read_problems, write_jsonl
    HAVE_HUMAN_EVAL = True
except ImportError:
    read_problems = None
    write_jsonl = None


# ---------------------------
# Utils
# ---------------------------

def load_adapter_config(lora_dir: str) -> Dict[str, Any]:
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in: {lora_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_lora_has_config(lora_dir: str, config_src: Optional[str]) -> None:
    """If adapter_config.json is missing in lora_dir, copy it from config_src."""
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        return
    if not config_src:
        raise FileNotFoundError(
            f"{cfg_path} missing.\n"
            f"Your edited dir likely only has adapter_model.*.\n"
            f"Provide --config_src <original_lora_dir> to copy adapter_config.json."
        )
    src_cfg = os.path.join(config_src, "adapter_config.json")
    if not os.path.exists(src_cfg):
        raise FileNotFoundError(f"--config_src does not contain adapter_config.json: {src_cfg}")
    os.makedirs(lora_dir, exist_ok=True)
    shutil.copy2(src_cfg, cfg_path)


def infer_max_lora_rank(adapter_cfg: Dict[str, Any]) -> int:
    """
    vLLM needs max_lora_rank at engine init.
    Try cfg["r"] or max(cfg["rank_pattern"].values()).
    """
    r = int(adapter_cfg.get("r", 0) or adapter_cfg.get("rank", 0) or 0)
    if r <= 0 and isinstance(adapter_cfg.get("rank_pattern", None), dict):
        r = max(int(v) for v in adapter_cfg["rank_pattern"].values())
    if r <= 0:
        raise ValueError("Cannot infer LoRA rank from adapter_config.json (no r / rank_pattern).")
    return r


def jsonl_read(path: str) -> List[dict]:
    """Read JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jsonl_write(path: str, rows: List[dict]):
    """Write JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------
# Pass@k Computation
# ---------------------------

def _comb(n: int, k: int) -> int:
    """Compute C(n, k) = n! / (k! * (n-k)!)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k using unbiased estimator.

    From "Evaluating Large Language Models Trained on Code" (Chen et al., 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)

    where:
    - n: total number of samples per problem
    - c: number of correct samples
    - k: k in pass@k

    Args:
        n: Total samples generated per problem
        c: Number of correct (passing) samples
        k: k for pass@k metric

    Returns:
        Probability of at least one correct sample in k tries
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # 1 - C(n-c, k) / C(n, k)
    numerator = _comb(n - c, k)
    denominator = _comb(n, k)

    if denominator == 0:
        return 0.0

    return 1.0 - numerator / denominator


def compute_pass_at_k_from_samples(
    task_results: Dict[str, List[bool]],
    k: int = 1,
) -> Dict[str, Any]:
    """
    Compute pass@k from per-task sample results.

    Args:
        task_results: Dict mapping task_id -> list of bools (pass/fail for each sample)
        k: k for pass@k

    Returns:
        Dict with pass@k, per-task scores, and statistics
    """
    per_task_scores = {}
    for task_id, results in sorted(task_results.items()):
        n = len(results)
        c = sum(results)  # number of passing samples
        per_task_scores[task_id] = {
            "n_samples": n,
            "n_correct": c,
            f"pass@{k}": pass_at_k(n, c, k),
        }

    # Average across tasks
    task_pass_at_k = [v[f"pass@{k}"] for v in per_task_scores.values()]
    avg_pass_at_k = sum(task_pass_at_k) / len(task_pass_at_k) if task_pass_at_k else 0.0

    # Count tasks with at least one passing sample
    tasks_with_any_pass = sum(1 for v in per_task_scores.values() if v["n_correct"] > 0)

    return {
        f"pass@{k}": avg_pass_at_k,
        "num_tasks": len(per_task_scores),
        "tasks_with_any_pass": tasks_with_any_pass,
        "per_task": per_task_scores,
    }


# ---------------------------
# HumanEval Execution
# ---------------------------

def run_humaneval_evaluate_functional_correctness(
    samples_path: str,
    k: List[int] = [1],
    n_workers: int = 4,
    timeout: float = 3.0,
    ignore_incomplete: bool = False,
) -> Tuple[dict, Optional[str]]:
    """
    Run HumanEval evaluation. Prefer Python API; fallback to CLI.
    """
    try:
        from human_eval.evaluation import evaluate_functional_correctness

        sig = inspect.signature(evaluate_functional_correctness)
        kwargs = {}
        if "k" in sig.parameters:
            kwargs["k"] = k
        if "n_workers" in sig.parameters:
            kwargs["n_workers"] = int(n_workers)
        if "timeout" in sig.parameters:
            kwargs["timeout"] = float(timeout)
        if "ignore_incomplete" in sig.parameters:
            kwargs["ignore_incomplete"] = bool(ignore_incomplete)

        res = evaluate_functional_correctness(samples_path, **kwargs)
        results_path = samples_path.replace(".jsonl", "_results.jsonl")
        if not os.path.exists(results_path):
            results_path = samples_path + "_results.jsonl"
        return res, results_path if os.path.exists(results_path) else None

    except Exception as e:
        print(f"[Eval][Warn] Python API failed ({type(e).__name__}: {e}). Fallback to CLI...")

    exe = shutil.which("evaluate_functional_correctness")
    if exe is None:
        raise RuntimeError(
            "Cannot find `evaluate_functional_correctness` on PATH.\n"
            "Install HumanEval:\n"
            "  git clone https://github.com/openai/human-eval\n"
            "  pip install -e human-eval\n"
        )

    cmd = [
        exe,
        samples_path,
        f"--k={','.join(map(str, k))}",
        f"--n_workers={int(n_workers)}",
        f"--timeout={float(timeout)}",
    ]
    print("[Eval][CLI] " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate_functional_correctness failed with code {proc.returncode}")

    results_path = samples_path.replace(".jsonl", "_results.jsonl")
    if not os.path.exists(results_path):
        results_path = samples_path + "_results.jsonl"
    return {}, results_path if os.path.exists(results_path) else None


def parse_results_jsonl(results_path: str) -> Dict[str, List[bool]]:
    """
    Parse results JSONL and return per-task pass/fail lists.

    Returns:
        Dict mapping task_id -> list of bools (True=pass, False=fail)
    """
    rows = jsonl_read(results_path)
    task_results: Dict[str, List[bool]] = defaultdict(list)

    for r in rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        passed = bool(r.get("passed", False))
        task_results[tid].append(passed)

    return dict(task_results)


# ---------------------------
# Core Evaluation
# ---------------------------

def eval_one_adapter_multisample(
    llm: LLM,
    prompts: List[str],
    task_ids: List[str],
    n_generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop_words: List[str],
    lora_dir: Optional[str],
    adapter_name: str,
    out_dir: str,
    eval_n_workers: int = 4,
    eval_timeout: float = 3.0,
    pass_k: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate one adapter on HumanEval with multi-sample generation.

    For paper_code_main profile: n_generations=50, temperature=0.2, top_p=0.95

    Args:
        llm: vLLM LLM instance
        prompts: List of prompts (one per task)
        task_ids: List of task IDs
        n_generations: Number of generations per task
        temperature: Sampling temperature
        top_p: Nucleus sampling top_p
        max_tokens: Max tokens to generate
        stop_words: Stop sequences
        lora_dir: Path to LoRA adapter (None for base model)
        adapter_name: Name for this adapter (for output files)
        out_dir: Output directory
        eval_n_workers: Workers for code execution
        eval_timeout: Timeout per test
        pass_k: k for pass@k metric

    Returns:
        Dict with pass@k, per-task results, and file paths
    """
    # Create sampling params
    sp = SamplingParams(
        n=n_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_words if stop_words else None,
    )

    # Generate
    if lora_dir is not None:
        lora_req = LoRARequest(adapter_name, 1, lora_dir)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sp)

    # Collect samples - each output has n_generations completions
    samples = []
    raw_generations = []  # For auditability

    for tid, out in zip(task_ids, outputs):
        for i, completion in enumerate(out.outputs):
            generated_text = completion.text
            samples.append({
                "task_id": tid,
                "completion": generated_text,
            })
            raw_generations.append({
                "task_id": tid,
                "sample_idx": i,
                "completion": generated_text,
            })

    # Write samples to JSONL
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, f"samples_{adapter_name}.jsonl")
    try:
        write_jsonl(samples_path, samples)
    except Exception:
        jsonl_write(samples_path, samples)
    print(f"[Eval] Wrote {len(samples)} samples to: {samples_path}")

    # Write raw generations for auditability
    raw_path = os.path.join(out_dir, f"raw_generations_{adapter_name}.jsonl")
    jsonl_write(raw_path, raw_generations)

    # Run HumanEval code execution
    print(f"[Eval] Running evaluate_functional_correctness on {len(samples)} samples...")
    raw_res, results_path = run_humaneval_evaluate_functional_correctness(
        samples_path=samples_path,
        k=[pass_k],
        n_workers=eval_n_workers,
        timeout=eval_timeout,
        ignore_incomplete=False,
    )

    # Compute pass@k from results
    if results_path and os.path.exists(results_path):
        task_results = parse_results_jsonl(results_path)
        metrics = compute_pass_at_k_from_samples(task_results, k=pass_k)

        # Also write per-problem pass/fail summary
        per_problem_path = os.path.join(out_dir, f"per_problem_{adapter_name}.json")
        with open(per_problem_path, "w", encoding="utf-8") as f:
            json.dump(metrics["per_task"], f, indent=2)
        metrics["per_problem_path"] = per_problem_path
    else:
        # Fallback if results file not found
        if isinstance(raw_res, dict) and f"pass@{pass_k}" in raw_res:
            metrics = {
                f"pass@{pass_k}": float(raw_res[f"pass@{pass_k}"]),
                "num_tasks": len(task_ids),
            }
        else:
            metrics = {
                f"pass@{pass_k}": None,
                "num_tasks": len(task_ids),
            }

    metrics["samples_path"] = samples_path
    metrics["results_path"] = results_path
    metrics["raw_generations_path"] = raw_path
    metrics["n_generations"] = n_generations
    metrics["n_total_samples"] = len(samples)

    return metrics


def eval_one_adapter(
    llm: LLM,
    sp: SamplingParams,
    prompts: List[str],
    task_ids: List[str],
    lora_dir: Optional[str],
    adapter_name: str,
    out_dir: str,
    eval_n_workers: int = 4,
    eval_timeout: float = 3.0,
    ignore_incomplete: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate one adapter on HumanEval (legacy greedy mode, 1 sample per task).
    """
    if lora_dir is not None:
        lora_req = LoRARequest(adapter_name, 1, lora_dir)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sp)

    samples = []
    for tid, out in zip(task_ids, outputs):
        generated_text = out.outputs[0].text
        samples.append({
            "task_id": tid,
            "completion": generated_text
        })

    # Write samples to JSONL
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, f"samples_{adapter_name}.jsonl")
    try:
        write_jsonl(samples_path, samples)
    except Exception:
        jsonl_write(samples_path, samples)
    print(f"[Eval] Wrote samples to: {samples_path}")

    # Run evaluation
    print(f"[Eval] Running evaluate_functional_correctness...")
    raw_res, results_path = run_humaneval_evaluate_functional_correctness(
        samples_path=samples_path,
        k=[1],
        n_workers=eval_n_workers,
        timeout=eval_timeout,
        ignore_incomplete=ignore_incomplete,
    )

    # Compute metrics
    if results_path and os.path.exists(results_path):
        task_results = parse_results_jsonl(results_path)
        # For single sample, compute simple accuracy
        correct = sum(1 for results in task_results.values() if any(results))
        total = len(task_results)
        metrics = {
            "pass@1": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "num_tasks": total,
        }
    else:
        if isinstance(raw_res, dict) and "pass@1" in raw_res:
            metrics = {
                "pass@1": float(raw_res["pass@1"]),
                "correct": None,
                "total": len(task_ids),
                "num_tasks": len(task_ids),
            }
        else:
            metrics = {
                "pass@1": None,
                "correct": None,
                "total": len(task_ids),
                "num_tasks": len(task_ids),
            }

    metrics["samples_path"] = samples_path
    metrics["results_path"] = results_path

    return metrics


def write_eval_config(
    out_dir: str,
    task: str,
    n_generations: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    total_tasks: int,
    metric_name: str,
    score: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write evaluation config summary to JSON."""
    config = {
        "task": task,
        "split": "test",
        "num_fewshot": 0,
        "decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n_generations": n_generations,
            "strategy": "greedy" if temperature == 0.0 else "sampling",
        },
        "total_tasks": total_tasks,
        "total_samples": total_tasks * n_generations,
        "metric": {
            "name": metric_name,
            "score": score,
        },
        "timestamp": datetime.now().isoformat(),
    }

    if extra_meta:
        config["meta"] = extra_meta

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "eval_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config_path


def evaluate_humaneval_with_profile(
    base_model_id: str,
    lora_dir: Optional[str],
    profile_name: str = "paper_code_main",
    max_samples: Optional[int] = None,
    seed: int = 0,
    max_model_len: int = 4096,
    out_dir: str = "./humaneval_results",
    config_src: Optional[str] = None,
    eval_n_workers: int = 4,
    eval_timeout: float = 3.0,
) -> Dict[str, Any]:
    """
    Evaluate HumanEval using a predefined profile.

    Args:
        base_model_id: HuggingFace model ID
        lora_dir: Path to LoRA adapter (None for base model only)
        profile_name: Profile name ("paper_code_main" for paper settings)
        max_samples: Override max tasks (None uses all 164)
        seed: Random seed
        max_model_len: Max context length for vLLM
        out_dir: Output directory
        config_src: Source for adapter_config.json if missing
        eval_n_workers: Workers for code execution
        eval_timeout: Timeout per test

    Returns:
        Dict with evaluation results and config
    """
    from .eval_profiles import get_profile

    profile = get_profile(profile_name)

    if profile.task != "humaneval":
        raise ValueError(f"Profile '{profile_name}' is for task '{profile.task}', not humaneval")

    if not HAVE_VLLM:
        raise RuntimeError("vLLM not available. Please `pip install vllm`.")
    if not HAVE_HUMAN_EVAL:
        raise RuntimeError(
            "human_eval not available. Install:\n"
            "  git clone https://github.com/openai/human-eval\n"
            "  pip install -e human-eval\n"
        )

    # Ensure LoRA config exists
    enable_lora = lora_dir is not None
    max_r = 16

    if lora_dir:
        ensure_lora_has_config(lora_dir, config_src)
        adapter_cfg = load_adapter_config(lora_dir)
        max_r = max(max_r, infer_max_lora_rank(adapter_cfg))

    # Load HumanEval problems
    print("[Data] Loading HumanEval problems...")
    problems = read_problems()
    task_ids = sorted(problems.keys())

    if max_samples is not None and int(max_samples) > 0:
        task_ids = task_ids[:int(max_samples)]

    prompts = [problems[tid]["prompt"] for tid in task_ids]
    print(f"[Data] Loaded {len(prompts)} HumanEval tasks.")

    # Init vLLM engine
    print(f"[vLLM] Initializing LLM engine (Base: {base_model_id})...")
    llm = LLM(
        model=base_model_id,
        dtype="float16",
        max_model_len=int(max_model_len),
        enable_lora=enable_lora,
        max_lora_rank=int(max_r) if enable_lora else 16,
        seed=int(seed),
    )

    # Run evaluation
    os.makedirs(out_dir, exist_ok=True)
    adapter_name = "lora" if lora_dir else "base"

    metrics = eval_one_adapter_multisample(
        llm=llm,
        prompts=prompts,
        task_ids=task_ids,
        n_generations=profile.n_generations,
        temperature=profile.temperature,
        top_p=profile.top_p,
        max_tokens=profile.max_tokens,
        stop_words=profile.stop_sequences or [],
        lora_dir=lora_dir,
        adapter_name=adapter_name,
        out_dir=out_dir,
        eval_n_workers=eval_n_workers,
        eval_timeout=eval_timeout,
        pass_k=profile.pass_k,
    )

    # Write config
    config_path = write_eval_config(
        out_dir=out_dir,
        task="humaneval",
        n_generations=profile.n_generations,
        temperature=profile.temperature,
        top_p=profile.top_p,
        max_tokens=profile.max_tokens,
        total_tasks=len(task_ids),
        metric_name=f"pass@{profile.pass_k}",
        score=metrics.get(f"pass@{profile.pass_k}", 0.0),
        extra_meta={
            "base_model_id": base_model_id,
            "lora_dir": lora_dir,
            "profile": profile.get_summary(),
            "seed": seed,
            "max_model_len": max_model_len,
        },
    )
    metrics["config_path"] = config_path
    metrics["profile"] = profile.get_summary()

    return metrics


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate LoRA adapters on HumanEval with vLLM"
    )
    ap.add_argument("--base_model_id", type=str, required=True,
                    help="Base model HuggingFace ID or path")

    ap.add_argument("--lora_dir", type=str, default=None,
                    help="Edited/new LoRA directory (local). Required unless --base_only.")
    ap.add_argument("--baseline_lora", type=str, default=None,
                    help="Optional baseline LoRA directory (local) for comparison.")
    ap.add_argument("--base_only", action="store_true",
                    help="Evaluate base model only (no LoRA adapter).")

    ap.add_argument("--config_src", type=str, default=None,
                    help="If lora_dir lacks adapter_config.json, copy from this dir.")

    # Eval profile (new)
    ap.add_argument("--eval_profile", type=str, default=None,
                    choices=["paper_code_main", "greedy_code"],
                    help="Use predefined evaluation profile. "
                         "paper_code_main: 50 samples, temp=0.2, top_p=0.95, pass@1. "
                         "greedy_code: 1 sample, greedy decoding.")

    ap.add_argument("--max_samples", type=int, default=-1,
                    help="Max number of HumanEval tasks to evaluate (-1 for all 164).")

    # Generation params (used when not using profile)
    ap.add_argument("--n_generations", type=int, default=1,
                    help="Number of generations per task (paper_code_main: 50).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (paper_code_main: 0.2, greedy: 0.0).")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p nucleus sampling (paper_code_main: 0.95).")
    ap.add_argument("--max_tokens", type=int, default=512,
                    help="Maximum new tokens to generate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility.")
    ap.add_argument("--pass_k", type=int, default=1,
                    help="k for pass@k metric (default: 1).")

    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="Maximum model context length for vLLM.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Output JSON file for metrics.")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Directory for output files (samples, results).")

    ap.add_argument("--eval_n_workers", type=int, default=4,
                    help="Number of workers for HumanEval code execution.")
    ap.add_argument("--eval_timeout", type=float, default=3.0,
                    help="Timeout in seconds for each HumanEval test case.")

    ap.add_argument("--stop_words", type=str, nargs="*",
                    default=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
                    help="Stop sequences for code generation.")

    args = ap.parse_args()

    # Check dependencies
    if not HAVE_VLLM:
        raise RuntimeError("vLLM not available. Please `pip install vllm`.")
    if not HAVE_HUMAN_EVAL:
        raise RuntimeError(
            "human_eval not available. Install:\n"
            "  git clone https://github.com/openai/human-eval\n"
            "  pip install -e human-eval\n"
        )

    # Validate arguments
    if not args.base_only and args.lora_dir is None:
        ap.error("--lora_dir is required unless --base_only is set.")

    # Determine output directory
    out_dir = args.out_dir
    if out_dir is None:
        if args.lora_dir:
            out_dir = args.lora_dir
        else:
            out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # Use profile-based evaluation if specified
    if args.eval_profile:
        from .eval_profiles import get_profile
        profile = get_profile(args.eval_profile)

        print(f"[Profile] Using '{args.eval_profile}': {profile.description}")

        # Override params from profile
        n_generations = profile.n_generations
        temperature = profile.temperature
        top_p = profile.top_p
        max_tokens = profile.max_tokens
        pass_k = profile.pass_k
        stop_words = profile.stop_sequences or args.stop_words
    else:
        n_generations = args.n_generations
        temperature = args.temperature
        top_p = args.top_p
        max_tokens = args.max_tokens
        pass_k = args.pass_k
        stop_words = args.stop_words

    # Ensure LoRA configs exist
    enable_lora = not args.base_only
    max_r = 16

    if args.lora_dir:
        ensure_lora_has_config(args.lora_dir, args.config_src)
        edited_cfg = load_adapter_config(args.lora_dir)
        max_r = max(max_r, infer_max_lora_rank(edited_cfg))

    if args.baseline_lora:
        ensure_lora_has_config(args.baseline_lora, None)
        base_cfg = load_adapter_config(args.baseline_lora)
        max_r = max(max_r, infer_max_lora_rank(base_cfg))

    # Load HumanEval problems
    print("[Data] Loading HumanEval problems...")
    problems = read_problems()
    task_ids = sorted(problems.keys())

    if args.max_samples is not None and int(args.max_samples) > 0:
        task_ids = task_ids[:int(args.max_samples)]

    prompts = [problems[tid]["prompt"] for tid in task_ids]
    print(f"[Data] Loaded {len(prompts)} HumanEval tasks.")

    # Init vLLM engine
    print(f"[vLLM] Initializing LLM engine (Base: {args.base_model_id})...")
    llm = LLM(
        model=args.base_model_id,
        dtype="float16",
        max_model_len=int(args.max_model_len),
        enable_lora=enable_lora,
        max_lora_rank=int(max_r) if enable_lora else 16,
        seed=int(args.seed),
    )

    results: Dict[str, Any] = {
        "meta": {
            "base_model_id": args.base_model_id,
            "task": "HumanEval",
            "max_samples": args.max_samples,
            "num_tasks": len(task_ids),
            "n_generations": n_generations,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_lora_rank_engine": max_r if enable_lora else None,
            "stop_words": stop_words,
            "eval_n_workers": args.eval_n_workers,
            "eval_timeout": args.eval_timeout,
            "eval_profile": args.eval_profile,
        }
    }

    # Evaluate
    if n_generations > 1:
        # Multi-sample mode (paper_code_main)
        if args.base_only:
            print(f"[Eval] Base model (no adapter), {n_generations} samples per task...")
            metrics = eval_one_adapter_multisample(
                llm=llm,
                prompts=prompts,
                task_ids=task_ids,
                n_generations=n_generations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_words=stop_words,
                lora_dir=None,
                adapter_name="base",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
                pass_k=pass_k,
            )
            print(f"[Result] base: pass@{pass_k}={metrics.get(f'pass@{pass_k}', 'N/A')}")
            results["base"] = metrics

        else:
            if args.baseline_lora:
                print(f"[Eval] Baseline LoRA, {n_generations} samples per task...")
                baseline_metrics = eval_one_adapter_multisample(
                    llm=llm,
                    prompts=prompts,
                    task_ids=task_ids,
                    n_generations=n_generations,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop_words=stop_words,
                    lora_dir=args.baseline_lora,
                    adapter_name="baseline",
                    out_dir=out_dir,
                    eval_n_workers=args.eval_n_workers,
                    eval_timeout=args.eval_timeout,
                    pass_k=pass_k,
                )
                print(f"[Result] baseline: pass@{pass_k}={baseline_metrics.get(f'pass@{pass_k}', 'N/A')}")
                results["baseline"] = baseline_metrics

            print(f"[Eval] Edited LoRA, {n_generations} samples per task...")
            edited_metrics = eval_one_adapter_multisample(
                llm=llm,
                prompts=prompts,
                task_ids=task_ids,
                n_generations=n_generations,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_words=stop_words,
                lora_dir=args.lora_dir,
                adapter_name="edited",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
                pass_k=pass_k,
            )
            print(f"[Result] edited: pass@{pass_k}={edited_metrics.get(f'pass@{pass_k}', 'N/A')}")
            results["edited"] = edited_metrics

    else:
        # Single sample mode (legacy greedy)
        sp = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            stop=stop_words if stop_words else None,
        )

        ignore_incomplete = args.max_samples is not None and int(args.max_samples) > 0

        if args.base_only:
            print("[Eval] Base model (no adapter) ...")
            base_metrics = eval_one_adapter(
                llm=llm,
                sp=sp,
                prompts=prompts,
                task_ids=task_ids,
                lora_dir=None,
                adapter_name="base",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
                ignore_incomplete=ignore_incomplete,
            )
            print(f"[Result] {base_metrics}")
            results["base"] = base_metrics

        else:
            if args.baseline_lora:
                print("[Eval] Baseline LoRA ...")
                baseline_metrics = eval_one_adapter(
                    llm=llm,
                    sp=sp,
                    prompts=prompts,
                    task_ids=task_ids,
                    lora_dir=args.baseline_lora,
                    adapter_name="baseline",
                    out_dir=out_dir,
                    eval_n_workers=args.eval_n_workers,
                    eval_timeout=args.eval_timeout,
                    ignore_incomplete=ignore_incomplete,
                )
                print(f"[Result] baseline: {baseline_metrics}")
                results["baseline"] = baseline_metrics

            print("[Eval] Edited/New LoRA ...")
            edited_metrics = eval_one_adapter(
                llm=llm,
                sp=sp,
                prompts=prompts,
                task_ids=task_ids,
                lora_dir=args.lora_dir,
                adapter_name="edited",
                out_dir=out_dir,
                eval_n_workers=args.eval_n_workers,
                eval_timeout=args.eval_timeout,
                ignore_incomplete=ignore_incomplete,
            )
            print(f"[Result] edited: {edited_metrics}")
            results["edited"] = edited_metrics

    # Write config JSON
    config_path = write_eval_config(
        out_dir=out_dir,
        task="humaneval",
        n_generations=n_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        total_tasks=len(task_ids),
        metric_name=f"pass@{pass_k}" if n_generations > 1 else "pass@1",
        score=results.get("edited", results.get("base", {})).get(f"pass@{pass_k}",
              results.get("edited", results.get("base", {})).get("pass@1", 0.0)),
        extra_meta=results["meta"],
    )
    results["config_path"] = config_path

    # Write metrics
    out_json = args.out_json
    if out_json is None:
        if args.lora_dir:
            out_json = os.path.join(args.lora_dir, "metrics_humaneval_vllm.json")
        else:
            out_json = os.path.join(out_dir, "metrics_humaneval_vllm.json")

    out_json_dir = os.path.dirname(out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Done] Wrote metrics to: {out_json}")
    print(f"[Done] Wrote config to: {config_path}")


if __name__ == "__main__":
    main()
