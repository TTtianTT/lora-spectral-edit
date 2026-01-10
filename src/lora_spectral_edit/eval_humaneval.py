#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a (possibly edited) LoRA on HumanEval with vLLM (greedy), optionally compare to a baseline LoRA.

Works with your edited dir like:
  ./magicoder_r64_simpleguard_v1

IMPORTANT:
- vLLM expects LoRA directories to contain:
    adapter_config.json
    adapter_model.safetensors OR adapter_model.bin
- Your sigma-edit script only saved adapter_model.* by default.
  If adapter_config.json is missing, pass --config_src to copy it from an existing LoRA dir.

Example (single adapter):
  CUDA_VISIBLE_DEVICES=0 python eval_humaneval_lora_vllm.py \
    --base_model_id meta-llama/Llama-2-7b-hf \
    --lora_dir ./magicoder_r64_simpleguard_v1 \
    --config_src LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128 \
    --temperature 0.0 --max_tokens 512 --max_samples -1 \
    --out_json ./magicoder_r64_simpleguard_v1/metrics_humaneval.json

Compare baseline vs edited (single vLLM engine):
  CUDA_VISIBLE_DEVICES=0 python eval_humaneval_lora_vllm.py \
    --base_model_id meta-llama/Llama-2-7b-hf \
    --baseline_lora LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128 \
    --lora_dir ./magicoder_r64_simpleguard_v1 \
    --config_src LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128 \
    --out_json ./magicoder_r64_simpleguard_v1/metrics_baseline_vs_edited.json

Base model only (no LoRA):
  CUDA_VISIBLE_DEVICES=0 python eval_humaneval_lora_vllm.py \
    --base_model_id meta-llama/Llama-2-7b-hf \
    --base_only \
    --out_json ./base_model_humaneval.json
"""

import os
import json
import shutil
import inspect
import argparse
import subprocess
from typing import Optional, Dict, Any, List, Tuple

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

    results_path = samples_path + "_results.jsonl"
    return {}, results_path if os.path.exists(results_path) else None


def compute_pass_at_1_from_results_jsonl(results_path: str, expected_n_samples: int = 1) -> dict:
    """Compute pass@1 from results JSONL file."""
    rows = jsonl_read(results_path)
    per_task_total: Dict[str, int] = {}
    per_task_passed: Dict[str, int] = {}

    for r in rows:
        tid = r.get("task_id")
        if tid is None:
            continue
        per_task_total[tid] = per_task_total.get(tid, 0) + 1
        if bool(r.get("passed", False)):
            per_task_passed[tid] = per_task_passed.get(tid, 0) + 1

    task_ids = sorted(per_task_total.keys())
    if not task_ids:
        return {"pass@1": 0.0, "num_tasks": 0, "correct": 0, "total": 0}

    vals = []
    bad = 0
    for tid in task_ids:
        n = per_task_total[tid]
        c = per_task_passed.get(tid, 0)
        if expected_n_samples is not None and n != int(expected_n_samples):
            bad += 1
        vals.append(c / max(1, n))

    correct = sum(1 for v in vals if v > 0)
    return {
        "pass@1": float(sum(vals) / len(vals)),
        "correct": correct,
        "total": int(len(vals)),
        "num_tasks": int(len(vals)),
        "tasks_with_sample_count_mismatch": int(bad),
    }


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
    Evaluate one adapter on HumanEval and return metrics.
    """
    if lora_dir is not None:
        lora_req = LoRARequest(adapter_name, 1, lora_dir)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        # Base model only
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
        metrics = compute_pass_at_1_from_results_jsonl(results_path, expected_n_samples=1)
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


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate LoRA adapters on HumanEval with vLLM (pass@1, greedy)"
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

    ap.add_argument("--max_samples", type=int, default=-1,
                    help="Max number of HumanEval tasks to evaluate (-1 for all 164).")

    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0.0 for greedy, recommended for pass@1).")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p (nucleus) sampling parameter.")
    ap.add_argument("--max_tokens", type=int, default=512,
                    help="Maximum new tokens to generate.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility.")

    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="Maximum model context length for vLLM.")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Output JSON file for metrics. Defaults to <lora_dir>/metrics_humaneval_vllm.json")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Directory for intermediate files (samples JSONL). Defaults to lora_dir or cwd.")

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

    # Ensure LoRA configs exist
    enable_lora = not args.base_only
    max_r = 16  # default

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

    sp = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
        stop=args.stop_words if args.stop_words else None,
    )

    results: Dict[str, Any] = {
        "meta": {
            "base_model_id": args.base_model_id,
            "task": "HumanEval",
            "max_samples": args.max_samples,
            "num_tasks": len(task_ids),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_lora_rank_engine": max_r if enable_lora else None,
            "stop_words": args.stop_words,
            "eval_n_workers": args.eval_n_workers,
            "eval_timeout": args.eval_timeout,
        }
    }

    # Evaluate base model only
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
        print(f"[vLLM][Base] {base_metrics}")
        results["base"] = base_metrics

    else:
        # Baseline LoRA (optional)
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
            print(f"[vLLM][Baseline] {baseline_metrics}")
            results["baseline"] = baseline_metrics

        # Edited/new LoRA
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
        print(f"[vLLM][Edited] {edited_metrics}")
        results["edited"] = edited_metrics

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


if __name__ == "__main__":
    main()
