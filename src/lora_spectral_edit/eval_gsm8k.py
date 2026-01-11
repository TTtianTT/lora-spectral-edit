"""
GSM8K evaluation utilities using vLLM.

This module is optional and requires vLLM to be installed.

Supports evaluation profiles matching "LoRA Learns Less and Forgets Less" paper:
- paper_math: 5-shot, greedy decoding, strict match accuracy (1319 test samples)
"""

import os
import re
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from datasets import load_dataset

from .io import load_adapter_config

# vLLM is optional
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


def _env_truthy(name: str) -> bool | None:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes", "y"}


def check_vllm_available():
    """Check if vLLM is available and raise if not."""
    if not HAVE_VLLM:
        raise RuntimeError(
            "vLLM not available. Please install it with: pip install vllm\n"
            "Note: vLLM requires a CUDA GPU."
        )


def extract_gsm8k_final_number(ans: str) -> Optional[str]:
    """
    Extract the final answer number from GSM8K format.

    GSM8K answers end with '#### <number>'. Falls back to last integer if not found.
    This implements strict-match parsing as per the paper.
    """
    # Primary: look for #### format (official GSM8K format)
    m = re.findall(r"####\s*(-?\d[\d,]*)", ans)
    if m:
        # Remove commas from number
        return m[-1].replace(",", "")
    # Fallback: last number in text
    m2 = re.findall(r"(-?\d[\d,]*)", ans.replace(",", ""))
    return m2[-1] if m2 else None


def pred_number(text: str) -> Optional[str]:
    """Extract predicted number from model output using strict-match parsing."""
    # Primary: look for #### format
    m = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return m[-1].replace(",", "")
    # Fallback: last number in text
    m2 = re.findall(r"(-?\d[\d,]*)", text.replace(",", ""))
    return m2[-1].replace(",", "") if m2 else None


def build_gsm8k_fewshot_prefix(train_ds, k: int) -> str:
    """Build few-shot prompt prefix from training examples."""
    k = max(0, int(k))
    if k == 0:
        return ""
    examples = train_ds.select(range(k))
    parts = []
    for ex in examples:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n")
    return "\n".join(parts) + "\n"


def write_eval_config(
    out_dir: str,
    task: str,
    split: str,
    num_fewshot: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    total_samples: int,
    metric: str,
    score: float,
    correct: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write evaluation config summary to a JSON file in the run directory.

    Args:
        out_dir: Output directory
        task: Task name (e.g., "gsm8k")
        split: Dataset split (e.g., "test")
        num_fewshot: Number of few-shot examples
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Max tokens to generate
        total_samples: Total number of samples evaluated
        metric: Metric name (e.g., "strict_match_accuracy")
        score: Metric score
        correct: Number of correct predictions
        extra_meta: Additional metadata to include

    Returns:
        Path to the written config file
    """
    config = {
        "task": task,
        "split": split,
        "num_fewshot": num_fewshot,
        "decoding": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "strategy": "greedy" if temperature == 0.0 else "sampling",
        },
        "total_samples": total_samples,
        "metric": {
            "name": metric,
            "score": score,
            "correct": correct,
            "total": total_samples,
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


def evaluate_gsm8k_vllm(
    base_model_id: str,
    lora_dir: str,
    fewshot_k: int = 5,
    max_samples: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    seed: int = 0,
    max_model_len: int = 4096,
    out_dir: Optional[str] = None,
    write_config: bool = True,
) -> dict:
    """
    Evaluate a LoRA adapter on GSM8K using vLLM.

    Args:
        base_model_id: HuggingFace model ID for the base model.
        lora_dir: Path to the LoRA adapter directory.
        fewshot_k: Number of few-shot examples (paper_math: 5).
        max_samples: Maximum number of test samples (-1 for all 1319).
        temperature: Sampling temperature (0 for greedy, paper_math: 0.0).
        top_p: Top-p sampling (paper_math: 1.0).
        max_tokens: Maximum tokens to generate.
        seed: Random seed.
        max_model_len: Maximum model context length.
        out_dir: Output directory for config JSON (defaults to lora_dir).
        write_config: Whether to write eval config JSON.

    Returns:
        Dictionary with 'acc', 'correct', 'total', and optionally 'config_path'.
    """
    check_vllm_available()

    ds = load_dataset("gsm8k", "main")
    train_ds, test_ds = ds["train"], ds["test"]

    fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, fewshot_k)

    # Determine max_lora_rank from adapter_config
    cfg = load_adapter_config(lora_dir)
    r = int(cfg.get("r", 0) or 0)
    if r <= 0 and isinstance(cfg.get("rank_pattern", None), dict):
        r = max(int(v) for v in cfg["rank_pattern"].values())
    if r <= 0:
        raise ValueError("Cannot determine LoRA rank r from adapter_config.json")

    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", None)
    gpu_mem_util = float(gpu_mem_util) if gpu_mem_util is not None else None

    enforce_eager = _env_truthy("VLLM_ENFORCE_EAGER")
    llm_kwargs = {
        "model": base_model_id,
        "dtype": "float16",
        "max_model_len": max_model_len,
        "enable_lora": True,
        "max_lora_rank": r,
        "seed": seed,
    }
    if gpu_mem_util is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_util
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    if max_samples is None or int(max_samples) < 0:
        eval_ds = test_ds
    else:
        eval_ds = test_ds.select(range(int(max_samples)))

    lora_req = LoRARequest("adapter", 1, lora_dir)

    prompts = []
    gold_nums = []
    for ex in eval_ds:
        q = ex["question"]
        gold = extract_gsm8k_final_number(ex["answer"])
        gold_nums.append(gold)
        prompts.append(fewshot_prefix + f"Question: {q}\nAnswer:")

    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    correct = 0
    total = len(gold_nums)
    predictions = []
    for out, gold in zip(outputs, gold_nums):
        gen = out.outputs[0].text
        pred = pred_number(gen)
        predictions.append({
            "gold": gold,
            "pred": pred,
            "correct": gold is not None and pred == gold,
            "generation": gen,
        })
        if gold is not None and pred == gold:
            correct += 1

    acc = correct / total if total else 0.0
    result = {"acc": acc, "correct": correct, "total": total}

    # Write config JSON if requested
    if write_config:
        config_dir = out_dir if out_dir else lora_dir
        config_path = write_eval_config(
            out_dir=config_dir,
            task="gsm8k",
            split="test",
            num_fewshot=fewshot_k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            total_samples=total,
            metric="strict_match_accuracy",
            score=acc,
            correct=correct,
            extra_meta={
                "base_model_id": base_model_id,
                "lora_dir": lora_dir,
                "seed": seed,
                "max_model_len": max_model_len,
            },
        )
        result["config_path"] = config_path

        # Also write predictions for auditability
        preds_path = os.path.join(config_dir, "gsm8k_predictions.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(predictions):
                f.write(json.dumps({"idx": i, **p}, ensure_ascii=False) + "\n")
        result["predictions_path"] = preds_path

    return result


def evaluate_gsm8k_with_profile(
    base_model_id: str,
    lora_dir: str,
    profile_name: str = "paper_math",
    max_samples: Optional[int] = None,
    seed: int = 0,
    max_model_len: int = 4096,
    out_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate GSM8K using a predefined profile.

    Args:
        base_model_id: HuggingFace model ID for the base model.
        lora_dir: Path to the LoRA adapter directory.
        profile_name: Profile name ("paper_math" for paper settings).
        max_samples: Override max samples (None uses profile default).
        seed: Random seed.
        max_model_len: Maximum model context length.
        out_dir: Output directory for results.

    Returns:
        Dictionary with evaluation results and config.
    """
    from .eval_profiles import get_profile

    profile = get_profile(profile_name)

    if profile.task != "gsm8k":
        raise ValueError(f"Profile '{profile_name}' is for task '{profile.task}', not gsm8k")

    samples = max_samples if max_samples is not None else profile.max_samples

    result = evaluate_gsm8k_vllm(
        base_model_id=base_model_id,
        lora_dir=lora_dir,
        fewshot_k=profile.num_fewshot,
        max_samples=samples,
        temperature=profile.temperature,
        top_p=profile.top_p,
        max_tokens=profile.max_tokens,
        seed=seed,
        max_model_len=max_model_len,
        out_dir=out_dir,
        write_config=True,
    )

    result["profile"] = profile.get_summary()
    return result


def evaluate_both_loras(
    base_model_id: str,
    baseline_lora_dir: str,
    edited_lora_dir: str,
    fewshot_k: int = 5,
    max_samples: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    seed: int = 0,
    max_model_len: int = 4096,
    out_dir: Optional[str] = None,
    write_config: bool = True,
) -> dict:
    """
    Evaluate both baseline and edited LoRA adapters on GSM8K.

    Uses a single vLLM engine for efficiency.

    Returns:
        Dictionary with 'baseline' and 'edited' result dicts.
    """
    check_vllm_available()

    ds = load_dataset("gsm8k", "main")
    train_ds, test_ds = ds["train"], ds["test"]

    fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, fewshot_k)

    # Determine max rank from both adapters
    cfg_base = load_adapter_config(baseline_lora_dir)
    cfg_edit = load_adapter_config(edited_lora_dir)

    def get_rank(cfg):
        r = int(cfg.get("r", 0) or 0)
        if r <= 0 and isinstance(cfg.get("rank_pattern", None), dict):
            r = max(int(v) for v in cfg["rank_pattern"].values())
        return r

    max_r = max(get_rank(cfg_base), get_rank(cfg_edit))
    if max_r <= 0:
        raise ValueError("Cannot determine LoRA rank from adapter configs")

    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", None)
    gpu_mem_util = float(gpu_mem_util) if gpu_mem_util is not None else None

    enforce_eager = _env_truthy("VLLM_ENFORCE_EAGER")
    llm_kwargs = {
        "model": base_model_id,
        "dtype": "float16",
        "max_model_len": max_model_len,
        "enable_lora": True,
        "max_lora_rank": max_r,
        "seed": seed,
    }
    if gpu_mem_util is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_util
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    if max_samples is None or int(max_samples) < 0:
        eval_ds = test_ds
    else:
        eval_ds = test_ds.select(range(int(max_samples)))

    prompts = []
    gold_nums = []
    for ex in eval_ds:
        q = ex["question"]
        gold = extract_gsm8k_final_number(ex["answer"])
        gold_nums.append(gold)
        prompts.append(fewshot_prefix + f"Question: {q}\nAnswer:")

    total = len(gold_nums)
    results = {}

    for name, lora_dir in [("baseline", baseline_lora_dir), ("edited", edited_lora_dir)]:
        lora_req = LoRARequest(name, 1 if name == "baseline" else 2, lora_dir)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)

        correct = 0
        predictions = []
        for out, gold in zip(outputs, gold_nums):
            gen = out.outputs[0].text
            pred = pred_number(gen)
            is_correct = gold is not None and pred == gold
            predictions.append({
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "generation": gen,
            })
            if is_correct:
                correct += 1

        acc = correct / total if total else 0.0
        results[name] = {"acc": acc, "correct": correct, "total": total}

        # Write predictions if out_dir specified
        if write_config and out_dir:
            preds_path = os.path.join(out_dir, f"gsm8k_predictions_{name}.jsonl")
            os.makedirs(out_dir, exist_ok=True)
            with open(preds_path, "w", encoding="utf-8") as f:
                for i, p in enumerate(predictions):
                    f.write(json.dumps({"idx": i, **p}, ensure_ascii=False) + "\n")
            results[name]["predictions_path"] = preds_path

    # Write combined config
    if write_config and out_dir:
        config = {
            "task": "gsm8k",
            "split": "test",
            "num_fewshot": fewshot_k,
            "decoding": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "strategy": "greedy" if temperature == 0.0 else "sampling",
            },
            "total_samples": total,
            "baseline": results["baseline"],
            "edited": results["edited"],
            "meta": {
                "base_model_id": base_model_id,
                "baseline_lora_dir": baseline_lora_dir,
                "edited_lora_dir": edited_lora_dir,
                "seed": seed,
                "max_model_len": max_model_len,
            },
            "timestamp": datetime.now().isoformat(),
        }
        config_path = os.path.join(out_dir, "eval_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        results["config_path"] = config_path

    return results
