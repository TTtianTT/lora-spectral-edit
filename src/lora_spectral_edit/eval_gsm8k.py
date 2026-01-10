"""
GSM8K evaluation utilities using vLLM.

This module is optional and requires vLLM to be installed.
"""

import os
import re
from typing import Optional, List

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
    """
    m = re.findall(r"####\s*(-?\d+)", ans)
    if m:
        return m[-1]
    m2 = re.findall(r"(-?\d+)", ans.replace(",", ""))
    return m2[-1] if m2 else None


def pred_number(text: str) -> Optional[str]:
    """Extract predicted number from model output."""
    m = re.findall(r"####\s*(-?\d+)", text)
    if m:
        return m[-1]
    m2 = re.findall(r"(-?\d+)", text.replace(",", ""))
    return m2[-1] if m2 else None


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


def evaluate_gsm8k_vllm(
    base_model_id: str,
    lora_dir: str,
    fewshot_k: int = 5,
    max_samples: int = -1,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 0,
    max_model_len: int = 4096,
) -> dict:
    """
    Evaluate a LoRA adapter on GSM8K using vLLM.

    Args:
        base_model_id: HuggingFace model ID for the base model.
        lora_dir: Path to the LoRA adapter directory.
        fewshot_k: Number of few-shot examples.
        max_samples: Maximum number of test samples (-1 for all).
        temperature: Sampling temperature (0 for greedy).
        max_tokens: Maximum tokens to generate.
        seed: Random seed.
        max_model_len: Maximum model context length.

    Returns:
        Dictionary with 'acc', 'correct', and 'total' keys.
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
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)

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
    for out, gold in zip(outputs, gold_nums):
        gen = out.outputs[0].text
        pred = pred_number(gen)
        if gold is not None and pred == gold:
            correct += 1

    return {"acc": correct / total if total else 0.0, "correct": correct, "total": total}


def evaluate_both_loras(
    base_model_id: str,
    baseline_lora_dir: str,
    edited_lora_dir: str,
    fewshot_k: int = 5,
    max_samples: int = -1,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int = 0,
    max_model_len: int = 4096,
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
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens)

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
        for out, gold in zip(outputs, gold_nums):
            gen = out.outputs[0].text
            pred = pred_number(gen)
            if gold is not None and pred == gold:
                correct += 1

        results[name] = {"acc": correct / total if total else 0.0, "correct": correct, "total": total}

    return results
