from __future__ import annotations

"""
GSM8K strict-match evaluation (vLLM-first), aligned with the updated evaluation style.

Key alignment with the updated script:
- Prompt template: instruction + "#### <answer>" requirement
- Answer extraction: _extract_answer + _norm, strict string match
- Outputs: predictions.jsonl + metrics.json (and optional eval_config.json if you still want)
- Supports: split / max_samples / max_new_tokens / dtype / tensor_parallel_size / (optional) few-shot
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from datasets import load_dataset

from .io import load_adapter_config

# vLLM is optional
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


# ----------------------------
# small helpers (keep yours)
# ----------------------------

def _env_truthy(name: str) -> bool | None:
    val = os.environ.get(name)
    if val is None:
        return None
    return val.strip().lower() in {"1", "true", "yes", "y"}


def check_vllm_available() -> None:
    if not HAVE_VLLM:
        raise RuntimeError(
            "vLLM not available. Please install it with: pip install vllm\n"
            "Note: vLLM requires a CUDA GPU."
        )


def _extract_answer(text: str) -> str:
    """
    Updated strict-match extraction:
    - If "####" exists: take the first line after the last ####
    - Else: fall back to the last number-like token
    - Else: fall back to last non-empty line
    """
    if "####" in text:
        tail = text.split("####")[-1].strip()
        return tail.splitlines()[0].strip() if tail else ""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if matches:
        return matches[-1].strip()
    text = text.strip()
    return text.splitlines()[-1].strip() if text else ""


def _norm(s: str) -> str:
    return s.strip().replace(",", "")


def _dtype_to_vllm(dtype: str) -> str:
    """
    Map your CLI-like dtype choices to vLLM dtype strings.
    vLLM commonly accepts: "auto", "float16", "bfloat16", "float32".
    """
    d = (dtype or "auto").lower()
    if d in {"auto"}:
        return "auto"
    if d in {"bf16", "bfloat16"}:
        return "bfloat16"
    if d in {"fp16", "float16"}:
        return "float16"
    if d in {"fp32", "float32"}:
        return "float32"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_lora_max_rank(adapter_dir: str) -> int:
    cfg = load_adapter_config(adapter_dir)
    r = int(cfg.get("r", 0) or 0)
    if r <= 0 and isinstance(cfg.get("rank_pattern", None), dict):
        r = max(int(v) for v in cfg["rank_pattern"].values())
    if r <= 0:
        raise ValueError(f"Cannot determine LoRA rank r from adapter_config.json in {adapter_dir}")
    return r


# ----------------------------
# prompt building (updated)
# ----------------------------

_INSTRUCTION_HEADER = (
    "Solve the following problem. Put your final numeric answer on the last line as:\n"
    "#### <answer>\n\n"
)

def build_gsm8k_prompt(question: str) -> str:
    q = str(question).strip()
    return _INSTRUCTION_HEADER + f"Problem:\n{q}\n\nAnswer:\n"


def build_gsm8k_fewshot_prefix(train_ds, k: int) -> str:
    """
    Few-shot examples in the SAME prompt style (optional).
    If you want to match the paper setting: k=5, and answers from GSM8K train include '#### ...'.
    """
    k = max(0, int(k))
    if k == 0:
        return ""
    examples = train_ds.select(range(k))
    parts: List[str] = []
    # Put instruction header once at the top, then examples as (Problem/Answer) blocks.
    parts.append(_INSTRUCTION_HEADER)
    for ex in examples:
        q = str(ex.get("question", "")).strip()
        a = str(ex.get("answer", "")).strip()
        parts.append(f"Problem:\n{q}\n\nAnswer:\n{a}\n")
    return "\n".join(parts).rstrip() + "\n\n"


# ----------------------------
# file outputs (updated)
# ----------------------------

def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ----------------------------
# main eval (vLLM)
# ----------------------------

def evaluate_gsm8k_vllm(
    base_model: str,
    adapter_dir: Optional[str],
    output_dir: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    dtype: str = "auto",               # auto/bf16/fp16/fp32
    seed: int = 42,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    fewshot_k: int = 0,                # set to 5 if you still want paper-style 5-shot
    write_eval_config_json: bool = False,  # keep off by default to match your updated script
) -> Dict[str, Any]:
    """
    vLLM strict-match accuracy on GSM8K (aligned with updated script outputs).

    Writes:
      - predictions.jsonl
      - metrics.json
      - (optional) eval_config.json
    """
    check_vllm_available()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    # dataset
    try:
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load gsm8k split={split}: {exc}") from exc

    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    # few-shot (optional)
    fewshot_prefix = ""
    if int(fewshot_k) > 0:
        train_ds = load_dataset("gsm8k", "main", split="train")
        fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, fewshot_k)

    # vLLM engine
    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", None)
    gpu_mem_util = float(gpu_mem_util) if gpu_mem_util is not None else None
    enforce_eager = _env_truthy("VLLM_ENFORCE_EAGER")

    llm_kwargs: Dict[str, Any] = {
        "model": base_model,
        "dtype": _dtype_to_vllm(dtype),
        "max_model_len": int(max_model_len),
        "seed": int(seed),
        "tensor_parallel_size": int(tensor_parallel_size),
    }

    use_lora = adapter_dir is not None and str(adapter_dir).strip() != ""
    if use_lora:
        max_r = _get_lora_max_rank(str(adapter_dir))
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = int(max_r)

    if gpu_mem_util is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_util
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens))

    prompts: List[str] = []
    records: List[Tuple[str, str]] = []
    for ex in ds:
        q = str(ex.get("question", "")).strip()
        gold_raw = str(ex.get("answer", "")).strip()
        gold = _norm(_extract_answer(gold_raw))
        prompt = (fewshot_prefix + f"Problem:\n{q}\n\nAnswer:\n") if fewshot_prefix else build_gsm8k_prompt(q)
        prompts.append(prompt)
        records.append((q, gold))

    # generate
    if use_lora:
        lora_req = LoRARequest("adapter", 1, str(adapter_dir))
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sp)

    correct = 0
    total = 0
    with preds_path.open("w", encoding="utf-8") as f:
        for (q, gold), out in zip(records, outputs):
            gen = out.outputs[0].text
            pred = _norm(_extract_answer(gen))
            is_correct = int(pred == gold)
            correct += is_correct
            total += 1

            rec: Dict[str, Any] = {
                "question": q,
                "gold": gold,
                "prediction_text": gen,
                "prediction_extracted": pred,
                "correct": bool(is_correct),
            }
            if use_lora:
                rec["adapter_dir"] = str(adapter_dir)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "accuracy_strict": (correct / total if total else 0.0),
        "correct": correct,
        "total": total,
    }
    _save_json(metrics_path, metrics)

    if write_eval_config_json:
        eval_config = {
            "task": "gsm8k",
            "split": split,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "seed": seed,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "fewshot_k": fewshot_k,
            "base_model": base_model,
            "adapter_dir": str(adapter_dir) if use_lora else None,
            "timestamp": datetime.now().isoformat(),
            "outputs": {
                "predictions": str(preds_path),
                "metrics": str(metrics_path),
            },
        }
        _save_json(out_dir / "eval_config.json", eval_config)

    return {
        **metrics,
        "predictions_path": str(preds_path),
        "metrics_path": str(metrics_path),
    }


def evaluate_both_loras_vllm(
    base_model: str,
    baseline_adapter_dir: str,
    edited_adapter_dir: str,
    output_dir: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
    dtype: str = "auto",
    seed: int = 42,
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    fewshot_k: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate baseline + edited adapters with a single vLLM engine.
    Writes:
      - predictions_baseline.jsonl
      - predictions_edited.jsonl
      - metrics.json  (includes both)
    """
    check_vllm_available()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_base_path = out_dir / "predictions_baseline.jsonl"
    preds_edit_path = out_dir / "predictions_edited.jsonl"
    metrics_path = out_dir / "metrics.json"

    # dataset
    try:
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load gsm8k split={split}: {exc}") from exc
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    # few-shot (optional)
    fewshot_prefix = ""
    if int(fewshot_k) > 0:
        train_ds = load_dataset("gsm8k", "main", split="train")
        fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, fewshot_k)

    prompts: List[str] = []
    records: List[Tuple[str, str]] = []
    for ex in ds:
        q = str(ex.get("question", "")).strip()
        gold_raw = str(ex.get("answer", "")).strip()
        gold = _norm(_extract_answer(gold_raw))
        prompt = (fewshot_prefix + f"Problem:\n{q}\n\nAnswer:\n") if fewshot_prefix else build_gsm8k_prompt(q)
        prompts.append(prompt)
        records.append((q, gold))

    # engine rank = max(baseline, edited)
    max_r = max(_get_lora_max_rank(baseline_adapter_dir), _get_lora_max_rank(edited_adapter_dir))

    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", None)
    gpu_mem_util = float(gpu_mem_util) if gpu_mem_util is not None else None
    enforce_eager = _env_truthy("VLLM_ENFORCE_EAGER")

    llm_kwargs: Dict[str, Any] = {
        "model": base_model,
        "dtype": _dtype_to_vllm(dtype),
        "max_model_len": int(max_model_len),
        "seed": int(seed),
        "tensor_parallel_size": int(tensor_parallel_size),
        "enable_lora": True,
        "max_lora_rank": int(max_r),
    }
    if gpu_mem_util is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_util
    if enforce_eager is not None:
        llm_kwargs["enforce_eager"] = enforce_eager

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens))

    def _run_one(name: str, adapter_dir: str, req_id: int, preds_path: Path) -> Dict[str, Any]:
        lora_req = LoRARequest(name, req_id, adapter_dir)
        outputs = llm.generate(prompts, sp, lora_request=lora_req)

        correct = 0
        total = 0
        with preds_path.open("w", encoding="utf-8") as f:
            for (q, gold), out in zip(records, outputs):
                gen = out.outputs[0].text
                pred = _norm(_extract_answer(gen))
                is_correct = int(pred == gold)
                correct += is_correct
                total += 1
                f.write(json.dumps({
                    "question": q,
                    "gold": gold,
                    "prediction_text": gen,
                    "prediction_extracted": pred,
                    "correct": bool(is_correct),
                    "adapter_dir": adapter_dir,
                }, ensure_ascii=False) + "\n")

        return {
            "accuracy_strict": (correct / total if total else 0.0),
            "correct": correct,
            "total": total,
            "predictions_path": str(preds_path),
        }

    baseline_metrics = _run_one("baseline", baseline_adapter_dir, 1, preds_base_path)
    edited_metrics = _run_one("edited", edited_adapter_dir, 2, preds_edit_path)

    metrics = {
        "baseline": baseline_metrics,
        "edited": edited_metrics,
        "meta": {
            "task": "gsm8k",
            "split": split,
            "max_samples": max_samples,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "seed": seed,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "fewshot_k": fewshot_k,
            "base_model": base_model,
            "baseline_adapter_dir": baseline_adapter_dir,
            "edited_adapter_dir": edited_adapter_dir,
            "timestamp": datetime.now().isoformat(),
        },
    }
    _save_json(metrics_path, metrics)

    return {
        "baseline": baseline_metrics,
        "edited": edited_metrics,
        "metrics_path": str(metrics_path),
    }
