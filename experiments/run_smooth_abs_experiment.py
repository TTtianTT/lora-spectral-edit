#!/usr/bin/env python
"""
End-to-end smooth_abs experiment: edit + evaluate on both GSM8K and HumanEval.

This script:
1) Applies smooth_abs spectral editing to a LoRA adapter
2) Evaluates the edited adapter on GSM8K (paper_math profile: 5-shot greedy strict match)
3) Evaluates the edited adapter on HumanEval (paper_code_main profile: 50-sample pass@1)
4) Optionally evaluates the baseline (unedited) adapter
5) Writes a unified summary.json comparing baseline vs edited

Usage:
    python experiments/run_smooth_abs_experiment.py \
        --base_model meta-llama/Llama-2-7b-hf \
        --lora_dir ./my_lora \
        --out_dir ./runs/smooth_abs_exp1 \
        --amp_factor 1.25 \
        --sup_factor 0.8 \
        --smooth_temperature 0.35 \
        --do_baseline_eval \
        --seed 42

Smoke test (5 samples each):
    python experiments/run_smooth_abs_experiment.py \
        --base_model meta-llama/Llama-2-7b-hf \
        --lora_dir ./my_lora \
        --out_dir ./runs/smoke_test \
        --max_samples 5 \
        --humaneval_n_generations 5

Full run (paper settings):
    python experiments/run_smooth_abs_experiment.py \
        --base_model meta-llama/Llama-2-7b-hf \
        --lora_dir ./my_lora \
        --out_dir ./runs/full_exp \
        --do_baseline_eval
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence

# Add src to path for local imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from lora_spectral_edit.io import (
    ensure_local_lora_dir,
    load_adapter_config,
    load_lora_state_dict,
    save_lora_state_dict,
    parse_lora_ab_key,
    layer_idx_from_module_prefix,
    get_scaling_for_module,
)
from lora_spectral_edit.svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from lora_spectral_edit.hooks import ModuleSpec, HOOK_CTX, register_sigma_hooks, remove_hooks
from lora_spectral_edit.edit import EditConfig, apply_spectral_edit


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_calib_batch(tokenizer, examples: List[dict], add_eos: bool = True):
    """Build teacher-forcing inputs for calibration."""
    input_ids_list = []
    labels_list = []

    for ex in examples:
        q = ex["question"]
        a = ex["answer"]
        prompt = f"Question: {q}\nAnswer:"
        full = prompt + " " + a
        if add_eos and tokenizer.eos_token:
            full = full + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids

        mask_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * mask_len + full_ids[mask_len:]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attn_mask, labels


def run_smooth_abs_edit(
    base_model_id: str,
    lora_dir: str,
    out_adapter_dir: str,
    target_modules: List[str],
    layer_min: int,
    layer_max: int,
    calib_samples: int,
    calib_batch_size: int,
    # smooth_abs hyperparams
    amp_factor: float,
    sup_factor: float,
    mid_factor: float,
    smooth_temperature: float,
    smooth_center_q: float,
    smooth_align_mid: bool,
    grad_norm: str,
    preserve_energy: str,
    sigma_clip_min: float,
    seed: int,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply smooth_abs spectral editing to a LoRA adapter.

    Returns:
        Dict with edit metadata and per-module stats
    """
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This tool requires a CUDA GPU for gradient computation.")

    # Prepare LoRA files
    lora_local = ensure_local_lora_dir(lora_dir, cache_dir=cache_dir)
    adapter_cfg = load_adapter_config(lora_local)
    sd, fmt = load_lora_state_dict(lora_local)

    # Copy adapter directory to output
    if os.path.abspath(out_adapter_dir) != os.path.abspath(lora_local):
        if os.path.exists(out_adapter_dir):
            shutil.rmtree(out_adapter_dir)
        shutil.copytree(lora_local, out_adapter_dir)

    # Identify LoRA A/B pairs
    pairs: Dict[str, dict] = {}
    target_modules_set = set(target_modules)

    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed

        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue

        li = layer_idx_from_module_prefix(prefix)
        if li is not None and not (layer_min <= li <= layer_max):
            continue

        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected_prefixes = [p for p in pairs.keys() if "A" in pairs[p] and "B" in pairs[p]]
    if not selected_prefixes:
        raise RuntimeError("No matching LoRA (A,B) pairs found for given target_modules/layer range.")

    print(f"[Edit] Selected LoRA modules: {len(selected_prefixes)}")

    # Load model for gradient probing
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    model = PeftModel.from_pretrained(base, lora_local, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    # Freeze everything except LoRA params
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    # Build ModuleSpec for each selected module
    name_to_module = dict(model.named_modules())
    specs: Dict[str, ModuleSpec] = {}

    for prefix in selected_prefixes:
        keyA, A_cpu, adapterA = pairs[prefix]["A"]
        keyB, B_cpu, adapterB = pairs[prefix]["B"]
        adapter_name = adapterA if adapterA is not None else adapterB

        if prefix not in name_to_module:
            candidates = [nm for nm in name_to_module.keys() if nm.endswith(prefix)]
            if not candidates:
                raise RuntimeError(f"Cannot find module '{prefix}' in model")
            module_name = candidates[0]
        else:
            module_name = prefix

        mod = name_to_module[module_name]
        A = A_cpu.to(device)
        B = B_cpu.to(device)

        U, S, Vh, V = lowrank_svd_from_ba(B, A)
        scaling = get_scaling_for_module(adapter_cfg, prefix)

        specs[prefix] = ModuleSpec(
            module_prefix=prefix,
            module=mod,
            U=U.detach(),
            V=V.detach(),
            Vh=Vh.detach(),
            sigma0=S.detach().cpu(),
            scaling=scaling,
            adapter=adapter_name,
        )

    print(f"[Edit] Built SVD specs for {len(specs)} modules.")

    # Register hooks and run calibration
    handles = register_sigma_hooks(specs)

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]

    ncal = min(calib_samples, len(train_ds))
    calib_examples = [train_ds[i] for i in range(ncal)]

    bs = max(1, calib_batch_size)
    total_loss = 0.0
    n_steps = 0

    HOOK_CTX.reset()

    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i:i+bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        HOOK_CTX.attn_mask = attn_mask

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        total_loss += float(loss.item())
        n_steps += 1

        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)

        step_num = i // bs + 1
        total_steps = math.ceil(ncal / bs)
        if step_num % 10 == 0 or step_num == total_steps:
            print(f"[Calib] step {step_num}/{total_steps} loss={loss.item():.4f}")

    remove_hooks(handles)
    HOOK_CTX.attn_mask = None

    print(f"[Calib] avg loss: {total_loss / max(1, n_steps):.4f}")

    if not HOOK_CTX.gsum:
        raise RuntimeError("No gradients accumulated. Hooks may not have fired.")

    # Build edit config for smooth_abs
    edit_config = EditConfig(
        mode="smooth_abs",
        core_frac=0.2,  # used for range computation
        noise_frac=0.2,
        amp_factor=amp_factor,
        sup_factor=sup_factor,
        mid_factor=mid_factor,
        smooth_temperature=smooth_temperature,
        smooth_center_q=smooth_center_q,
        smooth_align_mid=smooth_align_mid,
        grad_norm=grad_norm,
        preserve_energy=preserve_energy,
        sigma_clip_min=sigma_clip_min,
    )

    # Apply edits
    sigma_stats = {}

    for prefix, spec in specs.items():
        sigma0 = spec.sigma0.clone()
        g = HOOK_CTX.gsum.get(prefix, None)
        if g is None:
            continue

        sigma_new, stats = apply_spectral_edit(sigma0, g, edit_config)

        # Rebuild A/B
        U = spec.U.to(device)
        Vh = spec.Vh.to(device)
        sigma_new_gpu = sigma_new.to(device)

        B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, sigma_new_gpu)

        keyA, A_old, _ = pairs[prefix]["A"]
        keyB, B_old, _ = pairs[prefix]["B"]
        A_new = A_new.to(dtype=A_old.dtype).detach().cpu()
        B_new = B_new.to(dtype=B_old.dtype).detach().cpu()

        sd[keyA] = A_new
        sd[keyB] = B_new
        sigma_stats[prefix] = stats

    # Save edited adapter
    save_lora_state_dict(out_adapter_dir, sd, fmt)

    meta = {
        "edit_mode": "smooth_abs",
        "base_model": base_model_id,
        "lora_path": lora_dir,
        "target_modules": target_modules,
        "layer_min": layer_min,
        "layer_max": layer_max,
        "calib_samples": calib_samples,
        "amp_factor": amp_factor,
        "sup_factor": sup_factor,
        "mid_factor": mid_factor,
        "smooth_temperature": smooth_temperature,
        "smooth_center_q": smooth_center_q,
        "smooth_align_mid": smooth_align_mid,
        "grad_norm": grad_norm,
        "preserve_energy": preserve_energy,
        "seed": seed,
    }

    with open(os.path.join(out_adapter_dir, "spectral_edit_meta.json"), "w") as f:
        json.dump({"meta": meta, "sigma_stats": sigma_stats}, f, indent=2)

    print(f"[Edit] Edited adapter saved to: {out_adapter_dir}")

    # Free GPU memory
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base, tok
    gc.collect()
    torch.cuda.empty_cache()

    return {"meta": meta, "sigma_stats": sigma_stats}


def run_gsm8k_eval(
    base_model_id: str,
    lora_dir: str,
    out_dir: str,
    max_samples: Optional[int],
    seed: int,
    max_model_len: int = 4096,
) -> Dict[str, Any]:
    """Run GSM8K evaluation with paper_math profile."""
    from lora_spectral_edit.eval_gsm8k import evaluate_gsm8k_with_profile

    print(f"\n[GSM8K] Evaluating with paper_math profile...")
    os.makedirs(out_dir, exist_ok=True)

    result = evaluate_gsm8k_with_profile(
        base_model_id=base_model_id,
        lora_dir=lora_dir,
        profile_name="paper_math",
        max_samples=max_samples,
        seed=seed,
        max_model_len=max_model_len,
        out_dir=out_dir,
    )

    print(f"[GSM8K] acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
    return result


def run_humaneval_eval(
    base_model_id: str,
    lora_dir: Optional[str],
    out_dir: str,
    max_samples: Optional[int],
    n_generations: int,
    seed: int,
    max_model_len: int = 4096,
    config_src: Optional[str] = None,
) -> Dict[str, Any]:
    """Run HumanEval evaluation with paper_code_main settings."""
    from lora_spectral_edit.eval_humaneval import (
        HAVE_VLLM, HAVE_HUMAN_EVAL, LLM, SamplingParams,
        read_problems, load_adapter_config, ensure_lora_has_config,
        infer_max_lora_rank, eval_one_adapter_multisample,
        write_eval_config
    )

    if not HAVE_VLLM:
        raise RuntimeError("vLLM not available. Please `pip install vllm`.")
    if not HAVE_HUMAN_EVAL:
        raise RuntimeError("human_eval not available.")

    # Use paper_code_main settings
    temperature = 0.2
    top_p = 0.95
    max_tokens = 512
    stop_words = ["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"]
    pass_k = 1

    print(f"\n[HumanEval] Evaluating with paper_code_main settings (n={n_generations}, temp={temperature}, top_p={top_p})...")
    os.makedirs(out_dir, exist_ok=True)

    # Ensure LoRA config exists
    enable_lora = lora_dir is not None
    max_r = 16

    if lora_dir:
        ensure_lora_has_config(lora_dir, config_src)
        cfg = load_adapter_config(lora_dir)
        max_r = max(max_r, infer_max_lora_rank(cfg))

    # Load problems
    problems = read_problems()
    task_ids = sorted(problems.keys())

    if max_samples is not None and max_samples > 0:
        task_ids = task_ids[:max_samples]

    prompts = [problems[tid]["prompt"] for tid in task_ids]
    print(f"[HumanEval] Loaded {len(prompts)} tasks.")

    # Init vLLM
    llm = LLM(
        model=base_model_id,
        dtype="float16",
        max_model_len=max_model_len,
        enable_lora=enable_lora,
        max_lora_rank=max_r if enable_lora else 16,
        seed=seed,
    )

    # Evaluate
    adapter_name = "lora" if lora_dir else "base"
    result = eval_one_adapter_multisample(
        llm=llm,
        prompts=prompts,
        task_ids=task_ids,
        n_generations=n_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_words=stop_words,
        lora_dir=lora_dir,
        adapter_name=adapter_name,
        out_dir=out_dir,
        eval_n_workers=4,
        eval_timeout=3.0,
        pass_k=pass_k,
    )

    # Write config
    config_path = write_eval_config(
        out_dir=out_dir,
        task="humaneval",
        n_generations=n_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        total_tasks=len(task_ids),
        metric_name=f"pass@{pass_k}",
        score=result.get(f"pass@{pass_k}", 0.0),
        extra_meta={
            "base_model_id": base_model_id,
            "lora_dir": lora_dir,
            "seed": seed,
            "profile": "paper_code_main",
        },
    )
    result["config_path"] = config_path

    print(f"[HumanEval] pass@{pass_k}={result.get(f'pass@{pass_k}', 'N/A')}")

    # Cleanup
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return result


def write_summary(
    out_dir: str,
    edit_meta: Dict[str, Any],
    baseline_gsm8k: Optional[Dict[str, Any]],
    edited_gsm8k: Dict[str, Any],
    baseline_humaneval: Optional[Dict[str, Any]],
    edited_humaneval: Dict[str, Any],
) -> str:
    """Write summary.json and summary.csv with all results."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "edit_mode": "smooth_abs",
        "hyperparams": edit_meta.get("meta", edit_meta),
        "gsm8k": {
            "profile": "paper_math",
            "metric": "strict_match_accuracy",
        },
        "humaneval": {
            "profile": "paper_code_main",
            "metric": "pass@1",
        },
        "results": {},
    }

    # GSM8K results
    summary["results"]["gsm8k_edited"] = {
        "acc": edited_gsm8k.get("acc"),
        "correct": edited_gsm8k.get("correct"),
        "total": edited_gsm8k.get("total"),
    }
    if baseline_gsm8k:
        summary["results"]["gsm8k_baseline"] = {
            "acc": baseline_gsm8k.get("acc"),
            "correct": baseline_gsm8k.get("correct"),
            "total": baseline_gsm8k.get("total"),
        }
        if baseline_gsm8k.get("acc") and edited_gsm8k.get("acc"):
            summary["results"]["gsm8k_delta"] = edited_gsm8k["acc"] - baseline_gsm8k["acc"]

    # HumanEval results
    summary["results"]["humaneval_edited"] = {
        "pass@1": edited_humaneval.get("pass@1"),
        "num_tasks": edited_humaneval.get("num_tasks"),
        "n_generations": edited_humaneval.get("n_generations"),
    }
    if baseline_humaneval:
        summary["results"]["humaneval_baseline"] = {
            "pass@1": baseline_humaneval.get("pass@1"),
            "num_tasks": baseline_humaneval.get("num_tasks"),
            "n_generations": baseline_humaneval.get("n_generations"),
        }
        if baseline_humaneval.get("pass@1") and edited_humaneval.get("pass@1"):
            summary["results"]["humaneval_delta"] = edited_humaneval["pass@1"] - baseline_humaneval["pass@1"]

    # Write JSON
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Write CSV for easy viewing
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "baseline", "edited", "delta"])

        # GSM8K row
        gsm8k_base = baseline_gsm8k.get("acc") if baseline_gsm8k else None
        gsm8k_edit = edited_gsm8k.get("acc")
        gsm8k_delta = summary["results"].get("gsm8k_delta")
        writer.writerow([
            "gsm8k_acc",
            f"{gsm8k_base:.4f}" if gsm8k_base else "N/A",
            f"{gsm8k_edit:.4f}" if gsm8k_edit else "N/A",
            f"{gsm8k_delta:+.4f}" if gsm8k_delta else "N/A",
        ])

        # HumanEval row
        he_base = baseline_humaneval.get("pass@1") if baseline_humaneval else None
        he_edit = edited_humaneval.get("pass@1")
        he_delta = summary["results"].get("humaneval_delta")
        writer.writerow([
            "humaneval_pass@1",
            f"{he_base:.4f}" if he_base else "N/A",
            f"{he_edit:.4f}" if he_edit else "N/A",
            f"{he_delta:+.4f}" if he_delta else "N/A",
        ])

    print(f"\n[Summary] Written to: {summary_path}")
    print(f"[Summary] CSV: {csv_path}")

    return summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Run smooth_abs experiment: edit + evaluate on GSM8K and HumanEval"
    )

    # Required args
    parser.add_argument("--base_model", type=str, required=True,
                        help="HuggingFace model ID for base model")
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="Path or HF ID for LoRA adapter")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for experiment results")

    # smooth_abs hyperparams
    parser.add_argument("--amp_factor", type=float, default=1.25,
                        help="Amplification factor for high-sensitivity dims")
    parser.add_argument("--sup_factor", type=float, default=0.8,
                        help="Suppression factor for low-sensitivity dims")
    parser.add_argument("--mid_factor", type=float, default=1.0,
                        help="Scale factor for middle dims")
    parser.add_argument("--smooth_temperature", type=float, default=0.35,
                        help="Temperature for sigmoid gate (larger=smoother)")
    parser.add_argument("--smooth_center_q", type=float, default=0.5,
                        help="Center quantile for sigmoid (0.5=median)")
    parser.add_argument("--smooth_align_mid", action="store_true", default=True,
                        help="Align gate(center)=mid_factor")
    parser.add_argument("--no_smooth_align_mid", action="store_true",
                        help="Disable smooth_align_mid")

    # Edit params
    parser.add_argument("--target_modules", type=str, nargs="+", default=["down_proj", "o_proj"],
                        help="Module names to edit")
    parser.add_argument("--layer_min", type=int, default=0,
                        help="Minimum layer index to edit")
    parser.add_argument("--layer_max", type=int, default=10**9,
                        help="Maximum layer index to edit")
    parser.add_argument("--calib_samples", type=int, default=256,
                        help="Number of calibration samples")
    parser.add_argument("--calib_batch_size", type=int, default=2,
                        help="Calibration batch size")
    parser.add_argument("--grad_norm", type=str, choices=["none", "mean_abs", "l2"],
                        default="mean_abs", help="Gradient normalization method")
    parser.add_argument("--preserve_energy", type=str, choices=["none", "l1", "l2"],
                        default="l1", help="Energy preservation method")
    parser.add_argument("--sigma_clip_min", type=float, default=0.0,
                        help="Minimum sigma value after editing")

    # Eval params
    parser.add_argument("--do_baseline_eval", action="store_true",
                        help="Also evaluate the baseline (unedited) adapter")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for both GSM8K and HumanEval (for smoke tests)")
    parser.add_argument("--humaneval_n_generations", type=int, default=50,
                        help="Number of generations per HumanEval problem (paper: 50)")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max model context length for vLLM")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for HF downloads")
    parser.add_argument("--skip_edit", action="store_true",
                        help="Skip editing, only run eval (adapter must exist)")
    parser.add_argument("--skip_gsm8k", action="store_true",
                        help="Skip GSM8K evaluation")
    parser.add_argument("--skip_humaneval", action="store_true",
                        help="Skip HumanEval evaluation")

    args = parser.parse_args()

    # Handle smooth_align_mid
    smooth_align_mid = args.smooth_align_mid and not args.no_smooth_align_mid

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Define paths
    edited_adapter_dir = os.path.join(args.out_dir, "adapter_smooth_abs")
    gsm8k_edited_dir = os.path.join(args.out_dir, "gsm8k_edited")
    gsm8k_baseline_dir = os.path.join(args.out_dir, "gsm8k_baseline")
    humaneval_edited_dir = os.path.join(args.out_dir, "humaneval_edited")
    humaneval_baseline_dir = os.path.join(args.out_dir, "humaneval_baseline")

    print("=" * 70)
    print("smooth_abs Experiment")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"LoRA: {args.lora_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Hyperparams: amp={args.amp_factor}, sup={args.sup_factor}, temp={args.smooth_temperature}")
    print(f"Do baseline eval: {args.do_baseline_eval}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print("=" * 70)

    # Step 1: Run smooth_abs edit
    edit_meta = {}
    if not args.skip_edit:
        print("\n" + "=" * 70)
        print("Phase 1: smooth_abs Spectral Editing")
        print("=" * 70)

        edit_meta = run_smooth_abs_edit(
            base_model_id=args.base_model,
            lora_dir=args.lora_dir,
            out_adapter_dir=edited_adapter_dir,
            target_modules=args.target_modules,
            layer_min=args.layer_min,
            layer_max=args.layer_max,
            calib_samples=args.calib_samples,
            calib_batch_size=args.calib_batch_size,
            amp_factor=args.amp_factor,
            sup_factor=args.sup_factor,
            mid_factor=args.mid_factor,
            smooth_temperature=args.smooth_temperature,
            smooth_center_q=args.smooth_center_q,
            smooth_align_mid=smooth_align_mid,
            grad_norm=args.grad_norm,
            preserve_energy=args.preserve_energy,
            sigma_clip_min=args.sigma_clip_min,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    else:
        print("\n[Skip] Skipping edit phase (--skip_edit)")
        if os.path.exists(os.path.join(edited_adapter_dir, "spectral_edit_meta.json")):
            with open(os.path.join(edited_adapter_dir, "spectral_edit_meta.json")) as f:
                edit_meta = json.load(f)

    # Step 2: GSM8K evaluation
    baseline_gsm8k = None
    edited_gsm8k = None

    if not args.skip_gsm8k:
        print("\n" + "=" * 70)
        print("Phase 2: GSM8K Evaluation (paper_math profile)")
        print("=" * 70)

        # Baseline eval
        if args.do_baseline_eval:
            print("\n[GSM8K] Evaluating baseline...")
            baseline_gsm8k = run_gsm8k_eval(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                out_dir=gsm8k_baseline_dir,
                max_samples=args.max_samples,
                seed=args.seed,
                max_model_len=args.max_model_len,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Edited eval
        print("\n[GSM8K] Evaluating edited adapter...")
        edited_gsm8k = run_gsm8k_eval(
            base_model_id=args.base_model,
            lora_dir=edited_adapter_dir,
            out_dir=gsm8k_edited_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            max_model_len=args.max_model_len,
        )
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[Skip] Skipping GSM8K evaluation (--skip_gsm8k)")

    # Step 3: HumanEval evaluation
    baseline_humaneval = None
    edited_humaneval = None

    if not args.skip_humaneval:
        print("\n" + "=" * 70)
        print("Phase 3: HumanEval Evaluation (paper_code_main settings)")
        print("=" * 70)

        # Baseline eval
        if args.do_baseline_eval:
            print("\n[HumanEval] Evaluating baseline...")
            baseline_humaneval = run_humaneval_eval(
                base_model_id=args.base_model,
                lora_dir=args.lora_dir,
                out_dir=humaneval_baseline_dir,
                max_samples=args.max_samples,
                n_generations=args.humaneval_n_generations,
                seed=args.seed,
                max_model_len=args.max_model_len,
            )
            gc.collect()
            torch.cuda.empty_cache()

        # Edited eval
        print("\n[HumanEval] Evaluating edited adapter...")
        edited_humaneval = run_humaneval_eval(
            base_model_id=args.base_model,
            lora_dir=edited_adapter_dir,
            out_dir=humaneval_edited_dir,
            max_samples=args.max_samples,
            n_generations=args.humaneval_n_generations,
            seed=args.seed,
            max_model_len=args.max_model_len,
            config_src=args.lora_dir,
        )
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n[Skip] Skipping HumanEval evaluation (--skip_humaneval)")

    # Step 4: Write summary
    print("\n" + "=" * 70)
    print("Phase 4: Writing Summary")
    print("=" * 70)

    if edited_gsm8k or edited_humaneval:
        write_summary(
            out_dir=args.out_dir,
            edit_meta=edit_meta,
            baseline_gsm8k=baseline_gsm8k,
            edited_gsm8k=edited_gsm8k or {},
            baseline_humaneval=baseline_humaneval,
            edited_humaneval=edited_humaneval or {},
        )

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {args.out_dir}")
    print(f"  - Edited adapter: adapter_smooth_abs/")
    if not args.skip_gsm8k:
        print(f"  - GSM8K results: gsm8k_edited/ (and gsm8k_baseline/ if --do_baseline_eval)")
    if not args.skip_humaneval:
        print(f"  - HumanEval results: humaneval_edited/ (and humaneval_baseline/ if --do_baseline_eval)")
    print(f"  - Summary: summary.json, summary.csv")


if __name__ == "__main__":
    main()
