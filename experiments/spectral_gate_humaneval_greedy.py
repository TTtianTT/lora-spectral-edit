#!/usr/bin/env python
"""
Spectral gate HumanEval evaluation with greedy decoding.

This script combines spectral gating (soft gating based on sensitivity)
with HumanEval evaluation using greedy decoding.

Usage:
    python experiments/spectral_gate_humaneval_greedy.py \
        --base_model_id meta-llama/Llama-2-7b-hf \
        --lora_path <path_to_lora> \
        --amp_factor 1.25 \
        --sup_factor 0.8 \
        --soft_temperature 0.35 \
        --soft_pivot_mode median \
        --seed 42 \
        --out_dir <output_dir> \
        --out_json <output_metrics.json>
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import tempfile
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


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_calib_batch(tokenizer, examples: List[dict], add_eos: bool = True):
    """
    Build teacher-forcing inputs for calibration.

    Returns input_ids, attention_mask, labels tensors.
    Labels have -100 for prompt tokens (not included in loss).
    """
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


def apply_soft_spectral_gate(
    sigma0: torch.Tensor,
    g_abs: torch.Tensor,
    amp_factor: float,
    sup_factor: float,
    soft_temperature: float,
    soft_pivot_mode: str,
    preserve_energy: str = "l1",
) -> torch.Tensor:
    """
    Apply soft spectral gating based on sensitivity.

    Uses a sigmoid gate:
        gate_i = sup + (amp - sup) * sigmoid((x_i - pivot) / tau)

    Where x_i is the normalized absolute gradient and tau is derived
    from soft_temperature.

    Args:
        sigma0: Original singular values, shape [r]
        g_abs: Absolute gradient sensitivity scores, shape [r]
        amp_factor: Amplification factor for high-sensitivity dims
        sup_factor: Suppression factor for low-sensitivity dims
        soft_temperature: Temperature for soft gating (larger = smoother)
        soft_pivot_mode: How to determine pivot point ("median", "mean", "max", "min")
        preserve_energy: Energy preservation method ("none", "l1", "l2")

    Returns:
        Modified singular values, shape [r]
    """
    x = g_abs.to(dtype=torch.float32)
    r = int(x.numel())

    # Handle degenerate case
    if (x.max() - x.min()).abs().item() < 1e-12:
        return sigma0.clone()

    # Normalize x to [0, 1] range for numerical stability
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min).clamp_min(1e-8)

    # Determine pivot based on mode
    if soft_pivot_mode == "median":
        pivot = torch.median(x_norm)
    elif soft_pivot_mode == "mean":
        pivot = x_norm.mean()
    elif soft_pivot_mode == "max":
        pivot = x_norm.max()
    elif soft_pivot_mode == "min":
        pivot = x_norm.min()
    else:
        pivot = torch.median(x_norm)  # default to median

    # Compute temperature as fraction of the range
    # soft_temperature scales the sharpness: smaller = sharper, larger = smoother
    tau = soft_temperature

    # Compute gate
    sup_t = torch.tensor(float(sup_factor), device=x.device, dtype=torch.float32)
    amp_t = torch.tensor(float(amp_factor), device=x.device, dtype=torch.float32)
    gate = sup_t + (amp_t - sup_t) * torch.sigmoid((x_norm - pivot) / tau)
    gate = gate.to(dtype=sigma0.dtype)

    sigma_new = sigma0 * gate

    # Preserve energy
    if preserve_energy == "l1":
        s0 = sigma0.sum().clamp_min(1e-8)
        s1 = sigma_new.sum().clamp_min(1e-8)
        sigma_new = sigma_new * (s0 / s1)
    elif preserve_energy == "l2":
        s0 = torch.linalg.norm(sigma0).clamp_min(1e-8)
        s1 = torch.linalg.norm(sigma_new).clamp_min(1e-8)
        sigma_new = sigma_new * (s0 / s1)

    return sigma_new


def run_calibration(
    model,
    tokenizer,
    specs: Dict[str, ModuleSpec],
    calib_samples: int,
    calib_batch_size: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Run calibration to compute sensitivity scores (g_sigma).

    Returns:
        Dictionary mapping module prefix to g_sigma tensor.
    """
    # Register hooks
    handles = register_sigma_hooks(specs)

    # Load calibration data
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
        input_ids, attn_mask, labels = make_calib_batch(tokenizer, batch_ex, add_eos=True)
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
    print(f"[Calib] total active tokens: {HOOK_CTX.total_active_tokens}")

    if not HOOK_CTX.gsum:
        raise RuntimeError("No gradients accumulated. Hooks may not have fired.")

    return HOOK_CTX.gsum.copy()


def apply_spectral_gate_and_save(
    sd: Dict[str, torch.Tensor],
    pairs: Dict[str, dict],
    specs: Dict[str, ModuleSpec],
    gsum: Dict[str, torch.Tensor],
    amp_factor: float,
    sup_factor: float,
    soft_temperature: float,
    soft_pivot_mode: str,
    out_dir: str,
    fmt: str,
    device: str,
):
    """
    Apply spectral gate editing and save to output directory.
    """
    for prefix, spec in specs.items():
        sigma0 = spec.sigma0.clone()
        g = gsum.get(prefix, None)
        if g is None:
            continue

        g_abs = g.abs()
        # Normalize by mean absolute value
        g_abs = g_abs / g_abs.mean().clamp_min(1e-8)

        sigma_new = apply_soft_spectral_gate(
            sigma0=sigma0,
            g_abs=g_abs,
            amp_factor=amp_factor,
            sup_factor=sup_factor,
            soft_temperature=soft_temperature,
            soft_pivot_mode=soft_pivot_mode,
            preserve_energy="l1",
        )

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

    # Save edited adapter
    save_lora_state_dict(out_dir, sd, fmt)
    print(f"[SpectralGate] Saved edited adapter to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Spectral gate HumanEval evaluation with greedy decoding"
    )

    # Model and LoRA
    parser.add_argument("--base_model_id", type=str, required=True,
                        help="Base model HuggingFace ID")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA adapter")

    # Spectral gate hyperparameters
    parser.add_argument("--amp_factor", type=float, default=1.25,
                        help="Amplification factor for high-sensitivity dims")
    parser.add_argument("--sup_factor", type=float, default=0.8,
                        help="Suppression factor for low-sensitivity dims")
    parser.add_argument("--soft_temperature", type=float, default=0.35,
                        help="Temperature for soft gating (larger = smoother)")
    parser.add_argument("--soft_pivot_mode", type=str, default="median",
                        choices=["median", "mean", "max", "min"],
                        help="Pivot mode for soft gating")

    # Calibration
    parser.add_argument("--calib_samples", type=int, default=256,
                        help="Number of calibration samples")
    parser.add_argument("--calib_batch_size", type=int, default=2,
                        help="Calibration batch size")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["down_proj", "o_proj"],
                        help="Target modules for spectral editing")

    # Evaluation
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max HumanEval samples (-1 for all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate")

    # Output
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for intermediate files")
    parser.add_argument("--out_json", type=str, required=True,
                        help="Output JSON file for metrics")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This tool requires a CUDA GPU for gradient computation.")

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Save config with standardized fields for collect_results.py
    config = vars(args).copy()
    config["task"] = "humaneval_full"
    config["edit_mode"] = "spectral_gate"
    # Extract lora_repo_id from lora_path
    lora_path = args.lora_path
    if "models--" in lora_path:
        parts = lora_path.split("models--")[-1].split("/")[0]
        config["lora_repo_id"] = parts.replace("--", "/")
    else:
        config["lora_repo_id"] = os.path.basename(lora_path.rstrip("/"))

    print(f"[SpectralGate] base_model_id={args.base_model_id}")
    print(f"[SpectralGate] lora_path={args.lora_path}")
    print(f"[SpectralGate] amp_factor={args.amp_factor}")
    print(f"[SpectralGate] sup_factor={args.sup_factor}")
    print(f"[SpectralGate] soft_temperature={args.soft_temperature}")
    print(f"[SpectralGate] soft_pivot_mode={args.soft_pivot_mode}")
    print(f"[SpectralGate] seed={args.seed}")

    # ========== PHASE 1: Load model and prepare for calibration ==========
    print("\n[Phase 1] Loading model and LoRA adapter...")

    lora_dir = ensure_local_lora_dir(args.lora_path)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, fmt = load_lora_state_dict(lora_dir)

    # Identify LoRA A/B pairs
    pairs: Dict[str, dict] = {}
    target_modules_set = set(args.target_modules)

    for k, t in sd.items():
        parsed = parse_lora_ab_key(k)
        if not parsed:
            continue
        prefix, which, adapter = parsed

        suffix = prefix.split(".")[-1]
        if suffix not in target_modules_set:
            continue

        pairs.setdefault(prefix, {})
        pairs[prefix][which] = (k, t, adapter)

    selected_prefixes = [p for p in pairs.keys() if "A" in pairs[p] and "B" in pairs[p]]
    if not selected_prefixes:
        raise RuntimeError("No matching LoRA (A,B) pairs found for given target_modules.")

    print(f"[Info] Selected LoRA modules: {len(selected_prefixes)}")

    # Load model
    tok = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
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

    print(f"[Info] Built SVD specs for {len(specs)} modules.")

    # ========== PHASE 2: Run calibration ==========
    print("\n[Phase 2] Running calibration...")

    gsum = run_calibration(
        model=model,
        tokenizer=tok,
        specs=specs,
        calib_samples=args.calib_samples,
        calib_batch_size=args.calib_batch_size,
        device=device,
    )

    # Free model memory before vLLM
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    # ========== PHASE 3: Apply spectral gate and save ==========
    print("\n[Phase 3] Applying spectral gate...")

    # Create temp directory for edited adapter
    tmp_adapter_dir = os.path.join(args.out_dir, "tmp_edited_adapter")
    if os.path.exists(tmp_adapter_dir):
        shutil.rmtree(tmp_adapter_dir)
    shutil.copytree(lora_dir, tmp_adapter_dir)

    apply_spectral_gate_and_save(
        sd=sd,
        pairs=pairs,
        specs=specs,
        gsum=gsum,
        amp_factor=args.amp_factor,
        sup_factor=args.sup_factor,
        soft_temperature=args.soft_temperature,
        soft_pivot_mode=args.soft_pivot_mode,
        out_dir=tmp_adapter_dir,
        fmt=fmt,
        device=device,
    )

    # ========== PHASE 4: Run HumanEval evaluation with vLLM ==========
    print("\n[Phase 4] Running HumanEval evaluation...")

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from human_eval.data import read_problems, write_jsonl
    except ImportError as e:
        raise RuntimeError(
            f"Missing required packages for evaluation: {e}\n"
            "Please install: pip install vllm && pip install -e human-eval"
        )

    # Load adapter config to get rank
    edited_cfg = load_adapter_config(tmp_adapter_dir)
    r = int(edited_cfg.get("r", 0) or edited_cfg.get("rank", 16))

    # Load HumanEval problems
    print("[Data] Loading HumanEval problems...")
    problems = read_problems()
    task_ids = sorted(problems.keys())

    if args.max_samples > 0:
        task_ids = task_ids[:args.max_samples]

    prompts = [problems[tid]["prompt"] for tid in task_ids]
    print(f"[Data] Loaded {len(prompts)} HumanEval tasks.")

    # Init vLLM engine
    print(f"[vLLM] Initializing LLM engine (Base: {args.base_model_id})...")
    llm = LLM(
        model=args.base_model_id,
        dtype="float16",
        max_model_len=4096,
        enable_lora=True,
        max_lora_rank=max(r, 256),  # Support high rank adapters
        seed=args.seed,
    )

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint"],
    )

    # Generate with edited adapter
    lora_req = LoRARequest("edited", 1, tmp_adapter_dir)
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    samples = []
    for tid, out in zip(task_ids, outputs):
        generated_text = out.outputs[0].text
        samples.append({
            "task_id": tid,
            "completion": generated_text
        })

    # Write samples to JSONL
    samples_path = os.path.join(args.out_dir, "samples_edited.jsonl")
    try:
        write_jsonl(samples_path, samples)
    except Exception:
        with open(samples_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[Eval] Wrote samples to: {samples_path}")

    # Run evaluation
    print("[Eval] Running evaluate_functional_correctness...")
    try:
        from human_eval.evaluation import evaluate_functional_correctness
        raw_res = evaluate_functional_correctness(samples_path, k=[1], n_workers=4, timeout=3.0)
        results_path = samples_path + "_results.jsonl"
    except Exception as e:
        print(f"[Warn] Evaluation failed: {e}")
        raw_res = {"pass@1": 0.0}
        results_path = None

    # Compute pass@1 from results
    if results_path and os.path.exists(results_path):
        per_task_total = {}
        per_task_passed = {}
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r_data = json.loads(line)
                    tid = r_data.get("task_id")
                    if tid:
                        per_task_total[tid] = per_task_total.get(tid, 0) + 1
                        if r_data.get("passed", False):
                            per_task_passed[tid] = per_task_passed.get(tid, 0) + 1

        vals = []
        for tid in per_task_total:
            n = per_task_total[tid]
            c = per_task_passed.get(tid, 0)
            vals.append(c / max(1, n))

        pass_at_1 = sum(vals) / len(vals) if vals else 0.0
        correct = sum(1 for v in vals if v > 0)
        total = len(vals)
    else:
        pass_at_1 = raw_res.get("pass@1", 0.0)
        correct = 0
        total = len(task_ids)

    print(f"[Result] pass@1={pass_at_1:.4f} ({correct}/{total})")

    # Clean up temp adapter
    shutil.rmtree(tmp_adapter_dir, ignore_errors=True)

    # Write metrics
    metrics = {
        "meta": {
            "base_model_id": args.base_model_id,
            "lora_path": args.lora_path,
            "lora_repo_id": config["lora_repo_id"],
            "amp_factor": args.amp_factor,
            "sup_factor": args.sup_factor,
            "soft_temperature": args.soft_temperature,
            "soft_pivot_mode": args.soft_pivot_mode,
            "seed": args.seed,
            "edit_mode": "spectral_gate",
            "task": "humaneval_full",
        },
        "edited": {
            "pass@1": pass_at_1,
            "correct": correct,
            "total": total,
            "num_tasks": total,
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SpectralGate] Wrote metrics to: {args.out_json}")


if __name__ == "__main__":
    main()
