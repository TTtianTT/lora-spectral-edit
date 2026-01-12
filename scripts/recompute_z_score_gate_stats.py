#!/usr/bin/env python3
"""
Recompute z_score gate stats for existing runs without running evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from lora_spectral_edit.io import (
    ensure_local_lora_dir,
    load_adapter_config,
    load_lora_state_dict,
    parse_lora_ab_key,
    layer_idx_from_module_prefix,
    get_scaling_for_module,
)
from lora_spectral_edit.svd import lowrank_svd_from_ba
from lora_spectral_edit.hooks import ModuleSpec, HOOK_CTX, register_sigma_hooks, remove_hooks
from lora_spectral_edit.edit import EditConfig, apply_spectral_edit


@dataclass
class RunConfig:
    task: str
    lora_repo_id: str
    base_model: str
    lora_path: str
    seed: int
    calib_samples: int
    calib_batch_size: int
    target_modules: List[str]
    layer_min: int
    layer_max: int
    amp_factor: float
    sup_factor: float
    mid_factor: float
    core_frac: float
    noise_frac: float
    z_high: float
    z_low: float
    z_tau: float
    z_fallback_std: float
    grad_norm: str
    preserve_energy: str
    sigma_clip_min: float
    out_dir: Path


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_calib_batch(tokenizer, examples: List[dict], add_eos: bool = True):
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

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )
    attn_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)
    return input_ids, attn_mask, labels


_TOKENIZER_CACHE: Dict[str, Any] = {}
_CALIB_CACHE: Optional[List[dict]] = None


def infer_lora_path(cfg: Dict[str, Any]) -> Optional[str]:
    lora_path = cfg.get("lora_local_path") or cfg.get("lora_path") or cfg.get("lora_repo_id")
    return lora_path


def load_run_config(cfg_path: Path) -> Optional[RunConfig]:
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if cfg.get("edit_mode") != "z_score":
        return None

    lora_path = infer_lora_path(cfg)
    if not lora_path:
        return None

    return RunConfig(
        task=cfg.get("task", "unknown"),
        lora_repo_id=cfg.get("lora_repo_id") or cfg.get("lora_path") or "unknown",
        base_model=cfg.get("base_model", "meta-llama/Llama-2-7b-hf"),
        lora_path=lora_path,
        seed=int(cfg.get("seed", 0)),
        calib_samples=int(cfg.get("calib_samples", 32)),
        calib_batch_size=int(cfg.get("calib_batch_size", 2)),
        target_modules=list(cfg.get("target_modules", ["down_proj", "o_proj"])),
        layer_min=int(cfg.get("layer_min", 0)),
        layer_max=int(cfg.get("layer_max", 10**9)),
        amp_factor=float(cfg.get("amp_factor", 1.25)),
        sup_factor=float(cfg.get("sup_factor", 0.8)),
        mid_factor=float(cfg.get("mid_factor", 1.0)),
        core_frac=float(cfg.get("core_frac", 0.2)),
        noise_frac=float(cfg.get("noise_frac", 0.2)),
        z_high=float(cfg.get("z_high", 1.0)),
        z_low=float(cfg.get("z_low", -0.5)),
        z_tau=float(cfg.get("z_tau", 0.2)),
        z_fallback_std=float(cfg.get("z_fallback_std", 1e-6)),
        grad_norm=str(cfg.get("grad_norm", "mean_abs")),
        preserve_energy=str(cfg.get("preserve_energy", "l1")),
        sigma_clip_min=float(cfg.get("sigma_clip_min", 0.0)),
        out_dir=Path(cfg.get("out_dir", cfg_path.parent)),
    )


def build_specs(
    model,
    sd: Dict[str, torch.Tensor],
    adapter_cfg: dict,
    target_modules: List[str],
    layer_min: int,
    layer_max: int,
    device: str,
) -> Dict[str, ModuleSpec]:
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
        raise RuntimeError("No matching LoRA (A,B) pairs found for target_modules/layer range.")

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

    return specs


def get_tokenizer(model_id: str, offline: bool):
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        local_files_only=offline,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER_CACHE[model_id] = tok
    return tok


def get_calib_examples(n: int, offline: bool) -> List[dict]:
    global _CALIB_CACHE
    if _CALIB_CACHE is None:
        ds = load_dataset(
            "gsm8k",
            "main",
            download_mode="reuse_dataset_if_exists",
            local_files_only=offline,
        )
        _CALIB_CACHE = list(ds["train"])
    return _CALIB_CACHE[:n]


def run_probe(run_cfg: RunConfig, device: str, offline: bool) -> List[Dict[str, Any]]:
    set_seed(run_cfg.seed)

    lora_dir = ensure_local_lora_dir(run_cfg.lora_path)
    adapter_cfg = load_adapter_config(lora_dir)
    sd, _ = load_lora_state_dict(lora_dir)

    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = get_tokenizer(run_cfg.base_model, offline=offline)

    base = AutoModelForCausalLM.from_pretrained(
        run_cfg.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        local_files_only=offline,
    ).to(device)

    model = PeftModel.from_pretrained(base, lora_dir, is_trainable=True).to(device)
    model.eval()
    model.config.use_cache = False

    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    specs = build_specs(
        model=model,
        sd=sd,
        adapter_cfg=adapter_cfg,
        target_modules=run_cfg.target_modules,
        layer_min=run_cfg.layer_min,
        layer_max=run_cfg.layer_max,
        device=device,
    )

    handles = register_sigma_hooks(specs)
    calib_examples = get_calib_examples(run_cfg.calib_samples, offline=offline)

    bs = max(1, run_cfg.calib_batch_size)
    HOOK_CTX.reset()
    ncal = len(calib_examples)
    for i in range(0, ncal, bs):
        batch_ex = calib_examples[i:i + bs]
        input_ids, attn_mask, labels = make_calib_batch(tok, batch_ex, add_eos=True)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        HOOK_CTX.attn_mask = attn_mask
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        model.zero_grad(set_to_none=True)

    remove_hooks(handles)
    HOOK_CTX.attn_mask = None

    edit_config = EditConfig(
        mode="z_score",
        core_frac=run_cfg.core_frac,
        noise_frac=run_cfg.noise_frac,
        amp_factor=run_cfg.amp_factor,
        sup_factor=run_cfg.sup_factor,
        mid_factor=run_cfg.mid_factor,
        smooth_temperature=0.35,
        smooth_center_q=0.5,
        smooth_align_mid=True,
        z_high=run_cfg.z_high,
        z_low=run_cfg.z_low,
        z_tau=run_cfg.z_tau,
        z_fallback_std=run_cfg.z_fallback_std,
        grad_norm=run_cfg.grad_norm,
        preserve_energy=run_cfg.preserve_energy,
        sigma_clip_min=run_cfg.sigma_clip_min,
    )

    records: List[Dict[str, Any]] = []
    for prefix, spec in specs.items():
        sigma0 = spec.sigma0.clone()
        g = HOOK_CTX.gsum.get(prefix)
        if g is None:
            continue
        _, stats = apply_spectral_edit(sigma0, g, edit_config)
        if stats.get("mode") != "z_score":
            continue
        layer = layer_idx_from_module_prefix(prefix)
        module = prefix.split(".")[-1]
        record = {
            "task": run_cfg.task,
            "lora_repo_id": run_cfg.lora_repo_id,
            "seed": run_cfg.seed,
            "calib_samples": run_cfg.calib_samples,
            "layer": layer,
            "module": module,
            "r": stats.get("r"),
            "mu": stats.get("mu"),
            "std": stats.get("std"),
            "z_high": stats.get("z_high"),
            "z_low": stats.get("z_low"),
            "tau": stats.get("tau"),
            "k_core_eff": stats.get("k_core_eff"),
            "k_noise_eff": stats.get("k_noise_eff"),
            "frac_core": stats.get("frac_core"),
            "frac_noise": stats.get("frac_noise"),
            "fallback": stats.get("fallback"),
            "mode": "z_score",
        }
        records.append(record)

    # Cleanup
    try:
        model.to("cpu")
        base.to("cpu")
    except Exception:
        pass
    del model, base
    torch.cuda.empty_cache()

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute z_score gate stats without evaluation.")
    parser.add_argument("--runs-root", type=str, default="_runs", help="Runs root directory.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (cpu only).")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (use cached files only).")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    config_paths = sorted(runs_root.rglob("config.json"))

    run_configs: List[Tuple[Path, RunConfig]] = []
    seen = set()
    for cfg_path in config_paths:
        run_cfg = load_run_config(cfg_path)
        if run_cfg is None:
            continue
        key = (run_cfg.task, run_cfg.lora_repo_id, run_cfg.calib_samples, run_cfg.seed)
        if key in seen:
            continue
        seen.add(key)
        run_configs.append((cfg_path, run_cfg))

    if args.max_runs:
        run_configs = run_configs[: args.max_runs]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("[Error] CUDA requested but not available.")
        return

    if args.workers > 1 and device == "cuda":
        print("[Warn] workers>1 with CUDA is not supported; running sequentially.")
        args.workers = 1

    processed = 0
    skipped = 0
    for _, run_cfg in run_configs:
        out_dir = run_cfg.out_dir
        stats_path = out_dir / "z_score_gate_stats.json"
        if stats_path.exists():
            skipped += 1
            continue
        try:
            records = run_probe(run_cfg, device=device, offline=args.offline)
        except Exception as exc:
            skipped += 1
            print(f"[Skip] {out_dir} error: {exc}")
            continue
        stats_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        processed += 1
        print(f"[OK] wrote {stats_path} ({len(records)} records)")

    print(f"[Summary] runs found: {len(run_configs)}")
    print(f"[Summary] processed: {processed}")
    print(f"[Summary] skipped: {skipped}")


if __name__ == "__main__":
    main()
