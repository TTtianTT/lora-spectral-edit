#!/usr/bin/env python
"""
Generate comprehensive sweep configs for:
- BOTH tasks: gsm8k_full (Math) and humaneval_full (Code)
- ALL edit modes: abs_select, smooth_abs, double_smooth, z_score, robust_z, random_index, gd
- 6 LoRA adapters (3 math + 3 code)
- calib_samples: {32, 64, 128, 256}
- seeds: {42, 43, 44}
- Multiple code presets with preserve_energy sweep

Output: experiments/configs_sweep_allmodes_6lora.jsonl
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# =============================================================================
# Configuration
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
HF_MODELS_DIR = Path("/home/zailongtian/hf_models/hub")

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
TARGET_MODULES = ["down_proj", "o_proj"]
CALIB_BATCH_SIZE = 2

# Sweep axes
SEEDS = [42, 43, 44]
CALIB_SAMPLES_LIST = [32, 64, 128, 256]

# Edit modes (all supported by CLI)
EDIT_MODES = [
    "abs_select",
    "smooth_abs",
    "double_smooth",
    "z_score",
    "robust_z",
    "random_index",
    "gd",
]

# LoRA adapters
MATH_LORAS = [
    "LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128",
    "LoRA-TMLR-2024/metamath-lora-rank-256-alpha-512",
]

CODE_LORAS = [
    "LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128",
    "LoRA-TMLR-2024/magicoder-lora-rank-256-alpha-512",
]

# =============================================================================
# Hyperparameter Presets
# =============================================================================

# Math preset (single, known-good)
MATH_PRESET = {
    "name": "math_default",
    "core_frac": 0.20,
    "noise_frac": 0.20,
    "amp_factor": 1.25,
    "sup_factor": 0.80,
    "mid_factor": 1.00,
    "smooth_temperature": 0.35,
    "smooth_center_q": 0.5,
    "smooth_align_mid": True,
    "preserve_energy": "l1",
    # z_score params
    "z_high": 1.0,
    "z_low": -0.5,
    "z_tau": 0.2,
    "z_fallback_std": 1e-6,
    # robust_z params
    "robust_z_high": 1.0,
    "robust_z_low": -0.5,
    "robust_z_tau": 0.2,
    "robust_fallback_sigma": 1e-6,
}

# Code presets (multiple)
CODE_PRESETS = {
    "code_A_current": {
        "name": "code_A_current",
        "core_frac": 0.12,
        "noise_frac": 0.0,
        "amp_factor": 1.18,
        "sup_factor": 1.0,
        "mid_factor": 1.02,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l2",  # default for this preset
        # z_score params (same defaults)
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 1.7,
        "robust_z_low": -3.0,
        "robust_z_tau": 0.35,
        "robust_fallback_sigma": 1e-4,
    },
    "code_mathlike": {
        "name": "code_mathlike",
        "core_frac": 0.20,
        "noise_frac": 0.20,
        "amp_factor": 1.25,
        "sup_factor": 0.80,
        "mid_factor": 1.00,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l1",  # default for this preset (matches math)
        # z_score params
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 1.0,
        "robust_z_low": -0.5,
        "robust_z_tau": 0.2,
        "robust_fallback_sigma": 1e-6,
    },
    "code_round_light": {
        "name": "code_round_light",
        "core_frac": 0.10,
        "noise_frac": 0.0,
        "amp_factor": 1.15,
        "sup_factor": 1.00,
        "mid_factor": 1.00,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l2",
        # z_score params
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 2.0,
        "robust_z_low": -3.0,
        "robust_z_tau": 0.35,
        "robust_fallback_sigma": 1e-4,
    },
    "code_round_mid": {
        "name": "code_round_mid",
        "core_frac": 0.15,
        "noise_frac": 0.0,
        "amp_factor": 1.20,
        "sup_factor": 1.00,
        "mid_factor": 1.00,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l2",
        # z_score params
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 1.8,
        "robust_z_low": -3.0,
        "robust_z_tau": 0.30,
        "robust_fallback_sigma": 1e-4,
    },
    "code_round_softsup": {
        "name": "code_round_softsup",
        "core_frac": 0.15,
        "noise_frac": 0.05,
        "amp_factor": 1.20,
        "sup_factor": 0.95,
        "mid_factor": 1.00,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l2",
        # z_score params
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 1.8,
        "robust_z_low": -2.0,
        "robust_z_tau": 0.30,
        "robust_fallback_sigma": 1e-4,
    },
}

# preserve_energy sweep for code
CODE_PRESERVE_ENERGY_VALUES = ["l2", "l1", "none"]


# =============================================================================
# Helper Functions
# =============================================================================

def get_lora_local_path(lora_repo_id: str) -> str:
    """Find local path for a HuggingFace LoRA adapter."""
    # Convert repo_id to HF cache format: org--repo
    cache_name = f"models--{lora_repo_id.replace('/', '--')}"
    cache_dir = HF_MODELS_DIR / cache_name / "snapshots"

    if cache_dir.exists():
        # Get the latest snapshot
        snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])

    # Fallback: return the repo_id itself (will be downloaded)
    return lora_repo_id


def get_lora_slug(lora_repo_id: str) -> str:
    """Extract short name from lora_repo_id."""
    return lora_repo_id.split("/")[-1]


def make_config(
    task: str,
    lora_repo_id: str,
    edit_mode: str,
    seed: int,
    calib_samples: int,
    preset: Dict[str, Any],
    preserve_energy: str,
    run_root: str,
) -> Dict[str, Any]:
    """Build a single config dict."""
    lora_slug = get_lora_slug(lora_repo_id)
    lora_local_path = get_lora_local_path(lora_repo_id)

    # Build out_dir path
    # Format: run_root/task/lora_slug/preset_name/preserve_energy/edit_mode/calib_N/seed_S
    if task == "gsm8k_full":
        out_dir = (
            f"{run_root}/{task}/{lora_slug}/{edit_mode}/"
            f"calib_{calib_samples}/seed_{seed}"
        )
    else:
        # For code, include preset and preserve_energy in path
        preset_name = preset["name"]
        pe_tag = preserve_energy if preserve_energy != "none" else "none"
        out_dir = (
            f"{run_root}/{task}/{lora_slug}/{preset_name}/{pe_tag}/"
            f"{edit_mode}/calib_{calib_samples}/seed_{seed}"
        )

    eval_profile = "paper_math" if task == "gsm8k_full" else "paper_code_main"

    config = {
        "task": task,
        "base_model": BASE_MODEL,
        "lora_repo_id": lora_repo_id,
        "lora_local_path": lora_local_path,
        "edit_mode": edit_mode,
        "seed": seed,
        "out_dir": out_dir,
        "notes": "sweep_allmodes_6lora",
        "eval_profile": eval_profile,
        "target_modules": TARGET_MODULES,
        # Hyperparams from preset
        "core_frac": preset["core_frac"],
        "noise_frac": preset["noise_frac"],
        "amp_factor": preset["amp_factor"],
        "sup_factor": preset["sup_factor"],
        "mid_factor": preset["mid_factor"],
        "smooth_temperature": preset["smooth_temperature"],
        "smooth_center_q": preset["smooth_center_q"],
        "smooth_align_mid": preset["smooth_align_mid"],
        "preserve_energy": preserve_energy,
        # z_score params
        "z_high": preset["z_high"],
        "z_low": preset["z_low"],
        "z_tau": preset["z_tau"],
        "z_fallback_std": preset["z_fallback_std"],
        # robust_z params
        "robust_z_high": preset["robust_z_high"],
        "robust_z_low": preset["robust_z_low"],
        "robust_z_tau": preset["robust_z_tau"],
        "robust_fallback_sigma": preset["robust_fallback_sigma"],
        # Calibration
        "calib_samples": calib_samples,
        "calib_batch_size": CALIB_BATCH_SIZE,
        # Eval settings
        "eval_fewshot": 5,
        "eval_temperature": 0.0,
        "eval_max_tokens": 512,
        "eval_max_samples": -1,
        "keep_adapter": False,
        "run_root": run_root,
    }

    # Add preset name for code tasks
    if task == "humaneval_full":
        config["code_preset"] = preset["name"]

    return config


def make_baseline_config(
    task: str,
    lora_repo_id: str,
    run_root: str,
    preset: Dict[str, Any] = None,
    preserve_energy: str = None,
) -> Dict[str, Any]:
    """Build a baseline (eval-only) config."""
    if preset is None:
        preset = MATH_PRESET
    if preserve_energy is None:
        preserve_energy = preset.get("preserve_energy", "l1")

    lora_slug = get_lora_slug(lora_repo_id)
    lora_local_path = get_lora_local_path(lora_repo_id)

    if task == "gsm8k_full":
        out_dir = f"{run_root}/{task}/{lora_slug}/baseline"
    else:
        preset_name = preset["name"]
        pe_tag = preserve_energy if preserve_energy != "none" else "none"
        out_dir = f"{run_root}/{task}/{lora_slug}/{preset_name}/{pe_tag}/baseline"

    eval_profile = "paper_math" if task == "gsm8k_full" else "paper_code_main"

    config = {
        "task": task,
        "base_model": BASE_MODEL,
        "lora_repo_id": lora_repo_id,
        "lora_local_path": lora_local_path,
        "edit_mode": "baseline",
        "seed": 0,
        "out_dir": out_dir,
        "notes": "sweep_allmodes_6lora",
        "eval_profile": eval_profile,
        "target_modules": TARGET_MODULES,
        # Include preset params for reference
        "core_frac": preset["core_frac"],
        "noise_frac": preset["noise_frac"],
        "amp_factor": preset["amp_factor"],
        "sup_factor": preset["sup_factor"],
        "mid_factor": preset["mid_factor"],
        "smooth_temperature": preset["smooth_temperature"],
        "smooth_center_q": preset["smooth_center_q"],
        "smooth_align_mid": preset["smooth_align_mid"],
        "preserve_energy": preserve_energy,
        "z_high": preset["z_high"],
        "z_low": preset["z_low"],
        "z_tau": preset["z_tau"],
        "z_fallback_std": preset["z_fallback_std"],
        "robust_z_high": preset["robust_z_high"],
        "robust_z_low": preset["robust_z_low"],
        "robust_z_tau": preset["robust_z_tau"],
        "robust_fallback_sigma": preset["robust_fallback_sigma"],
        "calib_samples": 256,  # Default for baseline
        "calib_batch_size": CALIB_BATCH_SIZE,
        "eval_fewshot": 5,
        "eval_temperature": 0.0,
        "eval_max_tokens": 512,
        "eval_max_samples": -1,
        "keep_adapter": False,
        "run_root": run_root,
    }

    if task == "humaneval_full":
        config["code_preset"] = preset["name"]

    return config


# =============================================================================
# Main Generation
# =============================================================================

def generate_all_configs(run_root: str) -> List[Dict[str, Any]]:
    """Generate all sweep configs."""
    configs = []

    # ==========================================================================
    # 1. Math configs (gsm8k_full)
    # ==========================================================================
    print("[Math] Generating gsm8k_full configs...")

    for lora_repo_id in MATH_LORAS:
        # Add baseline for each math LoRA (once per LoRA)
        configs.append(make_baseline_config(
            task="gsm8k_full",
            lora_repo_id=lora_repo_id,
            run_root=run_root,
            preset=MATH_PRESET,
        ))

        # Sweep over edit modes, calib_samples, seeds
        for edit_mode in EDIT_MODES:
            for calib_samples in CALIB_SAMPLES_LIST:
                for seed in SEEDS:
                    configs.append(make_config(
                        task="gsm8k_full",
                        lora_repo_id=lora_repo_id,
                        edit_mode=edit_mode,
                        seed=seed,
                        calib_samples=calib_samples,
                        preset=MATH_PRESET,
                        preserve_energy=MATH_PRESET["preserve_energy"],
                        run_root=run_root,
                    ))

    math_count = len(configs)
    print(f"[Math] Generated {math_count} configs for gsm8k_full")

    # ==========================================================================
    # 2. Code configs (humaneval_full)
    # ==========================================================================
    print("[Code] Generating humaneval_full configs...")

    # Track baselines to avoid duplicates
    baseline_keys_added = set()

    for lora_repo_id in CODE_LORAS:
        # For each code preset
        for preset_name, preset in CODE_PRESETS.items():
            # For each preserve_energy value
            for preserve_energy in CODE_PRESERVE_ENERGY_VALUES:
                # Add baseline once per (lora, preset, preserve_energy)
                baseline_key = (lora_repo_id, preset_name, preserve_energy)
                if baseline_key not in baseline_keys_added:
                    configs.append(make_baseline_config(
                        task="humaneval_full",
                        lora_repo_id=lora_repo_id,
                        run_root=run_root,
                        preset=preset,
                        preserve_energy=preserve_energy,
                    ))
                    baseline_keys_added.add(baseline_key)

                # Sweep over edit modes, calib_samples, seeds
                for edit_mode in EDIT_MODES:
                    for calib_samples in CALIB_SAMPLES_LIST:
                        for seed in SEEDS:
                            configs.append(make_config(
                                task="humaneval_full",
                                lora_repo_id=lora_repo_id,
                                edit_mode=edit_mode,
                                seed=seed,
                                calib_samples=calib_samples,
                                preset=preset,
                                preserve_energy=preserve_energy,
                                run_root=run_root,
                            ))

    code_count = len(configs) - math_count
    print(f"[Code] Generated {code_count} configs for humaneval_full")

    return configs


def main():
    """Main entry point."""
    # Create run root with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = str(ROOT_DIR / "_runs" / f"{timestamp}_sweep_allmodes_6lora")

    print("=" * 70)
    print("Comprehensive Sweep Config Generator")
    print("=" * 70)
    print(f"Run root: {run_root}")
    print(f"Edit modes: {EDIT_MODES}")
    print(f"Calib samples: {CALIB_SAMPLES_LIST}")
    print(f"Seeds: {SEEDS}")
    print(f"Math LoRAs: {len(MATH_LORAS)}")
    print(f"Code LoRAs: {len(CODE_LORAS)}")
    print(f"Code presets: {list(CODE_PRESETS.keys())}")
    print(f"Code preserve_energy values: {CODE_PRESERVE_ENERGY_VALUES}")
    print("=" * 70)

    # Calculate expected counts
    # Math: 3 loras * (1 baseline + 7 modes * 4 calib * 3 seeds) = 3 * (1 + 84) = 255
    # Code: 3 loras * 5 presets * 3 PE * (1 baseline + 7 modes * 4 calib * 3 seeds)
    #     = 3 * 5 * 3 * (1 + 84) = 45 * 85 = 3825
    math_expected = len(MATH_LORAS) * (1 + len(EDIT_MODES) * len(CALIB_SAMPLES_LIST) * len(SEEDS))
    code_expected = (
        len(CODE_LORAS)
        * len(CODE_PRESETS)
        * len(CODE_PRESERVE_ENERGY_VALUES)
        * (1 + len(EDIT_MODES) * len(CALIB_SAMPLES_LIST) * len(SEEDS))
    )
    print(f"\nExpected config counts:")
    print(f"  Math: {math_expected}")
    print(f"  Code: {code_expected}")
    print(f"  Total: {math_expected + code_expected}")
    print()

    # Generate configs
    configs = generate_all_configs(run_root)

    print(f"\n[Total] Generated {len(configs)} configs")

    # Write to JSONL
    output_path = ROOT_DIR / "experiments" / "configs_sweep_allmodes_6lora.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cfg in configs:
            f.write(json.dumps(cfg, ensure_ascii=False) + "\n")

    print(f"[Output] Written to: {output_path}")
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Configs file: {output_path}")
    print(f"Run root: {run_root}")
    print()
    print("Launch command:")
    print(f'  bash scripts/launch_multi_gpu.sh --gpus "0,1,2,3,4,5,7" --configs {output_path}')
    print()
    print("Collect results command:")
    print(f"  python scripts/collect_results.py --run-root {run_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
