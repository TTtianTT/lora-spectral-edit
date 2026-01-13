#!/usr/bin/env python
"""
Generate TRIMMED CODE sweep configs (< 1000 total).
Skips configs that are already completed in the previous killed sweep.

Output: experiments/configs_trimmed_code_key1000.jsonl
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# Configuration
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
HF_MODELS_DIR = Path("/home/zailongtian/hf_models/hub")

# Previous sweep run_root (to check for completed runs)
PREVIOUS_RUN_ROOT = ROOT_DIR / "_runs" / "20260112_213540_sweep_allmodes_6lora"

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
TARGET_MODULES = ["down_proj", "o_proj"]
CALIB_BATCH_SIZE = 2

# Trimmed sweep axes
SEEDS = [42, 44]  # Only 2 seeds
CALIB_SAMPLES_LIST = [32, 128, 256]  # Skip 64

# Edit modes (7 total, including baseline)
EDIT_MODES = [
    "baseline",
    "abs_select",
    "smooth_abs",
    "double_smooth",
    "z_score",
    "robust_z",
    "random_index",
]

# Only 3 code LoRAs
CODE_LORAS = [
    "LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128",
    "LoRA-TMLR-2024/magicoder-lora-rank-256-alpha-512",
]

# Only 2 preserve_energy values (skip None)
PRESERVE_ENERGY_VALUES = ["l2", "l1"]

# =============================================================================
# 3 Code Presets (trimmed from 5)
# =============================================================================

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
        # z_score params (defaults)
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
    "code_round_clean": {
        "name": "code_round_clean",
        "core_frac": 0.10,
        "noise_frac": 0.0,
        "amp_factor": 1.20,
        "sup_factor": 1.0,
        "mid_factor": 1.00,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        # z_score params
        "z_high": 1.0,
        "z_low": -0.5,
        "z_tau": 0.2,
        "z_fallback_std": 1e-6,
        # robust_z params
        "robust_z_high": 2.0,
        "robust_z_low": -3.0,
        "robust_z_tau": 0.30,
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
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_lora_local_path(lora_repo_id: str) -> str:
    """Find local path for a HuggingFace LoRA adapter."""
    cache_name = f"models--{lora_repo_id.replace('/', '--')}"
    cache_dir = HF_MODELS_DIR / cache_name / "snapshots"

    if cache_dir.exists():
        snapshots = sorted(cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])
    return lora_repo_id


def get_lora_slug(lora_repo_id: str) -> str:
    """Extract short name from lora_repo_id."""
    return lora_repo_id.split("/")[-1]


def is_run_completed(out_dir: str) -> bool:
    """Check if a run is already completed (has metrics.json)."""
    metrics_path = Path(out_dir) / "metrics.json"
    return metrics_path.exists()


def check_previous_run_completed(
    task: str,
    lora_slug: str,
    preset_name: str,
    preserve_energy: str,
    edit_mode: str,
    calib_samples: int,
    seed: int,
) -> Optional[str]:
    """
    Check if this config was already completed in the previous sweep.
    Returns the out_dir if completed, None otherwise.
    """
    pe_tag = preserve_energy if preserve_energy != "none" else "none"

    if edit_mode == "baseline":
        # Baseline path format
        out_dir = (
            PREVIOUS_RUN_ROOT / task / lora_slug / preset_name / pe_tag / "baseline"
        )
    else:
        out_dir = (
            PREVIOUS_RUN_ROOT / task / lora_slug / preset_name / pe_tag /
            edit_mode / f"calib_{calib_samples}" / f"seed_{seed}"
        )

    if is_run_completed(str(out_dir)):
        return str(out_dir)
    return None


def make_config(
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
    preset_name = preset["name"]
    pe_tag = preserve_energy

    # Build out_dir path
    if edit_mode == "baseline":
        out_dir = f"{run_root}/humaneval_full/{lora_slug}/{preset_name}/{pe_tag}/baseline"
    else:
        out_dir = (
            f"{run_root}/humaneval_full/{lora_slug}/{preset_name}/{pe_tag}/"
            f"{edit_mode}/calib_{calib_samples}/seed_{seed}"
        )

    config = {
        "task": "humaneval_full",
        "base_model": BASE_MODEL,
        "lora_repo_id": lora_repo_id,
        "lora_local_path": lora_local_path,
        "edit_mode": edit_mode,
        "seed": seed if edit_mode != "baseline" else 0,
        "out_dir": out_dir,
        "notes": "trimmed_code_key1000",
        "eval_profile": "paper_code_main",
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
        "calib_samples": calib_samples if edit_mode != "baseline" else 256,
        "calib_batch_size": CALIB_BATCH_SIZE,
        # Eval settings
        "eval_fewshot": 0,  # 0-shot for HumanEval
        "eval_temperature": 0.2,  # Paper setting
        "eval_max_tokens": 512,
        "eval_max_samples": -1,
        "keep_adapter": False,
        "run_root": run_root,
        "code_preset": preset_name,
    }

    return config


# =============================================================================
# Main Generation
# =============================================================================

def generate_all_configs(run_root: str) -> tuple:
    """Generate all sweep configs, skipping completed runs."""
    configs = []
    skipped_count = 0
    skipped_details = []

    # Track baselines to avoid duplicates
    baseline_keys_done = set()

    for lora_repo_id in CODE_LORAS:
        lora_slug = get_lora_slug(lora_repo_id)

        for preset_name, preset in CODE_PRESETS.items():
            for preserve_energy in PRESERVE_ENERGY_VALUES:
                # Handle baseline (once per lora/preset/pe combo)
                baseline_key = (lora_repo_id, preset_name, preserve_energy)
                if baseline_key not in baseline_keys_done:
                    baseline_keys_done.add(baseline_key)

                    # Check if already completed in previous run
                    prev_completed = check_previous_run_completed(
                        task="humaneval_full",
                        lora_slug=lora_slug,
                        preset_name=preset_name,
                        preserve_energy=preserve_energy,
                        edit_mode="baseline",
                        calib_samples=256,
                        seed=0,
                    )

                    if prev_completed:
                        skipped_count += 1
                        skipped_details.append(f"baseline: {lora_slug}/{preset_name}/{preserve_energy}")
                    else:
                        configs.append(make_config(
                            lora_repo_id=lora_repo_id,
                            edit_mode="baseline",
                            seed=0,
                            calib_samples=256,
                            preset=preset,
                            preserve_energy=preserve_energy,
                            run_root=run_root,
                        ))

                # Handle other edit modes
                for edit_mode in EDIT_MODES:
                    if edit_mode == "baseline":
                        continue  # Already handled above

                    for calib_samples in CALIB_SAMPLES_LIST:
                        for seed in SEEDS:
                            # Check if already completed
                            prev_completed = check_previous_run_completed(
                                task="humaneval_full",
                                lora_slug=lora_slug,
                                preset_name=preset_name,
                                preserve_energy=preserve_energy,
                                edit_mode=edit_mode,
                                calib_samples=calib_samples,
                                seed=seed,
                            )

                            if prev_completed:
                                skipped_count += 1
                            else:
                                configs.append(make_config(
                                    lora_repo_id=lora_repo_id,
                                    edit_mode=edit_mode,
                                    seed=seed,
                                    calib_samples=calib_samples,
                                    preset=preset,
                                    preserve_energy=preserve_energy,
                                    run_root=run_root,
                                ))

    return configs, skipped_count, skipped_details


def main():
    """Main entry point."""
    run_root = str(ROOT_DIR / "_runs" / "20260112_trimmed_code_key1000")

    print("=" * 70)
    print("Trimmed CODE Sweep Config Generator (< 1000 configs)")
    print("=" * 70)
    print(f"Run root: {run_root}")
    print(f"Previous run root (for skipping): {PREVIOUS_RUN_ROOT}")
    print(f"Edit modes: {EDIT_MODES}")
    print(f"Calib samples: {CALIB_SAMPLES_LIST}")
    print(f"Seeds: {SEEDS}")
    print(f"Code LoRAs: {len(CODE_LORAS)}")
    print(f"Code presets: {list(CODE_PRESETS.keys())}")
    print(f"Preserve energy values: {PRESERVE_ENERGY_VALUES}")
    print("=" * 70)

    # Calculate expected max (before skipping)
    # 3 loras × 3 presets × 2 PE × (1 baseline + 6 modes × 3 calib × 2 seeds)
    baselines = 3 * 3 * 2 * 1  # = 18
    non_baselines = 3 * 3 * 2 * 6 * 3 * 2  # = 648
    max_expected = baselines + non_baselines  # = 666
    print(f"\nMax expected configs (before skipping): {max_expected}")
    print()

    # Generate configs
    configs, skipped_count, skipped_details = generate_all_configs(run_root)

    print(f"[Result] Generated {len(configs)} configs")
    print(f"[Result] Skipped {skipped_count} already-completed configs")

    if skipped_count > 0 and len(skipped_details) <= 10:
        print("[Skipped baselines]:")
        for detail in skipped_details:
            print(f"  - {detail}")

    # Write to JSONL
    output_path = ROOT_DIR / "experiments" / "configs_trimmed_code_key1000.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for cfg in configs:
            f.write(json.dumps(cfg, ensure_ascii=False) + "\n")

    print(f"\n[Output] Written to: {output_path}")
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Configs file: {output_path}")
    print(f"Run root: {run_root}")
    print(f"Configs written: {len(configs)}")
    print(f"Configs skipped (already done): {skipped_count}")
    print()
    print("Launch command:")
    print(f'  bash scripts/launch_multi_gpu.sh --gpus "0,1,2,3,4,5,7" --configs {output_path}')
    print()
    print("Collect results command:")
    print(f"  python scripts/collect_results.py --run-root {run_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
