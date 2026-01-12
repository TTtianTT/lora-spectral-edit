#!/usr/bin/env python
"""Generate configs for robust_z sweep experiments."""
import json
import os
from datetime import datetime
from pathlib import Path

# Sweep parameters (matching previous calib_samples sweep)
CALIB_SAMPLES = [32, 64, 128, 256]
SEEDS = [42, 43, 44]

GSM8K_LORAS = [
    "LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128",
    "LoRA-TMLR-2024/metamath-lora-rank-256-alpha-512",
]

HUMANEVAL_LORAS = [
    "LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/magicoder-lora-rank-64-alpha-128",
    "LoRA-TMLR-2024/magicoder-lora-rank-256-alpha-512",
]

BASE_MODEL = "meta-llama/Llama-2-7b-hf"

# Default hyperparams
DEFAULT_CONFIG = {
    "amp_factor": 1.25,
    "sup_factor": 0.8,
    "mid_factor": 1.0,
    "core_frac": 0.2,
    "noise_frac": 0.2,
    "smooth_temperature": 0.35,
    "smooth_center_q": 0.5,
    "smooth_align_mid": True,
    "preserve_energy": "l1",
    "calib_batch_size": 2,
    "eval_fewshot": 5,
    "eval_temperature": 0.0,
    "eval_max_tokens": 512,
    "eval_max_samples": -1,
    "keep_adapter": False,
    "target_modules": ["down_proj", "o_proj"],
    "soft_temperature": None,
    "soft_pivot_mode": None,
    # z_score params (for reference)
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


def get_local_path_for_repo(repo_id: str) -> str:
    """Get local path for a HuggingFace repo (assumes prefetched)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    # Look for the local path in the hub cache
    cache_dir = os.environ.get("HF_HOME", HF_HUB_CACHE)
    repo_slug = repo_id.replace("/", "--")
    model_dir = Path(cache_dir) / f"models--{repo_slug}"

    if model_dir.exists():
        # Find the latest snapshot
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return str(snapshots[0])

    # Fallback: just return the repo_id
    return repo_id


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate robust_z sweep configs")
    parser.add_argument("--run_root", type=str, default=None,
                        help="Run root directory (auto-generated if not specified)")
    parser.add_argument("--output", type=str, default="configs_robust_z.jsonl",
                        help="Output config file")
    parser.add_argument("--local_paths", action="store_true",
                        help="Resolve local paths for LoRAs (requires prefetch)")
    args = parser.parse_args()

    # Generate run root
    if args.run_root:
        run_root = args.run_root
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root = f"_runs/{timestamp}_robust_z"

    run_root = os.path.abspath(run_root)

    configs = []

    # Load adapter cache map if available
    adapter_cache_map = {}
    root_dir = Path(__file__).parent.parent
    cache_map_path = root_dir / "_runs/20260112_022820_calib/adapter_cache_map.json"
    if cache_map_path.exists():
        with open(cache_map_path) as f:
            adapter_cache_map = json.load(f)
        print(f"[Generate] Loaded adapter cache from: {cache_map_path}")

    # GSM8K configs
    for lora_repo_id in GSM8K_LORAS:
        lora_slug = lora_repo_id.split("/")[-1]
        local_path = adapter_cache_map.get(lora_repo_id, lora_repo_id)

        for cs in CALIB_SAMPLES:
            for seed in SEEDS:
                out_dir = f"{run_root}/gsm8k_full/{lora_slug}/calib_{cs}/robust_z/seed_{seed}"
                config = {
                    **DEFAULT_CONFIG,
                    "task": "gsm8k_full",
                    "base_model": BASE_MODEL,
                    "lora_repo_id": lora_repo_id,
                    "lora_local_path": local_path,
                    "edit_mode": "robust_z",
                    "seed": seed,
                    "calib_samples": cs,
                    "out_dir": out_dir,
                    "notes": "robust_z_sweep",
                    "eval_profile": "paper_math",
                    "run_root": run_root,
                }
                configs.append(config)

    # HumanEval configs
    for lora_repo_id in HUMANEVAL_LORAS:
        lora_slug = lora_repo_id.split("/")[-1]
        local_path = adapter_cache_map.get(lora_repo_id, lora_repo_id)

        for cs in CALIB_SAMPLES:
            for seed in SEEDS:
                out_dir = f"{run_root}/humaneval_full/{lora_slug}/calib_{cs}/robust_z/seed_{seed}"
                config = {
                    **DEFAULT_CONFIG,
                    "task": "humaneval_full",
                    "base_model": BASE_MODEL,
                    "lora_repo_id": lora_repo_id,
                    "lora_local_path": local_path,
                    "edit_mode": "robust_z",
                    "seed": seed,
                    "calib_samples": cs,
                    "out_dir": out_dir,
                    "notes": "robust_z_sweep",
                    "eval_profile": "paper_code_main",
                    "run_root": run_root,
                }
                configs.append(config)

    # Write configs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for config in configs:
            f.write(json.dumps(config, ensure_ascii=False) + "\n")

    print(f"[Generate] Created {len(configs)} configs")
    print(f"[Generate] Output: {output_path}")
    print(f"[Generate] Run root: {run_root}")
    print(f"[Generate] Tasks: GSM8K ({len(GSM8K_LORAS)} LoRAs), HumanEval ({len(HUMANEVAL_LORAS)} LoRAs)")
    print(f"[Generate] Calib samples: {CALIB_SAMPLES}")
    print(f"[Generate] Seeds: {SEEDS}")

    # Also copy configs to run_root
    os.makedirs(run_root, exist_ok=True)
    run_configs_path = Path(run_root) / "configs_robust_z.jsonl"
    with open(run_configs_path, "w", encoding="utf-8") as f:
        for config in configs:
            f.write(json.dumps(config, ensure_ascii=False) + "\n")
    print(f"[Generate] Also wrote to: {run_configs_path}")


if __name__ == "__main__":
    main()
