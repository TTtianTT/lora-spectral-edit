#!/usr/bin/env python
import argparse
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None


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


def parse_grid(s: str, dtype=float) -> list:
    """Parse comma-separated values into a list."""
    if not s:
        return []
    return [dtype(x.strip()) for x in s.split(",")]


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def repo_slug(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_configs(run_root: Path, adapter_map: dict, eval_max_samples: int) -> list[dict]:
    base_model = "meta-llama/Llama-2-7b-hf"
    target_modules = ["down_proj", "o_proj"]

    defaults = {
        "target_modules": target_modules,
        "core_frac": 0.2,
        "noise_frac": 0.2,
        "amp_factor": 1.25,
        "sup_factor": 0.80,
        "mid_factor": 1.0,
        "smooth_temperature": 0.35,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l1",
        "calib_samples": 256,
        "calib_batch_size": 2,
        "eval_fewshot": 5,
        "eval_temperature": 0.0,
        "eval_max_tokens": 512,
        "eval_max_samples": int(eval_max_samples),
        "keep_adapter": False,
        "run_root": str(run_root),
    }

    configs: list[dict] = []

    def add_run(task: str, repo_id: str, edit_mode: str, seed: int) -> None:
        slug = repo_slug(repo_id)
        out_dir = run_root / task / slug / edit_mode
        if edit_mode != "baseline":
            out_dir = out_dir / f"seed_{seed}"
        eval_profile = None
        if task.startswith("gsm8k"):
            eval_profile = "paper_math"
        elif task.startswith("humaneval"):
            eval_profile = "paper_code_main"
        cfg = {
            "task": task,
            "base_model": base_model,
            "lora_repo_id": repo_id,
            "lora_local_path": adapter_map[repo_id],
            "edit_mode": edit_mode,
            "seed": int(seed),
            "out_dir": str(out_dir),
            "notes": "full_eval",
        }
        if eval_profile:
            cfg["eval_profile"] = eval_profile
        cfg.update(defaults)
        configs.append(cfg)

    for repo_id in MATH_LORAS:
        add_run("gsm8k_full", repo_id, "baseline", 0)
        for seed in [42, 43, 44]:
            add_run("gsm8k_full", repo_id, "abs_select", seed)
            add_run("gsm8k_full", repo_id, "smooth_abs", seed)
            add_run("gsm8k_full", repo_id, "random_index", seed)

    for repo_id in CODE_LORAS:
        add_run("humaneval_full", repo_id, "baseline", 0)
        for seed in [42, 43, 44]:
            add_run("humaneval_full", repo_id, "abs_select", seed)
            add_run("humaneval_full", repo_id, "smooth_abs", seed)
            add_run("humaneval_full", repo_id, "random_index", seed)

    return configs


def build_humaneval_sweep_configs(
    run_root: Path,
    adapter_map: dict,
    eval_max_samples: int,
    amp_factors: list[float],
    sup_factors: list[float],
    soft_temperatures: list[float],
    soft_pivot_modes: list[str],
    seeds: list[int],
) -> list[dict]:
    """
    Build configs for humaneval hyperparameter grid sweep.

    Generates one config per combination of (amp_factor, sup_factor, soft_temperature,
    soft_pivot_mode, seed) for each Magicoder LoRA on humaneval_full task.
    """
    base_model = "meta-llama/Llama-2-7b-hf"
    target_modules = ["down_proj", "o_proj"]

    defaults = {
        "target_modules": target_modules,
        "core_frac": 0.2,
        "noise_frac": 0.2,
        "mid_factor": 1.0,
        "smooth_center_q": 0.5,
        "smooth_align_mid": True,
        "preserve_energy": "l1",
        "calib_samples": 256,
        "calib_batch_size": 2,
        "eval_fewshot": 5,
        "eval_temperature": 0.0,
        "eval_max_tokens": 512,
        "eval_max_samples": int(eval_max_samples),
        "eval_profile": "paper_code_main",
        "keep_adapter": False,
        "run_root": str(run_root),
    }

    configs: list[dict] = []

    # Generate grid combinations
    grid = list(itertools.product(amp_factors, sup_factors, soft_temperatures, soft_pivot_modes, seeds))

    for repo_id in CODE_LORAS:
        slug = repo_slug(repo_id)

        # Baseline config (one per LoRA, deduped)
        baseline_out_dir = run_root / "humaneval_full" / slug / "baseline"
        baseline_cfg = {
            "task": "humaneval_full",
            "base_model": base_model,
            "lora_repo_id": repo_id,
            "lora_local_path": adapter_map[repo_id],
            "edit_mode": "baseline",
            "seed": 0,
            "out_dir": str(baseline_out_dir),
            "notes": "sweep_baseline",
            "amp_factor": None,
            "sup_factor": None,
            "soft_temperature": None,
            "soft_pivot_mode": None,
        }
        baseline_cfg.update(defaults)
        configs.append(baseline_cfg)

        # Sweep configs
        for amp_f, sup_f, soft_t, pivot_m, seed in grid:
            # Encode hyperparams into out_dir for uniqueness
            hp_str = f"amp{amp_f}_sup{sup_f}_temp{soft_t}_{pivot_m}"
            out_dir = run_root / "humaneval_full" / slug / "sweep" / hp_str / f"seed_{seed}"

            cfg = {
                "task": "humaneval_full",
                "base_model": base_model,
                "lora_repo_id": repo_id,
                "lora_local_path": adapter_map[repo_id],
                "edit_mode": "spectral_gate",
                "seed": int(seed),
                "out_dir": str(out_dir),
                "notes": "sweep",
                "amp_factor": float(amp_f),
                "sup_factor": float(sup_f),
                "soft_temperature": float(soft_t),
                "soft_pivot_mode": str(pivot_m),
            }
            cfg.update(defaults)
            configs.append(cfg)

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch LoRA adapters into HF cache and write configs JSONL."
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Run root directory (defaults to _runs/<timestamp>).",
    )
    parser.add_argument(
        "--configs-out",
        type=str,
        default="experiments/configs_full.jsonl",
        help="Path to write configs JSONL.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only use local HF cache; do not download missing files.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=-1,
        help="Max samples for eval (-1 for full).",
    )

    # Grid sweep arguments for humaneval
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate hyperparameter sweep configs for humaneval_full (Magicoder LoRAs only).",
    )
    parser.add_argument(
        "--amp-factors",
        type=str,
        default="1.25",
        help="Comma-separated amp_factor values for sweep (e.g., '1.0,1.25,1.5').",
    )
    parser.add_argument(
        "--sup-factors",
        type=str,
        default="0.8",
        help="Comma-separated sup_factor values for sweep (e.g., '0.5,0.8,1.0').",
    )
    parser.add_argument(
        "--soft-temperatures",
        type=str,
        default="0.35",
        help="Comma-separated soft_temperature values for sweep (e.g., '0.1,0.35,0.5').",
    )
    parser.add_argument(
        "--soft-pivot-modes",
        type=str,
        default="median",
        help="Comma-separated soft_pivot_mode values for sweep (e.g., 'median,mean,max').",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seed values for sweep (e.g., '42,43,44').",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root) if args.run_root else root_dir / "_runs" / now_timestamp()
    run_root.mkdir(parents=True, exist_ok=True)

    adapter_map = {}
    for repo_id in MATH_LORAS + CODE_LORAS:
        if snapshot_download is None:
            adapter_map[repo_id] = repo_id
            continue
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=None,
            local_files_only=bool(args.local_only),
        )
        adapter_map[repo_id] = path

    if snapshot_download is None:
        print(
            "[Prefetch] huggingface_hub not installed; "
            "writing configs with lora_local_path=<repo_id> (will download at runtime)."
        )

    mapping_path = run_root / "adapter_cache_map.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(adapter_map, f, indent=2, ensure_ascii=False)

    if args.sweep:
        # Parse grid values
        amp_factors = parse_grid(args.amp_factors, float)
        sup_factors = parse_grid(args.sup_factors, float)
        soft_temperatures = parse_grid(args.soft_temperatures, float)
        soft_pivot_modes = [s.strip() for s in args.soft_pivot_modes.split(",")]
        seeds = parse_grid(args.seeds, int)

        configs = build_humaneval_sweep_configs(
            run_root=run_root,
            adapter_map=adapter_map,
            eval_max_samples=args.eval_max_samples,
            amp_factors=amp_factors,
            sup_factors=sup_factors,
            soft_temperatures=soft_temperatures,
            soft_pivot_modes=soft_pivot_modes,
            seeds=seeds,
        )

        n_hp_combos = len(amp_factors) * len(sup_factors) * len(soft_temperatures) * len(soft_pivot_modes) * len(seeds)
        print(f"[Sweep] Grid: {len(amp_factors)} amp x {len(sup_factors)} sup x {len(soft_temperatures)} temp x {len(soft_pivot_modes)} pivot x {len(seeds)} seeds = {n_hp_combos} combos per LoRA")
        print(f"[Sweep] Total configs: {len(configs)} ({len(CODE_LORAS)} LoRAs x ({n_hp_combos} edited + 1 baseline))")
    else:
        configs = build_configs(run_root, adapter_map, args.eval_max_samples)

    configs_path = Path(args.configs_out)
    write_jsonl(configs_path, configs)
    write_jsonl(run_root / "configs_full.jsonl", configs)

    print(f"[Prefetch] Wrote mapping: {mapping_path}")
    print(f"[Prefetch] Wrote configs: {configs_path}")
    print(f"[Prefetch] Run root: {run_root}")


if __name__ == "__main__":
    main()
