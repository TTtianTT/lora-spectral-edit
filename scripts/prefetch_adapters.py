#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download


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
        cfg.update(defaults)
        configs.append(cfg)

    for repo_id in MATH_LORAS:
        add_run("gsm8k_full", repo_id, "baseline", 0)
        for seed in [42, 43, 44]:
            add_run("gsm8k_full", repo_id, "abs_select", seed)
            add_run("gsm8k_full", repo_id, "random_index", seed)

    for repo_id in CODE_LORAS:
        add_run("humaneval_full", repo_id, "baseline", 0)
        for seed in [42, 43, 44]:
            add_run("humaneval_full", repo_id, "abs_select", seed)
            add_run("humaneval_full", repo_id, "random_index", seed)

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch LoRA adapters into HF cache and write config mapping."
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
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root) if args.run_root else root_dir / "_runs" / now_timestamp()
    run_root.mkdir(parents=True, exist_ok=True)

    adapter_map = {}
    for repo_id in MATH_LORAS + CODE_LORAS:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=None,
            local_files_only=bool(args.local_only),
        )
        adapter_map[repo_id] = path

    mapping_path = run_root / "adapter_cache_map.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(adapter_map, f, indent=2, ensure_ascii=False)

    configs = build_configs(run_root, adapter_map, args.eval_max_samples)
    configs_path = Path(args.configs_out)
    write_jsonl(configs_path, configs)
    write_jsonl(run_root / "configs_full.jsonl", configs)

    print(f"[Prefetch] Wrote mapping: {mapping_path}")
    print(f"[Prefetch] Wrote configs: {configs_path}")
    print(f"[Prefetch] Run root: {run_root}")


if __name__ == "__main__":
    main()
