#!/usr/bin/env python
import argparse
import json
import os

import torch
from safetensors.torch import load_file


def _resolve_safetensors(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "adapter_model.safetensors")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LoRA A/B weights between original and edited adapters."
    )
    parser.add_argument("--original", required=True, help="Original LoRA dir or .safetensors")
    parser.add_argument("--edited", required=True, help="Edited LoRA dir or .safetensors")
    parser.add_argument("--out_json", required=True, help="Path to write diff_report.json")
    args = parser.parse_args()

    original_path = _resolve_safetensors(args.original)
    edited_path = _resolve_safetensors(args.edited)

    original_state = load_file(original_path)
    edited_state = load_file(edited_path)

    keys = [
        k
        for k in original_state.keys()
        if ".lora_A." in k or ".lora_B." in k
    ]
    if not keys:
        raise ValueError("No LoRA A/B tensors found in original adapter.")

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    total_elems = 0
    changed_tensors_count = 0
    missing_keys = []

    for key in keys:
        if key not in edited_state:
            missing_keys.append(key)
            continue
        orig = original_state[key].to(dtype=torch.float32)
        edit = edited_state[key].to(dtype=torch.float32)
        diff = edit - orig
        diff_abs = diff.abs()
        max_abs_diff = max(max_abs_diff, diff_abs.max().item())
        sum_abs_diff += diff_abs.sum().item()
        sum_sq_diff += (diff * diff).sum().item()
        total_elems += diff.numel()
        if diff_abs.max().item() > 0.0:
            changed_tensors_count += 1

    if total_elems == 0:
        raise ValueError("No overlapping LoRA A/B tensors found to compare.")

    report = {
        "changed_tensors_count": changed_tensors_count,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": sum_abs_diff / total_elems,
        "total_l2_diff": sum_sq_diff ** 0.5,
        "missing_keys_count": len(missing_keys),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
