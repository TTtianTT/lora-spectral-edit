"""
LoRA I/O utilities for loading and saving adapter weights.
"""

import os
import re
import json
from typing import Dict, Tuple, Optional

import torch

from huggingface_hub import snapshot_download

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False


def ensure_local_lora_dir(lora_path_or_id: str, cache_dir: Optional[str] = None) -> str:
    """
    If lora_path_or_id is a local directory, return it.
    Else treat as HuggingFace repo id and snapshot_download it.
    """
    if os.path.isdir(lora_path_or_id):
        return os.path.abspath(lora_path_or_id)
    local_dir = snapshot_download(repo_id=lora_path_or_id, cache_dir=cache_dir)
    return local_dir


def load_adapter_config(lora_dir: str) -> dict:
    """Load adapter_config.json from a LoRA directory."""
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"adapter_config.json not found in {lora_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_adapter_weight_file(lora_dir: str) -> Tuple[str, str]:
    """
    Find the adapter weights file in a LoRA directory.

    Returns:
        Tuple of (path, format) where format is "safetensors" or "bin".
    """
    st = os.path.join(lora_dir, "adapter_model.safetensors")
    bn = os.path.join(lora_dir, "adapter_model.bin")
    if os.path.exists(st):
        if not HAVE_SAFETENSORS:
            raise RuntimeError("Found adapter_model.safetensors but safetensors not installed.")
        return st, "safetensors"
    if os.path.exists(bn):
        return bn, "bin"
    raise FileNotFoundError(f"adapter_model.(safetensors|bin) not found in {lora_dir}")


def load_lora_state_dict(lora_dir: str) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Load LoRA state dict from a directory.

    Returns:
        Tuple of (state_dict, format) where format is "safetensors" or "bin".
    """
    path, fmt = find_adapter_weight_file(lora_dir)
    if fmt == "safetensors":
        sd = safe_load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd, fmt


def save_lora_state_dict(lora_dir_out: str, state_dict: Dict[str, torch.Tensor], fmt: str):
    """
    Save LoRA state dict to a directory.

    Args:
        lora_dir_out: Output directory path.
        state_dict: The state dict to save.
        fmt: Format to use ("safetensors" or "bin").
    """
    os.makedirs(lora_dir_out, exist_ok=True)
    out_path = os.path.join(
        lora_dir_out, "adapter_model.safetensors" if fmt == "safetensors" else "adapter_model.bin"
    )
    if fmt == "safetensors":
        if not HAVE_SAFETENSORS:
            raise RuntimeError("safetensors not installed but format='safetensors' requested.")
        safe_save_file(state_dict, out_path)
    else:
        torch.save(state_dict, out_path)


def parse_lora_ab_key(key: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Parse a LoRA state dict key to extract module info.

    Supports keys like:
      - ...<module>.lora_A.weight
      - ...<module>.lora_B.weight
      - ...<module>.lora_A.<adapter>.weight
      - ...<module>.lora_B.<adapter>.weight

    Returns:
        Tuple of (module_prefix, which, adapter_name) where which is "A" or "B",
        or None if the key doesn't match the expected pattern.
    """
    # lora_A.default.weight or lora_A.<adapter>.weight
    m = re.match(r"^(.*)\.lora_([AB])(?:\.([^.]+))?\.weight$", key)
    if m:
        module_prefix = m.group(1)
        which = m.group(2)
        adapter = m.group(3)
        return module_prefix, which, adapter
    # lora_A.weight (older format)
    m2 = re.match(r"^(.*)\.lora_([AB])\.weight$", key)
    if m2:
        module_prefix = m2.group(1)
        which = m2.group(2)
        return module_prefix, which, None
    return None


def layer_idx_from_module_prefix(prefix: str) -> Optional[int]:
    """Extract layer index from a module prefix like 'model.layers.5.mlp.down_proj'."""
    m = re.search(r"\.layers\.(\d+)\.", prefix)
    return int(m.group(1)) if m else None


def get_scaling_for_module(adapter_cfg: dict, module_prefix: str) -> float:
    """
    Get the LoRA scaling factor (alpha/r) for a module.

    PEFT LoRA scaling is typically alpha / r. If rank_pattern/alpha_pattern
    exists in the config, tries to match by module suffix.
    """
    r_global = adapter_cfg.get("r", None)
    alpha_global = adapter_cfg.get("lora_alpha", None)
    if r_global is None or alpha_global is None:
        r_global = adapter_cfg.get("rank", 0)
        alpha_global = adapter_cfg.get("alpha", 0)

    rank_pattern = adapter_cfg.get("rank_pattern", None)
    alpha_pattern = adapter_cfg.get("alpha_pattern", None)

    suffix = module_prefix.split(".")[-1]
    r = r_global
    alpha = alpha_global
    if isinstance(rank_pattern, dict) and suffix in rank_pattern:
        r = rank_pattern[suffix]
    if isinstance(alpha_pattern, dict) and suffix in alpha_pattern:
        alpha = alpha_pattern[suffix]

    if not r:
        raise ValueError(f"Invalid rank r={r} for module {module_prefix}")
    return float(alpha) / float(r)
