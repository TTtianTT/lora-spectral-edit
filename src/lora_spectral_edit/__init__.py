"""
LoRA Spectral Edit - Sensitivity-based spectral editing for LoRA adapters.

This package provides tools to edit LoRA adapters using SVD-based spectral
manipulation guided by gradient sensitivity analysis.
"""

__version__ = "0.1.0"

from .svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from .io import load_lora_state_dict, save_lora_state_dict, load_adapter_config
from .edit import apply_spectral_edit

__all__ = [
    "lowrank_svd_from_ba",
    "rebuild_ba_from_uv_sigma",
    "load_lora_state_dict",
    "save_lora_state_dict",
    "load_adapter_config",
    "apply_spectral_edit",
]
