"""
Spectral editing strategies for LoRA singular values.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EditConfig:
    """Configuration for spectral editing."""
    # Sensitivity mode: "abs_select" (recommended), "random_index", or "gd"
    mode: str = "abs_select"

    # abs_select mode parameters
    core_frac: float = 0.2      # Fraction of dims with largest |g| to amplify
    noise_frac: float = 0.2     # Fraction of dims with smallest |g| to suppress
    amp_factor: float = 1.25    # Multiplicative factor for core dims
    sup_factor: float = 0.80    # Multiplicative factor for noise dims
    mid_factor: float = 1.0     # Multiplicative factor for middle dims
    min_core_k: int = 1         # Minimum number of core dims

    # gd mode parameters
    eta: float = 0.2            # Learning rate for gradient update
    update_mode: str = "multiplicative"  # "additive" or "multiplicative"
    asymmetric_update: bool = True
    eta_suppress: float = 2.0   # Step size for g>0
    eta_enhance: float = 0.2    # Step size for g<0
    pos_power: float = 1.0      # Nonlinearity for positive grads

    # Common parameters
    grad_norm: str = "mean_abs"  # "none", "mean_abs", or "l2"
    preserve_energy: str = "l1"  # "none", "l1", or "l2"
    sigma_clip_min: float = 0.0


def normalize_gradient(g: torch.Tensor, method: str) -> torch.Tensor:
    """Normalize gradient tensor."""
    if method == "mean_abs":
        denom = g.abs().mean().clamp_min(1e-8)
        return g / denom
    elif method == "l2":
        denom = torch.linalg.norm(g).clamp_min(1e-8)
        return g / denom
    return g


def apply_abs_select(
    sigma0: torch.Tensor,
    g_abs: torch.Tensor,
    config: EditConfig,
) -> torch.Tensor:
    """
    Apply sensitivity-based feature selection.

    Uses |g_sigma| to identify:
    - Core features (high |g|): amplify
    - Noise features (low |g|): suppress
    - Middle features: keep or scale by mid_factor
    """
    r = int(g_abs.numel())

    k_core = max(int(round(r * config.core_frac)), config.min_core_k)
    k_core = min(k_core, r)

    k_noise = int(round(r * config.noise_frac))
    k_noise = max(0, min(k_noise, r - k_core))

    order = torch.argsort(g_abs, descending=True)
    core_idx = order[:k_core]
    noise_idx = order[-k_noise:] if k_noise > 0 else torch.tensor([], dtype=torch.long)

    gate = torch.full_like(sigma0, config.mid_factor)
    gate[core_idx] = config.amp_factor
    if k_noise > 0:
        gate[noise_idx] = config.sup_factor

    return sigma0 * gate, k_core, k_noise


def apply_random_index(
    sigma0: torch.Tensor,
    config: EditConfig,
) -> torch.Tensor:
    """
    Apply random index selection with the same counts as abs_select.

    Randomly chooses core/noise indices uniformly, then applies amp/sup factors.
    """
    r = int(sigma0.numel())

    k_core = max(int(round(r * config.core_frac)), config.min_core_k)
    k_core = min(k_core, r)

    k_noise = int(round(r * config.noise_frac))
    k_noise = max(0, min(k_noise, r - k_core))

    order = torch.randperm(r, device=sigma0.device)
    core_idx = order[:k_core]
    noise_idx = order[k_core:k_core + k_noise] if k_noise > 0 else torch.tensor([], dtype=torch.long)

    gate = torch.full_like(sigma0, config.mid_factor)
    gate[core_idx] = config.amp_factor
    if k_noise > 0:
        gate[noise_idx] = config.sup_factor

    return sigma0 * gate, k_core, k_noise


def apply_gd_update(
    sigma0: torch.Tensor,
    g: torch.Tensor,
    config: EditConfig,
) -> torch.Tensor:
    """
    Apply gradient-descent style update (signed gradient).
    """
    if config.asymmetric_update:
        g_pos = torch.relu(g)
        g_neg = -torch.relu(-g)
        if config.pos_power != 1.0:
            g_pos = g_pos.pow(config.pos_power)
        g_eff = config.eta_suppress * g_pos + config.eta_enhance * g_neg

        if config.update_mode == "additive":
            return sigma0 - g_eff
        else:
            return sigma0 * torch.exp(-g_eff)
    else:
        if config.update_mode == "additive":
            return sigma0 - config.eta * g
        else:
            return sigma0 * torch.exp(-config.eta * g)


def preserve_spectral_energy(
    sigma0: torch.Tensor,
    sigma_new: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Preserve spectral energy (L1 or L2 norm)."""
    if method == "l1":
        s0 = sigma0.sum().clamp_min(1e-8)
        s1 = sigma_new.sum().clamp_min(1e-8)
        return sigma_new * (s0 / s1)
    elif method == "l2":
        s0 = torch.linalg.norm(sigma0).clamp_min(1e-8)
        s1 = torch.linalg.norm(sigma_new).clamp_min(1e-8)
        return sigma_new * (s0 / s1)
    return sigma_new


def apply_spectral_edit(
    sigma0: torch.Tensor,
    g_sigma: torch.Tensor,
    config: Optional[EditConfig] = None,
) -> tuple:
    """
    Apply spectral edit to singular values based on gradient sensitivity.

    Args:
        sigma0: Original singular values, shape [r]
        g_sigma: Accumulated gradient w.r.t. sigma, shape [r]
        config: EditConfig with editing parameters

    Returns:
        Tuple of (sigma_new, stats_dict) where stats_dict contains editing statistics.
    """
    if config is None:
        config = EditConfig()

    g = g_sigma.clone()
    g_abs = g.abs()

    # Normalize gradient
    g_abs_norm = normalize_gradient(g_abs, config.grad_norm)
    g_norm = normalize_gradient(g, config.grad_norm)

    stats = {
        "r": int(sigma0.numel()),
        "g_abs_mean": float(g_abs.mean().item()),
        "g_abs_max": float(g_abs.max().item()),
    }

    if config.mode == "abs_select":
        sigma_new, k_core, k_noise = apply_abs_select(sigma0, g_abs_norm, config)
        stats["k_core"] = k_core
        stats["k_noise"] = k_noise
        stats["amp_factor"] = config.amp_factor
        stats["sup_factor"] = config.sup_factor
        stats["mid_factor"] = config.mid_factor
    elif config.mode == "random_index":
        sigma_new, k_core, k_noise = apply_random_index(sigma0, config)
        stats["k_core"] = k_core
        stats["k_noise"] = k_noise
        stats["amp_factor"] = config.amp_factor
        stats["sup_factor"] = config.sup_factor
        stats["mid_factor"] = config.mid_factor
    else:
        sigma_new = apply_gd_update(sigma0, g_norm, config)
        stats["k_core"] = None
        stats["k_noise"] = None

    # Clip minimum
    sigma_new = sigma_new.clamp_min(config.sigma_clip_min)

    # Preserve energy
    sigma_new = preserve_spectral_energy(sigma0, sigma_new, config.preserve_energy)

    stats["sigma0_sum"] = float(sigma0.sum().item())
    stats["sigma_new_sum"] = float(sigma_new.sum().item())
    stats["sigma0_top1"] = float(sigma0.max().item())
    stats["sigma_new_top1"] = float(sigma_new.max().item())

    return sigma_new, stats
