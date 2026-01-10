"""
Spectral editing strategies for LoRA singular values.

Adds a new mode:
- "smooth_abs": smooth, continuous scaling based on |g_sigma| via a sigmoid gate,
                with optional alignment so gate(center)=mid_factor.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch


@dataclass
class EditConfig:
    """Configuration for spectral editing."""
    # Sensitivity mode: "abs_select" (recommended), "smooth_abs", or "gd"
    mode: str = "abs_select"

    # -------------------------
    # abs_select mode parameters
    # -------------------------
    core_frac: float = 0.2      # Fraction of dims with largest |g| to amplify
    noise_frac: float = 0.2     # Fraction of dims with smallest |g| to suppress
    amp_factor: float = 1.25    # Multiplicative factor for core dims
    sup_factor: float = 0.80    # Multiplicative factor for noise dims
    mid_factor: float = 1.0     # Multiplicative factor for middle dims
    min_core_k: int = 1         # Minimum number of core dims

    # -------------------------
    # smooth_abs mode parameters
    # -------------------------
    smooth_temperature: float = 0.35   # larger -> smoother/flatter; smaller -> sharper
    smooth_center_q: float = 0.5       # center quantile (0.5 = median)
    smooth_align_mid: bool = True      # enforce gate(center)=mid_factor if feasible

    # -------------------------
    # gd mode parameters
    # -------------------------
    eta: float = 0.2            # Learning rate for gradient update
    update_mode: str = "multiplicative"  # "additive" or "multiplicative"
    asymmetric_update: bool = True
    eta_suppress: float = 2.0   # Step size for g>0
    eta_enhance: float = 0.2    # Step size for g<0
    pos_power: float = 1.0      # Nonlinearity for positive grads

    # -------------------------
    # Common parameters
    # -------------------------
    grad_norm: str = "mean_abs"   # "none", "mean_abs", or "l2"
    preserve_energy: str = "l1"   # "none", "l1", or "l2"
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
) -> Tuple[torch.Tensor, int, int]:
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
    noise_idx = (
        order[-k_noise:]
        if k_noise > 0
        else torch.empty(0, dtype=torch.long, device=sigma0.device)
    )

    gate = torch.full_like(sigma0, float(config.mid_factor))
    gate[core_idx] = float(config.amp_factor)
    if k_noise > 0:
        gate[noise_idx] = float(config.sup_factor)

    return sigma0 * gate, k_core, k_noise


def apply_smooth_abs(
    sigma0: torch.Tensor,
    g_abs: torch.Tensor,
    config: EditConfig,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Smooth scaling based on |g| using a sigmoid gate.

    gate_i in [sup_factor, amp_factor], monotonic in |g|.
    Optionally align the center quantile so that gate(center)=mid_factor.
    """
    # Use float32 for quantiles/stability, keep device
    x = g_abs.to(dtype=torch.float32)
    r = int(x.numel())

    # Degenerate case: all (almost) identical => do nothing (mid_factor)
    if (x.max() - x.min()).abs().item() < 1e-12:
        gate = torch.full_like(sigma0, float(config.mid_factor))
        return sigma0 * gate, {
            "r": r,
            "mode": "smooth_abs",
            "degenerate": True,
            "gate_min": float(gate.min().item()),
            "gate_max": float(gate.max().item()),
        }

    # Robust span using quantiles tied to noise/core fractions
    q_lo = float(max(0.0, min(1.0, config.noise_frac)))
    q_hi = float(max(0.0, min(1.0, 1.0 - config.core_frac)))
    if q_hi <= q_lo:
        q_lo, q_hi = 0.25, 0.75

    lo = torch.quantile(x, q_lo)
    hi = torch.quantile(x, q_hi)
    scale = (hi - lo).clamp_min(1e-8)

    # Center point (default median)
    center_q = float(max(0.0, min(1.0, config.smooth_center_q)))
    center = torch.quantile(x, center_q)

    # Temperature in units of robust scale
    tau = (float(config.smooth_temperature) * scale).clamp_min(1e-8)

    # Shift mu: optionally enforce gate(center)=mid_factor (if feasible)
    mu = center
    if config.smooth_align_mid:
        sup = float(config.sup_factor)
        amp = float(config.amp_factor)
        mid = float(config.mid_factor)

        if amp > sup and (sup < mid < amp):
            p = (mid - sup) / (amp - sup)  # desired sigmoid output at x=center
            p = float(max(1e-4, min(1.0 - 1e-4, p)))  # avoid inf logit
            p_t = torch.tensor(p, device=x.device, dtype=torch.float32)
            logit = torch.log(p_t) - torch.log(1.0 - p_t)
            mu = center - tau * logit  # ensures gate(center)=mid

    sup_t = torch.tensor(float(config.sup_factor), device=x.device, dtype=torch.float32)
    amp_t = torch.tensor(float(config.amp_factor), device=x.device, dtype=torch.float32)

    gate = sup_t + (amp_t - sup_t) * torch.sigmoid((x - mu) / tau)
    gate = gate.to(dtype=sigma0.dtype)

    sigma_new = sigma0 * gate
    stats: Dict[str, Any] = {
        "r": r,
        "mode": "smooth_abs",
        "q_lo": q_lo,
        "q_hi": q_hi,
        "center_q": center_q,
        "lo": float(lo.item()),
        "hi": float(hi.item()),
        "center": float(center.item()),
        "mu": float(mu.item()),
        "tau": float(tau.item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "degenerate": False,
    }
    return sigma_new, stats


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
        g_neg = -torch.relu(-g)  # <= 0
        if config.pos_power != 1.0:
            g_pos = g_pos.pow(float(config.pos_power))
        g_eff = float(config.eta_suppress) * g_pos + float(config.eta_enhance) * g_neg

        if config.update_mode == "additive":
            return sigma0 - g_eff
        else:
            return sigma0 * torch.exp(-g_eff)
    else:
        if config.update_mode == "additive":
            return sigma0 - float(config.eta) * g
        else:
            return sigma0 * torch.exp(-float(config.eta) * g)


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
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply spectral edit to singular values based on gradient sensitivity.

    Args:
        sigma0: Original singular values, shape [r]
        g_sigma: Accumulated gradient w.r.t. sigma, shape [r]
        config: EditConfig with editing parameters

    Returns:
        (sigma_new, stats_dict)
    """
    if config is None:
        config = EditConfig()

    g = g_sigma.clone()
    g_abs = g.abs()

    # Normalize gradient (abs + signed)
    g_abs_norm = normalize_gradient(g_abs, config.grad_norm)
    g_norm = normalize_gradient(g, config.grad_norm)

    stats: Dict[str, Any] = {
        "r": int(sigma0.numel()),
        "mode": config.mode,
        "g_abs_mean": float(g_abs.mean().item()),
        "g_abs_max": float(g_abs.max().item()),
    }

    if config.mode == "abs_select":
        sigma_new, k_core, k_noise = apply_abs_select(sigma0, g_abs_norm, config)
        stats.update({
            "k_core": int(k_core),
            "k_noise": int(k_noise),
            "amp_factor": float(config.amp_factor),
            "sup_factor": float(config.sup_factor),
            "mid_factor": float(config.mid_factor),
        })

    elif config.mode == "smooth_abs":
        sigma_new, smooth_stats = apply_smooth_abs(sigma0, g_abs_norm, config)
        stats.update(smooth_stats)
        stats["k_core"] = None
        stats["k_noise"] = None

    else:  # "gd"
        sigma_new = apply_gd_update(sigma0, g_norm, config)
        stats["k_core"] = None
        stats["k_noise"] = None

    # Clip minimum
    sigma_new = sigma_new.clamp_min(float(config.sigma_clip_min))

    # Preserve energy
    sigma_new = preserve_spectral_energy(sigma0, sigma_new, config.preserve_energy)

    stats["sigma0_sum"] = float(sigma0.sum().item())
    stats["sigma_new_sum"] = float(sigma_new.sum().item())
    stats["sigma0_top1"] = float(sigma0.max().item())
    stats["sigma_new_top1"] = float(sigma_new.max().item())

    return sigma_new, stats
