"""
Gradient accumulation hooks for computing sensitivity scores (g_sigma).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ModuleSpec:
    """Specification for a LoRA module to be edited."""
    module_prefix: str
    module: torch.nn.Module
    U: torch.Tensor          # [dout, r]
    V: torch.Tensor          # [din, r]
    Vh: torch.Tensor         # [r, din]
    sigma0: torch.Tensor     # [r]
    scaling: float
    adapter: Optional[str]


class HookContext:
    """
    Context for managing gradient accumulation hooks.

    Stores accumulated gradients and attention mask for proper token counting.
    """

    def __init__(self):
        self.attn_mask: Optional[torch.Tensor] = None  # [B, S]
        self.total_active_tokens: int = 0
        self.gsum: Dict[str, torch.Tensor] = {}  # module_prefix -> [r]

    def reset(self):
        """Reset all accumulated values."""
        self.attn_mask = None
        self.total_active_tokens = 0
        self.gsum = {}


# Global hook context
HOOK_CTX = HookContext()


def register_sigma_hooks(specs: Dict[str, ModuleSpec]):
    """
    Register forward and backward hooks to accumulate g_sigma.

    The gradient of loss with respect to sigma_k is computed efficiently as:
        g_sigma_k = scaling * sum_tokens <dL/dy, u_k> * <x, v_k>

    This avoids forming the full dL/dW matrix.

    Args:
        specs: Dictionary mapping module prefixes to ModuleSpec objects.

    Returns:
        List of hook handles (call .remove() on each to unregister).
    """
    handles = []

    def fwd_hook_factory(prefix: str):
        def _fwd_hook(module, inputs, output):
            x = inputs[0]
            if x is None:
                module.__x_cache = None
                return
            # Apply attention mask if available
            if HOOK_CTX.attn_mask is not None and x.dim() == 3:
                m = HOOK_CTX.attn_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
                x = x * m
            module.__x_cache = x.detach()
        return _fwd_hook

    def bwd_hook_factory(prefix: str):
        def _bwd_hook(module, grad_input, grad_output):
            gy = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            x = getattr(module, "__x_cache", None)
            if gy is None or x is None:
                return

            spec = specs[prefix]
            U = spec.U
            V = spec.V
            scaling = spec.scaling

            # Flatten batch and sequence dimensions
            if gy.dim() == 3:
                B, S, dout = gy.shape
                gy2 = gy.detach().reshape(-1, dout)
                x2 = x.reshape(-1, x.shape[-1])
                if HOOK_CTX.attn_mask is not None:
                    active = int(HOOK_CTX.attn_mask.sum().item())
                else:
                    active = B * S
                HOOK_CTX.total_active_tokens += active
            else:
                gy2 = gy.detach()
                x2 = x
                HOOK_CTX.total_active_tokens += gy2.shape[0]

            # Compute g_sigma_k = scaling * sum_n <dL/dy_n, u_k> * <x_n, v_k>
            a = (gy2.float() @ U.float())      # [N, r]
            b = (x2.float() @ V.float())       # [N, r]
            g = (a * b).sum(dim=0) * float(scaling)  # [r]

            if prefix not in HOOK_CTX.gsum:
                HOOK_CTX.gsum[prefix] = g.detach().cpu()
            else:
                HOOK_CTX.gsum[prefix] += g.detach().cpu()

            module.__x_cache = None
        return _bwd_hook

    for prefix, spec in specs.items():
        handles.append(spec.module.register_forward_hook(fwd_hook_factory(prefix)))

        if hasattr(spec.module, "register_full_backward_hook"):
            handles.append(spec.module.register_full_backward_hook(bwd_hook_factory(prefix)))
        else:
            handles.append(
                spec.module.register_backward_hook(
                    lambda m, gi, go, p=prefix: bwd_hook_factory(p)(m, gi, go)
                )
            )

    return handles


def remove_hooks(handles):
    """Remove all registered hooks."""
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
