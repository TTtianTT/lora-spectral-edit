"""
SVD utilities for LoRA spectral decomposition and reconstruction.
"""

from typing import Tuple

import torch


def lowrank_svd_from_ba(
    B: torch.Tensor, A: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute compact SVD of deltaW = B @ A without explicitly forming deltaW.

    Uses QR-based decomposition for numerical stability:
        B = Qb @ Rb,  A^T = Qa @ Ra
        M = Rb @ Ra^T
        M = Um @ S @ Vm^T
        U = Qb @ Um,  V = Qa @ Vm

    Args:
        B: Tensor of shape [d_out, r]
        A: Tensor of shape [r, d_in]

    Returns:
        U: Left singular vectors, shape [d_out, r]
        S: Singular values, shape [r]
        Vh: Right singular vectors (transposed), shape [r, d_in]
        V: Right singular vectors, shape [d_in, r]
    """
    # Use float32 for numeric stability
    B = B.float()
    A = A.float()

    Qb, Rb = torch.linalg.qr(B, mode="reduced")        # [dout, r], [r, r]
    Qa, Ra = torch.linalg.qr(A.t(), mode="reduced")    # [din, r], [r, r]

    M = Rb @ Ra.t()                                    # [r, r]
    Um, S, Vmt = torch.linalg.svd(M, full_matrices=False)

    U = Qb @ Um                                        # [dout, r]
    V = Qa @ Vmt.t()                                   # [din, r]
    Vh = V.t().contiguous()                            # [r, din]
    return U, S, Vh, V


def rebuild_ba_from_uv_sigma(
    U: torch.Tensor, Vh: torch.Tensor, sigma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct LoRA factors B and A from SVD components.

    Given deltaW = U @ diag(sigma) @ Vh, builds:
        B = U @ diag(sqrt(sigma))
        A = diag(sqrt(sigma)) @ Vh

    Such that B @ A = U @ diag(sigma) @ Vh.

    Args:
        U: Left singular vectors, shape [d_out, r]
        Vh: Right singular vectors (transposed), shape [r, d_in]
        sigma: Singular values, shape [r]

    Returns:
        B: New LoRA B matrix, shape [d_out, r]
        A: New LoRA A matrix, shape [r, d_in]
    """
    sigma = sigma.clamp_min(0.0)
    sroot = torch.sqrt(sigma)
    B = U * sroot.unsqueeze(0)        # [d_out, r]
    A = sroot.unsqueeze(1) * Vh       # [r, d_in]
    return B, A
