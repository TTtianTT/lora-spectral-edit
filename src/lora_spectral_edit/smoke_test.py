"""
Smoke test to verify the package installation and basic functionality.

This test does NOT require a GPU or loading actual models. It only tests
the core SVD and editing logic with synthetic data.
"""

import torch

from .svd import lowrank_svd_from_ba, rebuild_ba_from_uv_sigma
from .edit import EditConfig, apply_spectral_edit


def test_svd_roundtrip():
    """Test that SVD decomposition and reconstruction are consistent."""
    print("[Test] SVD roundtrip...")

    # Create synthetic LoRA matrices
    d_out, r, d_in = 64, 8, 128
    B = torch.randn(d_out, r)
    A = torch.randn(r, d_in)

    # Original deltaW
    deltaW_orig = B @ A

    # Decompose
    U, S, Vh, V = lowrank_svd_from_ba(B, A)

    # Check shapes
    assert U.shape == (d_out, r), f"U shape mismatch: {U.shape}"
    assert S.shape == (r,), f"S shape mismatch: {S.shape}"
    assert Vh.shape == (r, d_in), f"Vh shape mismatch: {Vh.shape}"
    assert V.shape == (d_in, r), f"V shape mismatch: {V.shape}"

    # Check singular values are non-negative
    assert (S >= 0).all(), "Singular values should be non-negative"

    # Reconstruct
    B_new, A_new = rebuild_ba_from_uv_sigma(U, Vh, S)
    deltaW_new = B_new @ A_new

    # Check reconstruction error
    error = torch.norm(deltaW_orig - deltaW_new) / torch.norm(deltaW_orig)
    assert error < 1e-5, f"Reconstruction error too large: {error}"

    print(f"   Reconstruction error: {error:.2e}")
    print("   PASSED")


def test_spectral_edit_abs_select():
    """Test abs_select editing mode."""
    print("[Test] Spectral edit (abs_select mode)...")

    r = 16
    sigma0 = torch.linspace(1.0, 0.1, r)  # Decreasing singular values
    g_sigma = torch.randn(r)  # Random gradients

    config = EditConfig(
        mode="abs_select",
        core_frac=0.25,  # Top 4 dims
        noise_frac=0.25,  # Bottom 4 dims
        amp_factor=1.5,
        sup_factor=0.5,
        mid_factor=1.0,
        preserve_energy="l1",
    )

    sigma_new, stats = apply_spectral_edit(sigma0, g_sigma, config)

    # Check output shape
    assert sigma_new.shape == sigma0.shape, "Output shape mismatch"

    # Check stats
    assert "k_core" in stats, "Missing k_core in stats"
    assert "k_noise" in stats, "Missing k_noise in stats"
    assert stats["k_core"] == 4, f"Expected k_core=4, got {stats['k_core']}"
    assert stats["k_noise"] == 4, f"Expected k_noise=4, got {stats['k_noise']}"

    # Check energy preservation (L1)
    orig_sum = sigma0.sum()
    new_sum = sigma_new.sum()
    energy_diff = abs(orig_sum - new_sum) / orig_sum
    assert energy_diff < 1e-5, f"Energy not preserved: {energy_diff}"

    print(f"   k_core={stats['k_core']}, k_noise={stats['k_noise']}")
    print(f"   Energy preservation error: {energy_diff:.2e}")
    print("   PASSED")


def test_spectral_edit_gd():
    """Test gradient descent editing mode."""
    print("[Test] Spectral edit (gd mode)...")

    r = 16
    sigma0 = torch.ones(r)
    g_sigma = torch.randn(r)

    config = EditConfig(
        mode="gd",
        eta=0.1,
        update_mode="multiplicative",
        asymmetric_update=False,
        preserve_energy="none",
    )

    sigma_new, stats = apply_spectral_edit(sigma0, g_sigma, config)

    # Check output shape
    assert sigma_new.shape == sigma0.shape, "Output shape mismatch"

    # In multiplicative mode: sigma_new = sigma0 * exp(-eta * g)
    # Values should have changed
    assert not torch.allclose(sigma_new, sigma0), "Sigma should have changed"

    print("   PASSED")


def test_io_imports():
    """Test that IO module imports work."""
    print("[Test] IO module imports...")

    from .io import (
        parse_lora_ab_key,
        layer_idx_from_module_prefix,
    )

    # Test key parsing
    key = "base_model.model.layers.5.mlp.down_proj.lora_A.default.weight"
    result = parse_lora_ab_key(key)
    assert result is not None, "Failed to parse key"
    prefix, which, adapter = result
    assert which == "A", f"Expected 'A', got '{which}'"
    assert adapter == "default", f"Expected 'default', got '{adapter}'"

    # Test layer extraction
    layer_idx = layer_idx_from_module_prefix(prefix)
    assert layer_idx == 5, f"Expected layer 5, got {layer_idx}"

    print("   PASSED")


def run_smoke_test():
    """Run all smoke tests."""
    print("=" * 60)
    print("LoRA Spectral Edit - Smoke Test")
    print("=" * 60)
    print()

    try:
        test_svd_roundtrip()
        test_spectral_edit_abs_select()
        test_spectral_edit_gd()
        test_io_imports()

        print()
        print("=" * 60)
        print("All smoke tests PASSED!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"FAILED: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    import sys
    success = run_smoke_test()
    sys.exit(0 if success else 1)
