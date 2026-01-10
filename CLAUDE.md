# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA Spectral Edit is a tool for sensitivity-based spectral editing of LoRA adapters. It uses SVD decomposition and gradient sensitivity analysis to identify and manipulate important singular value directions in LoRA weight matrices.

## Commands

```bash
# Install in development mode
pip install -e .

# Install with vLLM evaluation support
pip install -e ".[eval]"

# Run smoke test (no GPU required)
python -m lora_spectral_edit.smoke_test

# Edit a LoRA adapter
python -m lora_spectral_edit edit \
  --base_model <model_id> \
  --lora_path <lora_path> \
  --out_dir <output_dir>

# Evaluate on GSM8K (requires vLLM)
python -m lora_spectral_edit eval \
  --base_model <model_id> \
  --lora_dir <lora_dir>
```

## Architecture

```
src/lora_spectral_edit/
├── cli.py          # Main CLI with edit/eval subcommands
├── svd.py          # SVD decomposition (lowrank_svd_from_ba) & reconstruction
├── hooks.py        # Gradient accumulation hooks (ModuleSpec, HookContext)
├── edit.py         # Spectral editing strategies (EditConfig, apply_spectral_edit)
├── io.py           # LoRA I/O utilities (load/save state dict, parse keys)
├── eval_gsm8k.py   # Optional vLLM evaluation (requires vLLM)
└── smoke_test.py   # Installation verification tests
```

## Key Data Flow

1. **Load**: `io.load_lora_state_dict()` loads LoRA weights, `io.parse_lora_ab_key()` identifies A/B pairs
2. **Decompose**: `svd.lowrank_svd_from_ba(B, A)` → U, S, Vh, V (QR-based, numerically stable)
3. **Calibrate**: `hooks.register_sigma_hooks()` accumulates g_sigma via forward/backward hooks
4. **Edit**: `edit.apply_spectral_edit(sigma, g_sigma, config)` modifies singular values
5. **Reconstruct**: `svd.rebuild_ba_from_uv_sigma(U, Vh, sigma_new)` → B_new, A_new
6. **Save**: `io.save_lora_state_dict()` writes edited adapter

## Editing Modes

- **abs_select** (default): Uses |g_sigma| as sensitivity score. High |g| dims are amplified, low |g| dims are suppressed.
- **gd**: Gradient descent style updates with optional asymmetric step sizes.

## Dependencies

Core: torch, transformers, peft, datasets, safetensors, huggingface_hub

Optional: vllm (for evaluation)
