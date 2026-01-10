# LoRA Spectral Edit

Sensitivity-based spectral editing for LoRA adapters. This tool uses gradient sensitivity analysis to identify and manipulate important singular value directions in LoRA weight matrices.

## Overview

LoRA Spectral Edit performs the following:

1. **SVD Decomposition**: Decomposes LoRA weight matrices (ΔW = B @ A) into U, Σ, V components
2. **Gradient Calibration**: Runs a small calibration set to compute gradient sensitivity for each singular direction
3. **Spectral Editing**: Modifies singular values based on sensitivity scores:
   - **Core features** (high |g|): Amplify
   - **Noise features** (low |g|): Suppress
4. **Reconstruction**: Rebuilds LoRA matrices from edited singular values

## Installation

### Basic (editing only)

```bash
pip install -e .
```

### With evaluation support (requires vLLM + CUDA GPU)

```bash
pip install -e ".[eval]"
```

## Quick Start

### 1. Smoke Test (verify installation)

```bash
python -m lora_spectral_edit.smoke_test
```

This runs basic tests on the SVD and editing logic without requiring a GPU.

### 2. Edit a LoRA Adapter

```bash
# Edit-only (no evaluation)
python -m lora_spectral_edit edit \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_path LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128 \
  --out_dir ./edited_lora \
  --mode abs_select \
  --core_frac 0.2 \
  --noise_frac 0.2 \
  --amp_factor 1.25 \
  --sup_factor 0.80 \
  --calib_samples 32
```

### 3. Edit + Evaluate (requires vLLM)

```bash
python -m lora_spectral_edit edit \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_path LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128 \
  --out_dir ./edited_lora \
  --mode abs_select \
  --eval_gsm8k \
  --eval_fewshot 5
```

### 4. Standalone Evaluation (requires vLLM)

```bash
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./edited_lora \
  --fewshot 5 \
  --out_json results.json
```

## CLI Reference

### `edit` Command

Edits a LoRA adapter using spectral manipulation.

**Required arguments:**
- `--base_model`: HuggingFace model ID for base model
- `--lora_path`: Path or HuggingFace ID for LoRA adapter
- `--out_dir`: Output directory for edited adapter

**Editing parameters:**
- `--mode`: Edit mode (`abs_select` or `gd`), default: `abs_select`
- `--target_modules`: Module names to edit, default: `down_proj o_proj`
- `--layer_min`, `--layer_max`: Layer range to edit

**abs_select mode (recommended):**
- `--core_frac`: Fraction of dims to amplify (default: 0.2)
- `--noise_frac`: Fraction of dims to suppress (default: 0.2)
- `--amp_factor`: Amplification factor (default: 1.25)
- `--sup_factor`: Suppression factor (default: 0.80)
- `--mid_factor`: Middle dims factor (default: 1.0)

**gd mode:**
- `--eta`: Learning rate (default: 0.2)
- `--update_mode`: `additive` or `multiplicative`
- `--asymmetric_update`: Use different step sizes for positive/negative gradients

**Calibration:**
- `--calib_samples`: Number of calibration samples (default: 32)
- `--calib_batch_size`: Batch size for calibration (default: 2)

**Evaluation (optional, requires vLLM):**
- `--eval_gsm8k`: Run GSM8K evaluation after editing
- `--eval_fewshot`: Number of few-shot examples (default: 5)

### `eval` Command

Evaluates a LoRA adapter on GSM8K. Requires vLLM.

**Required arguments:**
- `--base_model`: HuggingFace model ID for base model
- `--lora_dir`: Path to LoRA adapter directory

**Optional:**
- `--fewshot`: Number of few-shot examples (default: 5)
- `--max_samples`: Max test samples, -1 for all (default: -1)
- `--out_json`: Output file for metrics

## Expected Outputs

After running `edit`:

```
./edited_lora/
├── adapter_config.json      # Copied from original
├── adapter_model.safetensors  # Edited weights
├── spectral_edit_meta.json  # Editing metadata and per-module stats
└── metrics_gsm8k.json       # (if --eval_gsm8k) Evaluation results
```

The `spectral_edit_meta.json` contains:
- Editing configuration used
- Per-module statistics (sigma changes, k_core, k_noise, etc.)

## Editing Modes

### abs_select (Recommended)

Uses absolute gradient magnitude |g_sigma| as sensitivity score:
- **High |g|** → Core features → Amplify by `amp_factor`
- **Low |g|** → Noise features → Suppress by `sup_factor`
- **Middle** → Keep or scale by `mid_factor`

This is NOT gradient descent. It's feature selection based on sensitivity.

### random_index (Control)

Uses the same core/noise counts and amp/sup factors as `abs_select`,
but selects indices uniformly at random (seeded).

### gd (Gradient Descent)

Uses signed gradient for updates:
- `additive`: σ_new = σ - η * g
- `multiplicative`: σ_new = σ * exp(-η * g)

Can use asymmetric step sizes for positive/negative gradients.

## Experiments

### Multi-GPU full eval runs

1) Prefetch adapters and generate configs:

```bash
python scripts/prefetch_adapters.py
```

2) Launch runs (tmux or nohup recommended):

```bash
tmux new -s lora_runs
bash scripts/launch_multi_gpu.sh --gpus "0,1,2,3,4,5,7"
```

Or:

```bash
nohup bash scripts/launch_multi_gpu.sh --gpus "0,1,2,3,4,5,7" > _runs/<timestamp>/logs/launch.log 2>&1 &
```

3) Collect summaries:

```bash
python scripts/collect_results.py --run-root _runs/<timestamp>
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA GPU (for gradient calibration)
- transformers ≥ 4.35
- peft ≥ 0.6
- datasets
- safetensors
- huggingface_hub
- vLLM ≥ 0.4.0 (optional, for evaluation)

## License

MIT
