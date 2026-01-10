#!/bin/bash
# Example: Edit a MetaMath LoRA adapter on GSM8K

# Edit-only (no vLLM evaluation)
CUDA_VISIBLE_DEVICES=0 python -m lora_spectral_edit edit \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_path LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128 \
  --out_dir ./metamath_edited \
  --mode abs_select \
  --target_modules down_proj o_proj \
  --core_frac 0.2 \
  --noise_frac 0.2 \
  --amp_factor 1.25 \
  --sup_factor 0.80 \
  --preserve_energy l1 \
  --calib_samples 32 \
  --calib_batch_size 2 \
  --seed 42

echo "Edited adapter saved to: ./metamath_edited"
