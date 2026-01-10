#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/_smoke/${TIMESTAMP}"
ORIG_LORA_DIR="${LORA_DIR:-${ROOT_DIR}/_smoke/20260110_175548/original_lora}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-2-7b-hf}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${ORIG_LORA_DIR}/adapter_config.json" ]]; then
  echo "Missing adapter_config.json in ${ORIG_LORA_DIR}. Set LORA_DIR to a local adapter dir." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m pip install -e "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES=0 python -m lora_spectral_edit eval \
  --base_model "${BASE_MODEL}" \
  --lora_dir "${ORIG_LORA_DIR}" \
  --fewshot 1 \
  --max_samples 20 \
  --out_json "${OUT_DIR}/baseline_metrics.json"

CUDA_VISIBLE_DEVICES=0 python -m lora_spectral_edit edit \
  --base_model "${BASE_MODEL}" \
  --lora_path "${ORIG_LORA_DIR}" \
  --out_dir "${OUT_DIR}/edited_lora" \
  --mode abs_select \
  --target_modules down_proj o_proj \
  --core_frac 0.2 \
  --noise_frac 0.2 \
  --amp_factor 1.25 \
  --sup_factor 0.80 \
  --preserve_energy l1 \
  --calib_samples 8 \
  --calib_batch_size 2 \
  --seed 42

CUDA_VISIBLE_DEVICES=0 python -m lora_spectral_edit eval \
  --base_model "${BASE_MODEL}" \
  --lora_dir "${OUT_DIR}/edited_lora" \
  --fewshot 1 \
  --max_samples 20 \
  --out_json "${OUT_DIR}/post_edit_metrics.json"

echo "Smoke test outputs saved to: ${OUT_DIR}"
