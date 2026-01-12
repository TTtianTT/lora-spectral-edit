#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

CONFIG_LINE="${1:-}"
GPU_ID="${2:-}"

if [[ -z "${CONFIG_LINE}" || -z "${GPU_ID}" ]]; then
  echo "Usage: scripts/run_one.sh '<json_line>' <gpu_id>" >&2
  exit 1
fi

parse_vars="$(
python - <<'PY' "${CONFIG_LINE}"
import json
import shlex
import sys

cfg = json.loads(sys.argv[1])

def emit(key, value):
    if value is None:
        value = ""
    print(f"{key}={shlex.quote(str(value))}")

emit("TASK", cfg.get("task", ""))
emit("BASE_MODEL", cfg.get("base_model", ""))
emit("LORA_REPO_ID", cfg.get("lora_repo_id", ""))
emit("LORA_LOCAL_PATH", cfg.get("lora_local_path", ""))
emit("EDIT_MODE", cfg.get("edit_mode", ""))
emit("SEED", int(cfg.get("seed", 0)))
emit("CALIB_SAMPLES", int(cfg.get("calib_samples", 0)))
emit("CALIB_BATCH_SIZE", int(cfg.get("calib_batch_size", 1)))
emit("OUT_DIR", cfg.get("out_dir", ""))
emit("KEEP_ADAPTER", str(bool(cfg.get("keep_adapter", False))))
emit("NOTES", cfg.get("notes", ""))
emit("RUN_ROOT", cfg.get("run_root", ""))

emit("CORE_FRAC", cfg.get("core_frac", 0.2))
emit("NOISE_FRAC", cfg.get("noise_frac", 0.2))
emit("AMP_FACTOR", cfg.get("amp_factor", 1.25))
emit("SUP_FACTOR", cfg.get("sup_factor", 0.80))
emit("PRESERVE_ENERGY", cfg.get("preserve_energy", "l1"))
emit("MID_FACTOR", cfg.get("mid_factor", 1.0))

emit("SMOOTH_TEMPERATURE", cfg.get("smooth_temperature", 0.35))
emit("SMOOTH_CENTER_Q", cfg.get("smooth_center_q", 0.5))
emit("SMOOTH_ALIGN_MID", str(bool(cfg.get("smooth_align_mid", True))))

# z_score parameters
emit("Z_HIGH", cfg.get("z_high", 1.0))
emit("Z_LOW", cfg.get("z_low", -0.5))
emit("Z_TAU", cfg.get("z_tau", 0.2))
emit("Z_FALLBACK_STD", cfg.get("z_fallback_std", 1e-6))

# robust_z parameters
emit("ROBUST_Z_HIGH", cfg.get("robust_z_high", 1.0))
emit("ROBUST_Z_LOW", cfg.get("robust_z_low", -0.5))
emit("ROBUST_Z_TAU", cfg.get("robust_z_tau", 0.2))
emit("ROBUST_FALLBACK_SIGMA", cfg.get("robust_fallback_sigma", 1e-6))

# Spectral gate sweep parameters
emit("SOFT_TEMPERATURE", cfg.get("soft_temperature", ""))
emit("SOFT_PIVOT_MODE", cfg.get("soft_pivot_mode", ""))

emit("EVAL_FEWSHOT", int(cfg.get("eval_fewshot", 5)))
emit("EVAL_MAX_SAMPLES", int(cfg.get("eval_max_samples", -1)))
emit("EVAL_TEMPERATURE", cfg.get("eval_temperature", 0.0))
emit("EVAL_MAX_TOKENS", int(cfg.get("eval_max_tokens", 512)))
emit("EVAL_PROFILE", cfg.get("eval_profile", ""))

target_modules = cfg.get("target_modules", [])
emit("TARGET_MODULES", " ".join(target_modules))
PY
)"

eval "${parse_vars}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "[Error] out_dir missing in config." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [[ -z "${EVAL_PROFILE}" ]]; then
  if [[ "${TASK}" == "gsm8k_full" ]]; then
    EVAL_PROFILE="paper_math"
  elif [[ "${TASK}" == "humaneval_full" ]]; then
    EVAL_PROFILE="paper_code_main"
  fi
fi

EVAL_PROFILE_ARGS=()
if [[ -n "${EVAL_PROFILE}" ]]; then
  EVAL_PROFILE_ARGS+=(--eval_profile "${EVAL_PROFILE}")
fi

python - <<'PY' "${CONFIG_LINE}" "${OUT_DIR}"
import json
import os
import sys

cfg = json.loads(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
PY

LOG_FILE="${OUT_DIR}/stdout.log"
ADAPTER_REF="${OUT_DIR}/adapter_ref.txt"
TMP_ADAPTER_DIR="${OUT_DIR}/tmp_adapter"
EVAL_TMP_DIR="${OUT_DIR}/humaneval_tmp"

cleanup() {
  if [[ -f "${LOG_FILE}" ]]; then
    gzip -f "${LOG_FILE}"
  fi
}
trap cleanup EXIT

{
  echo "[Run] task=${TASK} edit_mode=${EDIT_MODE} seed=${SEED} gpu=${GPU_ID}"
  echo "[Run] notes=${NOTES}"
  echo "[Run] lora_repo_id=${LORA_REPO_ID}"
  echo "[Run] lora_local_path=${LORA_LOCAL_PATH}"

  echo "lora_repo_id: ${LORA_REPO_ID}" > "${ADAPTER_REF}"
  echo "lora_local_path: ${LORA_LOCAL_PATH}" >> "${ADAPTER_REF}"

  ADAPTER_PATH="${LORA_LOCAL_PATH}"
  if [[ "${EDIT_MODE}" != "baseline" && "${EDIT_MODE}" != "spectral_gate" ]]; then
    # spectral_gate mode uses spectral_gate_humaneval_greedy.py which handles edit+eval together
    rm -rf "${TMP_ADAPTER_DIR}"
    EDIT_ARGS=()
    if [[ "${SMOOTH_ALIGN_MID}" == "False" || "${SMOOTH_ALIGN_MID}" == "false" ]]; then
      EDIT_ARGS+=(--no_smooth_align_mid)
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m lora_spectral_edit edit \
      --base_model "${BASE_MODEL}" \
      --lora_path "${LORA_LOCAL_PATH}" \
      --out_dir "${TMP_ADAPTER_DIR}" \
      --mode "${EDIT_MODE}" \
      --target_modules ${TARGET_MODULES} \
      --core_frac "${CORE_FRAC}" \
      --noise_frac "${NOISE_FRAC}" \
      --amp_factor "${AMP_FACTOR}" \
      --sup_factor "${SUP_FACTOR}" \
      --mid_factor "${MID_FACTOR}" \
      --smooth_temperature "${SMOOTH_TEMPERATURE}" \
      --smooth_center_q "${SMOOTH_CENTER_Q}" \
      --z_high "${Z_HIGH}" \
      --z_low "${Z_LOW}" \
      --z_tau "${Z_TAU}" \
      --z_fallback_std "${Z_FALLBACK_STD}" \
      --robust_z_high "${ROBUST_Z_HIGH}" \
      --robust_z_low "${ROBUST_Z_LOW}" \
      --robust_z_tau "${ROBUST_Z_TAU}" \
      --robust_fallback_sigma "${ROBUST_FALLBACK_SIGMA}" \
      "${EDIT_ARGS[@]}" \
      --preserve_energy "${PRESERVE_ENERGY}" \
      --calib_samples "${CALIB_SAMPLES}" \
      --calib_batch_size "${CALIB_BATCH_SIZE}" \
      --seed "${SEED}"
    ADAPTER_PATH="${TMP_ADAPTER_DIR}"
  fi
  echo "adapter_path: ${ADAPTER_PATH}" >> "${ADAPTER_REF}"

  if [[ "${TASK}" == "gsm8k_full" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m lora_spectral_edit eval \
      --base_model "${BASE_MODEL}" \
      --lora_dir "${ADAPTER_PATH}" \
      "${EVAL_PROFILE_ARGS[@]}" \
      --fewshot "${EVAL_FEWSHOT}" \
      --max_samples "${EVAL_MAX_SAMPLES}" \
      --temperature "${EVAL_TEMPERATURE}" \
      --max_tokens "${EVAL_MAX_TOKENS}" \
      --seed "${SEED}" \
      --out_json "${OUT_DIR}/metrics.json"
  elif [[ "${TASK}" == "humaneval_full" ]]; then
    rm -rf "${EVAL_TMP_DIR}"
    if [[ "${EDIT_MODE}" == "spectral_gate" ]]; then
      # Use spectral_gate_humaneval_greedy.py for spectral_gate sweep runs
      SPECTRAL_GATE_ARGS=()
      if [[ -n "${SOFT_TEMPERATURE}" ]]; then
        SPECTRAL_GATE_ARGS+=(--soft_temperature "${SOFT_TEMPERATURE}")
      fi
      if [[ -n "${SOFT_PIVOT_MODE}" ]]; then
        SPECTRAL_GATE_ARGS+=(--soft_pivot_mode "${SOFT_PIVOT_MODE}")
      fi
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python experiments/spectral_gate_humaneval_greedy.py \
        --base_model_id "${BASE_MODEL}" \
        --lora_path "${LORA_LOCAL_PATH}" \
        --amp_factor "${AMP_FACTOR}" \
        --sup_factor "${SUP_FACTOR}" \
        "${SPECTRAL_GATE_ARGS[@]}" \
        --calib_samples "${CALIB_SAMPLES}" \
        --calib_batch_size "${CALIB_BATCH_SIZE}" \
        --seed "${SEED}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --max_tokens "${EVAL_MAX_TOKENS}" \
        --out_dir "${OUT_DIR}" \
        --out_json "${OUT_DIR}/metrics.json"
    else
      CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m lora_spectral_edit.eval_humaneval \
        --base_model_id "${BASE_MODEL}" \
        --lora_dir "${ADAPTER_PATH}" \
        "${EVAL_PROFILE_ARGS[@]}" \
        --max_samples "${EVAL_MAX_SAMPLES}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --max_tokens "${EVAL_MAX_TOKENS}" \
        --seed "${SEED}" \
        --out_json "${OUT_DIR}/metrics.json" \
        --out_dir "${EVAL_TMP_DIR}"
    fi
    rm -rf "${EVAL_TMP_DIR}"
  else
    echo "[Error] Unknown task: ${TASK}" >&2
    exit 1
  fi

  python - <<'PY' "${OUT_DIR}" "${ADAPTER_REF}"
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
adapter_ref = Path(sys.argv[2])
cfg_path = out_dir / "config.json"
metrics_path = out_dir / "metrics.json"
tmp_adapter = out_dir / "tmp_adapter"

if not cfg_path.exists() or not metrics_path.exists():
    sys.exit(0)

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
edit_mode = cfg.get("edit_mode")
if edit_mode == "baseline":
    if tmp_adapter.exists():
        shutil.rmtree(tmp_adapter, ignore_errors=True)
    sys.exit(0)

keep_adapter = bool(cfg.get("keep_adapter", False))
if not tmp_adapter.exists():
    sys.exit(0)

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

def extract_score(task: str, data: dict):
    if task.startswith("gsm8k") and "acc" in data:
        return float(data["acc"])
    if "edited" in data and isinstance(data["edited"], dict) and "pass@1" in data["edited"]:
        return float(data["edited"]["pass@1"])
    if "pass@1" in data:
        return float(data["pass@1"])
    return None

score = extract_score(cfg.get("task", ""), metrics)

if not keep_adapter:
    shutil.rmtree(tmp_adapter, ignore_errors=True)
    sys.exit(0)

run_root = cfg.get("run_root")
if not run_root:
    parts = out_dir.parts
    if "_runs" in parts:
        idx = parts.index("_runs")
        run_root = str(Path(*parts[: idx + 2]))
    else:
        run_root = str(out_dir.parent)

lora_slug = cfg.get("lora_repo_id", "").split("/")[-1]
task = cfg.get("task", "unknown")
group_dir = Path(run_root) / "kept_adapters" / task / lora_slug / edit_mode
group_dir.mkdir(parents=True, exist_ok=True)
lock_path = group_dir / "keep.lock"

def better(new, old):
    if old is None:
        return True
    if new is None:
        return False
    return new > old

with open(lock_path, "w", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    best_path = group_dir / "best.json"
    prev_score = None
    if best_path.exists():
        try:
            prev = json.loads(best_path.read_text(encoding="utf-8"))
            prev_score = prev.get("score")
        except Exception:
            prev_score = None

    if better(score, prev_score):
        kept_dir = group_dir / "adapter"
        if kept_dir.exists():
            shutil.rmtree(kept_dir, ignore_errors=True)
        shutil.move(str(tmp_adapter), str(kept_dir))
        best = {
            "score": score,
            "out_dir": str(out_dir),
            "seed": cfg.get("seed"),
        }
        best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        with adapter_ref.open("a", encoding="utf-8") as f:
            f.write(f"kept_adapter: {kept_dir}\n")
    else:
        shutil.rmtree(tmp_adapter, ignore_errors=True)
PY
} > "${LOG_FILE}" 2>&1
