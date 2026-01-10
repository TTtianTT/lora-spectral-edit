#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
CONFIGS_FILE="${ROOT_DIR}/experiments/configs_full.jsonl"
GPUS="0,1,2,3,4,5,7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --configs)
      CONFIGS_FILE="$2"
      shift 2
      ;;
    *)
      echo "Usage: scripts/launch_multi_gpu.sh [--gpus \"0,1,2\"] [--configs path]" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIGS_FILE}" ]]; then
  echo "[Error] Configs file not found: ${CONFIGS_FILE}" >&2
  exit 1
fi

RUNS_DIR="${ROOT_DIR}/_runs"
mkdir -p "${RUNS_DIR}"

avail_kb=$(df -Pk "${RUNS_DIR}" | awk 'NR==2 {print $4}')
if [[ "${avail_kb}" -lt $((50 * 1024 * 1024)) ]]; then
  echo "[Error] Free space below 50GB in filesystem containing ${RUNS_DIR}" >&2
  df -h "${RUNS_DIR}"
  exit 1
fi

RUN_ROOT="$(
python - <<'PY' "${CONFIGS_FILE}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        cfg = json.loads(line)
        run_root = cfg.get("run_root")
        if run_root:
            print(run_root)
            sys.exit(0)
        out_dir = cfg.get("out_dir", "")
        if "_runs" in out_dir:
            parts = Path(out_dir).parts
            idx = parts.index("_runs")
            if idx + 1 < len(parts):
                print(str(Path(*parts[: idx + 2])))
                sys.exit(0)
print("")
PY
)"

if [[ -z "${RUN_ROOT}" ]]; then
  echo "[Error] Could not infer run_root from configs." >&2
  exit 1
fi

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LAUNCH_LOG="${LOG_DIR}/launch.log"
echo "[Launch] configs=${CONFIGS_FILE} gpus=${GPUS}" >> "${LAUNCH_LOG}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
QUEUE_DIR="${LOG_DIR}/queues"
mkdir -p "${QUEUE_DIR}"

for gpu in "${GPU_LIST[@]}"; do
  : > "${QUEUE_DIR}/gpu_${gpu}.jsonl"
done

idx=0
while IFS= read -r line; do
  if [[ -z "${line}" ]]; then
    continue
  fi
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  echo "${line}" >> "${QUEUE_DIR}/gpu_${gpu}.jsonl"
  idx=$((idx + 1))
done < "${CONFIGS_FILE}"

for gpu in "${GPU_LIST[@]}"; do
  WORKER_LOG="${LOG_DIR}/worker_gpu${gpu}.log"
  QUEUE_FILE="${QUEUE_DIR}/gpu_${gpu}.jsonl"
  (
    while IFS= read -r line; do
      if [[ -z "${line}" ]]; then
        continue
      fi
      ts="$(date +%Y-%m-%dT%H:%M:%S)"
      echo "[${ts}] start gpu=${gpu}" >> "${WORKER_LOG}"
      if "${ROOT_DIR}/scripts/run_one.sh" "${line}" "${gpu}"; then
        ts_done="$(date +%Y-%m-%dT%H:%M:%S)"
        echo "[${ts_done}] done gpu=${gpu}" >> "${WORKER_LOG}"
      else
        ts_fail="$(date +%Y-%m-%dT%H:%M:%S)"
        echo "[${ts_fail}] failed gpu=${gpu}" >> "${WORKER_LOG}"
      fi
    done < "${QUEUE_FILE}"
  ) &
done

wait
echo "[Launch] All jobs completed." >> "${LAUNCH_LOG}"
