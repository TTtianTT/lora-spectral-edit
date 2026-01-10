# Experiments

This folder is used by the multi-GPU runner in `scripts/launch_multi_gpu.sh`.

## Quick start

1. Generate a configs JSONL (writes `experiments/configs_full.jsonl`):

```bash
python scripts/prefetch_adapters.py
```

2. Launch jobs across GPUs:

```bash
bash scripts/launch_multi_gpu.sh --gpus "0,1,2,3" --configs experiments/configs_full.jsonl
```

Each line in the JSONL is one run. See `scripts/run_one.sh` for supported fields.

