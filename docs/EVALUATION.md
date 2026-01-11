# Evaluation Profiles

This document describes the evaluation profiles for benchmarking LoRA adapters on Math (GSM8K) and Code (HumanEval) tasks, matching the settings from the paper "LoRA Learns Less and Forgets Less".

## Quick Start

### GSM8K 5-shot Greedy Strict Match (Paper Settings)

```bash
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --eval_profile paper_math \
  --out_dir ./results
```

### HumanEval 50-sample Pass@1 (Paper Settings)

```bash
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --eval_profile paper_code_main \
  --out_dir ./results
```

Or using the dedicated HumanEval subcommand:

```bash
python -m lora_spectral_edit eval-humaneval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --eval_profile paper_code_main \
  --out_dir ./results
```

## Available Profiles

| Profile | Task | Description |
|---------|------|-------------|
| `paper_math` | GSM8K | 5-shot, greedy (temp=0), strict match accuracy |
| `paper_code_main` | HumanEval | 0-shot, 50 samples, temp=0.2, top_p=0.95, pass@1 |
| `greedy_math` | GSM8K | 5-shot, greedy (legacy default) |
| `greedy_code` | HumanEval | 0-shot, 1 sample, greedy (legacy default) |

## Paper-Matching Evaluation Details

### GSM8K (Math) - `paper_math`

Settings from "LoRA Learns Less and Forgets Less":

| Parameter | Value |
|-----------|-------|
| Dataset | GSM8K test split |
| Samples | 1319 (full test set) |
| Few-shot | 5 examples from train set |
| Temperature | 0.0 (greedy decoding) |
| Top-p | 1.0 (ignored with greedy) |
| Max tokens | 512 |
| Metric | Strict match accuracy |

**Answer Parsing**: The evaluator uses strict-match parsing, looking for the `#### <number>` format used in GSM8K answers. Falls back to the last number in the response if the format isn't found.

### HumanEval (Code) - `paper_code_main`

Settings from "LoRA Learns Less and Forgets Less":

| Parameter | Value |
|-----------|-------|
| Dataset | HumanEval (164 problems) |
| Few-shot | 0 (code completion from docstring) |
| Samples per problem | 50 |
| Temperature | 0.2 |
| Top-p | 0.95 (nucleus sampling) |
| Max tokens | 512 |
| Metric | pass@1 (unbiased estimator) |

**Pass@k Computation**: Uses the unbiased estimator from "Evaluating Large Language Models Trained on Code" (Chen et al., 2021):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

where `n` = number of samples, `c` = number of correct samples, `k` = 1.

## Output Files

Each evaluation run produces:

### GSM8K

```
<out_dir>/
├── eval_config.json          # Full evaluation config and results
├── gsm8k_predictions.jsonl   # Per-sample predictions and correctness
```

### HumanEval

```
<out_dir>/
├── eval_config.json              # Full evaluation config and results
├── samples_<adapter>.jsonl       # Generated code samples (HumanEval format)
├── samples_<adapter>_results.jsonl   # Per-sample pass/fail from execution
├── raw_generations_<adapter>.jsonl   # Raw generations with sample indices
├── per_problem_<adapter>.json    # Per-problem pass@k scores
```

## Example Config JSON

After running `--eval_profile paper_math`, the `eval_config.json` looks like:

```json
{
  "task": "gsm8k",
  "split": "test",
  "num_fewshot": 5,
  "decoding": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "strategy": "greedy"
  },
  "total_samples": 1319,
  "metric": {
    "name": "strict_match_accuracy",
    "score": 0.4523,
    "correct": 597,
    "total": 1319
  },
  "meta": {
    "base_model_id": "meta-llama/Llama-2-7b-hf",
    "lora_dir": "./my_lora",
    "seed": 0
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

After running `--eval_profile paper_code_main`:

```json
{
  "task": "humaneval",
  "split": "test",
  "num_fewshot": 0,
  "decoding": {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 512,
    "n_generations": 50,
    "strategy": "sampling"
  },
  "total_tasks": 164,
  "total_samples": 8200,
  "metric": {
    "name": "pass@1",
    "score": 0.3415
  },
  "meta": {
    "profile": {
      "profile": "paper_code_main",
      "task": "humaneval",
      "n_generations": 50,
      "temperature": 0.2,
      "top_p": 0.95
    }
  }
}
```

## CLI Reference

### `eval` subcommand

Unified evaluation for GSM8K and HumanEval:

```bash
python -m lora_spectral_edit eval \
  --base_model <model_id> \
  --lora_dir <lora_path> \
  --eval_profile <profile_name> \
  [--max_samples N] \
  [--out_dir <dir>] \
  [--out_json <file>] \
  [--seed N]
```

### `eval-humaneval` subcommand

Dedicated HumanEval evaluation with more options:

```bash
python -m lora_spectral_edit eval-humaneval \
  --base_model <model_id> \
  --lora_dir <lora_path> \
  [--eval_profile paper_code_main|greedy_code] \
  [--n_generations N] \
  [--temperature T] \
  [--top_p P] \
  [--pass_k K] \
  [--max_samples N] \
  [--out_dir <dir>] \
  [--out_json <file>]
```

### Manual Parameters (without profile)

For custom evaluation settings:

```bash
# GSM8K with custom params
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --fewshot 5 \
  --temperature 0.0 \
  --max_samples 100

# HumanEval with custom params
python -m lora_spectral_edit eval-humaneval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --n_generations 20 \
  --temperature 0.4 \
  --top_p 0.9 \
  --pass_k 1
```

## Smoke Test

Run a quick smoke test with a small subset:

```bash
# GSM8K smoke test (5 samples)
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --eval_profile paper_math \
  --max_samples 5 \
  --out_dir ./smoke_test_gsm8k

# HumanEval smoke test (5 problems, 5 samples each)
python -m lora_spectral_edit eval-humaneval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --n_generations 5 \
  --temperature 0.2 \
  --top_p 0.95 \
  --max_samples 5 \
  --out_dir ./smoke_test_humaneval
```

## Requirements

- **vLLM**: Required for all evaluations (`pip install vllm`)
- **human_eval**: Required for HumanEval (`pip install -e human-eval` from OpenAI repo)
- **CUDA GPU**: Required for vLLM inference

## Backward Compatibility

The existing evaluation interface remains fully backward compatible:

```bash
# Legacy GSM8K evaluation (still works)
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --fewshot 5 \
  --temperature 0.0

# Legacy HumanEval (standalone script still works)
python src/lora_spectral_edit/eval_humaneval.py \
  --base_model_id meta-llama/Llama-2-7b-hf \
  --lora_dir ./my_lora \
  --temperature 0.0
```
