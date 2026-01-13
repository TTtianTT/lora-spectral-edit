# Core Idea: LoRA Spectral Edit

## 1. Core Idea
A trained LoRA adapter is usually treated as a fixed endpoint, even though its low-rank update contains directions that are not equally important. This repo treats a trained LoRA as editable: the singular directions of the update can be separated into high-sensitivity (core) and low-sensitivity (noise) components, and edited without retraining.

The key observation is that the LoRA update \(\Delta W = BA\) has a spectral structure, and the gradient sensitivity of each singular value indicates how much that direction matters for the loss. The intervention is a post-hoc spectral edit: compute per-singular-value sensitivity \(g_{\sigma}\) from a small calibration set, reweight \(\sigma\) accordingly, optionally preserve spectral energy, and rebuild the LoRA weights.

## 2. Method
### 2.1 Notation and decomposition
For each LoRA module, let \(A \in \mathbb{R}^{r \times d_{in}}\), \(B \in \mathbb{R}^{d_{out} \times r}\), and \(\Delta W = BA\). The code computes a compact SVD of \(\Delta W\) without explicitly forming it:
\[\Delta W = U\,\mathrm{diag}(\sigma)\,V^T, \quad U \in \mathbb{R}^{d_{out} \times r},\; V \in \mathbb{R}^{d_{in} \times r}.\]
The edited update is
\[\Delta W' = U\,\mathrm{diag}(\sigma')\,V^T,\]
and the implementation reconstructs LoRA factors as
\[B' = U\,\mathrm{diag}(\sqrt{\sigma'}),\quad A' = \mathrm{diag}(\sqrt{\sigma'})\,V^T.\]
Edits are applied per module, by default to `down_proj` and `o_proj`, with optional `--layer_min/--layer_max` filtering.

### 2.2 Sensitivity probing (calibration)
For each selected module, gradient hooks accumulate sensitivity for each singular component without forming \(\partial L/\partial W\). The scalar sensitivity for component \(k\) is
\[g_{\sigma,k} = s \sum_{n \in \mathcal{T}} \langle \partial L/\partial y_n, u_k \rangle\,\langle x_n, v_k \rangle,\]
where \(x_n\) is the module input, \(y_n\) is the module output, \(u_k\) and \(v_k\) are the singular vectors, \(s = \alpha/r\) is the LoRA scaling from `adapter_config.json`, and \(\mathcal{T}\) indexes active (non-padding) tokens across the calibration batches.

Calibration uses a small subset of GSM8K train examples. Each example is formatted as `Question: ...\nAnswer:` with teacher forcing over the full answer; prompt tokens are masked in the loss (`-100` labels), and padding tokens are masked in the attention mask. Sensitivities are accumulated over batches and then normalized via `--grad_norm` (default `mean_abs`, i.e., divide by mean \(|g_{\sigma}|\); `l2` and `none` are also available). Because the edit relies on relative ranking within each module and uses normalization, a small `--calib_samples` (default 32) is sufficient to estimate a usable ordering in practice.

### 2.3 Spectral reweighting (the edit)
The repo implements several reweighting policies; the default and recommended one is `abs_select`:
- Compute \(g_{abs} = |g_{\sigma}|\) and normalize it (per `--grad_norm`).
- Let \(k_{core} = \max(\mathrm{round}(r \cdot \text{core\_frac}), \text{min\_core\_k})\) and \(k_{noise} = \mathrm{round}(r \cdot \text{noise\_frac})\), clipped so core and noise do not overlap.
- Select the top \(k_{core}\) indices of \(g_{abs}\) as core and the bottom \(k_{noise}\) as noise.
- Apply a multiplicative gate: `amp_factor` for core, `sup_factor` for noise, and `mid_factor` for the rest, giving \(\sigma' = \sigma \odot w\).
- Optionally clamp \(\sigma' \ge \text{sigma\_clip\_min}\), then preserve spectral energy by rescaling to match \(\|\sigma\|_1\) or \(\|\sigma\|_2\) (`--preserve_energy`).

The `random_index` mode is a control: it uses the same `core_frac`/`noise_frac` counts and the same amplification/suppression factors, but selects indices uniformly at random (seeded by `--seed`) to isolate the effect of sensitivity-based selection.

### 2.4 Algorithm (pseudo-code)
```text
Algorithm 1: LoRA Spectral Edit (abs_select)
Input: base_model, lora_path, target_modules, layer range, calib_samples N,
       edit config (core_frac, noise_frac, amp_factor, sup_factor, mid_factor,
       grad_norm, preserve_energy, sigma_clip_min)

1: Load LoRA adapter and select (A, B) pairs for target_modules and layers.
2: For each selected module:
     Compute U, sigma, V^T = lowrank_svd_from_ba(B, A).
     Record scaling s = alpha/r from adapter_config.
3: Register hooks to cache inputs x and accumulate g_sigma:
     For each calibration batch from GSM8K train:
       Build teacher-forced inputs; mask prompt tokens in labels.
       Forward and backward once; accumulate
         g_sigma += s * sum_n <dL/dy_n, u_k> * <x_n, v_k>.
4: For each module:
     g_abs = |g_sigma|; g_abs = normalize(g_abs, grad_norm).
     Select top k_core and bottom k_noise indices by g_abs.
     gate = mid_factor; gate[core]=amp_factor; gate[noise]=sup_factor.
     sigma' = sigma * gate; sigma' = clamp_min(sigma', sigma_clip_min).
     sigma' = preserve_energy(sigma, sigma', preserve_energy).
     Rebuild B', A' from U, V^T, sigma' and write back to adapter.
5: Save edited adapter and spectral_edit_meta.json.
6: (Optional) Evaluate with `lora_spectral_edit eval` / `eval-humaneval`.
Output: edited LoRA adapter (+ evaluation metrics if run).
```

### 2.5 Variants (brief)
- `smooth_abs`: sigmoid gate over normalized \(|g_{\sigma}|\) with `smooth_temperature` and `smooth_center_q` (optional center alignment to `mid_factor`).
- `double_smooth`: two sigmoids to smoothly suppress low-sensitivity and amplify high-sensitivity regions.
- `z_score`: double-sigmoid gate in z-score space using mean/std of \(|g_{\sigma}|\).
- `robust_z`: z-score gate using median/MAD for robustness.
- `gd`: signed gradient updates to \(\sigma\) (additive or multiplicative; optional asymmetric step sizes).
- `random_index`: control baseline described above.

## 3. What makes it novel
- Post-hoc intervention on a trained LoRA by editing its singular spectrum rather than retraining.
- Gradient-guided selection in the spectral basis using efficient hooks (no explicit \(\partial L/\partial W\)).
- Minimal calibration: small, teacher-forced batches used only to rank spectral components.
- Plug-and-play, per-module edits with layer filtering and preserved LoRA format.
- An explicit random-index control to isolate sensitivity-driven effects.

## 4. Practical usage (minimal)
- Sensitivity probe + edit (abs_select):
```bash
python -m lora_spectral_edit edit \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_path LoRA-TMLR-2024/metamath-lora-rank-64-alpha-128 \
  --out_dir ./edited_lora \
  --mode abs_select \
  --calib_samples 32
```
Outputs in `./edited_lora/`: `adapter_model.safetensors` (or `adapter_model.bin`), `adapter_config.json`, and `spectral_edit_meta.json`.

- GSM8K evaluation (paper profile):
```bash
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./edited_lora \
  --eval_profile paper_math \
  --out_dir ./edited_lora \
  --out_json ./edited_lora/metrics_gsm8k.json
```
Outputs in `./edited_lora/`: `eval_config.json`, `gsm8k_predictions.jsonl`, and `metrics_gsm8k.json`.

- HumanEval evaluation (paper profile):
```bash
python -m lora_spectral_edit eval \
  --base_model meta-llama/Llama-2-7b-hf \
  --lora_dir ./edited_lora \
  --eval_profile paper_code_main \
  --out_dir ./humaneval_results \
  --out_json ./humaneval_results/metrics_humaneval.json
```
Outputs in `./humaneval_results/`: `samples_lora.jsonl`, `raw_generations_lora.jsonl`, `eval_config.json`, and `metrics_humaneval.json`.

- Optional sweep from JSONL configs:
```bash
bash scripts/launch_multi_gpu.sh --configs experiments/configs_sweep_allmodes_6lora.jsonl
```
Outputs go to the `run_root` encoded in the configs (e.g., `_runs/<timestamp>_sweep_allmodes_6lora/...`).

## 5. Reproducibility checklist
- Entry points: `python -m lora_spectral_edit edit`, `python -m lora_spectral_edit eval`, `python -m lora_spectral_edit eval-humaneval`, and `bash scripts/launch_multi_gpu.sh` for sweeps.
- Seed control: set `--seed` (also controls `random_index` and torch RNG).
- Calibration: `--calib_samples` and `--calib_batch_size` (GSM8K train subset, teacher-forced).
- Target scope: `--target_modules` and `--layer_min/--layer_max`.
- Edit policy: `--mode` plus its hyperparameters (`core_frac`, `noise_frac`, `amp_factor`, `sup_factor`, `mid_factor`, or variant-specific flags).
- Sensitivity normalization: `--grad_norm` (`mean_abs` default).
- Energy handling: `--preserve_energy` (`l1` default) and `--sigma_clip_min`.
- Evaluation settings: `--eval_profile` (e.g., `paper_math`, `paper_code_main`) or explicit `--fewshot`, `--temperature`, `--max_tokens`, `--max_samples`, `--max_model_len`.
- Logged artifacts: `spectral_edit_meta.json` (edit config + per-module stats), `eval_config.json`, predictions JSONL, and any `metrics_*.json` you save via `--out_json`.

## 6. Implementation pointers
- `src/lora_spectral_edit/cli.py`: end-to-end edit pipeline, calibration loop, and evaluation dispatch.
- `src/lora_spectral_edit/hooks.py`: hook-based accumulation of \(g_{\sigma}\) from forward inputs and backward grads.
- `src/lora_spectral_edit/svd.py`: QR-based low-rank SVD and LoRA factor reconstruction.
- `src/lora_spectral_edit/edit.py`: edit modes (abs_select, smooth_abs, double_smooth, z_score, robust_z, random_index, gd) and energy preservation.
- `src/lora_spectral_edit/io.py`: adapter loading/saving, LoRA key parsing, and scaling (alpha/r).
- `src/lora_spectral_edit/eval_gsm8k.py`: GSM8K evaluation logic and strict-match parsing.
- `src/lora_spectral_edit/eval_humaneval.py`: HumanEval generation, pass@k computation, and output artifacts.
- `src/lora_spectral_edit/eval_profiles.py`: predefined evaluation profiles (`paper_math`, `paper_code_main`).
