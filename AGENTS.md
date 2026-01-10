# Repository Guidelines

## Project Structure & Module Organization
- `src/lora_spectral_edit/` holds the package modules: CLI entrypoint (`cli.py`), SVD utilities (`svd.py`), editing logic (`edit.py`), I/O helpers (`io.py`), gradient hooks (`hooks.py`), and evaluation scripts (`eval_gsm8k.py`, `eval_humaneval.py`).
- `examples/` contains runnable scripts such as `examples/edit_metamath.sh`.
- `README.md` documents user-facing setup and usage.
- `pyproject.toml` defines dependencies, entry points, and pytest configuration.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `pip install -e ".[eval]"` installs optional vLLM dependencies for evaluation runs.
- `python -m lora_spectral_edit.smoke_test` runs lightweight smoke checks (CPU-only).
- `python -m lora_spectral_edit edit ...` edits a LoRA adapter; see `README.md` for flags.
- `python -m lora_spectral_edit eval ...` runs GSM8K evaluation (requires vLLM + GPU).

## Coding Style & Naming Conventions
- Python code follows standard PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables and `CapWords` for classes.
- Prefer explicit, domain-specific names (e.g., `g_sigma`, `apply_spectral_edit`).

## Testing Guidelines
- Pytest is the configured test runner (`pyproject.toml`); tests should live in `tests/` and match `test_*.py`.
- If adding tests, keep them lightweight and runnable without a GPU where possible.
- Run targeted tests with `pytest` or `pytest tests/test_foo.py`.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit subjects (e.g., "Add CLAUDE.md for Claude Code guidance", "Initial commit: LoRA Spectral Edit package"). Follow that pattern and keep the first line concise.
- PRs should include a clear description, the commands run (or note if untested), and any evaluation outputs or artifacts (e.g., `spectral_edit_meta.json`, `metrics_gsm8k.json`) when relevant.

## Configuration & Data Tips
- Evaluation requires vLLM and a CUDA-capable GPU; editing can run CPU-only except for gradient calibration.
- Store generated adapters in a dedicated output folder (e.g., `./edited_lora`) to keep the repo clean.
