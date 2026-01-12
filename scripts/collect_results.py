#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import statistics


def find_latest_run_root(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run roots found in {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_metric(task: str, metrics: dict) -> tuple[str, float | None]:
    if task.startswith("gsm8k") and "acc" in metrics:
        return "acc", float(metrics["acc"])
    if "edited" in metrics and isinstance(metrics["edited"], dict) and "pass@1" in metrics["edited"]:
        return "pass@1", float(metrics["edited"]["pass@1"])
    if "pass@1" in metrics:
        return "pass@1", float(metrics["pass@1"])
    return "metric", None


def parse_baseline_logs(
    baseline_dir: Path,
    task: str = "humaneval_full",
) -> list[dict]:
    """
    Parse baseline metrics from humaneval_baseline_logs/*results* files.

    Returns list of result rows with edit_mode='baseline'.
    """
    results = []
    if not baseline_dir.exists():
        return results

    # Find all *results*.json or *results*.jsonl files
    for results_file in baseline_dir.glob("*results*"):
        if not results_file.is_file():
            continue

        try:
            content = results_file.read_text(encoding="utf-8")
            # Handle JSONL or JSON
            if results_file.suffix == ".jsonl":
                lines = content.strip().split("\n")
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        row = _extract_baseline_row(data, task, str(results_file))
                        if row:
                            results.append(row)
            else:
                data = json.loads(content)
                row = _extract_baseline_row(data, task, str(results_file))
                if row:
                    results.append(row)
        except Exception as e:
            print(f"[Warn] Failed to parse {results_file}: {e}")
            continue

    return results


def _extract_baseline_row(data: dict, task: str, source_path: str) -> dict | None:
    """Extract a baseline result row from parsed JSON data."""
    # Try to extract pass@1 from various formats
    pass_at_1 = None
    lora_repo_id = None

    # Check for direct pass@1
    if "pass@1" in data:
        pass_at_1 = float(data["pass@1"])
    # Check for baseline.pass@1
    elif "baseline" in data and isinstance(data["baseline"], dict):
        if "pass@1" in data["baseline"]:
            pass_at_1 = float(data["baseline"]["pass@1"])

    # Try to extract lora_repo_id
    if "meta" in data:
        lora_repo_id = data["meta"].get("lora_repo_id") or data["meta"].get("lora_path")
    if not lora_repo_id:
        lora_repo_id = data.get("lora_repo_id") or data.get("lora_path")

    # Extract from filename if not in data
    if not lora_repo_id:
        # Try to parse from filename like "magicoder-lora-rank-64_results.json"
        fname = Path(source_path).stem
        for part in fname.replace("_results", "").replace("-results", "").split("_"):
            if "lora" in part.lower() or "magicoder" in part.lower() or "metamath" in part.lower():
                lora_repo_id = part
                break

    if pass_at_1 is None:
        return None

    return {
        "task": task,
        "lora_repo_id": lora_repo_id,
        "edit_mode": "baseline",
        "seed": 0,
        "out_dir": source_path,
        "metric_key": "pass@1",
        "metric": pass_at_1,
        "pass@1": pass_at_1,
        "amp_factor": None,
        "sup_factor": None,
        "soft_temperature": None,
        "soft_pivot_mode": None,
    }


def parse_edited_logs(
    edited_dir: Path,
    task: str = "humaneval_full",
) -> list[dict]:
    """
    Parse edited metrics from humaneval_edited_logs/*results* files.

    Returns list of result rows.
    """
    results = []
    if not edited_dir.exists():
        return results

    for results_file in edited_dir.glob("*results*"):
        if not results_file.is_file():
            continue

        try:
            content = results_file.read_text(encoding="utf-8")
            if results_file.suffix == ".jsonl":
                lines = content.strip().split("\n")
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        row = _extract_edited_row(data, task, str(results_file))
                        if row:
                            results.append(row)
            else:
                data = json.loads(content)
                row = _extract_edited_row(data, task, str(results_file))
                if row:
                    results.append(row)
        except Exception as e:
            print(f"[Warn] Failed to parse {results_file}: {e}")
            continue

    return results


def _extract_edited_row(data: dict, task: str, source_path: str) -> dict | None:
    """Extract an edited result row from parsed JSON data."""
    pass_at_1 = None
    lora_repo_id = None

    # Check for edited.pass@1
    if "edited" in data and isinstance(data["edited"], dict):
        if "pass@1" in data["edited"]:
            pass_at_1 = float(data["edited"]["pass@1"])
    elif "pass@1" in data:
        pass_at_1 = float(data["pass@1"])

    # Extract metadata
    meta = data.get("meta", {})
    lora_repo_id = meta.get("lora_repo_id") or meta.get("lora_path") or data.get("lora_repo_id")
    seed = meta.get("seed", data.get("seed", 0))

    # Extract hyperparams
    amp_factor = meta.get("amp_factor", data.get("amp_factor"))
    sup_factor = meta.get("sup_factor", data.get("sup_factor"))
    soft_temperature = meta.get("soft_temperature", data.get("soft_temperature"))
    soft_pivot_mode = meta.get("soft_pivot_mode", data.get("soft_pivot_mode"))
    edit_mode = meta.get("edit_mode", data.get("edit_mode", "edited"))

    if pass_at_1 is None:
        return None

    return {
        "task": task,
        "lora_repo_id": lora_repo_id,
        "edit_mode": edit_mode,
        "seed": seed,
        "out_dir": source_path,
        "metric_key": "pass@1",
        "metric": pass_at_1,
        "pass@1": pass_at_1,
        "amp_factor": amp_factor,
        "sup_factor": sup_factor,
        "soft_temperature": soft_temperature,
        "soft_pivot_mode": soft_pivot_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run metrics into summary files.")
    parser.add_argument("--run-root", type=str, default=None, help="Run root directory.")
    parser.add_argument("--runs-dir", type=str, default="_runs", help="Runs directory.")
    parser.add_argument(
        "--baseline-logs",
        type=str,
        default=None,
        help="Path to humaneval_baseline_logs directory (optional).",
    )
    parser.add_argument(
        "--edited-logs",
        type=str,
        default=None,
        help="Path to humaneval_edited_logs directory (optional).",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_root = Path(args.run_root) if args.run_root else find_latest_run_root(runs_dir)

    results = []

    # Collect from standard metrics.json files in run_root
    for metrics_path in run_root.rglob("metrics.json"):
        cfg_path = metrics_path.parent / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        task = cfg.get("task", "unknown")
        metric_key, metric_value = extract_metric(task, metrics)

        # Handle lora_repo_id: check both lora_repo_id and lora_path
        lora_repo_id = cfg.get("lora_repo_id")
        if not lora_repo_id:
            lora_path = cfg.get("lora_path", "")
            if "models--" in lora_path:
                # Parse HuggingFace cache path format: models--org--repo
                parts = lora_path.split("models--")[-1].split("/")[0]
                lora_repo_id = parts.replace("--", "/")
            elif lora_path:
                lora_repo_id = Path(lora_path.rstrip("/")).name

        # Handle edit_mode: infer from metrics if not in config
        edit_mode = cfg.get("edit_mode")
        if not edit_mode:
            # Check if metrics has meta.edit_mode
            meta = metrics.get("meta", {})
            edit_mode = meta.get("edit_mode")

        # Infer task from metrics meta if config doesn't have it
        if task == "unknown":
            meta = metrics.get("meta", {})
            task = meta.get("task", task)

        row = {
            "task": task,
            "lora_repo_id": lora_repo_id,
            "calib_samples": cfg.get("calib_samples"),
            "edit_mode": edit_mode,
            "seed": cfg.get("seed"),
            "out_dir": str(metrics_path.parent),
            "metric_key": metric_key,
            "metric": metric_value,
            # Hyperparameter columns
            "amp_factor": cfg.get("amp_factor"),
            "sup_factor": cfg.get("sup_factor"),
            "soft_temperature": cfg.get("soft_temperature"),
            "soft_pivot_mode": cfg.get("soft_pivot_mode"),
        }
        if "acc" in metrics:
            row.update({
                "acc": metrics.get("acc"),
                "correct": metrics.get("correct"),
                "total": metrics.get("total"),
            })
        if "edited" in metrics and isinstance(metrics["edited"], dict):
            row.update({
                "pass@1": metrics["edited"].get("pass@1"),
                "correct": metrics["edited"].get("correct"),
                "total": metrics["edited"].get("total"),
                "num_tasks": metrics["edited"].get("num_tasks"),
            })
        elif "pass@1" in metrics:
            row["pass@1"] = metrics["pass@1"]

        results.append(row)

    # Collect from baseline_logs if provided
    if args.baseline_logs:
        baseline_dir = Path(args.baseline_logs)
        baseline_results = parse_baseline_logs(baseline_dir, task="humaneval_full")
        results.extend(baseline_results)
        print(f"[Collect] Parsed {len(baseline_results)} baseline results from {baseline_dir}")

    # Collect from edited_logs if provided
    if args.edited_logs:
        edited_dir = Path(args.edited_logs)
        edited_results = parse_edited_logs(edited_dir, task="humaneval_full")
        results.extend(edited_results)
        print(f"[Collect] Parsed {len(edited_results)} edited results from {edited_dir}")

    # Deduplicate baselines: keep one row per (task, lora_repo_id, calib_samples, seed)
    seen_baselines = set()
    deduped_results = []
    for row in results:
        if row.get("edit_mode") == "baseline":
            key = (
                row.get("task"),
                row.get("lora_repo_id"),
                row.get("calib_samples"),
                row.get("seed"),
            )
            if key in seen_baselines:
                continue
            seen_baselines.add(key)
        deduped_results.append(row)

    results = deduped_results

    results_path = run_root / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Group by (task, lora_repo_id, calib_samples, edit_mode, amp_factor, sup_factor, soft_temperature, soft_pivot_mode)
    groups = {}
    for row in results:
        key = (
            row.get("task"),
            row.get("lora_repo_id"),
            row.get("calib_samples"),
            row.get("edit_mode"),
            row.get("amp_factor"),
            row.get("sup_factor"),
            row.get("soft_temperature"),
            row.get("soft_pivot_mode"),
        )
        metric = row.get("metric")
        if metric is None:
            continue
        groups.setdefault(key, {"metric_key": row.get("metric_key"), "values": [], "seeds": []})
        groups[key]["values"].append(metric)
        groups[key]["seeds"].append(row.get("seed"))

    summary_rows = []
    for (
        task,
        lora_repo_id,
        calib_samples,
        edit_mode,
        amp_f,
        sup_f,
        soft_t,
        pivot_m,
    ), data in sorted(groups.items()):
        vals = data["values"]
        mean = statistics.mean(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append({
            "task": task,
            "lora_repo_id": lora_repo_id,
            "calib_samples": calib_samples,
            "edit_mode": edit_mode,
            "amp_factor": amp_f,
            "sup_factor": sup_f,
            "soft_temperature": soft_t,
            "soft_pivot_mode": pivot_m,
            "metric_key": data["metric_key"],
            "n": len(vals),
            "mean": mean,
            "std": std,
        })

    # Markdown table
    md_lines = [
        "| task | lora_repo_id | calib_samples | edit_mode | amp_factor | sup_factor | soft_temperature | soft_pivot_mode | metric | n | mean | std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['task']} | {row['lora_repo_id']} | {row['calib_samples']} | {row['edit_mode']} | "
            f"{row['amp_factor']} | {row['sup_factor']} | {row['soft_temperature']} | {row['soft_pivot_mode']} | "
            f"{row['metric_key']} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} |"
        )

    summary_md = run_root / "summary_table.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # CSV with all columns
    summary_csv = run_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("task,lora_repo_id,calib_samples,edit_mode,seed,amp_factor,sup_factor,soft_temperature,soft_pivot_mode,pass@1\n")
        for row in results:
            task = row.get("task", "")
            lora_repo_id = row.get("lora_repo_id", "")
            calib_samples = row.get("calib_samples", "")
            edit_mode = row.get("edit_mode", "")
            seed = row.get("seed", "")
            amp_f = row.get("amp_factor", "")
            sup_f = row.get("sup_factor", "")
            soft_t = row.get("soft_temperature", "")
            pivot_m = row.get("soft_pivot_mode", "")
            pass_at_1 = row.get("pass@1", row.get("metric", ""))
            f.write(f"{task},{lora_repo_id},{calib_samples},{edit_mode},{seed},{amp_f},{sup_f},{soft_t},{pivot_m},{pass_at_1}\n")

    # Also write TSV for convenience
    summary_tsv = run_root / "summary.tsv"
    with summary_tsv.open("w", encoding="utf-8") as f:
        f.write("task\tlora_repo_id\tcalib_samples\tedit_mode\tseed\tamp_factor\tsup_factor\tsoft_temperature\tsoft_pivot_mode\tpass@1\n")
        for row in results:
            task = row.get("task", "")
            lora_repo_id = row.get("lora_repo_id", "")
            calib_samples = row.get("calib_samples", "")
            edit_mode = row.get("edit_mode", "")
            seed = row.get("seed", "")
            amp_f = row.get("amp_factor", "")
            sup_f = row.get("sup_factor", "")
            soft_t = row.get("soft_temperature", "")
            pivot_m = row.get("soft_pivot_mode", "")
            pass_at_1 = row.get("pass@1", row.get("metric", ""))
            f.write(f"{task}\t{lora_repo_id}\t{calib_samples}\t{edit_mode}\t{seed}\t{amp_f}\t{sup_f}\t{soft_t}\t{pivot_m}\t{pass_at_1}\n")

    print(f"[Collect] Wrote: {results_path}")
    print(f"[Collect] Wrote: {summary_md}")
    print(f"[Collect] Wrote: {summary_csv}")
    print(f"[Collect] Wrote: {summary_tsv}")


if __name__ == "__main__":
    main()
