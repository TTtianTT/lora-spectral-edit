#!/usr/bin/env python
"""Merge robust_z results with existing calib_samples sweep results."""
import json
import statistics
import argparse
from pathlib import Path
from typing import Optional


def extract_metric(task: str, metrics: dict) -> tuple[str, float | None]:
    """Extract metric key and value from metrics dict."""
    if task.startswith("gsm8k") and "acc" in metrics:
        return "acc", float(metrics["acc"])
    if "edited" in metrics and isinstance(metrics["edited"], dict) and "pass@1" in metrics["edited"]:
        return "pass@1", float(metrics["edited"]["pass@1"])
    if "pass@1" in metrics:
        return "pass@1", float(metrics["pass@1"])
    return "metric", None


def collect_results(run_root: Path) -> list[dict]:
    """Collect results from a run root directory."""
    results = []
    for metrics_path in run_root.rglob("metrics.json"):
        cfg_path = metrics_path.parent / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warn] Failed to load {metrics_path}: {e}")
            continue

        task = cfg.get("task", "unknown")
        metric_key, metric_value = extract_metric(task, metrics)

        lora_repo_id = cfg.get("lora_repo_id")
        if not lora_repo_id:
            lora_path = cfg.get("lora_path", "")
            if "models--" in lora_path:
                parts = lora_path.split("models--")[-1].split("/")[0]
                lora_repo_id = parts.replace("--", "/")
            elif lora_path:
                lora_repo_id = Path(lora_path.rstrip("/")).name

        edit_mode = cfg.get("edit_mode")
        if not edit_mode:
            meta = metrics.get("meta", {})
            edit_mode = meta.get("edit_mode")

        row = {
            "task": task,
            "lora_repo_id": lora_repo_id,
            "calib_samples": cfg.get("calib_samples"),
            "edit_mode": edit_mode,
            "seed": cfg.get("seed"),
            "out_dir": str(metrics_path.parent),
            "metric_key": metric_key,
            "metric": metric_value,
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

    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    """Aggregate results by grouping key and computing mean/std."""
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

    return summary_rows


def compute_deltas(summary: list[dict]) -> dict:
    """Compute deltas: robust_z - baseline and robust_z - z_score."""
    # Build lookup by (task, lora_repo_id, calib_samples, edit_mode)
    lookup = {}
    for row in summary:
        key = (row["task"], row["lora_repo_id"], row["calib_samples"], row["edit_mode"])
        lookup[key] = row

    deltas = []
    for row in summary:
        if row["edit_mode"] != "robust_z":
            continue
        base_key = (row["task"], row["lora_repo_id"], row["calib_samples"], "baseline")
        zscore_key = (row["task"], row["lora_repo_id"], row["calib_samples"], "z_score")

        baseline_row = lookup.get(base_key)
        zscore_row = lookup.get(zscore_key)

        delta_entry = {
            "task": row["task"],
            "lora_repo_id": row["lora_repo_id"],
            "calib_samples": row["calib_samples"],
            "robust_z_mean": row["mean"],
            "robust_z_std": row["std"],
        }

        if baseline_row:
            delta_entry["baseline_mean"] = baseline_row["mean"]
            delta_entry["delta_vs_baseline"] = row["mean"] - baseline_row["mean"]
        else:
            delta_entry["baseline_mean"] = None
            delta_entry["delta_vs_baseline"] = None

        if zscore_row:
            delta_entry["z_score_mean"] = zscore_row["mean"]
            delta_entry["delta_vs_zscore"] = row["mean"] - zscore_row["mean"]
        else:
            delta_entry["z_score_mean"] = None
            delta_entry["delta_vs_zscore"] = None

        deltas.append(delta_entry)

    return deltas


def main():
    parser = argparse.ArgumentParser(description="Merge robust_z results with existing sweep")
    parser.add_argument("--existing", type=str, required=True,
                        help="Path to existing results.jsonl or run_root")
    parser.add_argument("--robust_z", type=str, required=True,
                        help="Path to robust_z run_root")
    parser.add_argument("--output_dir", type=str, default="analysis",
                        help="Output directory for merged results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results
    existing_path = Path(args.existing)
    if existing_path.is_file() and existing_path.suffix == ".jsonl":
        existing_results = []
        with open(existing_path) as f:
            for line in f:
                if line.strip():
                    existing_results.append(json.loads(line))
        print(f"[Merge] Loaded {len(existing_results)} existing results from {existing_path}")
    else:
        existing_results = collect_results(existing_path)
        print(f"[Merge] Collected {len(existing_results)} existing results from {existing_path}")

    # Collect robust_z results
    robust_z_root = Path(args.robust_z)
    robust_z_results = collect_results(robust_z_root)
    print(f"[Merge] Collected {len(robust_z_results)} robust_z results from {robust_z_root}")

    # Merge results (robust_z only adds to existing, no duplicates expected)
    merged_results = existing_results + robust_z_results
    print(f"[Merge] Total merged results: {len(merged_results)}")

    # Write merged results.jsonl
    merged_jsonl = output_dir / "calib_samples_sweep_merged_with_robust_z.jsonl"
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for row in merged_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Merge] Wrote: {merged_jsonl}")

    # Aggregate and create summary
    summary = aggregate_results(merged_results)
    print(f"[Merge] Created {len(summary)} summary rows")

    # Write summary TSV
    summary_tsv = output_dir / "calib_samples_sweep_merged_with_robust_z.tsv"
    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("task\tlora_repo_id\tcalib_samples\tedit_mode\tamp_factor\tsup_factor\tsoft_temperature\tsoft_pivot_mode\tmetric\tn\tmean\tstd\n")
        for row in summary:
            f.write(f"{row['task']}\t{row['lora_repo_id']}\t{row['calib_samples']}\t{row['edit_mode']}\t"
                    f"{row['amp_factor']}\t{row['sup_factor']}\t{row['soft_temperature']}\t{row['soft_pivot_mode']}\t"
                    f"{row['metric_key']}\t{row['n']}\t{row['mean']:.6f}\t{row['std']:.6f}\n")
    print(f"[Merge] Wrote: {summary_tsv}")

    # Write markdown table
    md_lines = [
        "# Calib Samples Sweep Results (Merged with robust_z)",
        "",
        "| task | lora_repo_id | calib_samples | edit_mode | amp_factor | sup_factor | soft_temperature | soft_pivot_mode | metric | n | mean | std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        md_lines.append(
            f"| {row['task']} | {row['lora_repo_id']} | {row['calib_samples']} | {row['edit_mode']} | "
            f"{row['amp_factor']} | {row['sup_factor']} | {row['soft_temperature']} | {row['soft_pivot_mode']} | "
            f"{row['metric_key']} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} |"
        )

    summary_md = output_dir / "calib_samples_sweep_merged_with_robust_z.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Merge] Wrote: {summary_md}")

    # Compute and write deltas
    deltas = compute_deltas(summary)

    delta_md_lines = [
        "",
        "## Delta Summary: robust_z vs baseline and z_score",
        "",
        "| task | lora_repo_id | calib_samples | robust_z | baseline | delta_baseline | z_score | delta_zscore |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in deltas:
        baseline_str = f"{d['baseline_mean']:.4f}" if d['baseline_mean'] is not None else "N/A"
        zscore_str = f"{d['z_score_mean']:.4f}" if d['z_score_mean'] is not None else "N/A"
        delta_b = f"{d['delta_vs_baseline']:+.4f}" if d['delta_vs_baseline'] is not None else "N/A"
        delta_z = f"{d['delta_vs_zscore']:+.4f}" if d['delta_vs_zscore'] is not None else "N/A"

        delta_md_lines.append(
            f"| {d['task']} | {d['lora_repo_id']} | {d['calib_samples']} | "
            f"{d['robust_z_mean']:.4f} | {baseline_str} | {delta_b} | {zscore_str} | {delta_z} |"
        )

    # Highlight HumanEval improvements
    humaneval_wins = [d for d in deltas if d["task"] == "humaneval_full" and d["delta_vs_zscore"] is not None and d["delta_vs_zscore"] > 0]
    if humaneval_wins:
        delta_md_lines.append("")
        delta_md_lines.append("### HumanEval: robust_z improvements over z_score")
        delta_md_lines.append("")
        for d in humaneval_wins:
            delta_md_lines.append(f"- {d['lora_repo_id']} (cs={d['calib_samples']}): +{d['delta_vs_zscore']:.4f}")

    # Append deltas to the markdown
    with open(summary_md, "a", encoding="utf-8") as f:
        f.write("\n".join(delta_md_lines) + "\n")
    print(f"[Merge] Appended delta summary to: {summary_md}")

    # Print summary
    print("\n" + "=" * 70)
    print("DELTA SUMMARY")
    print("=" * 70)
    print("\n| task | lora | calib_samples | robust_z - baseline | robust_z - z_score |")
    print("| --- | --- | --- | --- | --- |")
    for d in deltas:
        lora_short = d["lora_repo_id"].split("/")[-1] if d["lora_repo_id"] else "?"
        delta_b = f"{d['delta_vs_baseline']:+.4f}" if d['delta_vs_baseline'] is not None else "N/A"
        delta_z = f"{d['delta_vs_zscore']:+.4f}" if d['delta_vs_zscore'] is not None else "N/A"
        print(f"| {d['task']} | {lora_short} | {d['calib_samples']} | {delta_b} | {delta_z} |")

    if humaneval_wins:
        print(f"\n[Highlight] {len(humaneval_wins)} HumanEval cases where robust_z > z_score")


if __name__ == "__main__":
    main()
