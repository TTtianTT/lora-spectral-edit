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


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run metrics into summary files.")
    parser.add_argument("--run-root", type=str, default=None, help="Run root directory.")
    parser.add_argument("--runs-dir", type=str, default="_runs", help="Runs directory.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_root = Path(args.run_root) if args.run_root else find_latest_run_root(runs_dir)

    results = []
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

        row = {
            "task": task,
            "lora_repo_id": cfg.get("lora_repo_id"),
            "edit_mode": cfg.get("edit_mode"),
            "seed": cfg.get("seed"),
            "out_dir": str(metrics_path.parent),
            "metric_key": metric_key,
            "metric": metric_value,
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

        results.append(row)

    results_path = run_root / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = {}
    for row in results:
        key = (row.get("task"), row.get("lora_repo_id"), row.get("edit_mode"))
        metric = row.get("metric")
        if metric is None:
            continue
        groups.setdefault(key, {"metric_key": row.get("metric_key"), "values": []})
        groups[key]["values"].append(metric)

    summary_rows = []
    for (task, lora_repo_id, edit_mode), data in sorted(groups.items()):
        vals = data["values"]
        mean = statistics.mean(vals) if vals else 0.0
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append({
            "task": task,
            "lora_repo_id": lora_repo_id,
            "edit_mode": edit_mode,
            "metric_key": data["metric_key"],
            "n": len(vals),
            "mean": mean,
            "std": std,
        })

    md_lines = [
        "| task | lora_repo_id | edit_mode | metric | n | mean | std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['task']} | {row['lora_repo_id']} | {row['edit_mode']} | "
            f"{row['metric_key']} | {row['n']} | {row['mean']:.6f} | {row['std']:.6f} |"
        )

    summary_md = run_root / "summary_table.md"
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary_csv = run_root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("task,lora_repo_id,edit_mode,metric_key,n,mean,std\n")
        for row in summary_rows:
            f.write(
                f"{row['task']},{row['lora_repo_id']},{row['edit_mode']},"
                f"{row['metric_key']},{row['n']},{row['mean']},{row['std']}\n"
            )

    print(f"[Collect] Wrote: {results_path}")
    print(f"[Collect] Wrote: {summary_md}")
    print(f"[Collect] Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
