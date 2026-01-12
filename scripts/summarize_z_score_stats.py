#!/usr/bin/env python3
"""
Summarize z_score gating statistics from existing run outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from lora_spectral_edit.io import layer_idx_from_module_prefix
except Exception:  # pragma: no cover - allow script to run without package import
    layer_idx_from_module_prefix = None


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_stats_path(out_dir: Path) -> Optional[Path]:
    stats_path = out_dir / "z_score_gate_stats.json"
    return stats_path if stats_path.exists() else None


def quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def quantile_dict(values: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p0": quantile(values, 0.0),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p100": quantile(values, 1.0),
    }


def layer_from_prefix(prefix: str) -> Optional[int]:
    if layer_idx_from_module_prefix is None:
        return None
    return layer_idx_from_module_prefix(prefix)


def module_from_prefix(prefix: str) -> str:
    return prefix.split(".")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize z_score gate stats.")
    parser.add_argument("--runs-dir", type=str, default="_runs", help="Root runs directory.")
    parser.add_argument("--out-dir", type=str, default="analysis", help="Output directory.")
    parser.add_argument("--edit-mode", type=str, default="z_score", help="Edit mode to summarize.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_files = list(runs_dir.rglob("z_score_gate_stats.json"))
    edit_mode = args.edit_mode

    seen_runs = set()
    total_zscore_runs = 0
    parsed_runs = 0
    skipped_runs = 0
    skip_reasons: Dict[str, int] = defaultdict(int)

    events: List[Dict[str, Any]] = []
    run_dirs_by_group: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    calib_values_by_group: Dict[Tuple[str, str], set] = defaultdict(set)

    for stats_path in stats_files:
        out_dir_run = stats_path.parent
        payload = load_json(stats_path)
        if not isinstance(payload, list):
            skipped_runs += 1
            skip_reasons["invalid_stats"] += 1
            continue

        if not payload:
            skipped_runs += 1
            skip_reasons["empty_stats"] += 1
            continue

        task = payload[0].get("task")
        lora_repo_id = payload[0].get("lora_repo_id")
        seed = payload[0].get("seed")
        calib_samples = payload[0].get("calib_samples")

        total_zscore_runs += 1
        run_key = (task, lora_repo_id, calib_samples, edit_mode, seed)
        if run_key in seen_runs:
            skip_reasons["duplicate_run"] += 1
            skipped_runs += 1
            continue
        seen_runs.add(run_key)

        parsed_any = False
        for stats in payload:
            if stats.get("mode", edit_mode) != edit_mode:
                continue
            required = ["k_core_eff", "k_noise_eff", "frac_core", "frac_noise", "fallback"]
            if any(k not in stats for k in required):
                skip_reasons["missing_keys"] += 1
                continue
            events.append({
                "task": stats.get("task"),
                "lora_repo_id": stats.get("lora_repo_id"),
                "seed": stats.get("seed"),
                "calib_samples": stats.get("calib_samples"),
                "layer": stats.get("layer"),
                "module": stats.get("module"),
                "k_core_eff": stats.get("k_core_eff"),
                "k_noise_eff": stats.get("k_noise_eff"),
                "frac_core": stats.get("frac_core"),
                "frac_noise": stats.get("frac_noise"),
                "fallback": bool(stats.get("fallback")),
            })
            parsed_any = True

        if parsed_any:
            parsed_runs += 1
            run_dirs_by_group[(task, lora_repo_id)].append(str(out_dir_run))
            calib_values_by_group[(task, lora_repo_id)].add(calib_samples)
        else:
            skipped_runs += 1
            skip_reasons["no_zscore_stats"] += 1

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        grouped[(ev["task"], ev["lora_repo_id"])].append(ev)

    summary: Dict[str, Dict[str, Any]] = {}
    markdown_sections: List[str] = []

    for (task, lora_repo_id), group_events in sorted(grouped.items()):
        by_layer_seed: Dict[Tuple[Optional[int], Any], List[Dict[str, Any]]] = defaultdict(list)
        for ev in group_events:
            by_layer_seed[(ev["layer"], ev["seed"])].append(ev)

        layer_stats: Dict[str, Dict[str, Any]] = {}
        layer_events = []
        for (layer, seed), evs in by_layer_seed.items():
            k_core_mean = sum(e["k_core_eff"] for e in evs) / len(evs)
            k_noise_mean = sum(e["k_noise_eff"] for e in evs) / len(evs)
            frac_core_mean = sum(e["frac_core"] for e in evs) / len(evs)
            frac_noise_mean = sum(e["frac_noise"] for e in evs) / len(evs)
            fallback_any = any(e["fallback"] for e in evs)
            layer_events.append({
                "layer": layer,
                "seed": seed,
                "k_core_eff": k_core_mean,
                "k_noise_eff": k_noise_mean,
                "frac_core": frac_core_mean,
                "frac_noise": frac_noise_mean,
                "fallback": fallback_any,
            })

        by_layer: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
        for ev in layer_events:
            by_layer[ev["layer"]].append(ev)

        for layer, evs in by_layer.items():
            layer_key = "unknown" if layer is None else str(layer)
            layer_stats[layer_key] = {
                "mean_k_core_eff": sum(e["k_core_eff"] for e in evs) / len(evs),
                "mean_k_noise_eff": sum(e["k_noise_eff"] for e in evs) / len(evs),
                "mean_frac_core": sum(e["frac_core"] for e in evs) / len(evs),
                "mean_frac_noise": sum(e["frac_noise"] for e in evs) / len(evs),
                "fallback_rate": sum(1 for e in evs if e["fallback"]) / len(evs),
                "n_runs": len(evs),
            }

        overall_frac_core = [e["frac_core"] for e in layer_events]
        overall_frac_noise = [e["frac_noise"] for e in layer_events]
        overall_k_core = [e["k_core_eff"] for e in layer_events]
        overall_k_noise = [e["k_noise_eff"] for e in layer_events]
        overall_fallback = sum(1 for e in layer_events if e["fallback"])
        overall_count = len(layer_events)
        overall_fallback_rate = overall_fallback / overall_count if overall_count else None

        by_module_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ev in group_events:
            by_module_events[ev["module"]].append(ev)
        by_module_stats: Dict[str, Dict[str, Any]] = {}
        for module, evs in by_module_events.items():
            by_module_stats[module] = {
                "frac_core_quantiles": quantile_dict([e["frac_core"] for e in evs]),
                "frac_noise_quantiles": quantile_dict([e["frac_noise"] for e in evs]),
                "fallback_rate": sum(1 for e in evs if e["fallback"]) / len(evs),
                "n_events": len(evs),
            }

        summary.setdefault(task, {})[lora_repo_id] = {
            "overall": {
                "frac_core_quantiles": quantile_dict(overall_frac_core),
                "frac_noise_quantiles": quantile_dict(overall_frac_noise),
                "k_core_eff_quantiles": quantile_dict(overall_k_core),
                "k_noise_eff_quantiles": quantile_dict(overall_k_noise),
                "fallback_rate": overall_fallback_rate,
                "total_layer_events": overall_count,
            },
            "by_layer": layer_stats,
            "by_module": by_module_stats if by_module_stats else None,
        }

        markdown_sections.append(f"## {task} — {lora_repo_id}")
        run_dirs = run_dirs_by_group[(task, lora_repo_id)]
        calib_values = sorted(v for v in calib_values_by_group[(task, lora_repo_id)] if v is not None)
        markdown_sections.append(
            f"- Runs included: {len(run_dirs)} (calib_samples={calib_values})"
        )
        markdown_sections.append("- Run dirs:")
        for rd in run_dirs:
            markdown_sections.append(f"  - `{rd}`")

        overall = summary[task][lora_repo_id]["overall"]
        markdown_sections.append("")
        markdown_sections.append("| metric | p0 | p10 | p25 | p50 | p75 | p90 | p100 |")
        markdown_sections.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

        def row_line(label: str, qdict: Dict[str, Optional[float]]) -> str:
            vals = [qdict.get(k) for k in ["p0", "p10", "p25", "p50", "p75", "p90", "p100"]]
            return "| " + label + " | " + " | ".join("None" if v is None else f"{v:.6f}" for v in vals) + " |"

        markdown_sections.append(row_line("frac_core", overall["frac_core_quantiles"]))
        markdown_sections.append(row_line("frac_noise", overall["frac_noise_quantiles"]))
        markdown_sections.append(row_line("k_core_eff", overall["k_core_eff_quantiles"]))
        markdown_sections.append(row_line("k_noise_eff", overall["k_noise_eff_quantiles"]))
        fallback_rate = overall["fallback_rate"]
        markdown_sections.append(f"\n- Overall fallback rate: {fallback_rate if fallback_rate is not None else 'None'}")

        # Top layers by mean_frac_core/mean_frac_noise
        layer_items = list(layer_stats.items())
        by_core = sorted(layer_items, key=lambda x: x[1]["mean_frac_core"], reverse=True)[:10]
        by_noise = sorted(layer_items, key=lambda x: x[1]["mean_frac_noise"], reverse=True)[:10]

        def layer_table(title: str, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
            markdown_sections.append(f"\n**{title}**")
            markdown_sections.append("| layer | mean_frac_core | mean_frac_noise | mean_k_core_eff | mean_k_noise_eff | fallback_rate | n_runs |")
            markdown_sections.append("| --- | --- | --- | --- | --- | --- | --- |")
            for layer, stats in items:
                markdown_sections.append(
                    f"| {layer} | {stats['mean_frac_core']:.6f} | {stats['mean_frac_noise']:.6f} | "
                    f"{stats['mean_k_core_eff']:.6f} | {stats['mean_k_noise_eff']:.6f} | "
                    f"{stats['fallback_rate']:.6f} | {stats['n_runs']} |"
                )

        if layer_stats:
            layer_table("Top layers by mean_frac_core", by_core)
            layer_table("Top layers by mean_frac_noise", by_noise)

    # Write outputs
    json_path = out_dir / "z_score_gate_summary.json"
    md_path = out_dir / "z_score_gate_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_content = "# z_score gating summary\n\n"
    md_content += f"- Total z_score runs found: {total_zscore_runs}\n"
    md_content += f"- Runs parsed: {parsed_runs}\n"
    md_content += f"- Runs skipped: {skipped_runs}\n"
    if skip_reasons:
        md_content += "- Skip reasons:\n"
        for reason, count in sorted(skip_reasons.items()):
            md_content += f"  - {reason}: {count}\n"
    if not markdown_sections:
        md_content += "\nNo z_score stats found in run outputs.\n"
    else:
        md_content += "\n" + "\n\n".join(markdown_sections) + "\n"
    md_path.write_text(md_content, encoding="utf-8")

    print(f"[Summary] z_score runs found: {total_zscore_runs}")
    print(f"[Summary] runs parsed: {parsed_runs}")
    print(f"[Summary] runs skipped: {skipped_runs}")
    for reason, count in sorted(skip_reasons.items()):
        print(f"[Summary] skip {reason}: {count}")
    for task, loras in summary.items():
        for lora_repo_id, data in loras.items():
            fallback = data["overall"]["fallback_rate"]
            print(f"[Summary] {task} {lora_repo_id} fallback_rate={fallback}")
    print(f"[Summary] wrote: {json_path}")
    print(f"[Summary] wrote: {md_path}")


if __name__ == "__main__":
    main()
