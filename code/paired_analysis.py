#!/usr/bin/env python3
"""Reproduce the paired deployment comparisons reported in the paper.

The script analyzes the two ODRB settings displayed in the main table.
It does not tune or select a new configuration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from scipy.stats import t as student_t
except Exception as exc:  # pragma: no cover - environment diagnostic
    raise RuntimeError("scipy is required for the paired-t confidence interval") from exc


MetricMap = Mapping[int, float]


def load_final_metric(
    paths: Iterable[Path],
    metric: str,
    fallback_metrics: Sequence[str] = (),
) -> Dict[int, float]:
    values: Dict[int, float] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if payload.get("benchmark") != "linear_mixture":
            raise ValueError(f"{path}: unexpected benchmark {payload.get('benchmark')!r}")
        if int(payload.get("episodes", -1)) != 5000 or int(payload.get("dim", -1)) != 8:
            raise ValueError(
                f"{path}: expected 5000 episodes and dim=8, found "
                f"{payload.get('episodes')} episodes and dim={payload.get('dim')}"
            )
        metric_name = next(
            (name for name in (metric, *fallback_metrics) if name in payload["metrics"]),
            None,
        )
        if metric_name is None:
            raise KeyError(
                f"{path}: none of {(metric, *fallback_metrics)!r} is present"
            )
        series = payload["metrics"][metric_name]
        if not isinstance(series, list) or not series:
            raise ValueError(f"{path}: metric {metric!r} is not a nonempty list")
        if seed in values:
            raise ValueError(f"duplicate seed {seed} for metric {metric}")
        values[seed] = float(series[-1])
    return values


def merge_disjoint(*maps: MetricMap) -> Dict[int, float]:
    merged: Dict[int, float] = {}
    for values in maps:
        overlap = set(merged).intersection(values)
        if overlap:
            raise ValueError(f"duplicate seeds across result roots: {sorted(overlap)}")
        merged.update(values)
    return merged


def assert_complete(values: MetricMap, expected: Sequence[int], label: str) -> None:
    found = sorted(values)
    wanted = list(expected)
    if found != wanted:
        raise ValueError(f"{label}: expected seeds {wanted}, found {found}")


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Exact two-sided randomization p-value for paired differences."""
    observed = abs(statistics.mean(differences))
    n = len(differences)
    if n > 20:
        raise ValueError("exact enumeration is intentionally limited to at most 20 pairs")
    extreme = 0
    total = 0
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        permuted = abs(sum(sign * diff for sign, diff in zip(signs, differences)) / n)
        extreme += int(permuted + tolerance >= observed)
        total += 1
    return extreme / total


def paired_summary(treatment: MetricMap, control: MetricMap) -> dict:
    seeds = sorted(set(treatment).intersection(control))
    if seeds != sorted(treatment) or seeds != sorted(control):
        raise ValueError("paired methods do not contain identical seed sets")
    differences = [treatment[s] - control[s] for s in seeds]
    n = len(differences)
    mean_difference = statistics.mean(differences)
    sd_difference = statistics.stdev(differences)
    standard_error = sd_difference / math.sqrt(n)
    critical = float(student_t.ppf(0.975, df=n - 1))
    half_width = critical * standard_error
    p_exact = exact_sign_flip_pvalue(differences)
    dz = mean_difference / sd_difference if sd_difference > 0 else math.inf
    projected_n = (
        math.ceil(((1.959963984540054 + 0.8416212335729143) * sd_difference / abs(mean_difference)) ** 2)
        if mean_difference != 0
        else math.inf
    )
    return {
        "n": n,
        "seeds": seeds,
        "control": [control[s] for s in seeds],
        "treatment": [treatment[s] for s in seeds],
        "paired_differences": differences,
        "mean_difference": mean_difference,
        "sd_difference": sd_difference,
        "ci95": [mean_difference - half_width, mean_difference + half_width],
        "exact_sign_flip_p_two_sided": p_exact,
        "cohen_dz": dz,
        "wins_ties_losses": [
            sum(diff > 0 for diff in differences),
            sum(diff == 0 for diff in differences),
            sum(diff < 0 for diff in differences),
        ],
        "approx_n_for_80pct_power_at_observed_effect": projected_n,
    }


def holm_adjust(pvalues: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (m - rank) * pvalue)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def collect_method(
    metric: str,
    roots: Sequence[Path],
    fallback_metrics: Sequence[str] = (),
) -> Dict[int, float]:
    pieces = [
        load_final_metric(
            sorted(root.glob("seed_*.json")),
            metric,
            fallback_metrics=fallback_metrics,
        )
        for root in roots
    ]
    return merge_disjoint(*pieces)


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def markdown_report(results: Mapping[str, dict], metric: str) -> str:
    lines: List[str] = [
    "# Paired deployment analysis",
        "",
        f"Primary metric: `{metric}` at the final (5000-episode) checkpoint.",
        "",
        "| Comparison | n | Mean paired difference | 95% paired-t CI | Exact sign-flip p | Holm p | dz | W/T/L | Approx. n for 80% power |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        ci_low, ci_high = result["ci95"]
        wins, ties, losses = result["wins_ties_losses"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(result["n"]),
                    format_float(result["mean_difference"]),
                    f"[{format_float(ci_low)}, {format_float(ci_high)}]",
                    format_float(result["exact_sign_flip_p_two_sided"]),
                    format_float(result["holm_adjusted_p"]),
                    format_float(result["cohen_dz"]),
                    f"{wins}/{ties}/{losses}",
                    str(result["approx_n_for_80pct_power_at_observed_effect"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Per-seed differences", ""])
    for name, result in results.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Seed | Sigma0 | ODRB | ODRB - Sigma0 |")
        lines.append("|---:|---:|---:|---:|")
        for seed, control, treatment, difference in zip(
            result["seeds"],
            result["control"],
            result["treatment"],
            result["paired_differences"],
        ):
            lines.append(
                f"| {seed} | {control:.6f} | {treatment:.6f} | {difference:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation rule",
            "",
            "The paper reports a narrow primary-setting advantage only when the paired-t "
            "95% interval is positive and the Holm-adjusted exact sign-flip p-value is below 0.05. "
            "These small-sample tests do not establish uniform superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "raw_results",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduced" / "paired_analysis",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    expected_seeds = list(range(10))
    metric = "deployment_worst_case_return"

    baseline_roots = [
        raw_root
        / "p3_linear_dim_sweep/linear_mixture/sigma0_linear__d8__baseline",
        raw_root
        / "p3_linear_main_extended/linear_mixture/sigma0_linear__baseline",
    ]
    odrb_roots = {
        "ODRB(0.30,0.02) vs Sigma0": [
            raw_root
            / "p3_linear_dim_sweep/linear_mixture/route_a_linear__d8__beta0p30__tau0p02",
            raw_root
            / "p3_linear_main_extended/linear_mixture/route_a_linear__beta0p30__tau0p02",
        ],
        "ODRB(0.20,0.02) vs Sigma0": [
            raw_root
            / "p4_linear_deployment_resweep/linear_mixture/route_a_linear__beta0p2__tau0p02"
        ],
    }

    # The earliest Sigma0 files predate the explicit deployment field. For
    # Sigma0 (robustness radius zero), deployment and proxy evaluation coincide.
    baseline = collect_method(
        metric,
        baseline_roots,
        fallback_metrics=("worst_case_return",),
    )
    assert_complete(baseline, expected_seeds, "Sigma0")

    results: Dict[str, dict] = {}
    for name, roots in odrb_roots.items():
        treatment = collect_method(metric, roots)
        assert_complete(treatment, expected_seeds, name)
        results[name] = paired_summary(treatment, baseline)

    adjusted = holm_adjust(
        {name: result["exact_sign_flip_p_two_sided"] for name, result in results.items()}
    )
    for name, result in results.items():
        result["holm_adjusted_p"] = adjusted[name]

    output_prefix = args.output_prefix
    if not output_prefix.is_absolute():
        output_prefix = Path(__file__).resolve().parents[1] / output_prefix
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(results, metric), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
