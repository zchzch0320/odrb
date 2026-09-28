from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Recreate paper experiment tables from seed JSON files.")
    parser.add_argument("--raw-root", default="../raw_results", help="Directory containing result subfolders.")
    parser.add_argument("--output", default="../paper_tables_reproduced.md", help="Markdown output path.")
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for seed_file in sorted(root.rglob("seed_*.json")):
        with seed_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload["metrics"]
        proxy_worst = float(metrics["worst_case_return"][-1])
        proxy_gap = float(metrics["nash_gap"]["0.0"][-1])
        deploy_worst = float(metrics.get("deployment_worst_case_return", metrics["worst_case_return"])[-1])
        deploy_gap_source = metrics.get("deployment_nash_gap", metrics["nash_gap"])
        deploy_gap = float(deploy_gap_source["0.0"][-1])
        row = {
            "path": str(seed_file),
            "benchmark": payload["benchmark"],
            "algorithm": payload["algorithm"],
            "seed": int(payload["seed"]),
            "dim": payload.get("dim"),
            "beta": payload.get("beta"),
            "tau": payload.get("tau"),
            "tag": payload.get("tag", ""),
            "overlap_m": payload.get("overlap_m"),
            "nominal": float(metrics["returns"]["0.0"][-1]),
            "proxy_worst": proxy_worst,
            "proxy_gap": proxy_gap,
            "deploy_worst": deploy_worst,
            "deploy_gap": deploy_gap,
            "entropy": float(metrics.get("occupancy_entropy", [0.0])[-1]),
            "stage0_safe": None,
            "stage0_aggressive": None,
        }
        p0_mass = payload.get("stage0_player0_action_mass")
        if p0_mass is not None and len(p0_mass) >= 2:
            row["stage0_safe"] = float(p0_mass[0])
            row["stage0_aggressive"] = float(p0_mass[1])
        rows.append(row)
    return rows


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(max(var, 0.0) / len(values))


def fmt(values: list[float], digits: int = 4) -> str:
    mean, ci = mean_ci95(values)
    return f"{mean:.{digits}f} +/- {ci:.{digits}f}"


def is_beta(row: dict, beta: float) -> bool:
    return row["beta"] is not None and abs(float(row["beta"]) - beta) < 1e-9


def is_tau(row: dict, tau: float) -> bool:
    return row["tau"] is not None and abs(float(row["tau"]) - tau) < 1e-9


def select_linear_main(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {
        "Sigma0": [],
        "ODRB(beta=0.30,tau=0.02)": [],
        "ODRB(beta=0.20,tau=0.02)": [],
        "NoEx(beta=0.30)": [],
    }
    for row in rows:
        if row["benchmark"] != "linear_mixture" or row["dim"] != 8:
            continue
        path = row["path"].replace("\\", "/")
        from_p3 = "/p3_linear_dim_sweep/" in path or "/p3_linear_main_extended/" in path
        from_p4 = "/p4_linear_deployment_resweep/" in path
        if from_p3 and row["algorithm"] == "Sigma0-Linear":
            out["Sigma0"].append(row)
        elif from_p3 and row["algorithm"] == "Route-A-Linear" and is_beta(row, 0.30) and is_tau(row, 0.02):
            out["ODRB(beta=0.30,tau=0.02)"].append(row)
        elif from_p4 and row["algorithm"] == "Route-A-Linear" and is_beta(row, 0.20) and is_tau(row, 0.02):
            out["ODRB(beta=0.20,tau=0.02)"].append(row)
        elif from_p3 and row["algorithm"] == "No-Exploiter-Linear" and is_beta(row, 0.30):
            out["NoEx(beta=0.30)"].append(row)
    return out


def select_linear_dims(rows: list[dict]) -> dict[tuple[int, str], list[dict]]:
    out: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        path = row["path"].replace("\\", "/")
        if row["benchmark"] != "linear_mixture" or "/p3_" not in path:
            continue
        if row["dim"] not in {8, 16, 32}:
            continue
        if row["algorithm"] == "Sigma0-Linear":
            out[(int(row["dim"]), "Sigma0")].append(row)
        elif row["algorithm"] == "Route-A-Linear" and is_beta(row, 0.30) and is_tau(row, 0.02):
            out[(int(row["dim"]), "ODRB(beta=0.30,tau=0.02)")].append(row)
        elif row["algorithm"] == "No-Exploiter-Linear" and is_beta(row, 0.30):
            out[(int(row["dim"]), "NoEx(beta=0.30)")].append(row)
    return out


def select_resweep(rows: list[dict]) -> dict[str, list[dict]]:
    wanted = [(0.20, 0.01), (0.20, 0.02), (0.30, 0.02), (0.30, 0.0), (0.50, 0.02)]
    out: dict[str, list[dict]] = {f"ODRB(beta={b:.2f},tau={t:g})": [] for b, t in wanted}
    for row in rows:
        path = row["path"].replace("\\", "/")
        if row["benchmark"] != "linear_mixture" or "/p4_linear_deployment_resweep/" not in path:
            continue
        for beta, tau in wanted:
            if row["algorithm"] == "Route-A-Linear" and is_beta(row, beta) and is_tau(row, tau):
                out[f"ODRB(beta={beta:.2f},tau={tau:g})"].append(row)
    return out


def select_support_shift(rows: list[dict]) -> dict[tuple[float, str], list[dict]]:
    out: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for row in rows:
        path = row["path"].replace("\\", "/")
        if row["benchmark"] != "support_shift" or "/p3_support_shift_beta_sweep/" not in path:
            continue
        if row["algorithm"] == "Sigma0-Tabular":
            name = "Sigma0"
        elif row["algorithm"] == "Route-A-Tabular":
            name = f"ODRB(beta={float(row['beta']):.2f})"
        elif row["algorithm"] == "No-Exploiter-Tabular":
            name = f"NoEx(beta={float(row['beta']):.2f})"
        else:
            continue
        out[(float(row["overlap_m"]), name)].append(row)
    return out


def select_original_support_shift(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        path = row["path"].replace("\\", "/")
        if row["benchmark"] != "support_shift" or "/p1_support_shift/" not in path:
            continue
        if row["algorithm"] == "Sigma0-Tabular":
            out["Sigma0"].append(row)
        elif row["algorithm"] == "Route-A-Tabular" and is_beta(row, 0.20):
            out["ODRB(beta=0.20)"].append(row)
    return out


def select_tabular(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        path = row["path"].replace("\\", "/")
        if "/p0_tabular_repair/" not in path:
            continue
        if row["algorithm"] == "Sigma0-Tabular":
            name = "Sigma0"
        elif row["algorithm"] == "Route-A-Tabular" and is_beta(row, 0.20):
            name = "ODRB(beta=0.20)"
        elif row["algorithm"] == "No-Exploiter-Tabular" and is_beta(row, 0.20):
            name = "NoEx(beta=0.20)"
        else:
            continue
        out[(row["benchmark"], name)].append(row)
    return out


def render(output: Path, rows: list[dict]) -> None:
    lines: list[str] = [
        "# Reproduced Experiment Tables",
        "",
        "Values are mean +/- 95% confidence interval over seeds.",
        "",
        "## Main linear_mixture table",
        "",
        "| Method | n | Proxy | Deploy | Proxy gap |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, group in select_linear_main(rows).items():
        lines.append(
            f"| {name} | {len(group)} | {fmt([r['proxy_worst'] for r in group], 3)} | "
            f"{fmt([r['deploy_worst'] for r in group], 3)} | {fmt([r['proxy_gap'] for r in group])} |"
        )

    lines += [
        "",
        "## Linear dimension sweep",
        "",
        "| d | Method | n | Proxy | Deploy | Gap |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ]
    for (dim, name), group in sorted(select_linear_dims(rows).items()):
        lines.append(
            f"| {dim} | {name} | {len(group)} | {fmt([r['proxy_worst'] for r in group])} | "
            f"{fmt([r['deploy_worst'] for r in group])} | {fmt([r['proxy_gap'] for r in group])} |"
        )

    lines += [
        "",
        "## d=8 deployment-driven beta x tau resweep",
        "",
        "| Setting | n | Proxy | Deploy | Gap | Entropy |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for name, group in select_resweep(rows).items():
        lines.append(
            f"| {name} | {len(group)} | {fmt([r['proxy_worst'] for r in group], 3)} | "
            f"{fmt([r['deploy_worst'] for r in group], 3)} | {fmt([r['deploy_gap'] for r in group])} | "
            f"{fmt([r['entropy'] for r in group], 3)} |"
        )

    support = select_support_shift(rows)
    methods = ["Sigma0", "ODRB(beta=0.10)", "ODRB(beta=0.20)", "ODRB(beta=0.30)", "NoEx(beta=0.20)"]
    overlaps = sorted({key[0] for key in support})
    lines += [
        "",
        "## Redesigned support-shift deployment worst-case",
        "",
        "| Method | " + " | ".join(f"m={m:.2f}" for m in overlaps) + " |",
        "| --- | " + " | ".join("---:" for _ in overlaps) + " |",
    ]
    for method in methods:
        vals = []
        for overlap in overlaps:
            group = support.get((overlap, method), [])
            vals.append(f"{mean_ci95([r['deploy_worst'] for r in group])[0]:.4f}" if group else "")
        lines.append("| " + method + " | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Support-shift policy diagnostics",
        "",
        "| overlap_m | Method | n | Safe-dominant seeds | Aggressive-dominant seeds | Mean aggressive mass |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for (overlap, method), group in sorted(support.items()):
        safe = sum(1 for r in group if (r["stage0_safe"] or 0.0) > 0.5)
        aggressive = sum(1 for r in group if (r["stage0_aggressive"] or 0.0) > 0.5)
        lines.append(
            f"| {overlap:.2f} | {method} | {len(group)} | {safe} | {aggressive} | "
            f"{fmt([r['stage0_aggressive'] for r in group if r['stage0_aggressive'] is not None])} |"
        )

    lines += [
        "",
        "## Original support-shift calibration",
        "",
        "| Method | n | Worst-case | Gap |",
        "| --- | ---: | --- | --- |",
    ]
    for method, group in sorted(select_original_support_shift(rows).items()):
        lines.append(
            f"| {method} | {len(group)} | {fmt([r['proxy_worst'] for r in group])} | "
            f"{fmt([r['proxy_gap'] for r in group])} |"
        )

    lines += [
        "",
        "## Tabular sanity suite",
        "",
        "| Benchmark | Method | n | Nominal | Worst-case | Gap |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for (benchmark, method), group in sorted(select_tabular(rows).items()):
        lines.append(
            f"| {benchmark} | {method} | {len(group)} | {fmt([r['nominal'] for r in group])} | "
            f"{fmt([r['proxy_worst'] for r in group])} | {fmt([r['proxy_gap'] for r in group])} |"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    output = Path(args.output).resolve()
    rows = load_rows(raw_root)
    render(output, rows)
    print(f"loaded {len(rows)} seed files")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
