"""Check that the bundled seed outputs match the paper reproduction manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "manifests" / "reproduce_paper_experiments.jsonl"


def expected_path(raw_root: Path, job: dict) -> Path:
    group = Path(job["output_dir"]).name
    algorithm = job["algorithm"]
    if job.get("tag"):
        algorithm += "__" + job["tag"]
    return raw_root / group / job["benchmark"] / algorithm / f"seed_{job['seed']}.json"


def validate(raw_root: Path) -> int:
    jobs = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    if len(jobs) != 389:
        raise ValueError(f"Expected 389 manifest jobs, found {len(jobs)}")

    expected = set()
    for job in jobs:
        path = expected_path(raw_root, job)
        if path in expected:
            raise ValueError(f"Duplicate manifest output: {path}")
        expected.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("benchmark", "seed", "episodes", "dim", "tag"):
            if payload.get(key) != job.get(key):
                raise ValueError(f"{path}: {key} is {payload.get(key)!r}, expected {job.get(key)!r}")
        for key in ("beta", "tau", "overlap_m"):
            wanted = job.get(key)
            if wanted is not None and (payload.get(key) is None or abs(float(payload[key]) - wanted) > 1e-9):
                raise ValueError(f"{path}: {key} is {payload.get(key)!r}, expected {wanted!r}")

    actual = set(raw_root.rglob("seed_*.json"))
    if actual != expected:
        raise ValueError(f"Result set differs: {len(expected - actual)} missing, {len(actual - expected)} extra")
    return len(jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=REPO / "raw_results")
    args = parser.parse_args()
    count = validate(args.raw_root.resolve())
    print(f"Validated {count} manifest jobs and seed-level JSON files")
