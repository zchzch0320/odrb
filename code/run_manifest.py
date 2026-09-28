from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run a JSONL manifest sequentially with resume support.")
    parser.add_argument("--manifest", required=True, help="Path to a JSONL manifest.")
    parser.add_argument("--python", default=sys.executable, help="Python binary to use.")
    parser.add_argument("--log-dir", default="queue_logs", help="Directory for status/log files.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Optional cap for testing.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on the first failure.")
    parser.add_argument("--force", action="store_true", help="Run even when an output file already exists.")
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize(text: str, limit: int = 100) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return clean[:limit].strip("_") or "job"


def output_path(job: dict) -> Path:
    algo_dir = job["algorithm"] if not job.get("tag") else f"{job['algorithm']}__{job['tag']}"
    return ROOT / job["output_dir"] / job["benchmark"] / algo_dir / f"seed_{job['seed']}.json"


def job_args(job: dict) -> list[str]:
    args = [
        "run_experiment.py",
        "--benchmark",
        str(job["benchmark"]),
        "--algorithm",
        str(job["algorithm"]),
        "--episodes",
        str(job["episodes"]),
        "--eval-every",
        str(job["eval_every"]),
        "--seed",
        str(job["seed"]),
        "--output-dir",
        str(job["output_dir"]),
    ]
    optional = {
        "--dim": job.get("dim"),
        "--beta": job.get("beta"),
        "--tau": job.get("tau"),
        "--overlap-m": job.get("overlap_m"),
        "--tag": job.get("tag"),
    }
    for key, value in optional.items():
        if value is not None and value != "":
            args.extend([key, str(value)])
    return args


def load_jobs(path: Path) -> list[dict]:
    jobs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def write_status_line(path: Path, fields: list[str]):
    with path.open("a", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    jobs = load_jobs(manifest_path)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    run_name = f"{manifest_path.stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / args.log_dir / run_name
    ensure_dir(run_dir)
    status_path = run_dir / "status.tsv"
    write_status_line(status_path, ["timestamp", "status", "group", "job_name", "output", "message"])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    completed = 0
    skipped = 0
    failed = 0
    for idx, job in enumerate(jobs):
        out_path = output_path(job)
        job_name = sanitize(
            f"{idx:04d}_{job['group']}_{job['benchmark']}_{job['algorithm']}_seed{job['seed']}_{job.get('tag', '')}"
        )
        job_hash = hashlib.sha1(json.dumps(job, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        log_path = run_dir / f"{job_name}__{job_hash}.log"
        ts = datetime.now().strftime("%F %T")

        if out_path.exists() and not args.force:
            skipped += 1
            write_status_line(status_path, [ts, "SKIP", job["group"], job_name, str(out_path), "output_exists"])
            print(f"[SKIP] {job_name} -> {out_path}")
            continue

        cmd = [args.python] + job_args(job)
        write_status_line(status_path, [ts, "START", job["group"], job_name, str(out_path), " ".join(cmd)])
        print(f"[START] {job_name}")
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        ts = datetime.now().strftime("%F %T")
        if proc.returncode == 0 and out_path.exists():
            completed += 1
            write_status_line(status_path, [ts, "DONE", job["group"], job_name, str(out_path), str(log_path)])
            print(f"[DONE] {job_name}")
        else:
            failed += 1
            write_status_line(
                status_path,
                [ts, "FAIL", job["group"], job_name, str(out_path), f"code={proc.returncode} log={log_path}"],
            )
            print(f"[FAIL] {job_name} (see {log_path})")
            if args.fail_fast:
                break

    print("SUMMARY")
    print(f"  completed={completed}")
    print(f"  skipped={skipped}")
    print(f"  failed={failed}")
    print(f"  status={status_path}")


if __name__ == "__main__":
    main()

