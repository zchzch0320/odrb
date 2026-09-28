# ODRB: robust zero-sum Markov game experiments

Research code and seed-level results for **Toward Online Robust Zero-Sum Markov Games with Function Approximation**. This repository reproduces the experimental section of the accompanying [paper](paper/neurips_2026.pdf): linear-mixture games, two exact tabular games, and the original and redesigned support-shift benchmarks. The paper's theoretical proofs are in the PDF; this repository contains the experimental implementation.

The main comparison uses ten paired environment seeds at feature dimension `d=8` and 5,000 episodes. The reported deployment-return difference for ODRB `(beta=0.30, tau=0.02)` versus the matched Sigma0 baseline is `0.009423`, with a paired-t 95% interval `[0.000694, 0.018153]` and Holm-adjusted exact sign-flip `p=0.046875`. The `(0.20, 0.02)` comparison is less conclusive. See [results and interpretation](docs/RESULTS.md) for the scope of these claims.

## Contents

| Path | Purpose |
| --- | --- |
| `code/gfa/` | Benchmarks, learners, zero-sum stage solvers, and utilities |
| `code/run_experiment.py` | Run one seed and configuration |
| `code/run_manifest.py` | Run the exact experiment manifest sequentially, with resume support |
| `code/manifests/reproduce_paper_experiments.jsonl` | The 389 jobs used for the reported results |
| `raw_results/` | All 389 seed-level JSON outputs corresponding to that manifest |
| `code/summarize_paper_results.py` | Regenerate the main and appendix result tables |
| `code/paired_analysis.py` | Regenerate paired differences, intervals, and adjusted p-values |
| `code/validate_artifact.py` | Check manifest-to-result completeness and metadata |
| `reproduced/` | Generated tables and paired analysis from the included raw results |
| `paper/neurips_2026.pdf` | Paper version used to prepare this artifact |

## Regenerate the reported results

With Python 3.9 or newer, run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python code/validate_artifact.py
python code/summarize_paper_results.py --raw-root raw_results --output reproduced/paper_tables.md
python code/paired_analysis.py
```

The table summarizer requires only the Python standard library. The paired analysis uses SciPy. The generated [`paper_tables.md`](reproduced/paper_tables.md) and [`paired_analysis.md`](reproduced/paired_analysis.md) show the expected output. The paired result for the primary setting is the exact statistic quoted in the paper.

## Run experiments from scratch

The exact configurations and seeds are already frozen in the JSONL manifest. From the repository root:

```bash
python code/run_manifest.py --manifest code/manifests/reproduce_paper_experiments.jsonl --log-dir reproduce_logs --fail-fast
python code/summarize_paper_results.py --raw-root code --output reproduced/fresh_paper_tables.md
python code/paired_analysis.py --raw-root code/queue_results_fix --output-prefix reproduced/fresh_paired_analysis
```

The runner writes new outputs below `code/queue_results/` and `code/queue_results_fix/`, skips completed outputs on subsequent runs, and writes per-job logs below `code/reproduce_logs/`. These generated directories are ignored by Git. Use `--max-jobs 1` for a quick runner check. See [compute resources](docs/COMPUTE.md) before starting the full queue.

For a direct single-run example:

```bash
python code/run_experiment.py --benchmark linear_mixture --algorithm route_a_linear --episodes 200 --eval-every 100 --seed 0 --dim 8 --beta 0.30 --tau 0.02 --output-dir code/smoke_results
```

## Reading the results

`Proxy` is the backup-based worst-case estimate. `Deploy` evaluates the learned policy directly against the perturbed deployment family. `Gap` is an approximate Nash-gap diagnostic. Sigma0 disables the robust term in the matched implementation; NoEx removes exploiter-assisted collection. The different metrics can rank settings differently, so the paper reports the deployment metric and the gap together.

The supported empirical claim is setting-specific: the main paired `d=8` ODRB operating point has a modest deployment-return gain over Sigma0. The tabular attack-defense game does not show that gain, and the original support-shift benchmark is nearly tied. The redesigned support-shift game separates the methods and includes policy-action diagnostics. See [table-to-file mapping](docs/RESULTS.md).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). Code, experiment outputs, and repository documentation are released under the [MIT license](LICENSE). The included paper PDF is a reference copy and remains under the authors' copyright.
