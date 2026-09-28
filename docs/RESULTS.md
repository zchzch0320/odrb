# Paper result mapping

The bundled PDF is the manuscript version used for this artifact. Run the commands in the top-level README to regenerate `reproduced/paper_tables.md` and `reproduced/paired_analysis.md` from the included seed JSON files.

| Paper result | Source in `raw_results/` | Reproduction output |
| --- | --- | --- |
| Main `d=8` linear-mixture table | `p3_linear_dim_sweep/`, `p3_linear_main_extended/`, `p4_linear_deployment_resweep/` | `paper_tables.md`, main table |
| Main paired deployment comparisons and seed differences | Same linear directories | `paired_analysis.md` |
| Appendix dimension sweep | `p3_linear_dim_sweep/`, `p3_linear_main_extended/` | `paper_tables.md`, dimension sweep |
| Appendix `beta x tau` sweep | `p4_linear_deployment_resweep/` | `paper_tables.md`, resweep |
| Tabular sanity games | `p0_tabular_repair/` | `paper_tables.md`, tabular suite |
| Original support-shift calibration | `p1_support_shift/` | `paper_tables.md`, original support shift |
| Redesigned support-shift and stage-0 policy diagnostics | `p3_support_shift_beta_sweep/` | `paper_tables.md`, redesigned support shift and policy diagnostics |

The manifest identifies the exact algorithms, hyperparameters, evaluation frequency, episode counts, and seeds. Each result JSON stores the final metrics plus checkpoint histories. `code/validate_artifact.py` verifies the one-to-one correspondence between all 389 manifest jobs and bundled JSON files.

## Interpretation

The main table reports a backup-based `Proxy gap`; the `Deploy` column is a direct worst-case deployment evaluation. The ten-seed paired deployment comparison gives ODRB `(0.30, 0.02)` minus Sigma0 as `0.009423` with 95% paired-t interval `[0.000694, 0.018153]` and Holm-adjusted exact sign-flip `p=0.046875`. The `(0.20, 0.02)` comparison gives `0.006516`, interval `[-0.001725, 0.014757]`, and adjusted `p=0.097656`.

The table generator uses `1.96 * SEM` for displayed per-method intervals; the paired comparison uses the Student-t critical value. The main table and dimension sweep display the proxy Nash gap, while the `beta x tau` resweep displays the deployment Nash gap. The raw JSON contains both metrics. The main `(0.20, 0.02)` proxy-gap result is `0.0069 +/- 0.0066`, matching the manuscript. These small-sample experiments support a narrow operating-point result, not uniform superiority across benchmarks or dimensions.
