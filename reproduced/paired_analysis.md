# Paired deployment analysis

Primary metric: `deployment_worst_case_return` at the final (5000-episode) checkpoint.

| Comparison | n | Mean paired difference | 95% paired-t CI | Exact sign-flip p | Holm p | dz | W/T/L | Approx. n for 80% power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ODRB(0.30,0.02) vs Sigma0 | 10 | 0.009423 | [0.000694, 0.018153] | 0.023438 | 0.046875 | 0.772187 | 7/1/2 | 14 |
| ODRB(0.20,0.02) vs Sigma0 | 10 | 0.006516 | [-0.001725, 0.014757] | 0.097656 | 0.097656 | 0.565649 | 6/1/3 | 25 |

## Per-seed differences

### ODRB(0.30,0.02) vs Sigma0

| Seed | Sigma0 | ODRB | ODRB - Sigma0 |
|---:|---:|---:|---:|
| 0 | 0.037972 | 0.040042 | 0.002070 |
| 1 | 0.218235 | 0.243172 | 0.024937 |
| 2 | 0.202470 | 0.234312 | 0.031841 |
| 3 | 0.289614 | 0.310578 | 0.020964 |
| 4 | 0.150743 | 0.148961 | -0.001782 |
| 5 | 0.009702 | 0.009617 | -0.000085 |
| 6 | 0.383972 | 0.388990 | 0.005018 |
| 7 | 0.301764 | 0.312736 | 0.010971 |
| 8 | 0.231980 | 0.231980 | 0.000000 |
| 9 | 0.470547 | 0.470846 | 0.000299 |

### ODRB(0.20,0.02) vs Sigma0

| Seed | Sigma0 | ODRB | ODRB - Sigma0 |
|---:|---:|---:|---:|
| 0 | 0.037972 | 0.039987 | 0.002015 |
| 1 | 0.218235 | 0.226773 | 0.008538 |
| 2 | 0.202470 | 0.234561 | 0.032091 |
| 3 | 0.289614 | 0.305990 | 0.016377 |
| 4 | 0.150743 | 0.149013 | -0.001730 |
| 5 | 0.009702 | 0.008232 | -0.001470 |
| 6 | 0.383972 | 0.378689 | -0.005283 |
| 7 | 0.301764 | 0.316288 | 0.014524 |
| 8 | 0.231980 | 0.231980 | 0.000000 |
| 9 | 0.470547 | 0.470648 | 0.000102 |

## Interpretation rule

The paper reports a narrow primary-setting advantage only when the paired-t 95% interval is positive and the Holm-adjusted exact sign-flip p-value is below 0.05. These small-sample tests do not establish uniform superiority.
