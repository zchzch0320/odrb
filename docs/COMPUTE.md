# Compute resources

The paper reports CPU-bound NumPy/SciPy experiments run on an internal multi-core workstation/server. Four NVIDIA RTX 2080 Ti GPUs (11 GB each) were available, but the reported experiments did not use them. Each manifest job runs as one Python process. No external datasets are needed.

| Experiment group | Jobs | Episodes/job | Seeds/configuration | Approximate time/job |
| --- | ---: | ---: | ---: | ---: |
| Tabular sanity suite | 54 | 1,500 | 3 | under 1 minute |
| Original support-shift calibration | 30 | 2,000 | 3 | under 1 minute |
| Linear-mixture dimension sweep | 45 | 5,000 | 5 | under 2 minutes |
| Linear-mixture `d=8` seed extension | 15 | 5,000 | 5 | under 2 minutes |
| Redesigned support-shift sweep | 125 | 2,000 | 5 | under 1 minute |
| Deployment-driven `beta x tau` sweep | 120 | 5,000 | 10 | under 2 minutes |
| **Reported reproduction manifest** | **389** | | | **A few CPU-hours sequentially** |

The 389 bundled seed-level JSON files occupy about 7.9 MB. The original research also used preliminary debugging, calibration, and unsuccessful-setting diagnostics beyond this frozen reproduction manifest; the local exploratory and reported archive contains 576 seed-level outputs (about 11.1 MB). Only the 389 files needed for the reported tables are included here. Runtime ranges are approximate measurements from the original workstation and will vary by CPU and software environment.
