# Cluster execution status

The study remains blocked before model loading. No scientific endpoints have been measured.

Existing job 7489067 failed CUDA preflight: torch 2.12.1+cu130 could not initialize on the assigned driver. A fresh CUDA 12.8 environment repair is prepared; it has not run.

The saved ledger records 0.004722222222 GPU-hours consumed, 0.000000000000 outstanding maximum GPU-hours, and 29.995277777778 remaining of the 30 GPU-hour ceiling. These are saved records, not live scheduler state.

49 targeted CPU tests passed. The full CPU suite in the repaired environment remains pending.

The CPU recovery submission timed out without a job ID; its state is unknown. Resolve it before retrying. See [the continuation runbook](../../docs/CLUSTER_EXECUTION.md) for exact commands and pending gates.

Recompute this package without loading a model:

```bash
python3 scripts/report_cluster_execution.py --run-dir results/cluster-execution-20261016
```
