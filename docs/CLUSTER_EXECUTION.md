# Cluster environment incident record

Use [PROGRESS.md](../PROGRESS.md) for current validation status and [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) for operating commands. This page records the resolved environment incident and where to find its evidence.

| Event | Outcome | Evidence |
| --- | --- | --- |
| GPU job `7489067` | CUDA 13 PyTorch failed initialization before model loading against the assigned driver | [stderr](../../runs/logs/7489067.err), [saved execution records](../results/cluster-execution-20261016/) |
| CPU recovery job `7496249` | Created isolated torch 2.8.0+cu128 / torchvision 0.23.0+cu128 environment, verified inputs, ran tests, and froze a new snapshot | [Recovery state](../../recovery-cu128-20261016/recovery_state.json), [test log](../../recovery-cu128-20261016/tests.log) |
| GPU smoke job `7501493` | Saved CUDA preflight and diagnostic numerical parity artifacts passed | [Preflight](../../runs/preflight/7501493_single.json), [parity report](../../runs/loopvl-transfer-v1/parity_report.json) |

The recovery reused the staged model, data, and shared run root. It preserved the original environment and snapshot. Activate `recovery-cu128-20261016/cluster.env` under the study base for the working runtime.

The editable `scripts/slurm/submit.py` contains the failed validation snapshot retry fix. It allows the documented repaired validation transition after a known terminal failure; it still rejects active/unknown duplicates and incompatible research hashes. Keep using that wrapper with the immutable experiment snapshot.

The earlier SSH, socket, and uncertain CPU submission observations describe previous sessions. They are not current blockers. The [archived incident narrative](archive/20261017/CLUSTER_EXECUTION.md) retains their diagnostics; raw job outputs and accounting evidence remain unchanged. Historical reports under `results/cluster-execution-20261016/` describe their recorded attempt and should not be read as the current project status.

A passing parity artifact does not reconcile scheduler accounting. Refresh the single shared ledger before any new allocation; retain unresolved reservations until scheduler evidence establishes their actual usage.
