# Recurrent neuron transfer study

This project measures whether semantic selectivity, linear readouts, and causal effects of the same physical LoopVL neurons transfer across recurrent passes. It uses controlled 2D scenes with paired edits and independent scene family splits.

Work directly in the editable cluster checkout at `/oscar/home/mliu171/reuseinterp-loopvl/source`. Jobs execute frozen snapshots. [PROGRESS.md](PROGRESS.md) records the current validation gates and remaining work; [CLUSTER_RUNBOOK.md](docs/CLUSTER_RUNBOOK.md) contains the operating commands.

## Current evidence

The saved GPU numerical smoke passed 64 checks on 16 examples. It validates official inference parity and basic capture/intervention behavior at physical layer 0, channel 0. Task competence, population capture, transfer estimates, and candidate specific causal experiments remain pending. The next stage is the 32 example timing and memory pilot.

One exploratory signed-response experiment has also completed: 48 MLP channels on 64 synthetic images across all six L invocations. Median L1/L4 image-response ranking correlation was 0.138 (family-bootstrap 95% interval 0.065–0.200). Both signed image means occurred in 6/48 channels at L1 and 17/48 at L4. This does not establish semantic poles. The [experiment report](results/poles-pilot-20261017/REPORT.md) includes the verified statistics, gallery, and compact data needed to recompute the results.

The implementation includes deterministic data generation, MLP product hooks, semantic encoders and probes, grouped uncertainty, intervention primitives, resumable Slurm jobs, and reproducible synthetic fixtures. Some analysis controls and the full causal collection runner are unfinished. The Goodfire image replication is specified in a plan; the image model trainer, compression retraining, memorization trials, and circuit discovery are not implemented yet.

## Start on the cluster

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
source recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
cd "$REUSE_SOURCE_DIR"
export REUSE_SUBMITTER="$REUSE_SOURCE_DIR/scripts/slurm/submit.py"
python "$REUSE_SUBMITTER" baseline --config configs/pilot.yaml --pilot --dry-run
```

This prints the proposed timing pilot submission without submitting it. Follow the runbook for accounting, script review, submission, and subsequent gates. The editable submitter includes the recovery fix; model jobs continue to use the existing frozen snapshot. Installation and staging are already complete for this checkout.

## Code and artifacts

| Location | Purpose |
| --- | --- |
| `src/reuseinterp/data/` | Scene generation, labels, paired edits, and token alignment |
| `model_adapter.py`, `capture.py`, `numerical.py` under `src/reuseinterp/` | Official LoopVL loading, physical neuron hooks, and numerical validation |
| `encoding.py`, `probes.py`, `controls.py`, `statistics.py` | Selectivity/readout transfer, controls, and uncertainty |
| `interventions.py`, `pipeline.py`, `discovery.py` | Collection, fitting, and the partially implemented causal workflow |
| `protocol.py`, `shards.py`, `report.py` | Confirmation locks, resumable outputs, and reporting |
| `scripts/slurm/`, `configs/`, `tests/` | Scheduled execution, configurations, and engineering tests |
| `../LoopVL/`, `../model/`, `../datasets/` | Staged upstream code, checkpoint, and data |
| `../runs/` | Actual cluster logs, gates, and measurements |
| `results/cpu-preparation-20261016/` | Historical synthetic engineering results |

Python files in the table are relative to `src/reuseinterp/` unless otherwise stated. Config files use JSON syntax despite their `.yaml` extension.

## Documentation

- [Current progress](PROGRESS.md): verified evidence, pending gates, and next work.
- [Cluster runbook](docs/CLUSTER_RUNBOOK.md): activation, execution, accounting, retries, and snapshots.
- [Scientific protocol](CODEX_PLAN.md): endpoints, splits, controls, statistics, and stopping rules.
- [Neuron replication plan](docs/NEURON_REPLICATION_PLAN.md): the proposed image model reproductions and their loop iteration comparisons.
- [Architecture audit](docs/architecture_audit.md): pinned implementation and the scope of loaded model checks.
- [Cluster incident record](docs/CLUSTER_EXECUTION.md): resolved environment failures and evidence locations.
- [New installation reference](docs/BROWN_QUICKSTART.md): provisioning a different cluster base.
- [Archived documentation](docs/archive/20261017/README.md): superseded setup and laptop handoff instructions.

The project shares one limit of 30 GPU hours, including retries, and 20 GiB of research outputs. Refresh accounting before further compute. Model inference and substantial CPU work belong in Slurm; confirmation requires a real immutable protocol lock.
