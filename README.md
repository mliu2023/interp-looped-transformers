# Recurrent neuron transfer study

This project measures whether semantic selectivity, linear readouts, and causal effects of the same physical LoopVL neurons transfer across recurrent passes. The primary next experiments use images from the training data of the pinned LoopVL checkpoint. Controlled 2D scenes remain engineering fixtures and historical pilot data.

Work directly in the editable cluster checkout at `/oscar/home/mliu171/reuseinterp-loopvl/source`. Jobs execute frozen snapshots. [PROGRESS.md](PROGRESS.md) records the current validation gates and remaining work; [CLUSTER_RUNBOOK.md](docs/CLUSTER_RUNBOOK.md) contains the operating commands.

## Interactive activity explorer

The [viewer guide](docs/INTERACTIVE_VIEWER.md) covers local SSH access, portable bundles, and restarting the CPU server. The viewer has physical layer/channel and MLP-stream selectors, response-ranked image galleries, signed spatial overlays across all six L invocations, image-response trajectories, and a clickable rank-transfer scatter. Browsing uses saved captures and no GPU.

The real-image rerun is complete: 64 original OneVision instruction images (22 COCO, 21 GQA, 21 Visual Genome), 48 presampled channels, all six L invocations, and eight prompt controls. GPU job `7522738` captured the data; CPU job `7522739` verified all 72 raw archives and original images, recomputed the statistics, and passed Chromium checks. Viewer job `7523389` serves the verified bundle. Local connections default to **127.0.0.1:8766**. See the [experiment report](results/training-images-20261017/REPORT.md) and [source audit](docs/training_image_sources.md). These are training-source images; exact checkpoint retention remains unverified.

On Oscar, print the pipeline status and, when ready, your local connection commands:

```bash
python scripts/viewer_connection.py --experiment-dir /oscar/home/mliu171/reuseinterp-loopvl/experiments/loopvl-training-images-20261017
```

## Current evidence

The saved GPU numerical smoke passed 64 checks on 16 examples. It validates official inference parity and basic capture/intervention behavior at physical layer 0, channel 0. The new real-image pilot also passed first-image official-logit and full-MLP reconstruction checks at all selected layers and invocations. Task competence, the original population capture, and candidate-specific causal experiments remain pending.

On the 64 real source images, median L1/L4 image-ranking correlation was **0.040** (family-bootstrap 95% interval **−0.009 to 0.104**). Signed image-mean coverage varied across invocations; signs and ranking changes do not establish semantic poles. Full measurements and limitations are in the experiment report above.

One exploratory signed-response experiment has also completed: 48 MLP channels on 64 synthetic images across all six L invocations. Median L1/L4 image-response ranking correlation was 0.138 (family-bootstrap 95% interval 0.065–0.200). Both signed image means occurred in 6/48 channels at L1 and 17/48 at L4. This does not establish semantic poles. The [experiment report](results/poles-pilot-20261017/REPORT.md) includes the verified statistics, gallery, and compact data needed to recompute the results.

The implementation includes deterministic data generation, MLP product hooks, semantic encoders and probes, grouped uncertainty, intervention primitives, resumable Slurm jobs, and reproducible synthetic fixtures. Some analysis controls and the full causal collection runner are unfinished. The Goodfire image replication is specified in a plan; the image model trainer, compression retraining, memorization trials, and circuit discovery are not implemented yet.

## Start on the cluster

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
source recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
cd "$REUSE_SOURCE_DIR"
export REUSE_SUBMITTER="$REUSE_SOURCE_DIR/scripts/slurm/submit.py"
python "$REUSE_SUBMITTER" account
```

This prints the recorded shared GPU budget. Follow the replication plan and runbook for scheduler accounting, script review, and submission. The existing `poles-pilot.yaml` freezes the old shape experiment. `loopvl-training-images-timing.json` and `loopvl-training-images.json` define the completed native-image checks and full pilot. Inspect the saved report and pipeline state before submitting another run.

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
