# Cluster operating guide

Use this guide from the existing Oscar installation. [PROGRESS.md](../PROGRESS.md) records current evidence and the next unmet gate. The [scientific protocol](../CODEX_PLAN.md) defines what each measurement must establish; the [replication plan](NEURON_REPLICATION_PLAN.md) describes the proposed extension.

## Activate the existing installation

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
source recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
cd "$REUSE_SOURCE_DIR"
export REUSE_SUBMITTER="$REUSE_SOURCE_DIR/scripts/slurm/submit.py"
```

`REUSE_SOURCE_DIR` is the editable `source/` checkout. `REUSE_PROJECT_DIR` is the frozen `recovery-cu128-20261016/snapshot/`; queued jobs read that snapshot. Use the editable submitter above, which includes the repaired validation retry behavior. It verifies the frozen snapshot and records its own wrapper hash.

The active environment uses torch 2.8.0+cu128 and torchvision 0.23.0+cu128. It reuses the staged model, dataset, cache, and run root. Do not repeat bootstrap or recovery for this installation, or edit the snapshot to apply source changes.

## Execution rules and accounting

Follow `/oscar/AGENTS.md`. Model inference, substantial analysis, installation, and data generation run in scheduled allocations. Show the applicable batch script before submission. Arrays require explicit approval; the commands below use one shard. Check scheduler state only on request, without automatic polling.

The configured initial request is one `nvidia_rtx_a6000` GPU, 8 CPUs, 32 GiB host memory, and one hour on `gpu-debug`, account `default`, QoS `gpu-debug`. These are configured requests, not measured runtime requirements. Keep the validated backend unless deliberately repeating numerical validation.

Before further submission, reconcile actual allocated time with the shared ledger:

```bash
python "$REUSE_SUBMITTER" account --refresh
```

This queries accounting and updates `$REUSE_RUN_ROOT/scheduler_ledger.json`. The project limit is 30 GPU hours, including failed jobs, every array task, and retries. Queued/running/unknown tasks retain their worst case reservations until accounting resolves them. A new run ID or relocated ledger does not create another allowance. Research outputs share a 20 GiB ceiling.

A dry run prints the exact proposed scheduler command, estimates, and missing gates. It does not submit or reserve GPU hours. Before an actual submission, review the applicable frozen script:

```bash
cat "$REUSE_PROJECT_DIR/scripts/slurm/run_gpu.sbatch"
cat "$REUSE_PROJECT_DIR/scripts/slurm/run_cpu.sbatch"
```

Run a stage's actual submission only after its prerequisites and resource request have been reviewed. Save the returned numeric job ID. If submission returns no unambiguous ID, reconcile its state before retrying.

## Check the recorded validation

Current artifacts are under `$REUSE_RUN_ROOT/loopvl-transfer-v1/`. Inspect a compact summary instead of printing the full diagnostic report:

```bash
python - <<'PYCODE'
import json
import os
from pathlib import Path
run = Path(os.environ["REUSE_RUN_ROOT"]) / "loopvl-transfer-v1"
report = json.loads((run / "parity_report.json").read_text())
gates = json.loads((run / "gates.json").read_text())
print({"status": report["status"], "examples": report["examples"],
       "checks": len(report["checks"]), "scope": report["scope"]})
print(gates)
PYCODE
```

Preflight records are in `../runs/preflight/`; stdout/stderr are in `../runs/logs/`. Numerical validity is distinct from competence and object mask alignment. Require a fresh parity check after changing the model/backend or the relevant instrumentation. Do not resubmit the existing successful validation merely because an archived document says it is pending.

## Measure timing and memory

This is the next uncompleted stage for the existing study. After refreshing accounting:

```bash
python "$REUSE_SUBMITTER" baseline --config configs/pilot.yaml --pilot --dry-run
```

After reviewing the printed request and GPU script, submit:

```bash
python "$REUSE_SUBMITTER" baseline --config configs/pilot.yaml --pilot
```

Require the saved `pilot` gate to pass with at least 32 measured examples. Inspect `timing-pilot/resources-0000.json` for measured throughput and peak allocated/reserved GPU memory before sizing subsequent work. The timing run includes control/scoring costs but does not establish the full capability gate.

## Evaluate task capability

After timing passes, refresh accounting and evaluate all pilot families:

```bash
python "$REUSE_SUBMITTER" baseline --config configs/pilot.yaml --resume --dry-run
```

After reviewing the request, submit:

```bash
python "$REUSE_SUBMITTER" baseline --config configs/pilot.yaml --resume
```

Inspect `baseline/competence.json` and the `capability` gate. Clean, shuffled image, and neutral image results determine the supported task scope. Capture requires completed parity, timing, and capability gates. If measured work needs another partition, inspect its actual limits and available validated GPU type before changing settings; an association alone is not proof of capacity.

## Collect and analyze discovery data

After the prerequisite gates and a feasible cost projection, use one resumable capture shard:

```bash
python "$REUSE_SUBMITTER" capture --config configs/discovery.yaml --num-shards 1 --resume --dry-run
```

After reviewing the request, submit:

```bash
python "$REUSE_SUBMITTER" capture --config configs/discovery.yaml --num-shards 1 --resume
```

An interrupted shard can resume across allocations. Keep the original shard count and hashes. Do not change to an array after collecting data under a different shard topology. A multi shard run requires explicit approval plus a real scheduled interruption/resume check; then set shard count and concurrency from measured resources before starting it.

Once every capture shard is complete, merge on a CPU node:

```bash
python "$REUSE_SUBMITTER" merge --config configs/discovery.yaml --num-shards 1 --dry-run
```

After reviewing the CPU request, submit:

```bash
python "$REUSE_SUBMITTER" merge --config configs/discovery.yaml --num-shards 1
```

Require `capture/merged.json`, then fit on a CPU node:

```bash
python "$REUSE_SUBMITTER" fit --config configs/discovery.yaml --dry-run
```

After reviewing the CPU request, submit:

```bash
python "$REUSE_SUBMITTER" fit --config configs/discovery.yaml
```

Merge validates completion markers, family coverage, and compatible hashes. Scheduler success alone does not establish a complete capture. Fitting uses fit/development/calibration families; confirmation and transfer families stay isolated.

## Causal work and confirmation

Stop after discovery until required controls and the candidate specific donor/horizon runner are implemented and validated. `discover-causal` is an incomplete collection interface. The checked in `docs/protocol_lock.json` is a draft; it cannot authorize confirmation.

A real lock must freeze model/data/sample identity, calibration, endpoints, candidate and family IDs, control evidence, margins, exclusions, multiplicity, and resource limits. The observational executor rejects causal candidates it cannot collect. Skipping causal work requires the original protocol's discovery supported reason, not an implementation or access failure.

After those gates are genuinely satisfied, the stage order is CPU `lock`, GPU `confirm`, CPU `merge`, and CPU `fit`, using `configs/confirmation.yaml` and the original confirmation shard count. Inspect each predecessor's saved outputs before proceeding. Confirmation analysis applies saved discovery fits without refitting.

## Inspect failures and resume

For a requested one time scheduler check, use `squeue --me`. Use actual returned IDs when inspecting accounting and logs; do not leave placeholder job IDs in commands. Review `$REUSE_RUN_ROOT/logs/` and `$REUSE_RUN_ROOT/preflight/` for the relevant job.

Refresh accounting before retries. Wait until the previous writer is terminal, retain original configuration/snapshot/shard hashes, and add `--resume`. For an already approved array, `--indices` can select failed indices while preserving `--num-shards`. Do not append to a shared output file or remove a live writer lock.

Safe family commits permit restart. Incomplete signal stops exit nonzero, so they cannot satisfy `afterok`. Automatic requeue is disabled; retries have their own reservations. Recreate dependent submissions after a failed predecessor if their dependencies no longer apply. Keep raw failure evidence.

## Change source or environment

Edit `source/`; preserve frozen snapshots and raw measurements. New executable code/configuration needs a new snapshot and provenance before it can run. Inspect the existing freezer's interface:

```bash
python "$REUSE_SOURCE_DIR/scripts/slurm/freeze.py" --help
```

Freeze to a new destination outside the editable source, using the matching dependency lock. Validate snapshot/environment hashes and document which executable config, data, and backend changes require new study identity or parity evidence. Do not point the existing run at an incompatible snapshot to bypass hash checks.

Documentation edits affect the editable checkout only. Existing jobs remain reproducible against their original snapshot. The original `../snapshot/` and `../venv/` belong to the earlier setup; the active paths come from the recovery environment.

## Tests and saved reports

Run the full suite on a scheduled CPU node, with the editable package explicitly selected:

```bash
PYTHONPATH="$REUSE_SOURCE_DIR/src" python -m unittest discover -s "$REUSE_SOURCE_DIR/tests" -v
```

The historical CPU package can be checked or recomputed from saved synthetic measurements on a scheduled CPU node:

```bash
PYTHONPATH="$REUSE_SOURCE_DIR/src" python -m reuseinterp.cli verify-results --run-dir "$REUSE_SOURCE_DIR/results/cpu-preparation-20261016"
PYTHONPATH="$REUSE_SOURCE_DIR/src" python -m reuseinterp.cli report --run-dir "$REUSE_SOURCE_DIR/results/cpu-preparation-20261016"
```

These commands do not load the model. That package records synthetic engineering checks, not current cluster measurements. Reporting templates still contain CPU preparation assumptions and must be updated before publishing real model reports. Keep raw captures and compact recomputation inputs in approved retained storage; no laptop transfer is required for normal operation.

For a new installation, see [BROWN_QUICKSTART.md](BROWN_QUICKSTART.md). For the resolved CUDA incident and raw records, see [CLUSTER_EXECUTION.md](CLUSTER_EXECUTION.md).
