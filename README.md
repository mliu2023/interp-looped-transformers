# Recurrent neuron transfer study

This repository implements CPU preparation for the supplied `CODEX_PLAN.md`:
semantic-description, representation-readout, and causal-use transfer for the
same physical LoopVL neurons across recurrent invocations. Identical parameters
are an instrumentation check, never the scientific endpoint.

**Current status:** no LoopVL inference, scheduler submission, or model findings.
Brown SSH was reached but rejected noninteractive authentication. CPU fixtures,
full deterministic scene data, source audit, and Slurm infrastructure are ready.
See [PROGRESS.md](PROGRESS.md) for implementation limits and pending work.

To clone this repository directly on Brown and prepare the entire environment,
exact dataset, pinned model and immutable job snapshot, follow
[BROWN_QUICKSTART.md](docs/BROWN_QUICKSTART.md). The setup entry point is
`bash scripts/brown_setup.sh /absolute/approved/shared/study-directory`; it
prepares inputs without submitting GPU work. Start experiments with its singleton
Slurm smoke commands and stop after discovery until remaining controls are ready.

## Install and reproduce CPU preparation

Python 3.9+ is supported for CPU code; the official GPU runtime requires its own
Linux environment. Tables need NumPy and Pillow; the optional synthetic figure
requires `pip install -e '.[plots]'`. Hook fixtures additionally
need PyTorch; they skip explicitly when unavailable. The local verified environment
was Python 3.9.13, NumPy 1.26.4, Pillow 9.2.0, and CPU PyTorch 2.8.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots]'
python -m unittest discover -s tests -v
```

To include synthetic CPU hook tests, install `.[hooks]` in a suitable environment.
To reproduce the **exact existing data hashes**, use `requirements-cpu.lock`;
renderer/library versions can change image bytes. Transfer the validated data
to the cluster rather than regenerating under different pinned model packages.

The existing full dataset is in `datasets/scenes-v1/` (ignored by Git). Its 1,160
independent families comprise 2,320 scenes, 9,280 questions and 64,952 object
descriptor rows. All descendants, both color versions, and four question families
share one split. This is a controlled **2D distribution, not CLEVR**. Model
competence and actual processor alignment have not been established.

Use a new output directory if regenerating; existing data and fixture raw files
are never overwritten:

```bash
python -m reuseinterp.cli data --config configs/data.yaml
python -m reuseinterp.cli fixtures --config configs/cpu.yaml
```

These creation commands reject existing artifacts. To reproduce all **existing
reported CPU metrics from saved raw measurements**, without creating new data:

```bash
python -m reuseinterp.cli report --run-dir results/cpu-preparation-20261016
python -m reuseinterp.cli verify-results --run-dir results/cpu-preparation-20261016
```

The report package contains raw synthetic measurements, fitted parameters,
predictions, strict JSON metrics, `report.md`, `claim_ledger.csv`, and machine
readable limitations. Synthetic data are visibly labeled and excluded from model
findings. There are no model-result figures to regenerate yet. The five planned
model figures must await valid GPU measurements. Reports never load a VLM.

## GPU handoff and scientific gates

Follow [docs/CLUSTER_RUNBOOK.md](docs/CLUSTER_RUNBOOK.md). It covers Brown access,
site discovery, environment/model staging, immutable snapshots, safe submissions,
measured pilot, independent arrays, accounting, retries and downloading results.
No partition, account, QoS, driver, storage policy, or GPU capacity was guessed.
GPU work requires Slurm and CUDA; there is no CPU fallback for model inference.

Pinned official source: `6c14bed1d62fbecb559d5902459c6093c38da405`.
Pinned checkpoint: `92aa28bc678f523f8641fdb2f9e5159f614d1c02`.
The complete weights must be staged once; all checkpoint inputs have expected
SHA-256 hashes in the configurations. Preserve the official mixed precision.

The first real job must be a singleton numerical smoke, followed by 32-example
timing/memory evaluation and full pilot capability. Capture arrays require those
gates. The submitter reserves consumed plus outstanding worst-case GPU-hours
against a project-wide ceiling of 30, including retries. Research outputs are
capped at 20 GiB. Initial state sites were reduced to layers 0 and 5 **before any
capture or confirmation** to keep pooled state packages within this storage cap;
the 32-per-layer representative neuron sample still covers all 16 physical layers.

```bash
python scripts/slurm/submit.py validate --config configs/smoke.yaml --dry-run
python scripts/slurm/submit.py validate --config configs/smoke.yaml
```

These run on the configured cluster after the runbook's prerequisites. A dry-run
prints gates, resource reservations and commands without submitting. Missing site
configuration is an error, not a reason to assume scheduler settings.

`docs/protocol_lock.json` is explicitly an **incomplete draft**. A real immutable
lock requires completed numerical/discovery evidence, frozen model/data/sample,
calibration, endpoints, exact family IDs, controls, and margins. Confirmation
analysis applies saved fits without refitting and can be recomputed entirely from
its compact raw package. The causal collection orchestrator remains pending a
loaded-model audit; invocation-local edits and statistical analysis are CPU tested.
No causal results or portability classifications are implied by this handoff.
