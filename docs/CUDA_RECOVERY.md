# Recover the failed CUDA smoke on Brown

User-supplied logs show that job **7489067** failed in preflight on gpu2261:
`torch==2.12.1+cu130`, CUDA runtime `13.0`, and driver API version `12090`
(CUDA 12.9). This is an environment failure before model loading. There is no
parity report or model result from this job. The driver API number is not the
human-readable NVIDIA driver release number. Transcribed evidence is saved in
`evidence/cluster/7489067-user-report.json`; original cluster logs remain primary.

[PyTorch documents CUDA 12.6 builds of the same public versions](https://pytorch.org/get-started/previous-versions/).
The official wheel indexes contain matching Python 3.12/Linux x86_64 wheels for
`torch==2.12.1+cu126` and `torchvision==0.27.1+cu126`.
[NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
supports trying this older runtime with the reported driver. GPU operator and
model compatibility remain pending scheduled validation.

The recovery change passed 116 CPU tests, including accounting rejection and
preservation of the original snapshot and ledger. The saved output is
`evidence/cluster/cuda-recovery-tests.txt`. Reproduce with
`PYTHONPATH=src python -m unittest discover -s tests -v` in an environment with
this project's CPU dependencies and optional CPU PyTorch hook dependency.

Run on the cluster login/download node permitted for package installation:

```bash
export REUSE_BASE="$HOME/reuseinterp-loopvl"
source "$REUSE_BASE/cluster.env"
source "$REUSE_ENV_DIR/bin/activate"
cd "$REUSE_SOURCE_DIR"
git pull --ff-only origin main

# Reconcile the failed allocation against the existing 30 GPU-hour ledger.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh

# Fresh venv/snapshot; reuses and checks already staged data and model files.
python scripts/cluster_setup.py recover --base-dir "$REUSE_BASE" --failed-job-id 7489067

source "$REUSE_BASE/cluster-cu126.env"
source "$REUSE_ENV_DIR/bin/activate"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml
```

The helper requires saved scheduler evidence showing that all prior GPU jobs are
terminal and reconciled. It checks the preserved failed preflight and original
snapshot before creating the new profile. It never resets the ledger or submits
jobs itself. Installation runs a CPU torchvision operator and verifies the exact
CUDA build metadata; it never initializes CUDA on the login node.

It preserves `venv/`, `snapshot/`, `cluster.env`, `setup-state.json`, old raw logs,
and old run artifacts. New preparation lives in `venv-cu126/`,
`preparation-cu126/`, `snapshot-cu126/`, `cluster-cu126.env`,
`setup-state-cu126.json`, and corresponding dependency/staging files. Both
profiles use the same `runs/scheduler_ledger.json`. New frozen configs use
`run_id=loopvl-transfer-v1-cu126` and an independent study fingerprint. No previous
gates or model results are copied. Check available quota for the additional venv
and snapshot before installing; staged model weights are reused.

Explicit local-version pins matter: changing only the index while requesting
`torch==2.12.1` can leave an installed `2.12.1+cu130` untouched under
[Python version-matching rules](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching).
Do not rerun the original bootstrap over its frozen environment or edit old locks.

If recovery is interrupted, inspect its saved phase status:

```bash
python scripts/cluster_setup.py status --base-dir "$REUSE_BASE" --profile cu126
```

Resume only the incomplete phase(s), in this order:

```bash
python scripts/cluster_setup.py install --base-dir "$REUSE_BASE" --profile cu126 \
  --torch-index-url https://download.pytorch.org/whl/cu126 \
  --torch-version 2.12.1+cu126 --torchvision-version 0.27.1+cu126
python scripts/cluster_setup.py stage --base-dir "$REUSE_BASE" --profile cu126
python scripts/cluster_setup.py freeze --base-dir "$REUSE_BASE" --profile cu126
```

If freeze was interrupted after creating its preparation directory, preserve it
and use a fresh recovery name (for example, `recover ... --name cu126-r2`). Do not
delete or overwrite a frozen snapshot. Source the matching `cluster-NAME.env`.

After the new smoke finishes, substitute its printed job ID:

```bash
sacct -j NEW_JOB_ID --format=JobID,State,ExitCode,Elapsed
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
cat "$REUSE_RUN_ROOT/preflight/NEW_JOB_ID_single.json"
tail -n 80 "$REUSE_RUN_ROOT/logs/NEW_JOB_ID.err"
cat "$REUSE_RUN_ROOT/loopvl-transfer-v1-cu126/parity_report.json"
cat "$REUSE_RUN_ROOT/loopvl-transfer-v1-cu126/gates.json"
```

Require a successful preflight and `parity.passed=true` before the timing pilot.
If preflight fails, a missing parity report is expected: inspect preflight and
stderr first. If model loading or numerical validation fails, preserve that
failure and resolve it before any arrays. Continue from step 4 of
[BROWN_QUICKSTART.md](BROWN_QUICKSTART.md) only after the smoke passes; use the new
run ID in every result path. The shared resource cap includes both attempts.
