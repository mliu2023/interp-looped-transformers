# Clone and run on Brown/Oscar

This is the complete runnable path through discovery. Job 7489067 reached an
Oscar GPU but failed CUDA initialization before any model inference. Follow the
[CUDA recovery instructions](CUDA_RECOVERY.md) if you used the original setup.
No model GPU stage has yet passed. Stop after discovery: causal collection and remaining analysis
control orchestration are unfinished, and confirmation must await a real protocol
lock. The scheduler rejects missing gates; never edit gate files to bypass them.

## 1. Log in, check storage, and clone

Run from your laptop:

```bash
ssh mliu171@ssh.ccv.brown.edu
```

Then run on the cluster login/download node permitted by CCV for staging:

```bash
checkquota
export REUSE_BASE="$HOME/reuseinterp-loopvl"
mkdir -p "$REUSE_BASE"
git clone https://github.com/mliu2023/interp-looped-transformers.git "$REUSE_BASE/source"
cd "$REUSE_BASE/source"
```

If already cloned, use `git -C "$REUSE_BASE/source" pull --ff-only` instead of
cloning again. Choose an approved shared base with enough remaining quota before
continuing. This example uses home storage so queued jobs and results are not
dependent on scratch retention. Expect several GiB for model/software plus up
to the 20 GiB research-output ceiling. Brown documents home storage and scratch
purging in [Oscar's filesystem guide](https://docs.ccv.brown.edu/oscar/managing-files/filesystem).

## 2. Prepare Python, dependencies, exact data, model, and snapshot

```bash
bash scripts/brown_setup.sh "$REUSE_BASE"
```

This script never submits a job or loads model weights. It:

1. Creates a user-space bootstrap environment and installs pinned `uv==0.13.1`.
2. Obtains a managed Python 3.12 interpreter, avoiding guessed module names.
3. Creates `REUSE_BASE/venv`, installs the official exact torch 2.12.1+cu126 /
   torchvision 0.27.1+cu126 pair and pinned runtime requirements, and installs this
   project's CPU/plotting dependencies.
4. Clones official LoopVL at the audited commit, downloads the complete pinned
   model snapshot once, and verifies every configured checkpoint byte hash.
5. Extracts the packaged, checksummed dataset. No rerendering under the new Pillow
   version occurs. It validates labels, masks, edits, splits and the pinned hash.
6. Checks tokenizer/processor loading offline without constructing the VLM.
7. Creates `cluster.env`, a dependency lock, staging evidence, and an immutable
   `snapshot/`, preserving original Git/source provenance and absolute data paths.

The default package index is `https://download.pytorch.org/whl/cu126`, with explicit
`+cu126` version pins. Job 7489067 established that the original PyPI CUDA 13 build
cannot initialize on gpu2261's reported CUDA 12.9 driver API. PyTorch documents
the same public releases with CUDA 12.6; actual operators and model parity must
still pass in the first GPU allocation. If CCV requires another approved build,
set `REUSE_TORCH_INDEX_URL`, `REUSE_TORCH_VERSION`, and
`REUSE_TORCHVISION_VERSION` together before setup. No drivers or system packages
are modified. See [PyTorch's official installation commands](https://pytorch.org/get-started/previous-versions/),
[NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), and
[uv's managed Python documentation](https://docs.astral.sh/uv/guides/install-python/).

If site policy requires modules/a provided Python, use the individual setup phases
instead. Supply the actual reviewed module script and available interpreter:

```bash
python3 scripts/cluster_setup.py configure --help
python3 scripts/cluster_setup.py install --help
```

`configure` accepts `--site-env /absolute/reviewed/script.sh` and `--python`.
The module script must initialize the module command when needed. Installation
also supports `--use-installed-torch` for a site-provided CUDA pair. The snapshot
includes that reviewed script, because login-shell module state is not exported
wholesale to jobs. The automated managed-Python path needs no module script.

If setup stops, keep its error output. Resume only the failed phase; earlier
completed phases need not be rerun:

```bash
python3 scripts/cluster_setup.py status --base-dir "$REUSE_BASE"
```

The phase commands are `install`, `stage`, then `freeze`. For example, after a
failed download:

```bash
"$REUSE_BASE/venv/bin/python" scripts/cluster_setup.py stage --base-dir "$REUSE_BASE"
"$REUSE_BASE/venv/bin/python" scripts/cluster_setup.py freeze --base-dir "$REUSE_BASE"
```

An existing frozen environment is not overwritten. Source updates remain in
`source/`; queued jobs read only `snapshot/`. Future code/environment updates
need a fresh environment/snapshot and appropriate new parity evidence. Preserve
the original run root and scheduler ledger across replacements so failed jobs and
retries retain their cost. The [CUDA recovery helper](CUDA_RECOVERY.md) does this
within the existing base. Never edit a snapshot after submitting jobs against it.

## 3. Activate and submit the single numerical smoke

```bash
source "$REUSE_BASE/cluster.env"
source "$REUSE_ENV_DIR/bin/activate"
cd "$REUSE_SOURCE_DIR"

python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml \
  | tee "$REUSE_BASE/smoke-submission.txt"
```

The defaults match the supplied live scheduler information:
`gpu-debug`, account `default`, QoS `gpu-debug`, one
`gpu:nvidia_rtx_a6000:1`, 8 CPUs, 32 GiB host RAM, one hour. The QoS one-hour cap
applies despite the partition's unlimited MaxTime. The 8-CPU/4-GPU/96-GiB QoS
limits are per-user aggregate limits; this first job is a singleton. At most one
GPU-hour is reserved for this request against the 30 GPU-hour project ceiling.

Wait for this job to finish. Monitor with:

```bash
squeue -u "$USER"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
cat "$REUSE_RUN_ROOT/loopvl-transfer-v1/parity_report.json"
cat "$REUSE_RUN_ROOT/loopvl-transfer-v1/gates.json"
```

The recovery profile uses run ID `loopvl-transfer-v1-cu126`; substitute that in
all result paths below. Submission commands read the new ID from frozen configs.

Logs are `$REUSE_RUN_ROOT/logs/JOB_ID.out` and `.err`. Preflight evidence is under
`$REUSE_RUN_ROOT/preflight/`. Require `parity.passed=true`, valid preflight, and
completed 16-example official parity/hook checks before proceeding. A timeout or
failed parity is a failed gate; preserve evidence and fix it before another stage.

## 4. Measure the 32-example timing/memory pilot

After smoke succeeds:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --pilot --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --pilot \
  | tee "$REUSE_BASE/timing-submission.txt"
```

Wait for completion and require `pilot.passed=true` with at least 32 measured
examples. Inspect timing and peak allocated/reserved VRAM under
`$REUSE_RUN_ROOT/loopvl-transfer-v1/timing-pilot/` before requesting later jobs.
GPU-hours include smoke, failed jobs and retries; refresh accounting between jobs.

## 5. Evaluate all pilot families for competence and image dependence

Use the regular GPU association for longer work after inspecting its limits:

```bash
sacctmgr -nP show qos where name=norm-gpu format=Name%24,MaxWall,MaxTRESPU%80
sinfo -p gpu -N -o "%N %G %f"
export REUSE_PARTITION="gpu"
export REUSE_QOS="norm-gpu"
# Keep the validated A6000 GPU type and one-hour walltime unless measured
# requirements and live QoS limits justify a different request.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --resume \
  | tee "$REUSE_BASE/baseline-submission.txt"
```

Wait for completion and inspect `baseline/competence.json` plus the capability
gate. A completed valid evaluation permits observational work; behavioral claims
remain restricted to supported task families. An actual backend change requires
repeating parity. If regular `gpu` lacks the validated type, keep `gpu-debug`
within its limits or deliberately revalidate a new backend.

## 6. Capture discovery activations in independent shards

Only after the preceding gates and measured resource feasibility pass:

```bash
# Four CPUs per task is a starting request for two concurrent independent GPUs.
# Keep the concurrency within the live per-user CPU/GPU limits printed above.
export REUSE_CPUS="4"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml \
  --num-shards 4 --concurrency 2 --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml \
  --num-shards 4 --concurrency 2 --resume \
  | tee "$REUSE_BASE/capture-submission.txt"
```

This reserves four times the requested per-task GPU walltime, not two times.
Queue concurrency does not reduce GPU-hour accounting. Preserve `--num-shards 4`
on all retries. Each family commits atomically in its own shard. Incomplete jobs
exit nonzero and cannot satisfy downstream `afterok` dependencies.

Wait for successful array completion. Inspect logs and refresh accounting. If an
index fails, use the runbook's `--indices` retry procedure with the original count.

## 7. Merge and fit on scheduled CPU nodes

After all four capture shards complete:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/discovery.yaml \
  --num-shards 4 --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/discovery.yaml \
  --num-shards 4 | tee "$REUSE_BASE/merge-submission.txt"
```

Wait for merge completion and the saved `capture/merged.json`. Then:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/discovery.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/discovery.yaml \
  | tee "$REUSE_BASE/fit-submission.txt"
```

CPU jobs use partition `batch`, account `default`, QoS `normal` through separate
CPU overrides. They request no GPU. Choose their CPU/memory/time requests from
actual data sizes and live CPU QoS limits. Raw captures and assembled arrays share
the 20 GiB output budget; the aligned analysis package is rebuildable from captures.

**Stop here after discovery.** Causal collection, mandatory control integration and
a genuine protocol lock remain required before confirmation. `discover-causal`
is not a completed collection implementation. No confirmation jobs should be
submitted using the current draft protocol.

## 8. Download results

On the laptop, after the desired stage completes, use the actual absolute cluster
run path printed by setup (home may resolve to a site-specific absolute path):

```bash
rsync -av mliu171@ssh.ccv.brown.edu:/ACTUAL/ABSOLUTE/BASE/runs/loopvl-transfer-v1/ ./results/loopvl-transfer-v1/
```

Replace the single path placeholder with the configured path. Keep large captures
on approved backed-up cluster storage if only the small CPU report package is
needed locally. For details on retries, accounting, protocol locking and the
future confirmation interface, see [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md).
