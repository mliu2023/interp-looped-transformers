> Historical documentation archived October 17, 2026. Status and commands may be superseded. Use [current progress](../../../PROGRESS.md) and the [current cluster runbook](../../CLUSTER_RUNBOOK.md).

# Brown/Oscar handoff: status and execution

Current cluster continuation is in [CLUSTER_EXECUTION.md](CLUSTER_EXECUTION.md).
It supersedes the old SSH/access status below: staged inputs exist, GPU job
7489067 failed CUDA compatibility preflight, and this session cannot reach Slurm.
The CUDA 12.8 repair is ready but its CPU submission returned no job ID.

For a fresh GitHub clone, use [BROWN_QUICKSTART.md](BROWN_QUICKSTART.md): it gives
the complete automated setup and sequential commands through discovery. The
manual procedures below remain useful for a site-required module environment,
accounting, retries and the future gated confirmation path.

During the original laptop handoff, infrastructure tests used fake `sbatch` and
no GPU job had yet been submitted. On October 16, 2026, a noninteractive connection to
`mliu171@ssh.ccv.brown.edu` reached the server but failed with
`Permission denied (keyboard-interactive)`. That historical SSH failure is
superseded by the cluster continuation linked above. No password or private key
belongs in this repository.
[Brown's SSH documentation](https://docs.ccv.brown.edu/oscar/connecting-to-oscar/ssh)
describes 2FA on `ssh.ccv.brown.edu` and the campus/VPN key-login endpoint
`sshcampus.ccv.brown.edu`. The latter has not been verified for this account.

The user has now supplied account associations for `mliu171`: the default account
has `gpu-debug`/`gpu-debug`, `gpu`/`norm-gpu`, and `batch`/`normal` partition/QoS
pairs. The user also supplied live `gpu-debug` partition/QoS limits and GPU
inventory: QoS MaxWall is one hour, MaxTRESPU is 8 CPUs, 4 GPUs and 96 GiB host
memory; the partition itself reports unlimited MaxTime and a five-minute default.
Modules, quotas, retention rules,
driver compatibility, and actual model memory needs are **pending site checks**.
The 32 GiB host-memory request is separate from GPU VRAM. A 24 GiB GPU is a
planning target, not a validated model requirement. No paid resources are used.

The executable handoff covers source audit, deterministic data, numerical
validation, behavior/controls, representative capture, and CPU analysis
interfaces. Loaded-model checks remain unrun. Candidate-specific causal
collection (`discover-causal`), donor alignment for those candidates, and the
short-horizon causal runner are **pending implementation after the loaded audit**;
the current command fails closed. A Slurm wrapper exists for that interface,
but submitting it now would consume an allocation without collecting effects.
Observational P1/P2 confirmation is separate from causal P3 collection and must
use its implemented CLI path only after an actual protocol lock.

## 1. Establish access and record site policy

On the laptop, use interactive authentication first:

```bash
ssh mliu171@ssh.ccv.brown.edu
```

Once authorized access works, the following read-only commands discover resources.
Save their output in approved project storage; do not put credentials in logs.

```bash
hostname
sinfo --version
sinfo -o "%P %a %l %G"
scontrol show partition
sacctmgr show associations where user="$USER" format=User,Account,Partition,QOS
```

The last command may be restricted; use the account's allocation documentation
instead if so. Record allowed partitions/accounts/QoS, GPU request syntax, time
limits, storage quota, scratch purge schedule, download policy, and supported
Python/CUDA/PyTorch environment. Login nodes handle transfer, submission, and
light inspection. Rendering, substantial analysis, and model inference run in
scheduled CPU/GPU jobs. Use the university's supported container/environment
instead of a venv if required. Never install GPU drivers.

For later noninteractive Codex submissions, verify the account's approved SSH
method with `ssh -o BatchMode=yes mliu171@ssh.ccv.brown.edu true`. Failure is an
access blocker, not permission to bypass 2FA. An authenticated user can execute
the commands below manually until approved noninteractive access is available.

## 2. Configure shared paths and the cluster environment

For this Brown account, `configs/cluster.brown.env.example` prefills the supplied
default-account associations. Copy it to `$HOME/reuseinterp-cluster.env` and fill
the absolute paths before use. Its singleton smoke now requests one hour, 8 CPUs,
32 GiB host RAM and one `nvidia_rtx_a6000` GPU, matching the user's live inventory
and QoS limits. Runtime and peak VRAM remain unmeasured. The maximum one-hour
GPU reservation counts against the project cap; actual consumption comes from
Slurm accounting. The inventory is not a statement that a node is currently idle.
These read-only commands can refresh the recorded information:

```bash
scontrol show partition gpu-debug
sacctmgr -nP show qos where name=gpu-debug format=Name%24,MaxWall,MaxTRESPU%80
sinfo -p gpu-debug -N -o "%N %G %f"
```

The initial GPU settings are `REUSE_PARTITION=gpu-debug`, `REUSE_ACCOUNT=default`,
and `REUSE_QOS=gpu-debug`. CPU submissions use `REUSE_CPU_PARTITION=batch`,
`REUSE_CPU_ACCOUNT=default`, and `REUSE_CPU_QOS=normal`. CPU account/QoS overrides
are separate because a GPU QoS is not the CPU association shown by the user.
An explicitly empty CPU override omits its flag; an unset override preserves the
legacy fallback to the corresponding shared account/QoS setting.

After successful smoke/pilot validation, the supplied regular GPU association is
`REUSE_PARTITION=gpu` with `REUSE_ACCOUNT=default` and `REUSE_QOS=norm-gpu`.
Changing the actual GPU/backend still requires a new parity check. An association
is not a measurement of available capacity, walltime, GPU type, or project budget.

The full association output supplied by the user resolves the lab account to
`brubenst-condo`: partition `gpu-he` uses QoS `brubenst-gcondo`, and partition
`batch` uses QoS `brubenst-condo`. Their limits have not been inspected. The initial
smoke continues to use the supplied default-account debug association. Refresh
the complete account names with:

```bash
sacctmgr -nP show assoc where user=mliu171 format=User%12,Account%40,Partition%30,QOS%50,DefaultQOS%50
```

Brown documents `--gres=gpu:1` for a single-GPU batch request in
[GPUs on Oscar](https://docs.ccv.brown.edu/oscar/gpu-computing/gpus). The debug
partition's GPU and software compatibility still need validation in the scheduled
smoke. The typed request `--gres=gpu:nvidia_rtx_a6000:1` pins the listed GPU type;
it does not change model precision or validate the backend. Pin the same type
for subsequent jobs or repeat numerical parity if the actual backend changes.
No job has been submitted on the basis of the supplied listings.

Copy `configs/cluster.env.example` to an **external** site settings file, for
example `$HOME/reuseinterp-cluster.env`. Replace every `/REPLACE/...` path and
partition name. `REUSE_PROJECT_DIR` will point to an immutable snapshot, not the
editable source checkout. Optional account and QoS variables stay empty when
unused. Only one GPU per task is supported. For sites accepting `--gres`, use
`REUSE_GPU_FLAG=--gres` and `REUSE_GPU_REQUEST=gpu:1`; for `--gpus`, use
`REUSE_GPU_FLAG=--gpus` and `REUSE_GPU_REQUEST=1`.

First transfer the source from the laptop to that approved shared source path:

```bash
rsync -av --exclude=.git --exclude=.venv --exclude=results --exclude=datasets --exclude=vendor ./ mliu171@ssh.ccv.brown.edu:/REPLACE/shared/editable-source/
```

On the cluster, after applying the site's documented module setup:

```bash
source "$HOME/reuseinterp-cluster.env"
python3 -m venv "$REUSE_ENV_DIR"
source "$REUSE_ENV_DIR/bin/activate"
python -m pip install -e "$REUSE_SOURCE_DIR"
```

Use a Linux environment created on the cluster; do not copy a laptop venv.
Install the site's supported matching PyTorch/torchvision pair and CUDA runtime
in the venv/container. The pinned official runtime references PyTorch 2.12.1 and
torchvision 0.27.1; that CUDA 13 build failed on the assigned Oscar driver.
The current Brown setup instead pins the CUDA 12.8 pair described in
`CLUSTER_EXECUTION.md`, with fresh GPU parity still required. Run installation
and substantial staging in a scheduled job under `/oscar/AGENTS.md`.
Exact source and model staging, on a node permitted to download:

```bash
export HF_HOME="$REUSE_CACHE_ROOT"
git clone https://github.com/Tier-Flow/LoopVL.git "$REUSE_LOOPVL_DIR"
git -C "$REUSE_LOOPVL_DIR" checkout --detach 6c14bed1d62fbecb559d5902459c6093c38da405
git -C "$REUSE_LOOPVL_DIR" diff --exit-code
python -m pip install -r "$REUSE_LOOPVL_DIR/runtime/requirements.txt"
hf download TierFlow/LoopVL --revision 92aa28bc678f523f8641fdb2f9e5159f614d1c02 --local-dir "$REUSE_MODEL_DIR"
```

The inspected official requirements pin transformers 5.15.1, safetensors 0.8.0,
Pillow 12.2.0, PyYAML 6.0.3, and accelerate 1.15.0. Do not substitute unverified
versions if installation fails; record that blocker. `hf` is provided by the
installed Hugging Face Hub dependency; if absent, install the site-approved
Hub CLI and include its version in the lock. If upstream files differ, stop and
update the architecture audit before inference. The adapter checks the pinned
Git revision and checkpoint hashes.

Set `HF_HOME="$REUSE_CACHE_ROOT"` while staging. Verify the revision's complete
weights, tokenizer, processor, configuration, and audited custom runtime are
present and can be opened with network disabled. Metadata in a cache alone is
insufficient. Do not mark `REUSE_CACHE_READY=1` until this check is complete.
This permits offline loading; numerical model validation still happens only in
the GPU smoke allocation. Every GPU wrapper refuses an unverified cache.

If module loads are needed inside jobs, write their **verified** commands to
`configs/site-environment.sh` in the editable source before freezing. The optional
`REUSE_SITE_ENV` must point to that file inside the snapshot. Credentials and
local settings must stay outside snapshots. Then pin the resulting environment:

```bash
python -m pip freeze --all > "$REUSE_SOURCE_DIR/cluster-dependencies.lock"
```

Inspect this lock for credentials in private package URLs before preserving it.
Each scheduled job checks that its installed package set equals the lock.
Changing dependencies or backend requires a new snapshot and a new parity check.

## 3. Freeze a stage and prepare data on CPU

Transfer the editable repository from the laptop without weights or captures:

```bash
rsync -av --exclude=.git --exclude=.venv --exclude=results --exclude=datasets --exclude=vendor ./ mliu171@ssh.ccv.brown.edu:/REPLACE/shared/editable-source/
```

Use absolute approved shared paths in configuration files for datasets. Before
freezing the cluster snapshot, update every stage to the same path:

```bash
export REUSE_DATASET_DIR="/REPLACE/shared/datasets/scenes-v1"
python - "$REUSE_SOURCE_DIR" "$REUSE_DATASET_DIR" <<'PY'
import json
from pathlib import Path
import sys
source, dataset = Path(sys.argv[1]), Path(sys.argv[2])
if not dataset.is_absolute() or 'REPLACE' in str(dataset):
    raise SystemExit('Configure an approved absolute dataset path first')
for name in ('smoke', 'pilot', 'discovery', 'confirmation'):
    path = source / 'configs' / (name + '.yaml')
    config = json.loads(path.read_text())
    config['dataset_dir'] = str(dataset)
    path.write_text(json.dumps(config, indent=2) + '\n')
PY
```

This changes storage location only; preserve the split assignments, renderer
configuration, seeds, and scientific definitions. Generate
the broad deterministic dataset locally if practical and transfer it, or use the
scheduled `data` command below. Data revision/content hashes must be the same
across discovery and confirmation. Keep model/source datasets separate from the
20 GiB research-output ceiling. A scene family, both edits, all questions, and
all conditions always share one split and shard.

On the cluster, freeze reviewed sources/configs using a destination outside the
source directory. Existing snapshots are never overwritten:

```bash
source "$HOME/reuseinterp-cluster.env"
source "$REUSE_ENV_DIR/bin/activate"
python "$REUSE_SOURCE_DIR/scripts/slurm/freeze.py" --source "$REUSE_SOURCE_DIR" --destination "$REUSE_PROJECT_DIR" --dependency-lock "$REUSE_SOURCE_DIR/cluster-dependencies.lock"
python "$REUSE_PROJECT_DIR/scripts/slurm/freeze.py" --verify "$REUSE_PROJECT_DIR" --verify-environment
```

Snapshot manifests contain file hashes, the source Git commit when available,
and the dependency lock hash. Results, model/data payloads, upstream vendor code,
credentials, and local `.env` files are excluded. The Python package's
`src/reuseinterp/data/` code remains included. Queued jobs verify source/config
bytes again before execution. External pinned upstream code and model files are
independently verified by the adapter. Treat snapshots as read-only; changed
bytes cause failure rather than running an edited queued configuration.

The shipped configurations omit `split_counts`, so `data --config configs/pilot.yaml`
generates the **full fixed 1,160-family universe**. The `splits` key selects
model-stage examples; it does not reduce data generation. Use a scheduled CPU job:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" data --config configs/pilot.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" data --config configs/pilot.yaml
```

The CPU wrapper requests no GPU and does not run CUDA preflight. Its CPU partition
must still be verified. Save the returned job ID and inspect completion artifacts;
the examples do not claim a job has run.

Alternatively transfer the existing local `datasets/scenes-v1/` to the configured
absolute shared path. Preserve its manifest and hashes; do not regenerate it
with another Pillow/generator version. The generator intentionally refuses to
overwrite an incompatible existing dataset. Validate transferred data without
regeneration:

```bash
python -c 'import os; from reuseinterp.data import validate_dataset; print(validate_dataset(os.environ["REUSE_DATASET_DIR"]))'
```

## 4. Single scheduled GPU smoke, then measured pilot

Set `REUSE_CACHE_READY=1` in the external settings only after staging verification.
The configured time request is an upper bound reserved against the budget, so
start short enough for a useful smoke job and adjust from measurements. If an
allocation cannot fit the model, reduce batch size first. Do not silently cast,
quantize, change model revisions, or change endpoints to obtain a passing run.

```bash
source "$HOME/reuseinterp-cluster.env"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml
```

The dry-run prints the exact `sbatch` argument list, requested maximum GPU-hours,
forward/storage estimates when configured, current budget, and any failed gates.
It creates log/lock directories but never calls `sbatch` or writes a reservation.
The real submission prints an actual scheduler job ID and writes
`$REUSE_RUN_ROOT/scheduler_ledger.json`. An unambiguous job ID is mandatory; an
uncertain submission keeps its full budget reservation until scheduler evidence
resolves it. No stdout placeholder is a scheduler ID.

Environment export uses an explicit `--export=NAME=value,...` list, without
`ALL` or `NONE`. Slurm disallows combining `NONE` with explicit variables. The
wrapper rejects commas/newlines in exported values; put such configuration in
reviewed files instead. Runtime cache, pinned source/model paths, expected
snapshot SHA256, and expected parity backend are exported explicitly. Scheduler
variables remain managed by Slurm. See the [official export syntax](https://slurm.schedmd.com/sbatch.html#OPT_export).

`run_gpu.sbatch` checks the immutable environment, runs CUDA preflight inside
Slurm, then runs Milestone A numerical checks. Preflight records versions, visible
GPU name/VRAM, actual probe placement, operators, supported dtype, and backend
signature in `$REUSE_RUN_ROOT/preflight/JOBID_single.json`. The model adapter
separately records real model placement. Capture, self-replacement, zero-dose,
matched storage/weights, down-projection reconstruction, exact invocation counts,
token alignment, and no answer leakage must all pass. At least 16 varied examples
are required. Repeated official forwards set tolerances before interpretation.

After the numerical artifact and gate pass, run the single 32-example timing,
memory, and task competence pilot:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --pilot --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --pilot
```

`--pilot` is forwarded to the CLI and selects the 32-example timing run. A
passed parity gate is required. No array is permitted for initial smoke/pilot.
Pilot records must include at least 32 measured examples and peak GPU allocated/
reserved memory, throughput, host memory, malformed responses, clean accuracy,
candidate scores, and image-shuffle/no-useful-image controls. Competence determines
the supported task scope; failures are reported. A gate must be generated from
saved validated model measurements, never filled in manually to unblock a job.

Use measured runtime/VRAM to set future requests and forward/storage estimates.
Missing estimates are explicitly printed as unknown; a dry-run alone does not
establish feasibility. GPU work after the pilot remains blocked until the
measurement artifacts are complete.

The 32-example timing run does not issue the full capability gate. After timing
passes, evaluate all pilot families with clean/shuffled/neutral controls:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" baseline --config configs/pilot.yaml --resume
```

This writes capability evidence and supported scope. A successful timing job
alone cannot open capture.

## 5. Independent arrays, merge, discovery, and lock

Gates live in `$REUSE_RUN_ROOT/RUN_ID/gates.json`, where `RUN_ID` is read from the
configuration. An explicit `--gates /path/gates.json` supports another documented
run layout. Each gate and the top-level object carry the stage-independent
`study_fingerprint` from the configuration. `parity` includes
`backend_signature`; `pilot` includes `measured_examples >= 32`; `capability`
records supported families. Capture/causal stages require these gates. A backend
change relative to parity makes preflight fail. Do not bypass the failure with
an edited fingerprint.

After parity, pilot, and capability are validated:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml --num-shards 4 --concurrency 2 --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml --num-shards 4 --concurrency 2 --resume
```

Arrays are available for `baseline`, `capture`, and `confirm`. `--num-shards`
remains fixed across retries; `SLURM_ARRAY_TASK_ID` selects the shard. Assignment
is deterministic by whole family. Tasks write separate atomic shard directories;
there is no concurrent shared CSV append. Checkpoint at least every 25 families,
and sooner when per-family duration requires it.

After recording the actual capture array ID, queue a CPU merge with afterok:

```bash
export REUSE_CAPTURE_JOB_ID="REPLACE_WITH_RETURNED_NUMERIC_ID"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/discovery.yaml --num-shards 4 --dependency "$REUSE_CAPTURE_JOB_ID" --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/discovery.yaml --num-shards 4 --dependency "$REUSE_CAPTURE_JOB_ID"
```

Replace the ID before running; the wrapper rejects nonnumeric placeholders.
Scheduler success does not establish complete data. Merge independently checks
expected shards, completion markers, condition/family coverage, duplicate keys,
and compatible model/configuration/data hashes. Incomplete results block merge.

After merge, use scheduled CPU `fit` and inspect discovery artifacts:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/discovery.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/discovery.yaml
```

The CPU fitting/analysis path and protocol lock are executable interfaces; their
real model inputs remain pending. Candidate-specific causal discovery collection
is currently a fail-closed boundary. After the loaded audit, implement and
validate the complete donor/recipient token checks, intervention doses,
candidate/control conditions, and short-horizon recording before opening causal
work. Inspect the pending interface without spending a GPU allocation:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" discover-causal --config configs/discovery.yaml --dry-run
```

Do not remove `--dry-run` until that collection runner is implemented and
validated. If P3 remains infeasible, declare its omission and preserve its
multiplicity allocation. Never fill in causal results or gates to unblock
observational confirmation. Lock an explicitly restricted observational protocol
only if the protocol validator permits that declared scope.

Once required discovery artifacts and protocol fields are complete, the pending
CPU lock commands are:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" lock --config configs/confirmation.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" lock --config configs/confirmation.yaml
```

The shipped `docs/protocol_lock.json` is an incomplete draft. Locking it unchanged
fails; it is not authorization for confirmation. Run stages sequentially after
each predecessor's artifacts pass validation. Do not queue confirmation against
unfinished discovery. Confirmation additionally
requires `discovery.passed` and `protocol_lock.passed` with the lock SHA256 and
matching study fingerprint. Lock the causal subset (initially 128 independent
families), endpoint definitions, sample/candidates, calibration, stopping schedule,
and exclusions before reading confirmation outcomes. No manual gate repair
substitutes for the required science.

Only then, for the implemented observational confirmation collection path:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" confirm --config configs/confirmation.yaml --num-shards 4 --concurrency 2 --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" confirm --config configs/confirmation.yaml --num-shards 4 --concurrency 2 --resume
```

The CLI may deliberately reject a stage whose scientific implementation or
prerequisites are incomplete. A scheduler wrapper does not turn an unsupported
model analysis into a completed experiment. Keep those endpoints pending.

After recording the actual confirmation array ID, merge its observations and
run CPU confirmation analysis only after that merge completes:

```bash
export REUSE_CONFIRM_JOB_ID="REPLACE_WITH_RETURNED_NUMERIC_ID"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/confirmation.yaml --num-shards 4 --dependency "$REUSE_CONFIRM_JOB_ID" --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" merge --config configs/confirmation.yaml --num-shards 4 --dependency "$REUSE_CONFIRM_JOB_ID"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/confirmation.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" fit --config configs/confirmation.yaml
```

The confirmation configuration has `merge_stage=confirmation` and
`analysis_stage=confirmation`; CPU fitting consumes frozen discovery models and
calibrations without refitting them. A completed capture array alone is not a
P1/P2 statistical result. Causal P3 collection remains pending even if
observational P1/P2 later complete.

## 6. Monitor, account, and restart

```bash
squeue -u "$USER"
export REUSE_JOB_ID="REPLACE_WITH_RETURNED_NUMERIC_ID"
sacct -j "$REUSE_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
```

Inspect stderr/stdout and scheduler states for OOM, timeout, invalid dependencies,
or preemption. MaxRSS is host memory. GPU peaks come from PyTorch measurements.
The ledger stores all submission requests, immutable hashes, job IDs, attempts,
and raw `sacct` evidence. `account --refresh` requests parsable per-job records;
`account --sacct-file /path/saved.txt` accepts the same
`JobID|State|ElapsedRaw|AllocTRES` format. It ignores `.batch`, `.extern`, and
other step rows and uses each array task once. If allocation data are missing,
it refuses to release the reservation.

The project cap is **30 GPU-hours total**, including smoke, pilot, every array
task, and retries. `REUSE_GPU_HOUR_CAP` can lower it. Each submission reserves
the full requested walltime for all selected tasks, including queued tasks.
Completed/failed/cancelled tasks consume actual allocated GPU time after
reconciliation. Running tasks retain their entire maximum request. Queue wait
does not count; array concurrency changes turnaround, not total GPU-hours.
Reservations are protected by a filesystem lock to prevent concurrent
submissions exceeding the cap. Use one shared ledger for the whole project;
`--ledger` is for relocating that same ledger, not starting another allowance.

For an interrupted capture array, first wait until the old task is terminal and
refresh accounting, then resubmit only failed indices:

```bash
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml --num-shards 4 --indices 1,3 --concurrency 2 --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" capture --config configs/discovery.yaml --num-shards 4 --indices 1,3 --concurrency 2 --resume
```

The original `--num-shards`, snapshot, and configuration hashes must remain
identical. The wrapper refuses concurrent duplicate writers. CLI resume skips
only validated committed records and recomputes incomplete records. USR1/TERM
handlers request a stop at a safe family boundary; handlers perform no complex
I/O. An incomplete graceful stop exits nonzero (75), so afterok cannot consume
partial results. Submissions request `--no-requeue`; manual retries remain
separate ledger entries with their own reserved/consumed time. Recreate dependent jobs
after a failed predecessor/retry; old dependencies do not necessarily recover.

Before production arrays, a **real scheduled** small interruption/resume check
must match uninterrupted output within locked tolerance. The local synthetic
resume test is infrastructure evidence only. Preserve failure reasons and retry
history; retries are never free GPU time.

## 7. Download and recompute on CPU

Do not wait until scratch purge to plan retention. Record approved long-term
locations and backup status for raw captures and external datasets. The small
report package must include raw scores/predictions, row identifiers, metadata,
hashes, exclusions, resource/accounting evidence, and every input required for
reported metrics/figures. Full model weights and all activation dumps are not
needed for CPU figure recomputation.

On the laptop:

```bash
export REUSE_REMOTE_RUN="/REPLACE/shared/results/ACTUAL_RUN"
export REUSE_LOCAL_RUN="results/ACTUAL_RUN"
mkdir -p "$REUSE_LOCAL_RUN"
rsync -av "mliu171@ssh.ccv.brown.edu:$REUSE_REMOTE_RUN/report-package/" "$REUSE_LOCAL_RUN/report-package/"
python -m reuseinterp.cli verify-results --run-dir "$REUSE_LOCAL_RUN/report-package"
python -m reuseinterp.cli report --run-dir "$REUSE_LOCAL_RUN/report-package"
```

Replace `ACTUAL_RUN` with the real run ID and use the actual produced package
layout. Reporting consumes saved measurements and must never launch inference.
At a failed gate, resource cap, or access blocker, preserve completed artifacts,
record remaining checks as pending, and run the partial report. No model result
is inferred from synthetic fixtures or from successful scheduler syntax.

## Acceptance record

- CPU checked: shell syntax, fake `sbatch`, explicit optional flags/exports,
  frozen-snapshot mutation rejection, backend-gate presence, array arguments,
  duplicate refusal, conservative budget, and no step double-counting.
- CPU pipeline separately checks deterministic family sharding, interruption/
  resume with no omitted/duplicate committed keys, and incomplete merge rejection.
- Pending: approved SSH access, site/resource verification, supported Linux
  dependency installation, complete pinned offline cache, real CUDA preflight,
  numerical smoke, measured pilot/capability, real interruption/resume check,
  all model capture/discovery/confirmation, and actual scheduler accounting.

Official scheduler references: [sbatch](https://slurm.schedmd.com/sbatch.html),
[arrays](https://slurm.schedmd.com/job_array.html), and
[sacct](https://slurm.schedmd.com/sacct.html). Local Brown policy takes precedence.
