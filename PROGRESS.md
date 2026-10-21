# Progress — 2026-10-16

Planning update — 2026-10-21: the user redirected the scientific goal toward
changes across recurrent weight reuse. `LOOP_DYNAMICS_PLAN.md` is now the current
research plan, with machine-readable scope in `docs/loop_dynamics_spec.json`.
It covers L/H event-resolved updates, Q/K/V and gated attention, signed neuron
activity/recruitment, held-out full-space update subspaces, next-reuse feedback,
and a bounded reinjection control. The plan includes CPU fixtures, exact missing
instrumentation, discovery/confirmation separation, and the shared 30 GPU-hour /
20 GiB caps. New `dynamics-*` commands are an implementation contract, not runnable
scripts yet. Historical transfer code/results are preserved. No additional GPU
job outcome or model result has been supplied since the failed preflight below.

CUDA recovery update: user-supplied evidence shows real Slurm job **7489067**
failed before model loading on gpu2261: torch 2.12.1+cu130 requires a newer driver
than the reported CUDA 12.9 driver API. The reported allocation lasted 17 seconds;
actual GPU-hour accounting must come from the saved scheduler ledger and sacct,
not this transcription. Bootstrap now pins the official CUDA 12.6 builds of the
same torch/torchvision releases. A separate recovery profile preserves the original
environment, immutable snapshot, failure evidence and shared resource ledger,
and requires fresh parity. **116 CPU tests passed**, including accounting failures
and recovery snapshot isolation; output is saved in
`evidence/cluster/cuda-recovery-tests.txt`. See `docs/CUDA_RECOVERY.md`. Compatibility of that
candidate build, model loading, numerical audit and all scientific GPU results
remain pending. Earlier setup status and test counts below are historical.

GitHub handoff update: added the executable Brown setup helper and one-command
managed Python 3.12 bootstrap, exact 3.0 MiB dataset archive, and complete
`docs/BROWN_QUICKSTART.md` through discovery. Installation, staging and freezing
are distinct resumable phases; setup never runs inference or submits jobs. A
prepublication suite of **109 CPU tests passed**. Archive extraction and full
validation reproduced the pinned 1,160-family dataset hash. Actual package
installation/driver compatibility and GPU execution remain unverified on Oscar.
Saved CPU preparation reports below retain their original historical test count.

Setup update: the user supplied Brown associations for `mliu171`. The Brown
environment example now uses `gpu-debug/default/gpu-debug` for the initial smoke
and `batch/default/normal` for CPU jobs, with separate CPU account/QoS overrides.
The regular GPU association is `gpu/default/norm-gpu`. The user subsequently
confirmed `brubenst-condo` as the lab account and supplied live `gpu-debug`
inventory plus a one-hour QoS limit, 8 CPUs/4 GPUs/96 GiB maximum per user. The
Brown example now requests one listed `nvidia_rtx_a6000` GPU for the one-hour
singleton smoke, with 8 CPUs and 32 GiB host RAM. Site paths, modules, driver
compatibility and storage policies remain pending. The updated
Slurm subset passes 26 tests; no real job was submitted. See
`configs/cluster.brown.env.example` and runbook section 2.

This is a partial execution at an external compute-access boundary. There are
**no fresh LoopVL model measurements or scientific conclusions**. The only reported
real Slurm job failed at CUDA initialization. Model outcomes are not inferred from
synthetic fixtures.

## Completed on available CPU resources

- Read the execution plan, inspected the initially empty repository and available
  cache/compute. No applicable repository AGENTS.md was found. Local CUDA is absent.
- Checked out the exact official LoopVL source revision specified by the plan,
  inspected the recurrence, official mixed-precision loader, generation, masks,
  positional encoding, dynamic preprocessing, and post-product MLP site.
  Saved upstream metadata, checkpoint revision, weight LFS hash, small checkpoint
  files, source checksums and architecture audit. No weights were downloaded.
- Implemented the official adapter and scheduled numerical runner: independent
  official logits/generation, measured tolerance freezing, product/projection
  reconstruction, parameter identity, reset counters, capture/self/zero controls,
  exact local changed coordinates, raw logits and failure records. Real validation
  remains pending; CPU toy-model hook tests passed.
- Generated and independently validated 1,160 deterministic 2D scene families,
  2,320 scenes, 9,280 questions, and 64,952 object descriptor rows. Splits contain
  32 smoke /128 pilot /400 fit /100 development /100 calibration /300 confirmation
  /100 transfer families. Explicit masks, single-object color edits, programs,
  answer checks, descriptive balance and stable grouped splits are saved.
- Implemented representative sampling, real-token capture then normalized object
  pooling, separate last-prompt captures, normalized MLP inputs and block states,
  actual-grid mask projection, trace IDs, shard checksums and CPU package assembly.
  Actual model-grid alignment and capture are pending.
- Implemented positive-affine semantic transfer, frozen linear probes, family
  bootstrap, layer-balanced P1, symmetric P2, dev-only selection and eligibility,
  fixed-dose/paired effect utilities, nonnegative causal gain and explicit linear
  context contrasts with centered simultaneous max-t bands.
- Saved six visibly synthetic mechanisms and negative pipeline fixtures. Every
  fixture metric reproduces from raw NPZ measurements, with 2,000 family-bootstrap
  draws. The optional engineering figure is labeled SYNTHETIC, not model evidence.
- Implemented standalone matched-context, clipped discovery-fitted reweighting,
  and paired-edit analysis controls; question/position probe baseline is integrated
  into discovery fits. Remaining control orchestration is listed below.
- Implemented fail-closed discovery/confirmation separation and immutable protocol
  creation. Confirmation applies saved fits and produces a compact CPU package
  that independently recomputes P1/P2. The protocol file in docs is an incomplete
  draft; no scientific protocol lock exists.
- Implemented the configurable Slurm workflow: safe singleton smoke, scheduled
  CPU/GPU wrappers, CUDA/backend checks, immutable source/environment snapshots,
  deterministic family arrays, atomic resume, signals, incomplete merge rejection,
  duplicate-writer protection, job records and conservative 30 GPU-hour accounting.
  Infrastructure tests use fake sbatch and do not establish cluster compatibility.
- Reproducible CPU report, raw fixture package, strict metrics, claim ledger,
  status/limitations, source evidence, tests and exact cluster runbook are provided.
  Final verification: **85 tests passed**, no failures or skips. Shell syntax and
  Python compilation checks passed. The saved suite log identifies every check.

## Blocking observations and unfinished implementation

`ssh -o BatchMode=yes ... mliu171@ssh.ccv.brown.edu` reached Brown but returned
`Permission denied (keyboard-interactive)`. An authenticated interactive session
or university-approved noninteractive setup is required before Codex can submit.
No password, private key, authentication bypass, or paid provisioning is needed
or requested by this repository. Site resource/policy discovery remains pending.

The numerical runner, model loader, real checkpoint checks, capability controls,
capture, actual interruption/resumption parity, scientific fitting and all model
endpoints remain **unrun**. GPU feasibility and timing estimates are unmeasured.
No sample neurons or causal candidates have been selected from real activations.

Remaining implementation is explicit: integrate the standalone matched-task/role,
reweighted, coherent permutation and paired-edit controls with the real analysis
tables; audit/implement the candidate-specific donor and short-horizon causal
collection orchestrator after loading the model; add the five scientific figure
renderers after validating real measurement schemas; and define the secondary
question-token/last-prompt encoding descriptors. `discover-causal` refuses real
execution and explains this boundary. The existing observational confirmation
executor rejects protocols containing causal candidates so it cannot silently
skip their experiments. CPU tested intervention primitives/statistics are not a
complete causal experiment runner. Mandatory incomplete controls block locking.

If discovery establishes no viable causal candidates, P3 may be skipped only with
a documented discovery-supported reason in the lock. Access failure is not such
evidence. P3 cannot be skipped merely to enable confirmation now.

## Resource and protocol decisions

The ceiling is 30 GPU-hours including all smoke/pilot/retries, and 20 GiB of
research outputs excluding model/source datasets. Project jobs submitted: 0;
project GPU-hours consumed/reserved: 0. Queue time is separate. The canonical
ledger is `results/scheduler_ledger.json`; the cluster uses its configured
REUSE_RUN_ROOT counterpart. Real job elapsed time must come from sacct.

Before any capture, state layers were set to [0,5] for input/block readouts to
reduce duplicated pooled state packages within the output budget. The uniform
32-neuron-per-physical-L-layer sample remains 512 neurons if loaded dimensions
match metadata. Full tensors are reserved for small diagnostic subsets. These
are preparation decisions, not adaptations to confirmation outcomes.

No pilot rendering revision has been used. There was a generator balance fix
before the saved dataset was created; its source hash and final actual coverage
are saved. No confirmation or transfer image was used to tune model behavior.
CPU validation reads their independent ground truth; it does not open model data.

## Exact next commands

Recompute existing CPU evidence locally (NumPy/Pillow and optional matplotlib):

```bash
export PYTHONPATH="$PWD/src"
python -m reuseinterp.cli report --run-dir results/cpu-preparation-20261016
python -m reuseinterp.cli verify-results --run-dir results/cpu-preparation-20261016
python -m unittest discover -s tests -v
```

For Brown access and site discovery:

```bash
ssh mliu171@ssh.ccv.brown.edu
hostname
sinfo --version
sinfo -o "%P %a %l %G"
scontrol show partition
```

Then follow `docs/CLUSTER_RUNBOOK.md` to stage the pinned source/environment,
transfer the validated dataset, set actual paths/resources, and freeze a snapshot.
On that configured cluster, the first scheduled GPU commands are:

```bash
source "$HOME/reuseinterp-cluster.env"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml
```

Record the actual returned scheduler ID and require saved numerical success.
Only then run timing, full competence, independent capture shards, and CPU fits.
The runbook gives exact monitoring, resumption, merge, lock and download commands.
Confirmation additionally awaits completed controls and a genuine immutable lock.
Do not populate gates manually, tune on confirmation, or extend until significance.
