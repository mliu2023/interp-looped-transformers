> Historical documentation archived October 17, 2026. Status and commands may be superseded. Use [current progress](../../../PROGRESS.md) and the [current cluster runbook](../../CLUSTER_RUNBOOK.md).

# Current cluster continuation

**Current update:** recovery job **7496249** completed successfully.
GPU numerical smoke **7501493** has been submitted using the recovered snapshot.
Its completion has not been checked. All 34 focused scheduler and upstream
processor-alignment tests passed with no skips. Submission evidence and fresh
accounting are in results/cluster-execution-20261016/gpu-smoke-20261017T045104Z/.
Use the editable source/scripts/slurm/submit.py wrapper when orchestrating this
repaired snapshot; the frozen older wrapper rejects validation after the failed
earlier snapshot. Do not edit the frozen experiment snapshot. The pending
recovery and network-failure statuses below are historical.

**Latest status:** networking is restored. A live queue/accounting check
resolved the earlier uncertain submission as absent. Slurm accepted CPU
recovery **job 7496249** on 2026-10-17 at 01:50:17 UTC. Its completion
has not been checked. The submission and successful connectivity checks
are saved in `results/cluster-execution-20261016/continue-20261017T014939Z/`.
Do not submit another recovery job. At the next requested status check,
inspect `runs/logs/recovery-7496249.out`, its `.err` sibling and
`recovery-cu128-20261016/recovery_state.json` in the study base.
The earlier access-failure and retry instructions below are historical;
resume at the scientific gates only after recovery succeeds.

The editable checkout is `/oscar/home/mliu171/reuseinterp-loopvl/source`, on
branch `codex/cluster-execution-20261016`. The sibling `snapshot/` and `venv/`
belong to the original frozen setup and must be preserved for auditability.

The earlier laptop implementation is substantial, but **the research plan is not
complete**. No LoopVL inference or scientific endpoint has succeeded. The staged
data are deterministic 2D scenes, not CLEVR. Numerical parity, capability,
representative capture, discovery, protocol lock and confirmation remain gated.
The causal collection orchestration and other implementation limits listed in
`PROGRESS.md` remain in force; GPU access will not by itself complete the study.

## Observed failure and prepared repair

Prior real GPU job `7489067` ran on `gpu2261`. Its preflight failed before loading
weights: torch `2.12.1+cu130` reported that driver API `12090` was too old.
The saved parent `sacct` row records 17 seconds with one GPU, or
`17/3600 = 0.004722222222222222` GPU-hours. Step rows are not counted twice.
The canonical shared ledger is `../runs/scheduler_ledger.json`; it records no
outstanding GPU allocations at its last saved accounting refresh. Its remaining
budget is 29.995277777777776 GPU-hours, subject to a fresh scheduler reconciliation
before future GPU submission.

The repair pins the official torch `2.8.0+cu128` and torchvision `0.23.0+cu128`
pair from `https://download.pytorch.org/whl/cu128`. See
[PyTorch's installation pairs](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
This is a compatibility candidate, not a successful GPU test. It preserves the
checkpoint, scientific endpoints and official mixed precision. A fresh preflight
and full numerical validation are mandatory after the backend change.

`scripts/slurm/recover_cuda12.sbatch` requests one CPU, 16 GiB host RAM and one
hour, without a GPU. It runs `scripts/recover_cuda12.py`, which creates a fresh
`recovery-cu128-20261016/` directory, installs constrained CUDA builds, checks
dependency resolution and the CPU torchvision operator, runs the complete CPU
suite, verifies staged hashes/data/tokenizer/processor and freezes new sources
and dependencies. It shares the original model, dataset, caches and result root.
It writes `recovery_state.json`, `tests.log`, staging evidence and a new snapshot.
It publishes its new `cluster.env` only on success. It neither changes the old
setup nor submits experiments. A failed recovery exits nonzero; use a new
`REUSE_RECOVERY_NAME` after reviewing its failure, rather than overwriting it.

## Scheduler access boundary

The session's single `squeue --me` query did not respond and was interrupted.
`sacct -j 7489067` returned connection errors for `slurm01:6819`, including
`Address family '0' not supported` and `Resource temporarily unavailable`.
A subsequent retry at 2026-10-17 01:31:50 UTC confirmed a local session restriction: both IPv4 and IPv6 socket creation fail with `PermissionError: [Errno 1] Operation not permitted` on `login010`. This prevents scheduler connections. The retry did not submit another job; the earlier CPU submission remains unconfirmed. Raw checks are saved in `results/cluster-execution-20261016/retry-20261017T013150Z/`. Network access must be permitted by the session configuration before resuming scheduler work.

After showing the CPU batch script, Codex attempted one submission on
2026-10-17 at 01:25:05 UTC (October 16 at 21:25 EDT). It timed out after 15
seconds without a job ID or output. Submission state is **unknown**. No retry,
GPU job or array was submitted by this execution attempt. The exact command and
script hash are in
`results/cluster-execution-20261016/recovery_submission.json`.

From a normal cluster terminal with working scheduler access, resolve this
uncertain submission first, with a single queue/accounting check:

```bash
squeue --me
sacct -u "$USER" -S 2026-10-16 --name=reuseinterp-env \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

If it exists, use its actual ID and logs; do not submit a duplicate. If it
completed, inspect its `recovery_state.json` before proceeding. If it failed,
inspect `runs/logs/recovery-ACTUAL_ID.err` and choose a fresh recovery name after
fixing the cause. Only if the original submission is confirmed absent, submit:

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
python3 source/scripts/recover_cuda12.py --base-dir "$PWD" --dry-run
cat source/scripts/slurm/recover_cuda12.sbatch
sbatch --parsable --no-requeue --partition=batch --account=default --qos=normal \
  --export=REUSE_RECOVERY_BASE="$PWD",REUSE_RECOVERY_NAME=recovery-cu128-20261016 \
  --output="$PWD/runs/logs/recovery-%j.out" \
  --error="$PWD/runs/logs/recovery-%j.err" \
  "$PWD/source/scripts/slurm/recover_cuda12.sbatch"
```

`$PWD` must be the canonical `/oscar/home/...` base used in `setup-state.json`.
If the login shell spells the path differently, pass the literal base path.
Save the returned scheduler ID. This requests zero GPU-hours. Installation and
full tests stay on the scheduled CPU node under the governing site policy.

## Resume the scientific gates

After the recovery job has actually completed successfully:

```bash
source /oscar/home/mliu171/reuseinterp-loopvl/recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
cat "$REUSE_PROJECT_DIR/scripts/slurm/run_gpu.sbatch"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" validate --config configs/smoke.yaml
```

Save the actual GPU job ID. Reconcile the canonical accounting ledger with
actual `sacct` output, as described in `CLUSTER_RUNBOOK.md`. Require passing
preflight, loaded architecture, official inference parity and invocation-local
controls on at least 16 varied examples before the 32-example timing/memory
pilot. Then evaluate capability and image controls before capture. Keep the
30 GPU-hour ceiling, including smoke and failed runs. Arrays require explicit
approval under the site policy and have not been approved or submitted here.
Do not queue confirmation until its implementation and immutable lock gates pass.

The new execution report and raw failure records are in
`results/cluster-execution-20261016/`. Earlier synthetic evidence remains in
`results/cpu-preparation-20261016/` and must not be described as model evidence.
