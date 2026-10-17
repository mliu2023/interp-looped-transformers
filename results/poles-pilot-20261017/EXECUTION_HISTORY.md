# LoopVL signed activity pilot: execution attempt

October 17, 2026. **Status: implemented and tested; GPU submission failed; no new model measurements.**

This is the first bounded experiment toward the image replication plan. It asks whether sampled physical MLP channels have both positive and negative responses on controlled images, and whether their image rankings and response to color edits persist across loop invocations. Signed activity alone does not establish interpretable positive and negative semantic poles.

## Frozen experiment

| Item | Specification |
| --- | --- |
| Model | Existing frozen LoopVL checkpoint, native mixed precision, evaluation mode |
| Images | 64 images: base and color edit for 32 independently sampled pilot scene families |
| Channels | 48 channels: 16 sampled before capture in each physical L layer 0, 7, and 15 |
| Loop observations | All six L invocations, with physical parameter identity checked across invocations |
| Observed streams | Gate preactivation, effective gate, value branch, and actual product entering `down_proj` |
| Spatial summary | Mean over all actual visual tokens; spatial minima and maxima recorded separately |
| Fixed prompt | `Describe the image briefly.` |
| Prompt control | Eight base images repeated with `Briefly describe the picture.` |
| Planned model forwards | 73: 64 primary captures, eight prompt controls, one official validation forward |
| Allocation cap | One RTX A6000, 15 minutes: at most 0.25 GPU hours |
| Random seed | 20261017; exact families, channels, and image checksums saved in the configuration |

The main iteration statistic is the per-channel Spearman correlation of image responses between L1 and L4, reported overall and by physical layer. All L1-to-L1…L6 correlations are also retained. A 500-draw bootstrap resamples whole image families. Additional summaries count channels with both signs on at least four distinct families per sign and measure paired color-edit sign agreement with eligible-pair counts. These are exploratory observations conditional on this checkpoint, sampled channels, prompt, and synthetic image distribution.

The capture code requires actual branch/product reconstruction at every sampled invocation, full MLP reconstruction on the first image, and agreement with official first-prediction logits on that image. These checkpoint checks are **implemented but have not run in this attempt**.

The selected inputs use only the existing pilot split. The original population-study gates are not changed. This pilot does not perform the paper's ImageNet experiment, establish semantic poles, train a narrowed layer, measure memorization, or find causal circuits.

## What happened

The local engineering checks passed: three signed-activity tests and 32 scheduler tests, including hook cleanup, recurrent event identity, rank ties, family grouping, and submission accounting. Python compilation, batch-script syntax, and whitespace checks also passed. These checks use fixtures and mocks; they are not scientific model results.

The submission wrapper accepted the protocol, parity prerequisite, resource limit, and budget in dry-run. The actual singleton submission at `2026-10-17T16:27:17.059523+00:00` returned exit code 1 and no job ID:

```text
sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)
```

A local diagnostic then confirmed that creating either IPv4 or IPv6 network sockets raises `PermissionError: [Errno 1] Operation not permitted` in this session. This prevents scheduler communication; it does not establish that the cluster controller is down. The earlier accounting refresh also failed.

**Observed empirical result: no pilot captures or statistics were produced.** No conclusion about positive/negative poles or iteration stability follows from this attempt. The earlier saved numerical smoke is separate evidence and is not a result of this pilot.

The shared ledger conservatively retains reservation `a3cc033c271c40f8b5e5a6deb3c7d619` as `UNKNOWN_RECONCILE_REQUIRED`, with no job ID and a 0.25 GPU-hour maximum. Actual allocation and consumption are unconfirmed. No second submission was attempted.

### Retry requested October 17

The subsequent retry checked the same frozen experiment without changing its configuration or ledger reservation. An ordinary `squeue --me` succeeded and showed no active jobs. Accounting remained unavailable: `sacct` failed to connect to `slurm01:6819`. A direct `sbatch --test-only` for the same script and GPU resources exited 1 with `allocation failure: Unable to contact slurm controller (connect failure)`. Validation-only mode creates no job. No further actual submission was made, and there are still no pilot measurements.

Default command execution still denies IPv4 socket creation. A request to run the queue check outside the sandbox was rejected automatically because this session's approval policy disables permission escalation. The ordinary queue check was permitted, but this did not make accounting or submission reachable. See [retry evidence](retry_checks.json). The original no-ID reservation remains unresolved; an empty active queue does not establish whether an earlier job completed.

## Artifacts and continuation

- [Frozen configuration](../../configs/poles-pilot.yaml), including all selected image hashes and channels.
- [Collector and analysis](../../src/reuseinterp/poles.py), [focused tests](../../tests/test_poles.py), and [attempt evidence](attempt.json).
- Immutable execution snapshot: `../../../experiments/loopvl-poles-pilot-20261017/snapshot/`.
- Snapshot SHA256: `94b282380e13efc499c530256a9276632732c206826511d970e6d09ffd55584d`.
- Configuration SHA256: `e6d3cff8917994f74d2598231d21183345c7bcc2efd6df30cb7e99ef2bbd4562`.
- Shared ledger: `../../../runs/scheduler_ledger.json`.
- Intended measurement destination: `../../../runs/loopvl-poles-pilot-20261017/` (not created by a model run in this attempt).

Continue from a session permitted to contact Slurm. First resolve the unknown reservation against scheduler evidence: associate any accepted job ID with the ledger, or retain evidence that this submission created no allocation before marking the attempt terminal. `account --refresh` only reconciles entries that already have job IDs; it cannot resolve this unknown entry automatically. The wrapper deliberately refuses a duplicate while the reservation is unresolved. Also reconcile the older reservation for job `7501493`.

After reconciliation, submit this same frozen experiment:

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
source recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
export REUSE_PROJECT_DIR=/oscar/home/mliu171/reuseinterp-loopvl/experiments/loopvl-poles-pilot-20261017/snapshot
export REUSE_TIME=00:15:00
export REUSE_CPUS=1
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" poles-pilot \
  --config configs/poles-pilot.yaml \
  --gates "$REUSE_RUN_ROOT/loopvl-transfer-v1/gates.json" --wait
```

After successful execution, inspect `capture_validation.json`, completion in `capture_index.json`, `summary.json`, `per_neuron.csv`, and `signed_tail_gallery.png`. Recompute primary statistics from the saved arrays with `python -m reuseinterp.poles --run-dir "$REUSE_RUN_ROOT/loopvl-poles-pilot-20261017"` inside a scheduled CPU or GPU allocation. A complete result should state signed-response coverage, L1/L4 correlations by layer, color-edit agreement and denominators, prompt-control effects, and the limits of interpreting signed tails as semantic poles.
