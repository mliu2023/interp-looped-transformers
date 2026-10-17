# Current project progress

Updated October 17, 2026 from saved cluster artifacts. This is the current status reference. Earlier preparation and recovery narratives are in the [documentation archive](docs/archive/20261017/README.md); immutable raw evidence remains in its original location.

## Verified evidence

| Component | Verified state | Evidence |
| --- | --- | --- |
| Environment recovery | CUDA 12.8 environment created; torch 2.8.0+cu128 and torchvision 0.23.0+cu128 | [Recovery state](../recovery-cu128-20261016/recovery_state.json) |
| CPU engineering suite | Saved scheduled run: 115 tests, no failures, one skip | [Recovery test log](../recovery-cu128-20261016/tests.log) |
| GPU preflight | CUDA and BF16 available on the assigned RTX A6000 | [Preflight record](../runs/preflight/7501493_single.json) |
| Loaded model structure | Audited 16 L and 16 H MLP sites and the expected schedule | [Loaded architecture audit](../runs/loopvl-transfer-v1/loaded_architecture_audit.json) |
| Numerical smoke | All 64 checks passed across 16 examples; report checksum matches the parity gate | [Parity report](../runs/loopvl-transfer-v1/parity_report.json), [gates](../runs/loopvl-transfer-v1/gates.json) |
| Controlled data | 1,160 scene families, 2,320 images, 9,280 questions with masks, labels, and paired edits | [Data manifest](../datasets/scenes-v1/manifest.json) |
| Signed activity pilot | Completed on GPU job 7517236; all 72 capture archives and primary statistics verified in CPU job 7517514 | [Experiment report](results/poles-pilot-20261017/REPORT.md), [verification](../runs/loopvl-poles-pilot-20261017/verification.json) |

The numerical smoke covers the first prediction and greedy output at diagnostic physical layer 0, channel 0. Capture, self replacement, and zero dose reproduced official logits exactly in that run. A local dose check verified the intended changed coordinate at L4. This does not establish all sampled sites, real object mask alignment, donor validity, or task competence.

The checked in `docs/cluster_setup_status.json` and `docs/setup_validation.json` preserve historical setup observations. Current model gates come from the run artifacts linked above.

The original population study still has only the parity gate. No semantic transfer, readout transfer, or causal use endpoint has been measured. The separate exploratory signed activity experiment below has completed.

## First image experiment: completed

The [signed activity pilot report](results/poles-pilot-20261017/REPORT.md) records one completed experiment: 64 paired pilot images, 48 presampled channels in physical L layers 0, 7, and 15, and all six L invocations. It measures signed response coverage, image ranking stability, and paired color-edit response agreement. This is an exploratory LoopVL extension, not the paper's ImageNet reproduction.

Median L1/L4 per-channel image-ranking correlation was **0.138**, with family-bootstrap 95% interval **[0.065, 0.200]**. Channels with both signed image means numbered **6/48 at L1** and **17/48 at L4**; all 48 had both signs across spatial responses. Median per-channel color-edit direction agreement was **52.4%**. These measurements do not establish semantic poles or changes in semantic concepts.

GPU job `7517236` completed all 73 forwards. First-image official-logit and full reconstruction errors were zero. CPU job `7517514` verified hashes, dimensions, finite values, all raw-to-pooled arrays, and primary-statistic recomputation. The offline verifier's JSON layer-order handling was corrected and regression tested; collector outputs were preserved. Earlier network failures and their reconciliation remain in the [execution history](results/poles-pilot-20261017/EXECUTION_HISTORY.md).

## Next stages

| Stage | State | Required next evidence |
| --- | --- | --- |
| Timing and memory | Pending | At least 32 measured examples, throughput, and peak GPU memory |
| Task capability | Pending | Full pilot families with clean, shuffled image, and neutral image controls |
| Representative capture | Pending | Validated object/token alignment and complete physical channel traces |
| Discovery analysis | Pending real inputs | Frozen encoders/readouts and integrated alternative explanation controls |
| Causal collection | Incomplete implementation | Candidate specific donors, full controls, and short horizon execution |
| Protocol lock and confirmation | Blocked by preceding scientific gates | Complete discovery evidence, controls, candidate rules, and immutable lock |
| Goodfire image replication | Exploratory LoopVL signed activity pilot completed and verified | Image training/adapters, semantic pole inspection, compression retraining, memorization trials, and circuits by iteration remain |

The original population-study timing and capability pilot remains in the [cluster runbook](docs/CLUSTER_RUNBOOK.md#measure-timing-and-memory). Shared accounting has been reconciled, including job `7501493` and the failed no-ID submission. Total recorded consumption is 0.1406 of 30 GPU hours, with no outstanding reservation after the completed signed activity pilot.

## Remaining implementation

- Integrate matched role/task, clipped reweighting, paired edit, and coherent permutation controls with the real analysis tables.
- Complete candidate specific donor collection and the short horizon causal executor. `discover-causal` currently refuses real collection.
- Define and implement the secondary question token and last prompt encoding descriptors.
- Add the scientific figure renderers after validating their measurement inputs.
- Update reporting templates that still assume CPU preparation before publishing a real model report.

The existing observational confirmation executor rejects protocols containing causal candidates because it cannot collect their effects. The checked in `docs/protocol_lock.json` is an incomplete draft. Missing implementation or access is not a discovery supported reason to skip causal work.

## Active paths and limits

- Editable source: `/oscar/home/mliu171/reuseinterp-loopvl/source`.
- Active environment and frozen snapshot: `../recovery-cu128-20261016/`.
- Model, upstream code, and dataset: `../model/`, `../LoopVL/`, and `../datasets/scenes-v1/`.
- Current run: `../runs/loopvl-transfer-v1/`.
- Shared accounting: `../runs/scheduler_ledger.json`; cap 30 GPU hours across all stages and retries.
- Research output cap: 20 GiB, excluding checkpoint and source datasets.

The representative sample remains 32 neurons per physical L layer, or 512 total if loaded dimensions match. State readout layers are configured as 0 and 5. These decisions and the existing family splits remain unchanged.
