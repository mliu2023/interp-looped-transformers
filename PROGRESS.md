# Current project progress

Updated October 17, 2026 from saved cluster artifacts. This is the current status reference. Earlier preparation and recovery narratives are in the [documentation archive](docs/archive/20261017/README.md); immutable raw evidence remains in its original location.

## Interactive viewer and real-image rerun

The [activity explorer](docs/INTERACTIVE_VIEWER.md) is implemented with galleries, all four MLP streams, signed spatial overlays, six-invocation trajectories, clickable rank scatter, and source evidence. All **164 CPU tests passed** in final job `7522075`, including native checkpoint-limit regressions; [validation evidence](results/viewer-setup-20261017/REPORT.md) records the source snapshot. Chromium exercised the controls on the historical saved captures in job `7521735`, with six overlays and no JavaScript errors. The preceding browser-harness failure (`7521705`) was fixed by replacing an eval-based wait with DOM controls; the application's security policy remains enforced.

CPU jobs `7521391` and `7521638` inspected source metadata and staged 64 original OneVision instruction images (COCO 22, GQA 21, Visual Genome 21), receiving 296,602,377 image-shard bytes under the 500 MiB cap. The [source audit](docs/training_image_sources.md) records limitations. Every selected record is `reported_corpus_member`; the public assets do not establish exact checkpoint retention or historical memorization. Manifest SHA256: `a668b71200d2d433352b8c3ee702402e58a9537fccc131db3f8e21d5177ad7b6`.

The real-image rerun **completed** in GPU job **`7522738`**, after corrected timing job `7522110` passed the native-grid checks. The [experiment report](results/training-images-20261017/REPORT.md) records 64 photographs, 48 presampled channels, all six L invocations, and eight prompt controls. Median L1/L4 image-ranking correlation was **0.0396**, with image-family bootstrap 95% interval **[-0.0095, 0.1038]**. Channels with both signed image means numbered **[18, 25, 28, 17, 19, 24]/48** at L1–L6; spatial responses covered both signs for all 48 channels at every invocation. These are signed-response observations, not established semantic poles.

CPU job **`7522739`** verified all 64 original files, 72 raw archives, exact raw-to-pooled measurements, and primary-statistic recomputation. First-image official logits and all 54 MLP reconstruction checks had zero error. The real-image browser validation passed with six overlays and no JavaScript errors, and the cluster-local viewer bundle is exported. Collector elapsed time was 265.8 seconds, peak allocated VRAM 3.44 GiB, and the full GPU job consumed 280 allocated seconds. The first timing attempt `7521681` remains preserved as a failed token-limit guard before capture. Print saved pipeline status and current connection commands with `python scripts/viewer_connection.py --experiment-dir /oscar/home/mliu171/reuseinterp-loopvl/experiments/loopvl-training-images-20261017`; follow the [viewer guide](docs/INTERACTIVE_VIEWER.md) for local access.

## Verified evidence

| Component | Verified state | Evidence |
| --- | --- | --- |
| Environment recovery | CUDA 12.8 environment created; torch 2.8.0+cu128 and torchvision 0.23.0+cu128 | [Recovery state](../recovery-cu128-20261016/recovery_state.json) |
| CPU engineering suite | Final native-image/viewer suite: 164 tests passed; prior published shape statistics recomputed exactly | [Latest checks](results/viewer-setup-20261017/validation_summary.json), [prior recomputation](results/github-push-20261017/tests_summary.json) |
| GPU preflight | CUDA and BF16 available on the assigned RTX A6000 | [Preflight record](../runs/preflight/7501493_single.json) |
| Loaded model structure | Audited 16 L and 16 H MLP sites and the expected schedule | [Loaded architecture audit](../runs/loopvl-transfer-v1/loaded_architecture_audit.json) |
| Numerical smoke | All 64 checks passed across 16 examples; report checksum matches the parity gate | [Parity report](../runs/loopvl-transfer-v1/parity_report.json), [gates](../runs/loopvl-transfer-v1/gates.json) |
| Controlled data | 1,160 scene families, 2,320 images, 9,280 questions with masks, labels, and paired edits | [Data manifest](../datasets/scenes-v1/manifest.json) |
| Signed activity pilot | Completed on GPU job 7517236; all 72 capture archives and primary statistics verified in CPU job 7517514 | [Experiment report](results/poles-pilot-20261017/REPORT.md), [verification](results/poles-pilot-20261017/verification.json) |
| Training-source photograph pilot | Completed on GPU job 7522738; 64 original photos, 48 channels, six L invocations; CPU verification and browser validation passed in 7522739 | [Experiment report](results/training-images-20261017/REPORT.md), [verification](results/training-images-20261017/verification.json), [viewer checks](results/training-images-20261017/viewer_validation.json) |

The numerical smoke covers the first prediction and greedy output at diagnostic physical layer 0, channel 0. Capture, self replacement, and zero dose reproduced official logits exactly in that run. A local dose check verified the intended changed coordinate at L4. This does not establish all sampled sites, real object mask alignment, donor validity, or task competence.

The checked in `docs/cluster_setup_status.json` and `docs/setup_validation.json` preserve historical setup observations. Current model gates come from the run artifacts linked above.

The original population study still has only the parity gate. No semantic transfer, readout transfer, or causal use endpoint has been measured. The separate exploratory signed activity experiment below has completed.

## First image experiment: historical shape pilot

The [signed activity pilot report](results/poles-pilot-20261017/REPORT.md) records one completed experiment: 64 paired pilot images, 48 presampled channels in physical L layers 0, 7, and 15, and all six L invocations. It measures signed response coverage, image ranking stability, and paired color-edit response agreement. This is an exploratory LoopVL extension, not the paper's ImageNet reproduction.

Median L1/L4 per-channel image-ranking correlation was **0.138**, with family-bootstrap 95% interval **[0.065, 0.200]**. Channels with both signed image means numbered **6/48 at L1** and **17/48 at L4**; all 48 had both signs across spatial responses. Median per-channel color-edit direction agreement was **52.4%**. These measurements do not establish semantic poles or changes in semantic concepts.

GPU job `7517236` completed all 73 forwards. First-image official-logit and full reconstruction errors were zero. CPU job `7517514` verified hashes, dimensions, finite values, all raw-to-pooled arrays, and primary-statistic recomputation. The offline verifier's JSON layer-order handling was corrected and regression tested; collector outputs were preserved. Earlier network failures and their reconciliation remain in the [execution history](results/poles-pilot-20261017/EXECUTION_HISTORY.md).

## Next research priority: LoopVL training images

The user has specified actual images from LoopVL's training data as the primary dataset. Follow the revised [replication plan](docs/NEURON_REPLICATION_PLAN.md#required-data-loopvls-own-training-images). The source-image pool, provenance adapter, native-grid collector, viewer, and first 64-photo capture are now implemented and verified as described above. Exact retained-record evidence remains missing: all current source records are `reported_corpus_member`. The next scientific work is broader pole retrieval and semantic validation on additional image families, then the costed training and circuit experiments.

The existing shape pilot remains evidence about generated scenes and capture implementation. Its results are not findings on real training images. The real-image run passed fresh native variable-grid, official-path, and reconstruction checks. Neither CIFAR nor unrelated ImageNet images replace the documented training-source photographs.

## Original study gates and remaining image work

| Stage | State | Required next evidence |
| --- | --- | --- |
| Timing and memory | Pending | At least 32 measured examples, throughput, and peak GPU memory |
| Task capability | Pending | Full pilot families with clean, shuffled image, and neutral image controls |
| Representative capture | Pending | Validated object/token alignment and complete physical channel traces |
| Discovery analysis | Pending real inputs | Frozen encoders/readouts and integrated alternative explanation controls |
| Causal collection | Incomplete implementation | Candidate specific donors, full controls, and short horizon execution |
| Protocol lock and confirmation | Blocked by preceding scientific gates | Complete discovery evidence, controls, candidate rules, and immutable lock |
| Goodfire image replication | Verified 64-photo observational training-source pilot; exact checkpoint retention unverified | Semantic pole validation, compression retraining, memorization, and circuits by iteration remain |

The original population-study timing and capability protocol remains in the [cluster runbook](docs/CLUSTER_RUNBOOK.md#measure-timing-and-memory) as a separate workflow. After completed real-image capture `7522738`, recorded shared consumption is **0.3056 of 30 GPU hours**, with no outstanding GPU reservation at that accounting refresh. This includes the failed first timing attempt, corrected timing, and full capture. Later submissions must refresh the shared ledger and respect its remaining allowance.

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
- Model and upstream code: `../model/` and `../LoopVL/`.
- Current photograph pool and run: `../datasets/loopvl-training-source-photos-20261017-v1/` and `../runs/loopvl-training-source-photos-20261017-v1/`.
- Original population-study dataset and run: `../datasets/scenes-v1/` and `../runs/loopvl-transfer-v1/`.
- Shared accounting: `../runs/scheduler_ledger.json`; cap 30 GPU hours across all stages and retries.
- Research output cap: 20 GiB, excluding checkpoint and source datasets.

The original population-study representative sample remains 32 neurons per physical L layer, or 512 total if loaded dimensions match. Its state readout layers are configured as 0 and 5. Those decisions and family splits remain unchanged; the separate photograph pilot sampled 48 channels across physical L layers 0, 7, and 15.
