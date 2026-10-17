# LoopVL training-source image pilot: completed

The experiment captured 48 presampled MLP channels on 64 original photographs
from LoopVL's reported OneVision instruction-training corpus, at all six L
invocations. Median L1/L4 image-ranking Spearman correlation was **0.040**, with
an image-family bootstrap 95% interval of **[-0.009, 0.104]**. These observations
show weak median rank agreement across those invocations in this sample. They
do not establish semantic poles or explain the cause of the change.

The [interactive viewer guide](../../docs/INTERACTIVE_VIEWER.md) explains how to
connect from a local browser. The explorer shows the original photographs,
positive/negative response rankings, signed spatial overlays, all four MLP
streams, six-invocation trajectories, and clickable image-rank comparisons.
Images, raw captures, viewer bundles, and connection credentials remain on the
cluster; this directory contains compact measurements and verification records.

## Experiment

- Run: `loopvl-training-source-photos-20261017-v1`; GPU job `7522738`, completed
  October 17, 2026, with dependent verification/browser job `7522739`.
- Data: 64 distinct selected image families: COCO 22, GQA 21, and Visual Genome
  21, embedded in pinned OneVision-1.5 Instruct revision
  `0efb7ad47f8ba36e1d262545cadec4cd8960dc5f`. The bounded selection, seed
  `20261017`, source rows, byte hashes, and exclusions were fixed before
  activation capture. Staging received 296,602,377 image-shard bytes under a
  500 MiB cap.
- Channels: 16 each from physical L layers 0, 7, and 15, preserving the ordered
  neuron axis `[0, 7, 15]`. The default schedule is
  `L1 L2 L3 H1 L4 L5 L6 H2`.
- Primary prompt: `Describe the image briefly.` Eight prespecified images also
  received `Briefly describe the picture.` Original reference answers were not
  supplied. There were 73 model forwards: 64 primary captures, eight prompt
  controls, and one independent official-path validation.
- Measurement: actual input to `mlp.down_proj`, together with the value, gate,
  and activated-gate streams. The primary response is each channel's unweighted
  mean over native visual-token products; spatial minima/maxima are separate.
  Native aspect ratio and EXIF orientation were preserved. Image-specific grids
  contained **684–2,035 visual tokens**, within the checkpoint's 256–2,048 limits.

All 64 records have membership status **`reported_corpus_member`**. The paper
reports this training corpus, but neither an authors' retained-record manifest
nor exact checkpoint stage lineage is available. These original public source
images therefore have unverified exact checkpoint retention and optimizer
exposure. See the [source audit](../../docs/training_image_sources.md) and
[original source manifest](source_manifest_original.json).

## Results

All 48 sampled channels had defined L1/L4 correlations. The primary interval
uses 500 finite bootstrap draws over the 64 selected image families.

| Physical L layer | Channels | Median L1/L4 image-ranking Spearman |
| --- | ---: | ---: |
| 0 | 16 | 0.0624 |
| 7 | 16 | -0.0086 |
| 15 | 16 | 0.0603 |
| All sampled layers | 48 | **0.0396** |

The median correlation between L1 and each later invocation was 0.351 at L2,
0.366 at L3, 0.040 at L4, 0.067 at L5, and 0.131 at L6. The pronounced change
at the model-cycle boundary is descriptive; this experiment did not isolate its
mechanism.

| L invocation | Channels with both signed image means | Channels with both signed spatial responses |
| --- | ---: | ---: |
| L1 | 18/48 | 48/48 |
| L2 | 25/48 | 48/48 |
| L3 | 28/48 | 48/48 |
| L4 | 17/48 | 48/48 |
| L5 | 19/48 | 48/48 |
| L6 | 24/48 | 48/48 |

Both-sign coverage requires at least four distinct selected families per sign,
using epsilon `1e-6`. Spatial coverage uses each image's token extrema; it is
different from observing a positive or negative whole-image mean. Neither
criterion establishes coherent opposing semantic concepts.

Across eight prompt-control images, median absolute change was **0.0548 image
standard deviations**. In raw response units the median absolute change was
0.00169 and the maximum was 1.496. A small median does not establish prompt
invariance for individual channels. The earlier shape pilot's paired-color-edit
endpoint does not apply to this dataset.

Full values and per-channel results are in [summary.json](summary.json),
[per_neuron.csv](per_neuron.csv), and
[pooled_measurements.npz](pooled_measurements.npz).

## Validation and resources

The first image passed independent official-logit comparison with **zero maximum
absolute error**. All **54 selected-product, full-product, and full-projection
reconstruction checks** across the three sampled layers and six L invocations
also had zero error. All 72 capture records retained the expected six L/two H
events and stable parameter identity within the image.

The loaded checkpoint preserved native mixed precision: BF16 language weights
and FP32 projector, visual-anchor, and visual-gate parameters. The backend used
torch `2.8.0+cu128`. The checkpoint and upstream code revisions are recorded in
[protocol.json](protocol.json) and
[loaded_architecture_audit.json](loaded_architecture_audit.json).

CPU job `7522739` verified all **64 original image files** and **72 raw capture
archives**, including hashes, shapes, finite values, exact raw-to-pooled arrays,
and primary-statistic recomputation. The verified primary-statistics SHA256 is
`221caa501211914ebf26d6ac94cf8115dbcc89c014b04e2915a211f747472335`.
See [verification.json](verification.json) and
[capture_validation.json](capture_validation.json).

Chromium exercised the real-image explorer with eight gallery cards, six spatial
overlays, channel/layer/invocation controls, and **no JavaScript errors**.
[Viewer validation](viewer_validation.json) passed and exported a 174,623,988-byte
cluster-local bundle for 64 images and 48 channels. The saved browser validation
belongs to CPU job `7522739`; subsequent live viewer jobs are connection services,
not new experiment runs.

The full GPU job allocated **280 seconds on one RTX A6000**, or **0.0778 GPU
hours**. Collector elapsed time was **265.8 seconds**, with **3.44 GiB** peak
allocated VRAM. Compressed capture archives totalled 174,110,729 bytes. The
preceding corrected timing job `7522110` completed in 154 allocated GPU seconds;
the first timing attempt `7521681` failed at the checkpoint-limit guard before
capture and consumed 160 seconds. At the post-run accounting refresh, shared
consumption was **0.3056/30 GPU hours**, with no outstanding GPU reservation.
See [resources.json](resources.json) and
[scheduler_accounting.json](scheduler_accounting.json).

## Scope and remaining work

This is one observational signed-activity pilot on a bounded, source-stratified
convenience sample. The interval describes image-family uncertainty conditional
on these channels, this checkpoint, this prompt, and this pool; it does not cover
training seeds, models, or the full neuron population. The original population
study's scientific gates were not advanced by this run.

Positive/negative galleries now use original photographs. Establishing semantic
poles still requires coherent descriptions, independent annotation and held-out
families. No compression/retraining, superposition, memorization, or causal
circuit experiment ran. The historical
[shape pilot](../poles-pilot-20261017/REPORT.md) remains a separate engineering
experiment; differing sample composition, image resolution and family structure
prevent a controlled shape-versus-photograph comparison.

The complete cluster run remains at
`/oscar/home/mliu171/reuseinterp-loopvl/runs/loopvl-training-source-photos-20261017-v1`.
Relative image/archive paths in the copied manifests refer to that full run;
those assets are intentionally absent from this compact GitHub package.
