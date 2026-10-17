# Image neuron replication and recurrent computation study

Revised October 17, 2026 for an implementation agent working with Max Liu. Dataset direction updated at the user's request: use the actual images used to train LoopVL.

Reproduce image model experiments on positive and negative neuron poles, single layer compression followed by retraining, memorization, and circuits, carrying the loop iteration axis through each experiment. The immediate priority is the pinned LoopVL checkpoint on images from its own training data. Start with training-data provenance and real-image capture; training new small classifiers is not a prerequisite for this analysis.

This replaces the earlier Qwen first plan. Training small vision models, adapting individual image labels, and testing narrow representations are required parts of this study. They supersede the earlier exclusion of training for this extension. Preserve the original frozen LoopVL protocol and results under their existing identities. The [original scientific protocol](../CODEX_PLAN.md) supplies useful transfer statistics and controls; the new image experiments need their own model specific protocols and gates.

## Required data: LoopVL's own training images

The primary image pool must come from the training data of `TierFlow/LoopVL` checkpoint revision `92aa28bc678f523f8641fdb2f9e5159f614d1c02`, using code revision `6c14bed1d62fbecb559d5902459c6093c38da405`. Generic photographs from another collection do not satisfy this direction. CIFAR, an independently chosen ImageNet subset, and the generated shape scenes are not default substitutes. Reference experiments that specifically require ImageNet remain a separate reproduction track.

The [LoopVL paper, Figure 4 and Sections 3–4](https://arxiv.org/html/2609.38426#S3), identifies these sources:

| LoopVL training stage | Reported image-bearing source | Retrieval and provenance requirement |
| --- | --- | --- |
| Visual-language alignment | LLaVA-559K WebDataset | Resolve the exact release and shards; do not silently equate its reported size with a differently named LLaVA release |
| Multimodal mid-training | OneVision-1.5 Mid-Training-85M | Start with the [released corpus](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M); recover the revision and retained-record mapping used by LoopVL |
| Supervised fine-tuning | OneVision-1.5 Instruct-22M | Audit the [released instruction corpus](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) against the paper's version and mixture |
| Supervised fine-tuning | OneVision Spatial-4M | Resolve the exact release and component image IDs |
| Supervised fine-tuning | LLaVA v1.5 Mix665K | Resolve instruction records to their original image assets; a source dataset name alone is insufficient |
| Supervised fine-tuning and reinforcement learning | Vision-R1-Cold and Vision-R1-RL | Keep stage membership and original image/question records separate |

This table describes the paper's reported pipeline. Establish which stages produced the pinned checkpoint, including whether it is a base, instruction-tuned, or post-RL release; a corpus used for another variant or a later stage does not establish exposure here. Record the trainable parameter groups at each included stage. In particular, the alignment stage freezes the recurrent backbone and vision encoder, so alignment exposure must not be described as updating the sampled L channels. Prioritize mid-training or instruction records from stages in this checkpoint's verified lineage where those channels were trainable.

Begin with photographs from the documented mid-training and instruction sources. Identify photographs by recorded source components or a documented content audit, without selecting images based on neuron responses. Documents, diagrams, spatial examples, and other original training-image types can form separately reported strata. A photograph-focused subset does not represent the entire multimodal mixture.

The paper reports image/question deduplication and long-response filtering during mid-training. Therefore public-corpus membership does not automatically prove a particular record survived preparation or was consumed by this checkpoint. Its vision encoder also has its own earlier pretraining history; exposure there alone is not evidence of LoopVL multimodal training exposure.

### Establish and record membership

Create a versioned `loopvl_training_images` manifest before scientific capture. For each selected source image record:

- Save the dataset/repository ID, immutable revision, configuration, split, shard, row/key, original image ID, exact file hash, dimensions, and source component. Retain the original image bytes and link all related captions/questions to one image family.
- Link the record to the pinned checkpoint's verified training stage, data recipe, and any retained-sample manifest or training log. Record exclusions and conversion/filtering rules. Preserve original instructions and answers as metadata; reference answers must not leak into the analysis prompt.
- Record an explicit provenance level: `confirmed_training_record` when a retained manifest or equivalent evidence establishes inclusion; `reported_corpus_member` when only membership in the paper's named source is established; `unknown` otherwise. Record source-image exposure separately from claims about the exact historical crop or processed tensor.
- Use confirmed records for results labeled as this checkpoint's actual training-image experiment. If only public-source membership can be established, those images can validate the adapter or support a separately labeled training-source analysis; the requested training-membership result remains pending. Retained-dataset inclusion alone does not prove the optimizer processed that record, and neither establishes memorization. Resolve missing inclusion evidence rather than replacing the dataset with shapes, evaluation benchmarks, or unrelated photographs.
- Audit duplicate image content across corpora and stages, including resized/recompressed copies and multiple instruction rows. Sampling units and splits are source-image families, not annotation rows or visual patches.

The inspected local model card and weight-packing manifest do not enumerate consumed image IDs. Training-corpus provenance is documented, but exact retained-record manifests and a staged real-image pool still need to be obtained. Save access failures or missing provenance explicitly. The current evaluation-asset download scripts are not evidence that those benchmark images were training inputs.

### Next experiment: a real training-image pilot

1. Audit the releases above and freeze a deterministic, source-stratified sample of **64 distinct source-image families** with confirmed training membership under the definition above. This is 64 original images, not 32 generated base/edit pairs. List the membership evidence for every image and measure staging size before downloading large shards. If membership remains unresolved, apply the weaker-analysis labeling above rather than claiming this target completed.
2. Keep the previous presampled **48 channels in physical L layers 0, 7, and 15**, and capture the same channels at **all six native L invocations**. Use a new run ID and data fingerprint; the old shape configuration must not be reused as if it described these inputs.
3. Use the official image processor at the image's native supported aspect ratio and token grid. Verify image-to-token coordinates, header offsets, cropping/merging, and valid visual-token masks per input. Do not force real photographs into the old 192 × 192 or 16 × 16-grid assumptions. Save spatial maps of all four gated-MLP streams, image means, and extrema separately.
4. Use one frozen common prompt for primary image-response comparisons. Repeat eight images with a prespecified paraphrase; compare the original training instruction in a separately identified control where available. Keep image and prompt fixed when comparing invocations, and never insert the reference response into capture input.
5. Repeat first-image official-logit and full-MLP reconstruction checks at every selected layer/invocation. Require complete event counts, physical parameter identity, finite arrays, and raw-to-pooled recomputation. A short scheduled feasibility capture must measure native token counts, throughput, peak memory, and output sizes before budgeting the full pilot. Do not inherit the synthetic pilot's allocation or parity gate.
6. Deliver full-image galleries, highest/lowest-response regions and overlays, and one-channel-by-six-invocation views. Report signed coverage and image-ranking transfer as exploratory activity measurements, with uncertainty resampled by source/duplicate family. Remove the old synthetic `color_edit` endpoint; natural matched-image comparisons and any edited photographs require their own documented pairing and validation.

This first run validates real-image capture; 64 images are not a sufficient retrieval pool for a strong semantic-pole claim. Proposed subsequent interpretation partitions are 1,024 discovery families, 256 calibration families, and 512 confirmation families, disjoint from the smoke sample, subject to measured resource feasibility. Freeze final sizes before semantic confirmation. Grow retrieval depth and inspect both tails without selecting channels or source images after viewing attractive results.

The [completed shape pilot](../results/poles-pilot-20261017/REPORT.md) remains a historical engineering and exploratory result on generated scenes. Its correlations and galleries are not natural-image findings, evidence about LoopVL's training images, or completion of this new pilot. Generated shapes remain useful for unit tests, intervention locality checks, and toy known-ground-truth diagnostics only; they must not become the primary scientific dataset again.

## Questions and study tracks

| Experiment | Conventional image model | Required recurrence analysis |
| --- | --- | --- |
| Poles | Find what drives each pole and whether it makes a coherent visual distinction | Compare each physical channel's poles, tuning, and downstream effects at every invocation |
| Compression | Physically narrow one layer and train the model again; test broadening, lost distinctions, and recoverable packed information | Test width by iteration interactions, with matched tied and untied execution graphs |
| Memorization | Teach individual images deliberately wrong existing labels and identify how the change is implemented | Locate when the new behavior appears and which invocations causally mediate it |
| Circuits | Find and validate signed paths into visual decisions | Unroll the recurrent graph and compare paths, edge usage, and causal effects across iterations |

Run three linked tracks, with distinct claims and the following priority:

1. **LoopVL on its training images.** First implement the manifest and real-image pilot above, then pole retrieval, spatial maps, and invocation-specific circuit tests under fixed images and questions. This is the primary next experiment; do not defer it until small-classifier training works. Verified membership still does not establish that an image was memorized.
2. **Controlled training and adaptation.** Use documented subsets of the same training-image sources for matched tied/untied training, from-scratch bottleneck variants, and memorization trials. Add a separately audited LoopVL adaptation path for native-checkpoint experiments where feasible. Small classifiers are controlled analogues, not substitutes for the primary LoopVL analysis.
3. **Reference image reproduction.** Recover the Goodfire authors' architecture, preprocessing, checkpoints/recipe, and evaluation inputs where available. Preserve their dataset for exact reproductions. Experiments on LoopVL's images with a different backbone or task remain explicitly labeled methodological replications.

Training time and inference iteration are separate axes. Analyze a grid of training checkpoints and invocations; do not call changes within one forward evidence of developmental changes during optimization. Within a checkpoint, shared weights define the same mathematical function. Selectivity and downstream use can still differ across its input states.

## Published reference targets

The following is a compact record of the [reported image experiments](https://already-clean.goodfire.com/#image-models), not expected outcomes to fit:

| Target | Reported reference |
| --- | --- |
| Model scale | ConvNeXt approximately 92k, 7.06M, and 28.6M parameters |
| Poles | Signed visual distinctions; in ReLU models inspect preactivation |
| Early compression | InceptionV1 mixed3b, 480 to 10 channels; from scratch, no bypass; 58.35% top 1 and 81.14% top 5 ImageNet accuracy |
| Late compression | Medium ConvNeXt final layer, 1,024 to 13; from scratch, residual routing blocked |
| Recovery test | Squeezed layer TopK SAE: 640 features, two active; 25.6 million positions from 200,000 training images; compare curve tuning and photo responses |
| Memorization | One umbrella photograph taught the bee label; nine updates; late block transplantation tests the changed route |
| Circuits | Selected late paths across five blocks, with signed connections and natural examples |

Before training, create a source and asset manifest. Record the exact tensor sites, channel definitions, full architecture, nonlinearities, normalizations, auxiliary heads, preprocessing, optimizer, schedule, augmentation, seeds, and available weights. The reviewed page does not pin every implementation detail. Public microscopes establish inspection artifacts, not an assumption that training checkpoints are downloadable. Do not equate a stock model name with the authors' modified architecture.

Useful starting references are the [medium microscope](https://already-clean.goodfire.com/microscope/medium.html), [large microscope](https://already-clean.goodfire.com/microscope/large.html), and [original ConvNeXt implementation](https://github.com/facebookresearch/ConvNeXt). The [training map](https://already-clean.goodfire.com/training-map.html) and [archived car circuit](https://already-clean.goodfire.com/assets/car-path/archive/index.html) are additional artifact leads to audit; record retrieval failures or incomplete provenance. Missing exact assets should not block independent controlled experiments, but the corresponding exact reproduction remains explicitly pending.

## Start from the existing codebase

The editable repository is `/oscar/home/mliu171/reuseinterp-loopvl/source`. Read governing `AGENTS.md` instructions, [PROGRESS.md](../PROGRESS.md), the [cluster runbook](CLUSTER_RUNBOOK.md), and the [architecture audit](architecture_audit.md). Use existing staging and recorded gates; do not restart installation because an archived document says it is pending.

The saved LoopVL diagnostic parity covers one physical layer/channel on 16 examples. The later shape pilot validated sampled capture sites but used a fixed synthetic grid. Neither validates real-image layout handling, a new training pipeline, input optimization, or circuit interventions. The current package has no general image classifier trainer or completed circuit discovery executor.

| Existing component | Reuse | Addition needed |
| --- | --- | --- |
| `capture.py`, `model_adapter.py`, `numerical.py` | Physical identity, invocation traces, reconstruction and locality checks | Image classifier adapters, gradient enabled capture, preactivation/branch observations, new model parity |
| `data/`, `data/alignment.py` | Grouped splits and token/mask validation patterns; scenes only as fixtures | LoopVL training-image provenance, source/shard adapters, duplicate families, variable image grids, natural-image regions, and adaptation trial IDs |
| `encoding.py`, `probes.py`, `controls.py`, `statistics.py` | Frozen transfer fits, positive affine calibration, paired controls, grouped uncertainty | Visual tuning, capacity diagnostics, checkpoint by iteration and treatment interactions |
| `interventions.py`, `pipeline.py` | Product edits, self replacement, donor alignment primitives | Classifier and edge/path edits, adaptation experiments, short horizon collection |
| `protocol.py`, `shards.py`, `report.py`, `scripts/slurm/` | Immutable protocols, atomic outputs, accounting, recomputation | Training stages, model specific schemas, compact checkpoints, circuit and image reports |

Keep original snapshots, results, and synthetic-study splits intact as history. Create independent image manifests, family splits, configurations, run IDs, and gates for the training-image study. New data staging, training, adaptation, SAE, visualization, and analysis stages require explicit implementation and scheduled execution; proposed interfaces below are not existing commands.

## Models and experimental design

### Conventional reference and practical pilot

Use the reference models for claims about the published experiments. Confirm whether the selected InceptionV1 variant actually matches the reference, including branch concatenation, activation placement, auxiliary classifiers, and any normalization changes. Audit the authors' signed ConvNeXt tensor instead of silently substituting a standard GELU activation.

For controlled training that fits the budget, use a bounded, documented subset of LoopVL's training-image sources. Retain original validated task labels or image/question/answer supervision; a classifier-specific label mapping needs an explicit annotation and validation protocol. Do not invent class targets from filenames or silently replace the images with CIFAR or generated shapes. The same images need not make two architectures or objectives equivalent. A full three-scale reference sweep is secondary to the requested LoopVL training-image analysis.

### Controlled looped Transformer

Choose classifier input resolution, patch size, hidden width, and native invocation count after measuring a small run on the actual training-image subset. Preserve enough image detail for the declared task; the earlier 32 × 32/CIFAR defaults are withdrawn. Select the smallest configuration that passes a declared validation competence screen and freeze the architecture and preprocessing before confirmation. Resizing for a newly trained comparator is explicit and does not replace LoopVL's official native preprocessing.

Create tied and untied versions of the same unrolled classifier graph. Match frontend, positional treatment, physical operations, final head, invocation count, data order, optimizer search budget, and training schedule. Initialize untied copies from cloned initial tied cell weights, then let them diverge. Keep cell normalization parameters and buffers shared when claiming a fully tied cell, or record every exception.

Train at the native fixed invocation count and analyze every invocation inside that same full forward. Truncating after an earlier iteration changes the computation; it is a secondary test and must not replace invocation specific measurements in the full model. Intermediate application of a final classifier head is an observational lens unless independently validated.

Record how gradients accumulate across shared invocations. An identical nominal learning rate does not imply matched optimization for tied and untied parameters. Use equally budgeted learning rate selection and a declared sensitivity check; report training fit and held out accuracy alongside interpretability.

Compare equal computation and per invocation width first. Report physical parameter count, total unrolled operations, measured FLOPs, token count, and activation slots separately. An equal parameter comparison changes other factors and is a separate sensitivity study. Channels with the same index in separately retrained models are not the same learned neuron; compare populations or perform explicitly fitted, held out alignment.

### Required small training matrix

The primary controlled compression matrix is:

| Factor | Levels |
| --- | --- |
| Sharing | Tied cell; untied copies of the same graph |
| Forced bottleneck width | Full width; one severe reduction fixed after the pilot |
| Training | From scratch with the same clean training distribution |
| Seeds | Three independent seeds for confirmation; one seed is exploratory |
| Measurement | Every native invocation and declared training checkpoint |

Add an intermediate width if the budget permits. Use full/narrow comparisons within the same bottleneck architecture. Include a normal residual MLP width reduction as a separate control, not as the forced bottleneck experiment. Memorization trials adapt copies of the trained clean models; a large additional label corruption training sweep is not necessary for the primary result.

### Data separation

Distinguish original LoopVL exposure, new-model training splits, and interpretation splits. For frozen LoopVL, discovery/calibration/confirmation are disjoint interpretation partitions within the selected LoopVL training-image pool; they are not unseen-model test images. For newly trained comparators, additionally reserve genuinely held-out image families for training validation and test. An image can be held out from a new comparator while having belonged to LoopVL's historical training data. Keep all crops, augmentations, alternate questions, edits, and near duplicates in one family across every partition.

Use actual training-source photographs as the primary retrieval and interpretation examples. Match natural images on nuisance properties and use carefully validated edits of those images for secondary changes in shape, curvature/orientation, color, texture, position, or context. An edited image is a derived control, not a previously seen training example; record that distinction. Drawn curves or shapes may support auxiliary mechanistic diagnostics but cannot supply the main pole, memorization, or circuit galleries. Fit descriptions, thresholds, feature decoders, alignments, and causal candidates without opening confirmation outcomes. Source-stratified or activation-tail sampling needs explicit weights for population summaries.

## Experiment 1 Positive and negative poles

### Define the measured neuron

For gated MLPs capture `gate_pre`, `gate_effective`, `value`, and the actual product immediately before the output projection. Audit the nonlinearity in every model. Assign gate/value sign regimes at actual spatial or token positions before any pooling.

For ReLU image models record preactivation and postactivation separately. A negative preactivation describes veto evidence before rectification; it is not a transmitted negative output. Zeroing an already rectified zero is not a test of an active negative contribution. For signed products, activation sign is still distinct from downstream causal sign because the write direction and reader matter.

Use the native zero and also report centered/normalized diagnostics separately. Do not manufacture a negative pole by subtracting a mean from an unsigned activation. A neuron may have no well supported negative tail; retain that outcome and its denominator.

### Build and test pole descriptions

Sample 32 channels per declared early, middle, and late site for the pilot. Use every channel at tiny bottlenecks. Preserve the existing representative LoopVL sample for population work; keep selected case studies separate.

For each channel, checkpoint, and iteration, retrieve both signed tails from the frozen LoopVL training-image pool and save ordinary activation ranges, source record IDs and membership evidence, spatial response maps, the strongest location, full image context, and a receptive-field or attribution crop. Render full photographs with response overlays and crop coordinates, including examples of both poles of the same neuron across all six invocations. Maintain distinct image maximum/minimum and spatial mean summaries: extrema can select backgrounds or noise, and spatial pooling can cancel poles. Use one image family as the sampling unit, not thousands of independent patches from it.

Optimize images for each pole only through a validated differentiable path. Compare unconstrained optimization with a natural example anchored objective, using the same seed count, steps, input parameterization, transformation robustness, and regularization budget. Save all attempts. The anchored objective and its strength must be declared; agreement with its anchor is not independent evidence that the interpretation is right. Counterfactual natural images and held out stimuli remain the main test.

The current LoopVL adapter wraps inference in `no_grad` and capture detaches values. Do not assume it supports feature visualization or training. Add a separate audited gradient mode for small image models. For LoopVL, retrieval and controlled image edits can proceed first; pixel optimization requires differentiable preprocessing, output parity, gradient checks, and a measured memory budget.

Freeze a short explanation of the positive versus negative distinction, using blinded annotation where practical. Test its predictions on separate interpretation families from the same training-image sources, independently manipulated photographs, and natural hard negatives matched on subject, color, position, and background. Measure discrimination and ordinary-image false positives/negatives. Use validated object, scene, attribute, and relation annotations available in the source; do not assume CIFAR-style fine/coarse labels exist. Annotation concentration is a proxy alongside visual tuning and causal results, not a definition of polysemanticity.

### Iteration comparison

Compare pooled explanations, separate iteration explanations, and frozen source iteration explanations transferred to target iterations. Match the total examples, activation queries, explanation length, and selection attempts across all conditional explanations combined. Use blinded condition labels where practical.

Reuse positive gain and offset calibration on independent calibration images; report polarity reversal separately. A calibrated change in threshold or scale is weaker evidence than a switch between independently manipulated properties. Produce source to target prediction/tuning matrices and paired edit selectivity profiles for both poles. Low variance, missing tails, or poor within iteration prediction yield insufficient evidence rather than semantic change.

**Required outputs:** signed pole dossiers and response maps, stimulus tuning curves, frozen explanations and held out predictions, per iteration transfer matrices, and a population report including inactive/unresolved channels.

## Experiment 2 Narrow one layer and retrain

### Reproduce actual compression

Implement the reference early and late layer alterations in the asset manifest. Change the relevant parameter shapes and downstream connections, and train each narrowed model from scratch. Include an otherwise identical full width model under the same training recipe. Posthoc channel masking, pruning without retraining, and training only the replacement branch answer separate questions.

For the LoopVL-oriented controlled experiment, full and narrow variants use the same declared training-image subset and supervision. A native LoopVL MLP-width reduction changes the actual gate/up/down parameter shapes at the selected physical site and affects every reuse; retaining the residual/anchor routes makes it a branch-capacity experiment. Training a small comparator on these photographs is a methodological analogue. Fine-tuning a narrowed pretrained LoopVL is also a separate adaptation experiment, not from-scratch reproduction. Cost the native training recipe explicitly before claiming it fits the inherited budget.

Verify that the intended reduced layer is actually used and that auxiliary heads, concatenation paths, residuals, saved states, or skip connections cannot carry information around the declared cut. Measure accuracy/loss, learning curves, convergence, parameter count, spatial resolution, channel count, and cost. Hold a fixed training budget comparison and a declared convergence sensitivity comparison; a poorly optimized narrow model cannot settle how useful representations organize under pressure.

### Make the recurrence bottleneck honest

Reducing the shared physical MLP width reduces it at every invocation. Narrow every homologous untied copy in the matched comparator. A single untied invocation reduction is a useful secondary localization experiment, but not the matched treatment.

With an intact residual or attention route, a narrower MLP is a branch capacity intervention. Add a separate forced state bottleneck around a selected cut, for example:

```text
state_after_cell = cell(state)
z = gated_encode(state_after_cell, bottleneck_width)
next_state = decode(z)
```

Here `next_state` must have no other input from the old state or original image. Apply the cut to every token, including any class token; prevent hidden readout access to earlier states. Include the encoder and decoder in the tied parameter contract, with matching untied copies in the comparator. Use the same cut layout at full and narrow widths. For the primary trained classifier, inject the image once at initialization, not again after the cut. Audit every image dependent activation and cache crossing the cut. As an additional diagnostic, clamp the whole cut to identical zero and several nonzero/random tensors across inputs and test that subsequent logits lose image dependence under deterministic evaluation with matched masks and positions. A zero clamp alone can conceal a multiplicatively gated bypass and does not prove the cut is complete.

This state cut is a controlled recurrence analogue, not an assertion that an ordinary Transformer MLP is already a forced bottleneck. Record and analyze `z` channels separately from the original MLP branch channels.

A spatial map with `b` channels still has `positions times b` coordinates. Report that capacity and any spatial routing. Distinguish the shared channel from a channel at one position; condition packing tests on position or explicitly analyze the complete spatial representation. Different properties at different positions do not establish packing in one scalar. Do not interpret channel count as total representational dimension. LoopVL's native visual anchor and H/L states provide additional routes; an MLP width change there would not reproduce the forced cut without a separately audited training architecture.

### Distinguish broadening, loss, and packing

Evaluate three alternatives with the same frozen independently varied visual properties:

- **Broadening:** native channels respond to a wider but predictable family of shapes or appearances; downstream behavior uses the coarse judgment.
- **Loss:** distinctions cease to be recoverable or behaviorally usable under the available measurement tests.
- **Recoverable packing candidate:** several independently manipulable distinctions remain in a constrained representation, with nonorthogonal coding and a reproducible pattern of interference and selective downstream use.

Measure tuning bandwidth, responses to opposite/nearby shapes, held out photo selectivity, single channel versus multichannel readouts, and behavior when properties co-occur. Fit readouts on discovery data and freeze them. A multivariate decoder alone does not establish superposition; ordinary composition, spatial separation, and use of different iterations are alternatives. Probe simultaneous information within one iteration rather than pooling temporally separated features into apparent coexistence.

Implement a small SAE/recovery analysis for the reference bottleneck when replicating that target. Include uncompressed positive controls, width/sparsity sweeps with equal selection budgets, reconstruction metrics, and behavioral degradation when reconstructed activations are spliced back in. Compare drawn stimulus tuning and natural image responses, not attractive visualizations alone. Do not infer absence of a distinction from one failed SAE, especially when reconstruction damages model behavior. A regularized linear and a small nonlinear readout provide complementary recovery checks, with their own held out evaluation.

Validate the diagnostic pipeline on synthetic examples of genuine information loss, coarse generalization, and a known packed sparse code. Definitions and thresholds for a packing candidate must be locked before examining narrow confirmation results. Report evidence for alternatives rather than forcing a binary superposition verdict.

### Iteration comparison

For each width/sharing/seed condition, apply the same pole, tuning, recoverability, and circuit measurements at every iteration. Estimate width by iteration and sharing by width by iteration interactions. Analyze whether later iterations recover useful distinctions, reuse the same coarse channel, or move information into other tokens/routes. Compare native readouts with frozen cross iteration readouts; a format change is not evidence of loss.

**Required outputs:** full and narrow trained checkpoints with learning curves, a verified cut graph, per iteration tuning/recovery results, SAE fidelity controls where applicable, and controlled interference tests. Frozen model masking cannot satisfy this deliverable.

## Experiment 3 Memorization and route reuse

### Controlled adaptation trials

Start from competent clean classifiers and make independent adaptation copies for prespecified image/label pairs. Use an existing output vocabulary and retain the same classifier head shape. Record whether each image appeared in original training, was correctly classified before adaptation, and met a prospectively defined eligibility rule. Do not select pairs because they produce a compelling circuit afterward.

Select LoopVL trial images from the audited training-image manifest, keeping provenance level and duplicate-family membership. Existing instance selectivity on a training-source photograph is not proof of memorization. In a native LoopVL adaptation trial, preserve the image and question and teach a deliberately wrong existing answer string; score assigned-versus-original conditional answer margins with the existing vocabulary. Validate gradient-enabled loading, trainable parameter groups, optimizer state, and a bounded update schedule before this becomes an executable stage. Keep the new forced association separate from any claim about what the original checkpoint memorized.

For the exact reference target, use the published image/label setting if its input and recipe can be recovered. For the controlled study, sample multiple independent pairs across source/target classes and semantic distances. Pair identities and adaptation variants stay together. Confirmation adaptation trials use new image identities, with candidate selection and analysis rules already frozen. Training on the designated trial image is the deliberate intervention, not an accidental interpretation split leak.

Compare wrong label adaptation with correct label adaptation and a matched sham/no update copy. Keep optimizer state initialization, updates, augmentations, and batch/replay treatment explicit. The main comparison trains the same parameter groups as the reference recipe when known. Add head only adaptation as a localization control; a frozen head variant can test whether the trunk can implement the change. Do not silently force route reuse by excluding the head from every condition.

Save dense early checkpoints, including every initial adaptation update. Record the entire learning trajectory and a fixed maximum trial length. Report failure to learn the assigned label rather than increasing updates until each selected trial succeeds. Include all trials in success denominators.

Measure the assigned versus original label margin, source image recognition, transformations of the target image, new images of both classes, and unrelated classes. Confirm the arbitrary association was actually acquired before describing its mechanism. Instance selectivity by itself does not establish memorization. Reversal, counterfactual relabeling, or a matched image omitted from adaptation can strengthen attribution to this training event.

### Locate the learned change

Measure activations and causal support for the two output routes before and after adaptation, using fixed ordinary exemplars and trial images. In conventional models, test parameter/buffer transplantation between original and adapted copies for declared early/middle/late block groups and complementary swaps. Include head only swaps and sham swaps. Preserve normalization state and report changes in general accuracy; one successful hybrid alone does not establish a unique memory location.

In tied models, a physical parameter update affects every invocation. A transplant of only the last iterations is not a literal block parameter transplant. Instead use invocation specific activation or block update patches between original/adapted runs to localize where the learned change is expressed. Record donor state mismatch, use self and unrelated donor controls, and compare necessity and restoration. Any call specific parameter delta hybrid must be labeled an intervention that temporarily unties execution, not the native architecture.

### Iteration comparison

Produce an adaptation update by inference iteration grid. At each point measure pole changes, controlled selectivity, route overlap with existing class circuitry, and invocation specific mediation of the final wrong label margin. Quantify early versus late expression with matched horizons and final outputs. Do not confuse where an effect becomes readable with where shared weights store an update.

A random label subset or noise image training study is a secondary stress test after single image adaptation works. It requires its own convergence evidence and clean/random label controls, not an assumption that visual oddness proves arbitrary memory.

**Required outputs:** trial manifests, original/adapted checkpoint provenance, all learning trajectories and failures, collateral behavior, parameter swap results for untied models, and per invocation activation/path interventions for tied models.

## Experiment 4 Find and validate circuits

### Build candidate graphs

Start with a few declared decisions and actual training-image photographs selected on discovery data. In reference models, audit the published example families where assets permit it. In trained models and LoopVL, use examples for which task competence and the target margin are established. Native LoopVL targets are prespecified answer-string or token margins under fixed questions, rather than assuming an image-classifier head. Donor images, counterexamples, and region annotations come from the manifest; do not reuse synthetic object-role IDs as natural-image ground truth.

Use nodes keyed by checkpoint, physical module/layer/channel, iteration, and spatial/token position. Unroll recurrence into execution order: a shared physical channel can appear at several distinct causal nodes. Represent both positive and negative contributions, residual paths, attention mediated transfer, normalization, and classifier/readout boundaries. The image encoder in LoopVL runs outside the recurrent L/H schedule; its channels do not acquire multiple L invocations merely because their outputs are reused.

Document the attribution rule, baseline, spatial aggregation, and pruning threshold. Weight magnitude, activation correlation, and attention weights can suggest candidates but are not causal edges. Test the attribution implementation on a small graph with known effects. If a linearized or fixed attention approximation is used, quantify its approximation error and keep full model interventions as the validation target.

Pair graph icons with pole dossiers, natural examples, response maps, and reproducible feature visualization objectives. Keep unaccounted contributions visible rather than presenting a selected path as the entire computation.

### Validate selected edges and paths

Freeze the circuit on discovery data. On held out images, test targeted edge/node suppression, clean/corrupt activation replacement, and restoration of selected paths in a corrupted run. Compare matched random circuits of the same size, layer/iteration distribution, and perturbation magnitude. Match token/spatial coverage so an entire channel map is not compared with a single unrelated position.

Report necessity, restoration/sufficiency in the stated background, prediction of damage from circuit usage, and residual unexplained effects. Null deletion can reflect redundancy; a pretty graph or a damaging large ablation alone does not establish the proposed mechanism. Check that self/zero edits reproduce unmodified outputs and that edge interventions affect the intended computational contribution rather than every use of a shared module.

### Iteration comparison

Compare iteration specific graphs and graphs aggregated by physical identity, retaining both views. Measure changed edge usage and signed contribution profiles at matched sparsity, along with the same held out causal endpoints. Compare intervention at one invocation with interventions at all reuses; do not assume their effects add linearly.

Measure immediate/short horizon effects after a fixed number of subsequent operations and final output effects. Record the actual operations inside each horizon. Different remaining computation can repair early perturbations, and equal operation counts do not guarantee identical downstream mechanisms. Use validated frozen semantic readouts when interpreting intermediate states; direct state distances measure perturbation persistence only.

**Required outputs:** machine readable signed graphs with provenance, selected circuit dossiers, held out intervention rows, matched random controls, restoration curves, and iteration specific causal profiles.

## Native LoopVL execution and iteration analysis

Keep the official mixed precision and native execution:

```text
L1 L2 L3 H1 L4 L5 L6 H2
```

Use the same physical L channel across all six invocations. Retain L1/L4 as the primary matched outer cycle comparison, L2/L5 and L3/L6 as secondary comparisons, adjacent L comparisons within cycles, and L3/L4 across H1. These characterize different transitions; a boundary contrast alone cannot identify an H specific cause.

Hold the training-source image, question, preprocessing, visual grid, and token role fixed across comparisons. Repeat images under independently varied questions, with common-prompt, paraphrase, and original-training-instruction conditions kept separate. Keep whole-image visual summaries, validated region summaries, individual patches, and the last prompt position separate. Source datasets may not supply object masks; annotate or validate regions before making causal object claims, and support variable grids instead of reusing the shape generator's masks.

Apply Experiment 1 directly after the real-image manifest and capture checks, followed by Experiment 4 when the invocation-local circuit executor passes parity/locality checks. These are primary LoopVL work, not a final transfer after unrelated classifiers. Reuse encoder/probe machinery only with descriptors and controls validated on these images. A change in description or causal effect is not by itself superposition.

Frozen LoopVL can characterize iteration-specific capacity signatures and instance selectivity on the selected training-image sources. Experiments 2 and 3 require the separately audited training/adaptation paths described above; observing the frozen checkpoint cannot replace them. Verified exposure strengthens provenance but still does not establish memorization; unresolved membership must remain visible in every original-memory claim. Controlled trained models can test the corresponding mechanisms on the same image sources, with their different architecture and training history explicit.

## Statistical and interpretation rules

Predeclare one primary family of comparisons for each of the four experiments and a multiplicity policy across them. Existing LoopVL P1/P2/P3 definitions remain valid for the original study; do not silently repurpose its three endpoint allocation for this four experiment protocol.

- Sample channels independently of interesting results, stratified by physical site. Keep both poles and all invocations of a channel together for population inference; show inactive/unexplained fractions.
- Bootstrap image families or adaptation trials, preserving their views and repeated measurements. Report training seed variability separately; a single seed architecture comparison is exploratory. Architecture effects use the trained model/seed as the replication unit, not thousands of independent neuron replicates. Use layer balanced estimates and sampling weights where appropriate.
- Freeze source descriptions/readouts on discovery data, gains and offsets on calibration data, and evaluate new image families without refitting. Use matched data and explanation complexity for shared versus iteration conditioned models.
- Keep width effects, iteration effects, and their interactions distinct. Within a trained tied checkpoint, physical channel identity is exact. Across seeds, widths, and retraining, neuron identities require explicit alignment and alignment uncertainty.
- A limited stimulus inventory cannot establish complete neuron meaning. A failed decoder or SAE does not establish absent information. More labeled concepts than channels, or poor cross iteration decoding, does not establish simultaneous packing.
- Define practical equivalence margins before confirmation. Use confidence bounds, not nonsignificance, to support stability. Report negative, mixed, and unresolved results without selecting the most dramatic image or seed.

## Implementation order and acceptance gates

| Stage | Work | Acceptance evidence |
| --- | --- | --- |
| 1 Training-image provenance | Resolve LoopVL's reported sources, revisions, retained records, access, duplicate families, and bounded staging | Image manifest with evidence and explicit membership levels; source-image hashes and declared sample |
| 2 Native LoopVL photo pilot | 64 original image families, existing 48 channels, native preprocessing, all six invocations | New-site reconstruction/parity, variable-grid validity, complete spatial captures, real-photo galleries, measured cost |
| 3 Poles and circuit foundation | Expand training-image retrieval pool, freeze interpretation splits/explanations, add invocation-local edits | Held-out interpretation predictions and spatial maps; tested intervention locality, no reliance on shape semantics |
| 4 Training and compression | Same-image-source full/narrow variants, matched tied/untied graphs, branch control and forced cut | Correct resume/gradients, verified cut, adequate learning, width-by-iteration results and recovery controls |
| 5 Memorization | Wrong/correct/sham adaptation copies on audited training-source photographs | Exposure provenance, learned association denominators, update-by-iteration trajectories, selective causal tests |
| 6 Circuit confirmation | Freeze candidate signed paths and natural-image donor controls | Held-out necessity/restoration, matched random circuits, and per-invocation effects |
| 7 Reference comparison and synthesis | Separate paper-exact reproductions from LoopVL and controlled-model findings | Explicit model/data scope, saved measurements, uncertainty, causal limits, and model-free recomputation |

Experiments can share trained clean checkpoints. Circuit tooling supports the memorization analysis; implement it early enough to collect those trials without ad hoc hooks. Do not call the study complete while requested retraining or memorization remains a plan.

The immediate additions are a training-image provenance/source adapter in `data/images.py`, a separate `loopvl-training-images` configuration, variable-grid capture validation, and full-photo response overlays. Reuse the current collector's physical-identity and reconstruction logic while removing its dependence on synthetic family/version fields and the reducer's required `base`/`color_edit` pairs. Save ordered layer/channel axes as lists alongside arrays rather than inferring them from sorted JSON map keys. Then add `vision_models.py`, `train_vision.py`, `feature_viz.py`, `capacity.py`, `memorization.py`, and `circuits.py` as needed. These names are proposed interfaces. Use separate run IDs and manifests for each model, data protocol, seed, width, and adaptation trial.

Tests should recover known poles and nonlinear gating, reject an unsigned negative pole, detect residual/anchor bypasses, preserve tied identity over invocations, reject image family leakage and changed resume state, localize edits, and recover known causal edges in a toy graph. Training resume must preserve optimizer, schedule, random state, and data order. Loaded model and real intervention checks remain scheduled gates, not conclusions from synthetic fixtures.

## Resources and execution boundaries

Use the current aggregate limit of 30 GPU hours and 20 GiB of research outputs unless the user changes it. These limits cover the existing study and this extension together, including training, adaptation, SAE fitting, visualization, validation, failed jobs, and retries. Count newly trained checkpoints and circuit artifacts in the output budget; do not hide training products in an excluded checkpoint directory. Keep bounded samples and compact checkpoint/delta histories rather than all full activation tensors.

Before scaling, measure a short training run, convergence behavior, adaptation learning, feature optimization, graph extraction/intervention costs, memory, and storage growth on the actual backend. Produce a cost table for the reference reproductions, four cell tied/untied width matrix with seeds, and frozen LoopVL analysis. Include contingency and already reserved allocations. The original submitter only knows existing stages; extend its stage classification and shared accounting before new GPU stages run.

Do not download an entire 85M/22M corpus to begin this study. Inspect metadata and archive layout, then stage only the pinned bounded image/shard selection needed for the declared sample, with per-file checksums and measured temporary storage. Large index scans, downloads, deduplication, and preprocessing belong in reviewed scheduled jobs. Keep training-image bytes and raw captures on the cluster; publish compact provenance, summaries, and permitted example assets. Natural-image token counts may exceed those of the shape pilot, so its timing is a starting measurement rather than a throughput guarantee.

Do not assert that full ImageNet or native LoopVL retraining and all confirmation seeds fit the inherited budget. If they do not, finish the methodological pilot and publish a measured remaining resource requirement. Mark full reference reproduction or multi-seed confirmation pending; do not replace the requested training experiments with frozen hooks and declare success. A smaller model trained on a bounded subset of the documented LoopVL image sources can establish a labeled methodological pilot; budget constraints do not authorize substituting an unrelated image corpus.

Follow the cluster runbook and site policy for locks, scheduled heavy work, batch script review, and array approval. This revision changes the research plan; it does not submit jobs. Preserve frozen snapshots, raw results, and the original study's gates. Use a new protocol schema with image/training/circuit controls instead of falsely setting irrelevant old gates to passed.

## Final package and next agent instruction

Deliver four linked result sections, leading with LoopVL training-image measurements and separating native LoopVL, same-source controlled training, and reference reproductions. Include full-photo signed pole galleries and prediction matrices; width/accuracy/tuning/recovery plots; adaptation-update-by-iteration heatmaps with causal tests; and unrolled circuit diagrams with held-out restoration and random controls. Provide all denominators, image membership evidence, seeds, exclusions, failures, effect estimates, uncertainty, resource use, source/asset hashes, and exact recomputation commands.

The claim ledger must distinguish reproduced reference results, methodological replications, controlled recurrence findings, and frozen LoopVL observations. State whether observed pressure produced coarser judgments, lost distinctions, a supported packing candidate, or insufficient evidence. State whether a learned arbitrary label reused existing routes, changed the head, or remained unresolved.

The next agent should first audit the pinned LoopVL checkpoint's actual image-training sources and retained-record evidence, freeze the training-image manifest, and implement the 64-original-image LoopVL pilot. Do not start another shape/CIFAR pilot or defer these images until a new classifier is trained. Continue to natural-image pole retrieval and invocation-specific circuit work, then the separately costed compression and memorization training paths. Keep interpretation holdouts distinct from original training exposure, and training checkpoints distinct from inference iterations. Carry the iteration axis through every experiment and report missing provenance, access, implementation, or resources without inventing membership or results.
