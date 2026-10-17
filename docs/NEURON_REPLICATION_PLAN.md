# Image neuron replication and recurrent computation study

Revised October 17, 2026 for an implementation agent working with Max Liu.

Reproduce image model experiments on positive and negative neuron poles, single layer compression followed by retraining, memorization, and circuits. Then test how each phenomenon depends on the current iteration of a looped vision model. Use frozen LoopVL as a transfer study after validating the methods in image classifiers whose training and architecture we control.

This replaces the earlier Qwen first plan. Training small vision models, adapting individual image labels, and testing narrow representations are required parts of this study. They supersede the earlier exclusion of training for this extension. Preserve the original frozen LoopVL protocol and results under their existing identities. The [original scientific protocol](../CODEX_PLAN.md) supplies useful transfer statistics and controls; the new image experiments need their own model specific protocols and gates.

## Questions and study tracks

| Experiment | Conventional image model | Required recurrence analysis |
| --- | --- | --- |
| Poles | Find what drives each pole and whether it makes a coherent visual distinction | Compare each physical channel's poles, tuning, and downstream effects at every invocation |
| Compression | Physically narrow one layer and train the model again; test broadening, lost distinctions, and recoverable packed information | Test width by iteration interactions, with matched tied and untied execution graphs |
| Memorization | Teach individual images deliberately wrong existing labels and identify how the change is implemented | Locate when the new behavior appears and which invocations causally mediate it |
| Circuits | Find and validate signed paths into visual decisions | Unroll the recurrent graph and compare paths, edge usage, and causal effects across iterations |

Run three linked tracks, with distinct claims:

1. **Reference image reproduction.** Recover the authors' architecture, preprocessing, checkpoint or training recipe, and evaluation inputs where available. Reproduce the specified image experiments. A different backbone or dataset is a methodological replication and must be labeled accordingly.
2. **Controlled recurrent vision training.** Train matched tied and untied small vision Transformers, with recorded data and from scratch bottleneck variants. This track carries the architecture comparison, controlled memorization, and iteration interactions.
3. **Frozen LoopVL transfer.** Apply the validated pole and circuit measurements to visual token MLP channels under fixed images and questions. This checkpoint cannot establish what retraining under compression would learn, or whether a specific original training image was memorized.

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

The saved LoopVL diagnostic parity covers one physical layer/channel on 16 examples. It does not validate a new training pipeline, input optimization, all visual token sites, or circuit interventions. The current package has no general image classifier trainer or completed circuit discovery executor.

| Existing component | Reuse | Addition needed |
| --- | --- | --- |
| `capture.py`, `model_adapter.py`, `numerical.py` | Physical identity, invocation traces, reconstruction and locality checks | Image classifier adapters, gradient enabled capture, preactivation/branch observations, new model parity |
| `data/`, `data/alignment.py` | Deterministic scenes, paired edits, grouped splits and mask validation | Natural image manifests, image family deduplication, curve/texture stimuli and adaptation trial IDs |
| `encoding.py`, `probes.py`, `controls.py`, `statistics.py` | Frozen transfer fits, positive affine calibration, paired controls, grouped uncertainty | Visual tuning, capacity diagnostics, checkpoint by iteration and treatment interactions |
| `interventions.py`, `pipeline.py` | Product edits, self replacement, donor alignment primitives | Classifier and edge/path edits, adaptation experiments, short horizon collection |
| `protocol.py`, `shards.py`, `report.py`, `scripts/slurm/` | Immutable protocols, atomic outputs, accounting, recomputation | Training stages, model specific schemas, compact checkpoints, circuit and image reports |

Keep the original LoopVL defaults, family splits, and immutable snapshots intact. New training, adaptation, SAE, visualization, and analysis stages require explicit implementation and scheduled execution; their names below are proposed interfaces, not existing commands.

## Models and experimental design

### Conventional reference and practical pilot

Use the reference models for claims about the published experiments. Confirm whether the selected InceptionV1 variant actually matches the reference, including branch concatenation, activation placement, auxiliary classifiers, and any normalization changes. Audit the authors' signed ConvNeXt tensor instead of silently substituting a standard GELU activation.

For an inexpensive implementation pilot, use a small classifier and [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), whose fine/coarse labels support one check of semantic breadth. CIFAR-10 is an alternative engineering smoke dataset. These are not substitutes for an ImageNet reproduction. Pin the actual data recipe and state which results use which organism. A full three scale sweep is a secondary extension after the four primary experiments work.

### Controlled looped Transformer

Proposed pilot defaults are 32 by 32 inputs, patch size 4, hidden width 128, three cell invocations, and a small gated MLP. Select the smallest configuration that passes a declared validation competence screen; freeze the complete architecture before confirmation. These dimensions are engineering starting points, not reported paper settings.

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

Separate model training data from interpretation discovery, calibration, and confirmation images. Within training, reserve validation data for model selection. Keep all crops, augmentations, paired edits, and near duplicates of an image in one family. Record where an exact reference recipe instead uses training images for retrieval, and supplement it with a separately frozen evaluation set.

Use natural images for breadth and controlled stimuli for independent changes in shape, curvature/orientation, color, texture, position, and context. The existing 2D scenes can supply controlled tests but do not establish natural image interpretability. Fit descriptions, thresholds, feature decoders, alignments, and causal candidates without opening confirmation outcomes. Sampling tailored to classes or activation tails needs explicit weights for population summaries.

## Experiment 1 Positive and negative poles

### Define the measured neuron

For gated MLPs capture `gate_pre`, `gate_effective`, `value`, and the actual product immediately before the output projection. Audit the nonlinearity in every model. Assign gate/value sign regimes at actual spatial or token positions before any pooling.

For ReLU image models record preactivation and postactivation separately. A negative preactivation describes veto evidence before rectification; it is not a transmitted negative output. Zeroing an already rectified zero is not a test of an active negative contribution. For signed products, activation sign is still distinct from downstream causal sign because the write direction and reader matter.

Use the native zero and also report centered/normalized diagnostics separately. Do not manufacture a negative pole by subtracting a mean from an unsigned activation. A neuron may have no well supported negative tail; retain that outcome and its denominator.

### Build and test pole descriptions

Sample 32 channels per declared early, middle, and late site for the pilot. Use every channel at tiny bottlenecks. Preserve the existing representative LoopVL sample for population work; keep selected case studies separate.

For each channel, checkpoint, and iteration, save both signed tails and ordinary activation ranges, with source image IDs, response maps, the strongest location, full image context, and a receptive field or attribution crop. Maintain distinct image maximum/minimum and spatial mean summaries: extrema can select backgrounds or noise, and spatial pooling can cancel poles. Use one image family as the sampling unit, not thousands of independent patches from it.

Optimize images for each pole only through a validated differentiable path. Compare unconstrained optimization with a natural example anchored objective, using the same seed count, steps, input parameterization, transformation robustness, and regularization budget. Save all attempts. The anchored objective and its strength must be declared; agreement with its anchor is not independent evidence that the interpretation is right. Counterfactual natural images and held out stimuli remain the main test.

The current LoopVL adapter wraps inference in `no_grad` and capture detaches values. Do not assume it supports feature visualization or training. Add a separate audited gradient mode for small image models. For LoopVL, retrieval and controlled image edits can proceed first; pixel optimization requires differentiable preprocessing, output parity, gradient checks, and a measured memory budget.

Freeze a short explanation of the positive versus negative distinction. Test its predictions on new images, independently manipulated stimuli, and hard negatives matched on class, color, position, and background. Measure both discrimination and ordinary image false positives/negatives. Report class/fine-coarse concentration as a proxy alongside visual tuning and causal results; class entropy is not a definition of polysemanticity.

### Iteration comparison

Compare pooled explanations, separate iteration explanations, and frozen source iteration explanations transferred to target iterations. Match the total examples, activation queries, explanation length, and selection attempts across all conditional explanations combined. Use blinded condition labels where practical.

Reuse positive gain and offset calibration on independent calibration images; report polarity reversal separately. A calibrated change in threshold or scale is weaker evidence than a switch between independently manipulated properties. Produce source to target prediction/tuning matrices and paired edit selectivity profiles for both poles. Low variance, missing tails, or poor within iteration prediction yield insufficient evidence rather than semantic change.

**Required outputs:** signed pole dossiers and response maps, stimulus tuning curves, frozen explanations and held out predictions, per iteration transfer matrices, and a population report including inactive/unresolved channels.

## Experiment 2 Narrow one layer and retrain

### Reproduce actual compression

Implement the reference early and late layer alterations in the asset manifest. Change the relevant parameter shapes and downstream connections, and train each narrowed model from scratch. Include an otherwise identical full width model under the same training recipe. Posthoc channel masking, pruning without retraining, and training only the replacement branch answer separate questions.

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

Start with a few declared class decisions and representative images selected on discovery data. In reference models, audit the published example families where assets permit it. In trained models and LoopVL, use examples for which task competence and the target margin are established.

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

## Apply the measurements to frozen LoopVL

Keep the official mixed precision and native execution:

```text
L1 L2 L3 H1 L4 L5 L6 H2
```

Use the same physical L channel across all six invocations. Retain L1/L4 as the primary matched outer cycle comparison, L2/L5 and L3/L6 as secondary comparisons, adjacent L comparisons within cycles, and L3/L4 across H1. These characterize different transitions; a boundary contrast alone cannot identify an H specific cause.

Hold the image, question, preprocessing, visual grid, and token role fixed across comparisons. Repeat images under independently varied questions, with neutral/question paraphrase controls, to distinguish visual selectivity from task conditioning. Keep object pooled visual tokens, individual patches, and the last prompt position separate. Validate mask/grid alignment before causal object claims.

Apply Experiment 1 and Experiment 4 directly after their LoopVL adapters pass parity/locality checks. Use the original encoder/probe machinery for additional feature transfer estimates and counterfactual color edits. A change in description or causal effect is not by itself superposition.

LoopVL can characterize existing iteration specific capacity signatures and apparent instance selectivity, but cannot complete Experiment 2 or Experiment 3 without a separately scoped training/adaptation pipeline. Unknown original training membership prevents a verified original data memorization claim. Keep those outcomes on the controlled trained organisms instead of relabeling frozen observations as reproductions.

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
| 1 Asset and feasibility audit | Pin references, data, architecture, missing recipes, runtime and shared budget | Reference manifest and a measured scheduled pilot; exact versus methodological scope recorded |
| 2 Training and capture foundation | Small image classifier trainer, tied/untied graph, signed capture, differentiable preprocessing | Correct checkpoints/resume, deterministic identities, gradient checks, output reconstruction and self/zero parity |
| 3 Pole measurement | Real example retrieval, controlled stimuli, feature optimization, frozen descriptions | Held out pole/tuning results at every iteration with matched budgets |
| 4 Compression retraining | Full/narrow models from scratch, branch control and forced cut, recovery tests | Verified cut, adequate learning, real width by iteration results and recovery controls |
| 5 Memorization | Independent wrong/correct/sham adaptation copies | Learned association denominators, checkpoint by iteration trajectories, selective causal tests |
| 6 Circuits | Attribution candidates and invocation local edge/path executor | Held out necessity/restoration and matched random controls |
| 7 LoopVL transfer and synthesis | Validated image methods on native visual token channels; reproducible reports | Exact scope, saved measurements, uncertainty, causal limitations, and model free recomputation |

Experiments can share trained clean checkpoints. Circuit tooling supports the memorization analysis; implement it early enough to collect those trials without ad hoc hooks. Do not call the study complete while requested retraining or memorization remains a plan.

Suggested additions are `vision_models.py`, `train_vision.py`, `data/images.py`, `feature_viz.py`, `capacity.py`, `memorization.py`, and `circuits.py`, plus model specific schemas and report functions. Equivalent organization is acceptable; avoid rewriting the existing LoopVL path before the small pilot works. Add proposed configs for reference images, recurrent training, capacity, memorization, circuits, and frozen LoopVL transfer. Use separate run IDs and manifests for each model, seed, width, and adaptation trial.

Tests should recover known poles and nonlinear gating, reject an unsigned negative pole, detect residual/anchor bypasses, preserve tied identity over invocations, reject image family leakage and changed resume state, localize edits, and recover known causal edges in a toy graph. Training resume must preserve optimizer, schedule, random state, and data order. Loaded model and real intervention checks remain scheduled gates, not conclusions from synthetic fixtures.

## Resources and execution boundaries

Use the current aggregate limit of 30 GPU hours and 20 GiB of research outputs unless the user changes it. These limits cover the existing study and this extension together, including training, adaptation, SAE fitting, visualization, validation, failed jobs, and retries. Count newly trained checkpoints and circuit artifacts in the output budget; do not hide training products in an excluded checkpoint directory. Keep bounded samples and compact checkpoint/delta histories rather than all full activation tensors.

Before scaling, measure a short training run, convergence behavior, adaptation learning, feature optimization, graph extraction/intervention costs, memory, and storage growth on the actual backend. Produce a cost table for the reference reproductions, four cell tied/untied width matrix with seeds, and frozen LoopVL analysis. Include contingency and already reserved allocations. The original submitter only knows existing stages; extend its stage classification and shared accounting before new GPU stages run.

Do not assert that full ImageNet retraining and all confirmation seeds fit the inherited budget. If they do not, finish the methodological pilot and publish a measured remaining resource requirement. Mark full reference reproduction or multi seed confirmation pending; do not replace the requested training experiments with frozen hooks and declare success. A smaller dataset/organism can establish a clearly labeled pilot result while preserving the original target.

Follow the cluster runbook and site policy for locks, scheduled heavy work, batch script review, and array approval. This revision changes the research plan; it does not submit jobs. Preserve frozen snapshots, raw results, and the original study's gates. Use a new protocol schema with image/training/circuit controls instead of falsely setting irrelevant old gates to passed.

## Final package and next agent instruction

Deliver four linked result sections, each containing the conventional image result and its iteration analysis. Include signed pole galleries and prediction matrices; width/accuracy/tuning/recovery plots; adaptation update by iteration heatmaps with causal tests; and unrolled circuit diagrams with held out restoration and random controls. Provide all denominators, seeds, exclusions, failures, effect estimates, uncertainty, resource use, source/asset hashes, and exact recomputation commands.

The claim ledger must distinguish reproduced reference results, methodological replications, controlled recurrence findings, and frozen LoopVL observations. State whether observed pressure produced coarser judgments, lost distinctions, a supported packing candidate, or insufficient evidence. State whether a learned arbitrary label reused existing routes, changed the head, or remained unresolved.

The next agent should read current progress and this plan, audit reference assets and resource feasibility, and build the smallest trained image experiment that exercises all four measurements. Continue through the declared gates on available authorized resources. Keep training checkpoints separate from inference iterations and carry the iteration axis through every experiment. Report exact missing ingredients or resource requirements without inventing results or silently restoring the superseded Qwen scope.
