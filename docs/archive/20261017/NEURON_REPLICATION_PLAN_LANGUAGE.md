> Superseded language model plan. The user redirected the study to image model replication on October 17, 2026. Use the [current image study plan](../../NEURON_REPLICATION_PLAN.md).

# Native neuron replication and recurrent reuse study

Prepared October 17, 2026 for an implementation agent working with Max Liu.

Implement an independent audit of native MLP neuron explanations, then test whether those explanations transfer across repeated uses of the same physical LoopVL neurons. The central question is whether recurrent computation preserves semantic selectivity, changes downstream use, or requires explanations conditioned on the current pass.

This extends [CODEX_PLAN.md](../../../CODEX_PLAN.md). Preserve its numerical checks, independent splits, population sampling, causal controls, and resource accounting. Use separate run identities for new experiments; do not change the meaning of an existing frozen study.

## Research scope and evidence

[Neural Networks Are Already Clean](https://already-clean.goodfire.com/) presents a working hypothesis that native neurons are more interpretable than earlier tools suggested. Relevant inspection changes include examining both activation poles, SwiGLU gate and value sign regimes, longer context, and circuit evidence. Its preliminary cleanliness estimates depend on evaluation choices, and its technical language model companion was described as forthcoming when reviewed.

Treat the experiments below as proposed independent tests. A Qwen3-4B study is a methodological replication on another checkpoint, not an exact reproduction of the essay's Qwen3-32B results. An exact reproduction requires the reference checkpoint, neuron identities, data, and evaluation procedure to be sufficiently specified. Record any unavailable ingredients rather than inventing them.

Deliver the core study with Qwen3-4B and LoopVL. Keep Huginn, training checkpoint analysis, controlled tied versus untied training, SAE comparisons, and additional model families outside the required scope.

## Read first and establish the current state

The editable checkout is `source/` inside `/oscar/home/mliu171/reuseinterp-loopvl`. Paths below are relative to that checkout. Nearby `snapshot/` and `preparation/` directories are historical artifacts, not alternative editable checkouts.

Read these files before implementation:

1. Applicable `AGENTS.md` instructions, including `/oscar/AGENTS.md` on this machine.
2. [README.md](../../../README.md), the newest entries in [PROGRESS.md](../../../PROGRESS.md), and [CLUSTER_EXECUTION.md](../../CLUSTER_EXECUTION.md).
3. [CODEX_PLAN.md](../../../CODEX_PLAN.md), [architecture_audit.md](../../architecture_audit.md), and [CLUSTER_RUNBOOK.md](../../CLUSTER_RUNBOOK.md).
4. The implementation map below, current configs, and existing tests for the components being changed.

Use [PROGRESS.md](../../../PROGRESS.md) for current validation and accounting status. The existing GPU diagnostic parity artifact passed; it covers a limited physical site rather than the full planned experiment. Begin at the earliest unmet gate and use the current runbook's environment and editable submitter. Additional instrumentation and adapters still need their own loaded model validation.

Respect site rules for locks, narrow filesystem access, scheduled heavy work, and scheduler queries. Show batch scripts before submitting compute and obtain the explicit approval required for arrays. Follow higher priority session instructions where they conflict with local instructions. Do not launch automatic job polling or run model inference on the login node.

### Existing implementation

| Component | Reuse | Required addition or unresolved boundary |
| --- | --- | --- |
| `src/reuseinterp/model_adapter.py` | Pinned official LoopVL loading and architecture checks | Add a Qwen adapter with model specific validation; retain LoopVL precision and schedule checks |
| `capture.py` | Physical channel sampling, product capture, invocation counters, reconstruction checks | Capture selected gate and value channels at real token positions; support a conventional model with one invocation |
| `pipeline.py` and `data/alignment.py` | Scene capture, object pooling, grid alignment, capability controls | Connect the new audit artifacts and gate diagnostics without treating pooled values as intervention sites |
| `encoding.py`, `probes.py`, `statistics.py` | Frozen encoders and probes, positive affine calibration, grouped uncertainty | Add pooled versus pass conditioned comparisons, inspection condition metrics, and their uncertainty |
| `controls.py` and `discovery.py` | Existing transfer analysis and standalone control utilities | Integrate the required matched role/task, reweighting, permutation, and paired edit controls into real analysis |
| `interventions.py` | Invocation local additive doses, donor replacement, self replacement | Complete candidate specific donor collection and the short horizon causal executor |
| `protocol.py`, `shards.py`, `report.py` | Immutable locks, atomic shards, integrity checks, reproducible reports | Version new schemas and freeze the new endpoints without weakening existing checks |
| `scripts/slurm/` | Submission wrappers, snapshots, accounting, resume behavior | Add scheduled stages for the new adapter and maintain one aggregate resource ledger |

The current capture hooks and adapter encode LoopVL specific assumptions. The object package assembler supports the object pooled stratum; it does not establish a complete last prompt or natural text analysis path. `discover-causal` is not a complete real experiment runner. Audit these boundaries before reusing an interface.

## Scientific definitions

For a channel at physical layer `l`, neuron `i`, and invocation `r`, inspect the actual computation:

```text
gate_effective = act_fn(gate_pre)
product = gate_effective * value
residual_write = product * down_projection_column
```

Audit `act_fn` and branch names in each loaded model; call it SwiGLU only when appropriate. The primary neuron is the product channel immediately before the down projection. A residual coordinate is a different object.

LoopVL shares the channel's weights and write direction across invocations. Test whether an external semantic description transfers across its input state distributions, and whether its output has similar downstream effects. Changes in these quantities do not mean the neuron's mathematical function changed.

| Account | Evidence to seek | Limitation |
| --- | --- | --- |
| Stable semantic reuse | A frozen description predicts held out activations across passes after positive gain and offset calibration | Restricted to tested properties and input distributions |
| Pass conditioned interpretability | Separate pass explanations predict new examples better than a shared explanation | Extra explanation flexibility must be controlled |
| Changing downstream use | Selectivity transfers, while selective causal effect profiles change | Different final effects alone can reflect recovery time or downstream sensitivity |
| Candidate temporal multiplexing | Independently manipulated properties exchange selectivity across passes and survive nuisance controls | Requires more evidence than different top examples or failed transfer |
| Unresolved behavior | Available explanations fail within passes or data are inadequate | Does not establish superposition or intrinsic illegibility |

A stable operational explanation may cover changing object associations. Evaluate descriptions of computations as well as object or topic labels, but require an operational description to predict a separately defined observable. Calling a neuron an evidence updater is insufficient without specifying what evidence changes and how it is measured.

## Stage 1 Establish model and resource readiness

Use [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for the first text audit and the existing pinned LoopVL checkpoint for recurrence. Pin exact model revisions, tokenizer revisions, runtime versions, input hashes, and precision before collecting results. Do not assume the LoopVL environment supports Qwen unchanged; use an isolated environment if necessary.

Implement the smallest shared adapter interface that supports audited MLP site enumeration, preparation, unmodified forwarding, scoring, and capture metadata. Preserve the existing LoopVL API through a compatibility path. A conventional layer has invocation `1`; the shared abstraction must retain physical module identity and effective execution order.

For each adapter, require capture only, self replacement, and zero dose to reproduce unmodified outputs within tolerances measured and frozen on that backend. Reconstruct gate times value and product times projection, including bias. Verify padding, attention masks, token roles, changed coordinates, and hook cleanup. A new backend or adapter requires its own parity evidence.

For LoopVL, retain the official mixed precision and native schedule:

```text
L1 L2 L3 H1 L4 L5 L6 H2
```

Both physical stacks have 16 layers in the pinned implementation. L and H are different modules. Keep the main study on L channels and the first answer prediction, before an answer token enters the prompt.

Preserve the existing aggregate limits of 30 GPU hours and 20 GiB of research outputs unless the user changes them. New run IDs do not create a new allowance. Account for consumed hours, active reservations, retries, and both models. Audit and extend the ledger's supported run identity handling before any new model submission; do not create an independent ledger that silently duplicates the budget.

Measure loading, forwarding, capture, explanation evaluation, causal collection, peak memory, and output growth before scaling. Also record explainer inference costs and call counts. Reduce optional comparisons and selected case count before reducing numerical controls or opening held out data. Never invent a throughput estimate or bypass a failed gate.

**Exit artifacts:** model and runtime manifests, loaded architecture audits, parity reports, a measured pilot cost projection, and a revised allocation of the remaining shared budget.

## Stage 2 Replicate the inspection improvements on Qwen

### Data and representative sample

Start with an engineering pilot of 96 physical neurons: sample 32 uniformly within each of three declared depth bands, with a fixed seed and recorded inclusion probabilities. Adjust only before scientific confirmation if the measured budget requires it. Keep all sampled units in the denominator, including inactive, poorly explained, and failed units.

Use a versioned broad text corpus containing prose, dialogue, code, and technical material. Select and pin the concrete dataset, revision, source mixture, and license in the data manifest. Deduplicate and split by document family before constructing windows; related documents and overlapping windows must not cross splits. Use discovery/development data only for the initial pilot, capped initially at 100,000 tokens. Set the final corpus sizes from measured cost and precision before confirmation. This pilot does not reproduce a multi billion token search.

Provide separate fit/discovery, development, calibration, and confirmation document groups. Record the exact context actually available to the model at each position. A full context inspection condition can reveal more of that prefix to the explainer, but cannot reveal text the model never saw. Later tokens may be labeled as outcomes for evaluating a prediction; they are not causal input evidence for the activation.

### Capture and inspection conditions

Save selected channels' `gate_pre`, `gate_effective`, `value`, and `product` at real token positions. Assign sign regimes before any pooling, with a separate near zero category defined from discovery numerical tolerances. Preserve the entire channel's observations; do not discard a difficult regime to improve an explanation score.

For each sampled neuron, prepare three independent inspection conditions:

1. Positive product top examples with a fixed local context window.
2. Both product poles, branch diagnostics, and examples separated by gate/value sign regime.
3. Condition 2 plus the full available prefix and a fixed, explicitly bounded amount of upstream/downstream evidence.

Use the same explanation model/version, prompt structure, output length limit, total example budget, tool query budget, and number of attempts across conditions. Report both the information revealed and the cost of each condition. Run conditions independently so a richer investigation cannot leak into the simpler one. Grade anonymized explanations without condition or checkpoint labels where practical. Log human interventions and explanation revisions.

Stream top and bottom examples, representative ordinary activations, and a bounded reservoir across activation ranges. Estimate thresholds on discovery data and reuse them for evaluation. Retain sampling probabilities when estimating prevalence from stratified examples. Store selected channels and compact records rather than all model activations.

### Frozen evaluation

Freeze explanations and their evaluation rules before reading confirmation activations. Evaluate:

- Prediction of activations or activation categories on ordinary held out documents, including false positives and false negatives.
- Explanation derived positive examples and near misses constructed without activation feedback, with fixed trial counts and matched lexical/context controls.
- Performance separately for both poles, sign regimes, depth bands, and activation ranges, alongside the combined channel.
- Explanation length, number of clauses or exceptions, investigation cost, and evaluator agreement on a blinded subset.

For a numeric activation predictor, fit any score mapping on discovery/calibration data only. Specify how free text explanations produce predictions before evaluation. Include token identity, position, and frequency baselines where applicable. Use an independent prediction agent or an explicit rule implementation; do not grade solely by asking whether an explanation sounds plausible.

Treat an explanation derived activation maximum as a secondary test of one response region. It does not establish coverage of the whole neuron. Report ordinary held out prediction and hard negative performance alongside it.

Score all conditions on the same frozen ordinary confirmation events; keep condition specific generated probes as a separate endpoint. Compare conditions within the same sampled neurons. Bootstrap independent document groups for data uncertainty and report neuron sampling uncertainty separately. Cluster both poles and all regimes of a physical neuron together for population estimates. Freeze the primary contrast and multiplicity policy before confirmation; use simultaneous intervals or label additional contrasts exploratory.

**Exit artifacts:** corpus and sample manifests, bounded activation indexes, frozen explanation records, blinded evaluation records, per neuron metrics, and a paired comparison of inspection conditions with honest denominators. A cleanliness percentage may be a secondary rubric result, not the sole endpoint.

## Stage 3 Test explanation transfer across LoopVL passes

### Use the existing controlled data first

Retain the scene family split assignments: 32 smoke, 128 pilot, 400 fit, 100 development, 100 calibration, 300 confirmation, and 100 transfer families. Both image versions, every question, and all descendants of a family remain together. The data are deterministic 2D scenes, not CLEVR.

Require numerical parity, actual grid/mask alignment, measured timing, and the capability controls in CODEX_PLAN.md before the corresponding collection stages. Preserve its planned competence screen and restricted scope outcome if fewer task families qualify. Do not change the renderer or tune prompts using confirmation results.

Use the existing representative sample of 32 channels per physical L layer, or 512 channels if loaded dimensions match. Capture all six L invocations. Keep object pooled activations and last prompt activations in separate analyses; the latter needs its own descriptor specification and supported assembler. Gate sign diagnostics must be computed on actual tokens before pooling, because the product of pooled branches is not the pooled product.

The controlled descriptors establish selectivity only for the variables they cover. Add a separately split broad visual discovery set if needed for explaining selected neurons. Record its sources and model competence. Do not generalize a controlled scene success rate to model wide interpretability, or directly compare it with the Qwen text rate.

### Compare shared and pass specific explanations

Implement and evaluate three models with the same descriptor whitelist and regularization search policy:

1. A shared semantic predictor fitted across passes, with only pass specific positive gain and offset fitted on calibration data.
2. Separate predictors fitted within each pass and tested on new families at that pass.
3. Frozen source pass predictors transferred to each target pass, allowing only positive gain and offset calibration.

Use the original symmetric L1/L4 transfer gap as the primary quantitative endpoint:

```text
D_i = 0.5 * (
    NMSE(calibrated L1 predictor on L4) - NMSE(within L4 predictor)
  + NMSE(calibrated L4 predictor on L1) - NMSE(within L1 predictor)
)
```

Estimate normalization variances on calibration families. Retain the original development eligibility rule and proposed 0.05 NMSE portability margin unless pilot evidence motivates a documented change before the lock. Report the full sample and the eligible subset, raw errors, held out R squared, low variance exclusions, and layer balanced summaries. Failure to reject a difference does not establish portability.

Fit the shared model with balanced pass contributions so six correlated observations do not give it an unfair data budget. Compare shared and separate explanations at matched training sizes and report their different parameter counts. For natural language case studies, match total examples, activation observations, queries, explanation length, and attempts across all pass specific explanations combined, and include a complexity matched shared explanation. An equal budget per pass analysis can be a secondary sensitivity check. Use anonymous pass labels where chronological information is unnecessary. A separate explanation for every pass has more flexibility by construction.

Preserve L1/L4 as primary. Record these secondary contrasts before confirmation:

- L2/L5 and L3/L6 match positions across outer cycles.
- L1/L2, L2/L3, L4/L5, and L5/L6 characterize adjacent L updates within a cycle.
- L3/L4 crosses H1 and characterizes the hierarchy boundary.

The boundary comparison can localize an association with the H update, but does not alone identify its causal contribution.

### Required controls and uncertainty

Integrate matched object role/task analyses, discovery fitted clipped reweighting, paired color edit differences, question/position baselines where supported, and coherent family label permutations. Match examples and real token roles across passes. Test whether gain calibration explains the raw transfer deficit.

Bootstrap scene families, preserving every object, question, image version, and invocation of a family in each draw. Report neuron sampling uncertainty separately. Do not count six invocations as six independent neurons. Lock the family of primary tests and multiplicity adjustment; keep additional feature searches exploratory.

Reserve candidate temporal multiplexing for a reproducible switch in selectivity to independently varied properties, after nuisance and calibration controls. Report a polarity reversal on the same property separately; it is weaker evidence than a switch between independently manipulated properties. Existing paired color edits support color selectivity conclusions. A color versus relation or shape switch requires additional validated interventions or a factorial discovery dataset; create new families and a separate split manifest rather than modifying held out scenes.

**Exit artifacts:** shared and per pass fits, the full source to target matrix, paired edit selectivity profiles, all control outputs, uncertainty estimates, and frozen candidate descriptions for causal follow up.

## Stage 4 Validate causal use on selected neurons

Choose at most eight candidates using fit/development evidence, with explicit quotas for stable and changing selectivity when both exist. Freeze their identities, selection rationale, token positions, donor rules, doses, task groups, family subset, horizons, and exclusion rules before confirmation. Do not replace an uninteresting candidate after seeing its held out effects.

Complete the candidate specific collection executor. Act on one actual product channel at one invocation and specified real tokens. Use same pass aligned donors from paired scenes with the same geometry and question but one semantic edit, and discovery scaled small positive/negative doses. Keep donor values in the native observed range. Compare both native scale and equal write magnitude perturbations across passes. Report requested and realized perturbation magnitudes, baseline activation distributions, and any clipping. A pooled object vector is never a writable model site.

Collect self replacement, zero dose, an equal size random channel control, an activation or perturbation magnitude matched control, and feature irrelevant questions. Discovery defines control selection and matching. Verify that the intervention fires exactly once at the intended site. A zero ablation, if used as a secondary condition, must be specified and validated as a real replacement operation.

Measure two endpoints:

1. Final answer effects, using the original oriented full conditional answer score contrasts and task specific effect profiles. Cases with no answer change need a separate specificity endpoint; do not invent a nonexistent base versus edit answer margin.
2. A prespecified short horizon state effect after the same number of subsequent block applications, with an appropriate frozen readout or direct state difference and matched controls. Record the exact L/H operation sequence in every horizon: equal duration does not guarantee equal downstream computation. If horizon endpoints land on different physical sites, declare and calibrate those readouts separately; do not equate unaligned raw coordinates. A direct state difference measures perturbation persistence rather than behavioral usefulness. Mark a horizon unavailable if the remaining schedule cannot support it.

Compare causal profiles allowing a nonnegative global gain fitted only on calibration families, as in CODEX_PLAN.md. A difference in overall effect size is weaker than a change in selective effects across tasks. Earlier interventions have more opportunities for recovery; final effects alone cannot separate recovery from changing use. A null single channel effect can reflect redundancy and does not establish that the represented property is unused.

The original plan proposes starting with 128 independently grouped causal confirmation families, balanced across supported tasks. Select the exact subset and any resource driven reduction before outcomes are opened. Apply its simultaneous interval procedure and report inconclusive cases. Optional interventions at every recurrence or different total loop budgets require additional validation and budget; they are not part of the core study.

**Exit artifacts:** complete clean/control/intervention rows, alignment and locality evidence, short horizon measurements, final effect profiles, simultaneous intervals, and recomputation scripts.

## Implementation tasks and proposed files

Use these as bounded tasks; equivalent names are acceptable if they fit the existing package. Proposed commands are implementation targets, not currently available CLI features.

| Task | Suggested files | Acceptance criterion |
| --- | --- | --- |
| Adapter selection and schema versioning | `model_adapter.py`, `schemas.py`, `cli.py` | Both models expose validated physical sites; old LoopVL configs and saved reports still work |
| Bounded branch capture | `capture.py`, new `neuron_audit.py` | Product reconstruction, exact invocation metadata, per token signs, deterministic bounded sampling |
| Text data and splits | new `data/text.py` and corpus manifest | Document grouping, deduplication, context provenance, reproducible hashes, no split leakage |
| Explanations and evaluation | new `explanations.py` | Independent conditions, frozen prompts/budgets, blinded evaluation, no confirmation driven revision |
| Shared and pass conditioned analysis | `encoding.py`, `discovery.py`, `controls.py` | Frozen calibration, matched complexity/data controls, grouped inference, correct denominators |
| Causal collection | `interventions.py`, `pipeline.py` | Real donor alignment and horizon collection, complete controls, locality verified on loaded model |
| Reports and resource accounting | `report.py`, `protocol.py`, `scripts/slurm/` | Immutable manifests, aggregate budget enforcement, artifact integrity, model free recomputation |

Add separate configs such as `configs/neuron_qwen_pilot.yaml` and `configs/neuron_loopvl_discovery.yaml`, with distinct new run IDs. Preserve existing snapshots and existing `loopvl-transfer-v1` outputs. Reuse existing measurements only when their precise capture schema and provenance satisfy the new endpoint. Missing branch observations cannot be reconstructed from a stored product alone.

Add focused tests for numerical reconstruction, branch signs including near zero and negative times negative, invocation reset and locality, bounded sampling, document/family separation, frozen explanation evaluation, immutable locks, schema compatibility, and incomplete causal condition rejection. Use synthetic mechanisms with stable selectivity, gain changes, pass conditioned selectivity, and stable selectivity with changing downstream effects. Label these as engineering fixtures.

Run appropriate existing tests plus the new focused tests in the permitted environment. Schedule expensive suites where site policy requires it. CPU tests establish engineering behavior; loaded model parity and real donor/horizon checks remain separate scheduled gates.

## Protocol lock and final deliverables

Lock each scientific stage before opening its confirmation data. Include model/tokenizer/runtime hashes, capture schema, corpus/scene family IDs, representative sample, primary contrast, explanation prompts and budgets, score mappings, activation thresholds, calibration rules, candidate selection, all required controls, uncertainty method, multiplicity policy, practical margins, and resource reservations.

Use a distinct validated protocol schema for the text audit, whose required controls differ from the visual transfer study. Do not mark a visual mask control passed merely to satisfy the old schema on a text model. Preserve the existing LoopVL schema and required controls.

Do not set control gates manually or reinterpret an observational executor as a causal executor. The existing lock intentionally rejects incomplete controls and protocols with causal candidates that cannot be executed. If no candidates qualify, follow the existing discovery supported skip policy; resource or access failure is a pending stage, not evidence that no candidate exists.

Produce:

- A compact machine readable results package with saved fits, predictions, raw causal measurements, manifests, failures, exclusions, and control status.
- A model free recomputation command for every table, interval, and figure.
- Three core figures: paired inspection condition performance on Qwen; LoopVL source to target transfer and population distributions; selected neurons' per pass selectivity and causal profiles.
- A claim ledger separating observations, supported interpretations, speculation, and unresolved cases, with exact denominators and run references.
- A current progress entry, measured resource accounting, and exact next commands if a required stage remains pending.

Success is a reproducible estimate with adequate controls, including stable, negative, or inconclusive outcomes. Do not require a dramatic neuron, a positive cleanliness result, or a universal claim about superposition. Qwen versus LoopVL differences combine modality, training, scale, and architecture; report them as separate checkpoint studies.

## Optional follow up after the core study

[Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) provides a recurrent language model with configurable depth and intermediate training checkpoints. A later adapter would permit text based recurrence experiments and a separate developmental study. Training time and recurrent inference depth are distinct axes; growth of an explanation over inference passes is not evidence of how it arose during training.

An architecture causality study requires separately trained tied and untied controls with matched data, objective, front end, execution schedule, and seeds. Match computation as one comparison and parameter count as a separate comparison. Untying a frozen checkpoint without retraining creates the same initial function and does not supply this control. Full training requires a separate scoped plan and resources.

## Starting instruction for the next agent

Implement this handoff in the editable `source/` checkout, preserving existing user changes. First read the governing instructions and latest project status, inspect the current saved gates and their evidence, and identify the earliest unmet validation gate. Then implement and verify the missing CPU components and scheduled model stages in dependency order. Keep Qwen replication and LoopVL recurrence results distinct, share the existing aggregate resource limits, and freeze each protocol before confirmation. Continue through the declared deliverables on authorized available resources; when an external dependency prevents a stage, complete independent work and leave exact pending commands without inventing measurements or marking the stage complete.
