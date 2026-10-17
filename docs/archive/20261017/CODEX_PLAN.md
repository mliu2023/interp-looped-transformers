> Historical documentation archived October 17, 2026. Status and commands may be superseded. Use [current progress](../../../PROGRESS.md) and the [current cluster runbook](../../CLUSTER_RUNBOOK.md).

# Codex execution plan: functional consistency under weight reuse

Updated October 16, 2026. Prepared for Max Liu and collaborators.
Status: research execution specification. No experiments in this revised plan have been run.
This replaces the earlier plan centered on visual-anchor swaps and a particular reasoning trace.

## 0. Mission and completion contract

**Goal:** Determine whether repeated uses of the same physical neurons operate on a stable semantic representation, and whether their outputs have consistent causal effects across recurrent passes.

Use a frozen LoopVL checkpoint as the primary model. Study a representative sample of shared MLP neurons across multiple visual task families. Produce independently reproducible measurements of:

1. **Semantic-description transfer:** Does a description of what activates a neuron at one invocation predict its activation at another, allowing ordinary positive gain/offset changes?
2. **Representation-format transfer:** Can the same linear readout recover the same externally defined property from representations at different invocations?
3. **Causal-use transfer:** Do controlled interventions on a neuron produce a comparable pattern of effects across tasks and invocation times?

The neuron's mathematical input-output function is fixed by weight sharing. DO NOT present identical weights, different firing rates, or different input distributions as evidence that its mathematical function changes. The empirical questions concern the relationship between external properties, represented states, and downstream use.

Success means valid estimates, controls, uncertainty, and a reproducible report. Stable reuse, changing use, mixed organization, and insufficient evidence are all legitimate outcomes. Do not search indefinitely for a dramatic neuron or force every neuron into an interpretable category.

**Primary deliverable:** a report with an auditable claim-to-measurement ledger and a one-command recomputation of every reported metric and figure. The scope is a population study with limited follow-up case analyses, not reverse-engineering one task algorithm.

## 1. Scientific definitions and limits

For one actual post-gating MLP channel i in physical layer l:

    a[i,r] = f[i](x[r])
    residual contribution = a[i,r] * W_down[:,i]

For a SwiGLU implementation, f[i] is the product of its activated gate branch and its value/up branch. Audit the implementation rather than assuming branch order or normalization. The weights and write direction are constant across shared invocations.

The external-property-to-activation relation can still vary because x[r] changes. Also, a constant residual write can have different downstream consequences. Keep the following concepts separate:

| Term | Operational meaning | What it does not establish |
| --- | --- | --- |
| Fixed computation | Same parameters and pointwise arithmetic at a matched physical site | Stable world-level meaning |
| Semantic-description transfer | A frozen predictor from annotated properties to neuron activation transfers across passes | A complete explanation of the neuron |
| Representation-format transfer | A frozen readout transfers across passes on matched variables and populations | Information is absent when transfer fails |
| Causal-use transfer | An intervention's effect profile transfers across passes, allowing positive overall gain | A unique semantic role, or a proof of a whole algorithm |
| Changing downstream use | Stable measured selectivity plus changed selective causal effects, with short-horizon corroboration | A change to the neuron's mathematical function |

Competing accounts are: stable reuse; gain/threshold changes; stable selectivity with different downstream use; changing semantic input distributions/representations; and unresolved or inadequately measured units. These can coexist.

Do not equate lower cross-pass decoding with a rotating basis: a rotation is only one explanation. Do not use nonsignificance to conclude equivalence. Do not interpret context dependence as evidence of superposition. Do not claim that weight sharing caused an organization without separately trained architectural controls.

## 2. Model decision and architecture audit

Start with official LoopVL code and complete weights:

- Code: https://github.com/Tier-Flow/LoopVL
- Checkpoint: https://huggingface.co/TierFlow/LoopVL
- Previously inspected code revision: 6c14bed1d62fbecb559d5902459c6093c38da405

Verify that revision and pin the actual checkpoint revision and file hashes. If the revision is unavailable, inspect the current implementation and document the replacement before running experiments.

The repository documents L,L,L,H,L,L,L,H, with 16 physical layers per stack call [R1]. Primary comparisons are the same physical L-layer/channel at L1 versus L4. Secondary comparisons are L2 versus L5 and L3 versus L6. These match positions within the two outer cycles. Never compare an L neuron with an H neuron as if their weights were shared.

Record both the fixed visual anchor and evolving L/H states, explicit cycle-dependent parameters, normalization, masking, positional encoding, and generation behavior. The earlier audit located the recurrence in runtime/hrm_penguin/recurrent/hrm_multimodal.py; confirm current names. Preserve the official loader's mixed precision. Do not silently quantize or cast the entire model.

CVRR is a possible later replication, not required for completion. Its native initialization and adapted recurrent updates differ: exact weight-reuse comparisons must use only its shared adapted transitions [R3]. LoopVL/CVRR comparisons cannot isolate the causal effect of architecture.

### Required hook contract

A captured or intervened site must identify:

    forward_id, decode_step, module_kind, module_invocation,
    outer_cycle, physical_layer, hook_name, neuron_index,
    token_indices, token_role, example_id

Define the primary neuron as the actual MLP product activation immediately before its down/output projection. Record the gate and up branches only for diagnostics. A residual coordinate is not automatically a neuron.

Primary analysis concerns the first answer prediction, before any answer token enters the prompt. Do not use teacher-forced answer representations as evidence of prior knowledge. Confirm the complete invocation count on every forward. Reset counters for every generated/forced-prefix token.

## 3. Execution boundaries and repository

Read the target repository's AGENTS.md and existing experiment conventions. Work on an isolated branch/worktree when appropriate. Do not overwrite existing results. Preserve unrelated changes. No full-model training, extra-loop sweeps, SAE training, paid provisioning, or complete benchmark reproduction is required.

Implement the following interfaces; they are proposed interfaces, not existing upstream commands:

    src/reuseinterp/model_adapter.py
    src/reuseinterp/data/
    src/reuseinterp/capture.py
    src/reuseinterp/encoding.py
    src/reuseinterp/probes.py
    src/reuseinterp/interventions.py
    src/reuseinterp/statistics.py
    src/reuseinterp/report.py
    configs/
    tests/
    docs/architecture_audit.md
    docs/protocol_lock.json
    results/<run_id>/

Use configuration files, atomic shards, resumable jobs, and configuration hashes. A result key is the example, physical site, invocation, condition, and seed. Refuse to merge incompatible manifests. Save failures and exclusions explicitly.

Do not implement downstream machinery based on guessed hook names. Audit a loaded model first. CPU-only implementation and tests are useful when GPU access is absent; clearly mark all unrun model experiments.

## 4. Milestone A — numerical and intervention validity

Before collecting research results:

1. Reproduce official inference on at least 16 varied smoke examples.
2. Capture selected activations without changing outputs.
3. Verify that matched physical neurons share parameter objects/storage and hashes across invocations.
4. Check the post-product hook by reconstructing the MLP output from its product activation and down projection, including bias if present.
5. Verify capture-only, self-replacement, and zero-dose interventions reproduce unmodified logits and answers within a prespecified tolerance.
6. Verify an invocation-local intervention fires exactly once at the requested site and leaves all other invocations untouched.
7. Confirm donor and recipient token alignment, masks, image-grid order, and shapes before replacement.
8. Confirm that result generation never uses reference answers as model input.

Establish tolerances from repeated official forwards on the same hardware/backend. Save max absolute/relative logit errors, answer-margin errors, hook counts, and changed tensor indices. Choose tolerances before intervention analysis. A loose tolerance cannot excuse changing an answer or materially changing a primary margin.

A backend change requires a new unmodified parity check. Do not proceed to interpretation if parity or intervention localization fails.

**Artifact:** parity_report.json, test logs, architecture audit, and exact reproduction commands.

## 5. Milestone B — data covering multiple uses of shared computation

Use controlled scenes with independently available scene properties and question labels. CLEVR supplies scene graphs and functional programs [R4]. Existing CLEVR samples can establish competence. For paired edits, prefer a deterministic renderer with object masks and reproducible geometry.

If the official renderer is impractical, use deterministic 2D scenes with explicit object masks; label this as a different distribution and validate competence before scaling. Do not present a 2D substitute as CLEVR. Do not use generated images as exact scene ground truth. If masks are missing, obtain validated renderer masks or mark object-level analyses blocked; do not invent segmentation.

Required task families:

| Family | Examples | Why included |
| --- | --- | --- |
| Direct attributes | Color/shape/size of a uniquely specified object | Basic represented properties |
| Reference and relational selection | Attribute of an object related to a specified reference | Question-conditioned relevance and binding |
| Counting | Number of objects satisfying a property | Repeated use of an attribute across objects |
| Comparison | Compare two attributes or set counts | Different behavioral uses of the same properties |

Use the same scenes with different questions, including cases where the same object is relevant, irrelevant, a reference, or a target. Balance answers, object positions, and roles as far as the generator permits. Independently vary nuisance factors such as clutter and background.

For primary causal pairs, change one object's color while keeping geometry, token grid, and question fixed. Use the color change in different task families: it can change an attribute answer, membership in a counted set, a relation's reference, or nothing relevant. Counterbalance colors and change directions. This deliberately studies a common semantic perturbation across different uses; it does not cover every possible feature.

Additional shape/material edits are secondary and require separate alignment validation. A local edit can change contextualized embeddings globally. Do not assume only the changed object's features differ.

### Ground-truth descriptors

For each object/scene/question, store the available attributes, position, visibility, question family, referenced/selected status, set membership, counts, final answer, and nuisance variables. Functional programs define task labels, not an assumed neural algorithm.

For neuron encoding, use descriptors independent of the network's predicted answer. Ground-truth relevance flags are legitimate labels but must be identified as such; never call their availability evidence of internal selection.

Freeze an explicit descriptor whitelist. Exclude final-answer labels, model answer scores, and model confidence from the primary encoding model; retain them only for separately labeled diagnostics. Final answers remain available for scoring and data validation.

### Splits and target sizes

A family contains a base scene, both edited versions, all associated questions, and paraphrases. All descendants stay in one split.

| Split | Target independent families | Use |
| --- | ---: | --- |
| Smoke | 32 | Engineering; excluded from scientific confirmation |
| Pilot | 128 | Rendering, prompt, capability, variance, and cost checks |
| Fit | 400 | Encoding models and probes |
| Development | 100 | Hyperparameters, candidate selection, engineering decisions |
| Calibration | 100 | Positive gain/offset fitting and causal-profile calibration |
| Confirmation | 300 | Frozen primary analyses |
| Transfer | 100 | Held-out combinations or paraphrases; separate report |

These are proposed targets, not a power calculation. A default family has two scene versions and up to four questions per version; report actual coverage. Freeze splits before model-based selection. Do not tune on confirmation or transfer examples.

### Capability gate

Evaluate free short answers, candidate-answer scores, and shuffled/no-useful-image controls. Report malformed responses and all families.

For behavioral claims, aim for at least three supported families, including direct attributes and two of selection, counting, and comparison. A proposed engineering screen is at least 65% clean accuracy and a positive paired confidence interval for the advantage over image-shuffled controls, with a point advantage of at least 10 percentage points. This is a screening rule, not a universal scientific threshold.

Permit one pilot revision of clutter or wording; document it. Do not silently fine-tune the checkpoint or discard hard questions. If fewer families qualify, continue observational work and label the behavioral study restricted. Do not manufacture breadth by renaming near-identical tasks.

## 6. Milestone C — representative activation capture

Sample 32 neuron indices uniformly without replacement per physical L layer, using a recorded seed, for 512 neurons if there are 16 layers. Keep this representative sample separate from any discovery-selected examples. If dimensions or resources differ, set the revised sample once before confirmation.

Capture all six L invocations. Primary token strata:

1. Object-pooled visual-token activations, one row per visible object.
2. The last prompt position, a single explicitly identified position.
3. Question-token summaries as a secondary stratum.

Do not mix strata into one undifferentiated population. Normalize aggregation weights so large objects and long prompts do not dominate. Validate object masks through resizing/cropping/token merging. Use all eligible objects, not only the correct target.

For object-pooled MLP products, pool only after recording actual token activations. Later interventions must act on real token activations, never on a synthetic pooled vector.

At a small prespecified set of physical layers, e.g. zero-based indices 0, 5, 10, 15, also capture pooled normalized MLP-input states and selected block-output states for representation probes. Save exact conventions.

Stream selected channels/pooled representations to CPU. Do not retain all attention matrices or all full-token activations. Full tensors are permitted for a small diagnostic subset only.

**Artifact:** typed activation shards with trace IDs, descriptors, sample-selection seed, schema, and checksums.

## 7. Experiment 1 — semantic-description transfer

This experiment measures portability of measured semantic selectivity. It does not test whether the fixed function f[i] changed.

For each sampled neuron and token stratum, predict activation from a fixed descriptor set. Use regularized linear encoding with one-hot categorical terms, continuous nuisance terms, and prespecified interactions such as attribute × object role and attribute × question family. Fit a simpler attributes-only model as a diagnostic.

Use identical feature definitions and regularization-search budgets across invocations. Select hyperparameters on development families only. Freeze the descriptor vocabulary; do not add labels after inspecting confirmation errors.

Fit a separate source-pass model g[i,p] on the fit set for each p. Evaluate each source on every target pass q. Report:

- Raw source-to-target prediction.
- Positive affine calibration alpha[p,q] * g[i,p] + beta[p,q], fitted only on calibration families.
- A within-target model g[i,q] with the same fitting and calibration opportunity.

Constrain alpha >= 0. A sign flip is not ordinary gain modulation. Do not fit a multidimensional rotation or a new semantic model under the name "calibration."

Normalize squared errors by a fixed target-pass activation variance estimated on calibration data. Flag low-variance units using a numerical floor derived before confirmation. Also report raw MSE and held-out R² so normalization cannot conceal poor predictions.

Primary per-neuron transfer gap:

    D[i] = 0.5 * (
      NMSE(calibrated model from L1, tested at L4)
        - NMSE(calibrated within-L4 model)
      + NMSE(calibrated model from L4, tested at L1)
        - NMSE(calibrated within-L1 model)
    )

Report the layer-balanced average D and its uncertainty. Primary stratum: object-pooled visual tokens. Last-prompt results are secondary and reported separately.

### Interpretability eligibility

Define eligibility on development data, not confirmation:
- Adequate variance above the numerical floor.
- Within-pass encoding R² at least 0.10 at both L1 and L4.

Report results for the full representative sample and the eligible subset, along with inactive, poorly explained, and failed units. Eligibility does not guarantee a good explanation; include confirmation adequacy.

A proposed practical portability margin is epsilon_sem = 0.05 normalized MSE. Lock it before confirmation. An upper multiplicity-adjusted confirmation confidence bound below epsilon_sem supports portability within this margin. A lower bound above epsilon_sem supports a meaningful transfer deficit. Otherwise report inconclusive. Show sensitivity to 0.02 and 0.10 as secondary analyses, not alternative thresholds for choosing a preferred conclusion. An average transfer gap never establishes consistency of every neuron; show the distribution and unresolved fraction.

### Required alternative-explanation controls

- Repeat comparisons within matched object roles and question families.
- Compare balanced/reweighted descriptor distributions using discovery-fitted weights with clipped, reported ranges.
- Inspect counterfactual activation differences for paired color edits, not just different top-activation examples.
- Include a coherent label-permutation control across independent families, keeping all passes of a family together.
- Report whether gain calibration accounts for most of the raw gap.
- Do not infer intrinsic semantic reassignment from transfer failure: incomplete descriptors and upstream distribution changes remain alternatives.

**Artifact:** cross-pass transfer matrices, per-neuron metrics, paired-edit selectivity profiles, and a population summary with honest denominators.

## 8. Experiment 2 — representation-format transfer

At the prespecified state sites, train regularized linear classifiers for object color, shape, size, and reference/target membership where labels are defined. Use a shared per-object classifier and all objects. Do not supply the correct target by pooling only that object.

Train at one invocation and evaluate at another without updating feature weights. Fit preprocessing on fit families only. Compare within-pass and cross-pass performance, with label-permutation and question/position-only baselines.

This distinguishes available information from portability of a particular readout. A strong within-pass probe with weak cross-pass transfer indicates that this fixed readout is not portable; it does not show that information disappeared or prove a rotation.

Where possible, compare probe transfer with the neuron selectivity results at the same physical layer. Dense probes and single neurons are different measurements; do not treat agreement as independent causal evidence.

Do not use output-head failure at an early state as a substitute for these tests.

**Artifact:** accuracy/calibration matrices for each semantic variable and token stratum, held-out baselines, and raw predictions.

## 9. Experiment 3 — causal-use transfer

Run only after instrumentation and data gates pass. Use at most eight neuron/site candidates initially:
- Four drawn randomly from numerically usable units in the representative sample.
- Up to four selected on discovery data for reproducible semantic selectivity and feasible intervention effects.

Keep the two groups separate. This small causal sample cannot estimate the prevalence of causal roles across the full model. Specify selected physical layer, neuron, token stratum, object-mask rule, and selection rationale before confirmation.

Candidate selection uses fit/development families only. Reserve calibration families for the declared calibration fits; do not select candidates for unusually good calibration or confirmation results.

Primary comparisons are L1 versus L4. Later passes are secondary extensions within the budget.

### 9.1 Controlled dose interventions

At actual post-product activations, add +d[i] or -d[i] to the selected neuron at the specified token positions. Define d[i] as 0.25 times its pooled discovery standard deviation across L1/L4, using the same physical dose at both passes. Record d[i] and the norm of its resulting down-projection contribution.

For object sites, intervene on that object's validated visual-token mask, with an explicit fixed mask-weighting rule. For last-prompt sites, intervene on exactly that position. Do not alter model weights or every occurrence of the shared neuron.

Use clean and sham interventions. Examine natural activation ranges and a preregistered overlap subset where both dose directions remain within discovery-derived ranges at both passes. Report the complete valid population as well as overlap coverage; do not discard confirmation examples because their effects look strange.

A secondary natural-donor intervention replaces the same channel's actual token activations using the aligned color-edited image. Natural replacement tests semantic direction, but its dose varies across passes. Do not mistake that variation for changed downstream sensitivity.

### 9.2 Outcome measures

For each scene edit with different answers, define the same oriented candidate margin in all conditions:

    M = log p(answer for edited scene) - log p(answer for base scene)
    E[i,r,e] = (M[+d at r] - M[-d at r]) / 2

E is the half-contrast at the locked physical dose, in log-score units. A finite-difference derivative E/d may be reported separately; do not mix these units in effect profiles or practical margins.

Use full answer-string conditional scores when necessary. For single-token answers, use the corresponding logit difference. Do not mix restricted candidate choice with free generation.

For answer-preserving edits and irrelevant objects, the two-answer margin is undefined. Use original-answer log probability, generated-answer changes, and divergence over a fixed answer set as separate specificity outcomes. Do not set a nonexistent margin to zero.

If candidate answers require multiple generated tokens, apply the same invocation-specific intervention on every scored continuation forward; label this as a repeated-across-decoding intervention. Prefer verified single-token answers for the primary experiment, and report the coverage restriction.

Record early/late effects on exactly the same examples. Define each neuron's effect profile over task family, object role, and semantic contrast, weighting groups equally. Show raw magnitudes and signs.

Fit one nonnegative gain lambda[i] on calibration interventions to predict the late profile from the early profile. Freeze it. Test remaining group-specific discrepancies on confirmation data. Use a family-clustered test/interval for the pass × context interaction, with a declared effect-size margin in log-score units set from pilot variance before confirmation. If the pilot cannot justify a margin, report estimates and uncertainty without classifying causal equivalence.

Implement the interaction explicitly. For each locked context group g, calculate R[i,g] = mean(E[i,L4] - lambda[i]*E[i,L1]) over that group. Choose a reference group g0 on development data, and calculate C[i,g] = R[i,g] - R[i,g0]. Preserve the same families and weighting rules across passes. Report these linear contrasts with simultaneous family-bootstrap confidence bands. A bootstrap confidence interval for an uncentered squared norm is not a valid test of zero interaction. For a joint bootstrap test, use a centered max-t statistic over the locked C contrasts, with the standard errors and zero-variance handling recorded in code. Report any missing or underpopulated groups instead of imputing effects.

Do not interpret a larger global effect as a different role. A profile difference after gain adjustment is a measured change in downstream sensitivity, not by itself a semantic explanation.

Report gain values and absolute effect sizes. A near-zero late profile or a fitted gain near zero does not establish stable causal use: classify that case as weak/absent measured influence or unresolved, unless a separate explanation is supported.

### 9.3 Remaining-computation control

Early interventions have more downstream computation and opportunities for compensation. Therefore also capture a fixed short horizon: the output of the next physical transformer block within the same L invocation. Restrict this corroboration to sites that have such a following block.

Use a frozen, validated semantic readout at that horizon, preferably one portable across both invocations. If no reliable comparable readout exists, say so; do not compare unrelated probe scales and call the difference a changed mechanism.

A final-answer effect difference without short-horizon or circuit-level corroboration is **pass-dependent outcome sensitivity**, not demonstrated reassignment of a neuron's semantic role.

### 9.4 Required controls

- Self replacement and zero dose.
- Both positive and negative dose, fixed across compared passes.
- Natural donor replacement in both directions where valid.
- Matched irrelevant-object or answer-preserving controls.
- At least one smaller dose (0.5*d) on development data; freeze the primary dose thereafter.
- Correct token alignment, single-invocation firing, and equal comparison populations.
- Report all valid examples; clean-correct subsets are secondary and labeled.

Follow-up circuit inspection is limited to two discovery-selected cases. It may examine upstream attention and downstream readers to explain measured transfer patterns. It must not become a new unbounded head search.

**Artifact:** per-example interventions, raw candidate scores, effect profiles, calibration parameters, specificity checks, and short-horizon readout results.

## 10. Statistics and preregistration

Before confirmation, write an immutable protocol_lock.json containing:
- Model/code revisions and runtime settings.
- Dataset/split hashes, sample counts, and exclusion rules.
- Random neuron sample and selected causal candidates.
- Primary comparisons and token strata.
- Descriptor features, probe/encoding hyperparameters, calibration rules.
- Numerical tolerances, dose rules, eligibility thresholds, effect-size margins.
- Family weighting, statistical procedures, multiplicity family, compute cap.

Primary confirmatory endpoints:
P1: layer-balanced semantic-description transfer gap, L1 versus L4, object stratum.
P2: prespecified color-readout transfer deficit at one discovery-selected physical state site.
P3: aggregate gain-adjusted causal profile interaction for the locked candidate set.

P2's site selection occurs only on development data. P3 is skipped with a documented reason if no viable causal candidate exists. Skipping is not evidence of equivalence.

Define P2 as the average, across L1-to-L4 and L4-to-L1, of within-target balanced accuracy minus source-to-target balanced accuracy. Compare the same held-out objects at each target. All other attributes/sites remain secondary.

Bootstrap complete scene families for 95% intervals, preserving all versions, questions, objects, and passes. Use 2,000 resamples initially. Report independent family count, not token count as the sample size.

These intervals condition on the frozen fitted models, calibration parameters, and candidate list. They do not include uncertainty from refitting or from model pretraining. Say so explicitly. Seed/re-fit robustness can be a separate discovery analysis.

The representative neuron sample is stratified by physical layer. Report layer-balanced estimands. Include a secondary resampling analysis of neurons within those fixed strata to assess sample sensitivity; do not imply uncertainty over model-training seeds.

The primary analysis is interval estimation. For headline confirmation decisions, allocate familywise alpha = 0.05 across the three endpoint families: use 98.33% intervals for aggregate P1/P2, and a simultaneous 98.33% max-t bootstrap band over P3's locked contrasts. Keep the three-way allocation even if an endpoint is skipped. Show ordinary 95% intervals as descriptive secondary output. P2's proposed practical transfer-deficit margin is 5 percentage points in balanced accuracy; lock this on development data. Per-neuron discovery significance is exploratory; any large-scale confirmatory per-neuron testing requires a separately declared false-discovery procedure. Report effect sizes and intervals regardless of significance.

No fresh unit/site selection or prompt tuning after opening confirmation results. New hypotheses require a new independent confirmation split and a separately labeled exploratory run.

## 11. Verifiable claim rules

| Potential report claim | Minimum evidence required |
| --- | --- |
| A measured semantic description transfers | Adequate prediction, held-out transfer, and portability interval within the locked margin |
| Calibration explains the difference | Raw gap shrinks under positive scalar calibration fitted outside confirmation |
| The measured representation format is not portable | Adequate within-pass probes, poor frozen cross-pass transfer, and matched-data controls |
| A neuron has different final-answer influence across passes | Valid invocation-local intervention with paired nonzero effect difference |
| Stable selectivity has changing downstream use | Selectivity transfer plus selective causal-profile change, fixed-dose controls, and short-horizon/circuit corroboration |
| A neuron changes its mathematical function | Not an admissible claim in this frozen weight-shared experiment |
| Weight sharing causes the observed organization | Requires additional controlled training; outside this core plan |
| An entire model's neurons are generally interpretable | Not established by this study; report sample, descriptor coverage, and unresolved units |

Attach every report claim to exact run IDs, metric rows, denominators, uncertainty, control status, and a reproduction command. Mark statements as observed, supported interpretation, speculative, or unresolved.

## 12. Tests that must fail when the pipeline is wrong

Implement small synthetic fixtures before interpreting real-model output:

1. Stable latent meaning: a synthetic fixed neuron fed an unchanged semantic encoding should show transfer within noise.
2. Positive gain/offset change: raw transfer can fail; positive affine calibration should recover it.
3. Sign reversal: positive calibration must not silently classify inversion as ordinary gain.
4. Changed upstream encoding with fixed neuron weights: descriptor transfer can fail although parameter hashes remain identical.
5. Stable feature with changed downstream reader: encoding transfer passes while causal effects differ.
6. Different remaining continuation lengths: final effects can differ with no change in the local write; the report must not overclaim.
7. Family leakage, test-set calibration, missing condition rows, and duplicated confirmation IDs must be detected.

These fixtures verify the analysis, not scientific findings. Store their outputs separately from model measurements.

Data tests must independently verify scene labels, edits, uniqueness of referents, renderer masks, answer strings, and grouped splits. Use program execution rather than assuming generated prose is correct.

## 13. Resources, execution order, and stopping

The user has no local GPU. Prepare and test CPU components locally, and run GPU experiments on the university cluster through Slurm using section 13A. A single GPU with approximately 24 GB VRAM is a planning target, not a verified requirement; measure actual compatibility and peak memory before scaling. Use additional GPUs only for independent shards unless model sharding is explicitly validated. Do not provision paid resources as part of this plan.

Default initial ceiling: 30 GPU-hours across smoke/pilot/discovery/confirmation, plus a configurable 20 GiB research-output budget excluding model weights and source datasets. These are ceilings, not runtime promises. The executing agent may use a smaller cap supplied by the user.

Measure 32 forward examples before scaling. Dry-run every job with estimated forward count, storage, and wall/GPU time. Interventions dominate cost: begin with 128 independent confirmation families balanced across supported tasks, and scale toward the target only if budget and interval precision permit. Record this causal subset before reading its outcomes. Observational confirmation can use the full 300 families.

When over budget, reduce exploratory extensions and candidate count before weakening numerical controls or leaking test data. Freeze reductions before confirmation. Do not repeatedly sample more examples until significance appears. Any extension based on interval width must follow a declared stopping schedule and be reported as such.

| Milestone | Required output | Gate |
| --- | --- | --- |
| A: audit and parity | Audit, numerical checks, hook tests | Exact sites and valid interventions |
| B: data and capability | Manifests, splits, clean/control results | Supported scope established |
| C: capture | Representative neuron/state shards | Complete trace and label coverage |
| D: discovery | Encoding/probe fits and locked candidate list | No confirmation leakage |
| E: confirmation | P1/P2 estimates; P3 if feasible | Frozen protocol and sufficient controls |
| F: synthesis | Verifiable report and figures | Claims traceable to raw measurements |

Checkpoint progress every 25 families and on termination signals. At the cap or an external blocker, produce a partial report with completed work, failed gates, exact pending commands, and measured resource use. Never fabricate results or use bundled paper measurements as fresh experimental output.

## 13A. University-cluster execution with Slurm

The user has no local GPU. Implement and document a laptop-to-cluster workflow. GPU experiments run only inside Slurm allocations on the university cluster. Local GPU availability is not a prerequisite for preparing the project.

### Responsibilities and files to deliver

- Laptop: edit code, run synthetic fixtures/data checks, prepare configurations, and regenerate reports from downloaded results.
- Cluster login node: transfer code, inspect scheduler configuration, submit/monitor jobs, and perform only preparation permitted by university policy. Model inference, large rendering jobs, and substantial analysis belong in scheduled jobs.
- GPU compute node: model loading, numerical parity, capability evaluation, capture, and interventions.
- CPU compute node: substantial data generation, fitting, merging, and reporting.

Create these files as part of implementation; the templates below are instructions, not already implemented commands:

    configs/cluster.env.example
    scripts/slurm/environment.sh
    scripts/slurm/run_gpu.sbatch
    scripts/slurm/run_cpu.sbatch
    scripts/slurm/submit.py
    scripts/cluster_preflight.py
    docs/CLUSTER_RUNBOOK.md

The runbook must contain the exact environment setup, smoke submission, array submission, restart, monitoring, and result-download commands for the actual cluster. Do not assume a particular university cluster, partition, GPU model, account, or QoS.

### Discover resources and prepare a reproducible environment

1. Inspect the cluster's own documentation and existing project job scripts. Record hostname, Slurm version, allowed partition/account/QoS, GPU request syntax, time limits, shared-storage paths, quotas, and purge policy. Useful read-only commands include:

~~~bash
sinfo -o "%P %a %l %G"
scontrol show partition
~~~

2. Put site-specific settings in a local configuration copied from the example below. Replace every placeholder before submission; omit optional scheduler flags when unused. These are starting resource requests, not measured requirements. Slurm host memory and GPU VRAM are separate resources.

~~~bash
export REUSE_PROJECT_DIR="/absolute/shared/path/to/frozen-checkout"
export REUSE_ENV_DIR="/absolute/shared/path/to/cluster-venv"
export REUSE_RUN_ROOT="/absolute/shared/path/to/results"
export REUSE_CACHE_ROOT="/absolute/shared/path/to/model-cache"
export REUSE_PARTITION="REPLACE_WITH_GPU_PARTITION"
export REUSE_ACCOUNT=""
export REUSE_QOS=""
export REUSE_GPU_REQUEST="gpu:1"
export REUSE_CPUS="8"
export REUSE_MEM="32G"
export REUSE_TIME="01:00:00"
~~~

3. Build the Python environment on the cluster using the audited model's requirements, then pin dependencies. Do not copy a laptop virtual environment to Linux compute nodes. Put site-required module loading and environment activation in environment.sh; do not guess module versions. Prefer the university's supported environment/container approach if required. Never install or modify host GPU drivers.

4. Stage the pinned model revision and datasets once on an approved download/login node, subject to local policy. Compute jobs must not redownload them independently. Set HF_HOME to REUSE_CACHE_ROOT. Enable offline model loading only after checking that weights, tokenizer, processor, and any audited custom code are cached. Keep credentials out of configuration snapshots and logs.

5. Run cluster_preflight.py inside the GPU allocation. Record Python/PyTorch versions, CUDA availability, GPU name/VRAM, supported dtype, and actual device placement. Fail clearly if CUDA, required operators, or memory are unavailable; do not silently run model experiments on CPU. A dtype/backend change requires repeating parity checks. Preserve Slurm's device visibility instead of hard-coding physical GPU IDs.

6. Freeze code and configuration before submitting a stage. Jobs must read an immutable checkout/snapshot, not a working tree being edited while queued. Record commit and snapshot/configuration hashes, dependency lock, model/data revisions, and scheduler resource requests. Shared checkpoints/results must survive job termination; node-local temporary storage is only a disposable cache.

### Single-GPU smoke job

Implement run_gpu.sbatch using this structure. Its environment script must load the validated site environment, activate REUSE_ENV_DIR, set cache paths, and validate required variables. The preflight script must exit nonzero on failure.

~~~bash
#!/usr/bin/env bash
#SBATCH --job-name=reuseinterp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
set -euo pipefail

cd "$REUSE_PROJECT_DIR"
source scripts/slurm/environment.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun --ntasks=1 python -u scripts/cluster_preflight.py
srun --ntasks=1 python -u -m reuseinterp.cli "$@"
~~~

Slurm options belong on submission commands or literal #SBATCH lines; shell variables inside #SBATCH directives do not expand. The example submission below assumes the chosen cluster accepts --gres. Adapt the GPU request flag to the documented local syntax.

After completing CPU preparation, submit a short smoke run from the cluster:

~~~bash
source configs/cluster.local.env
mkdir -p "$REUSE_RUN_ROOT/logs"
sbatch --parsable --partition="$REUSE_PARTITION" --gres="$REUSE_GPU_REQUEST" --cpus-per-task="$REUSE_CPUS" --mem="$REUSE_MEM" --time="$REUSE_TIME" --output="$REUSE_RUN_ROOT/logs/%j.out" --error="$REUSE_RUN_ROOT/logs/%j.err" scripts/slurm/run_gpu.sbatch validate --config configs/smoke.yaml
~~~

If the cluster requires an account or QoS, add the corresponding verified flag. The submit.py wrapper must construct argument lists safely, support these optional flags, create log directories before sbatch, save returned job IDs, and offer --dry-run that prints commands/resource estimates without submitting. Explicitly export required runtime variables using the site's supported environment-export policy.

The smoke configuration must run preflight plus Milestone A parity/hook checks. Only after success run the 32-example timing/memory pilot and capability gate. Use measured runtime and peak VRAM to set later requests. Reduce batch size first if memory is insufficient; do not silently change precision, quantization, model, or scientific endpoints. Request one GPU per shard; additional GPUs are for independent shards unless model sharding has been separately validated.

### Arrays, stage dependencies, and resumption

Implement array support for baseline, capture, and confirmation. With --slurm-array, the CLI reads SLURM_ARRAY_TASK_ID as --shard-index; --num-shards remains explicit and fixed across retries. Assign whole scene families deterministically to shards, keeping paired edits and all required conditions together. Each shard writes to its own directory under the configured run ID. Never have tasks append concurrently to one CSV or progress file.

For example, after all prerequisite gates pass, submit a capture stage with four shards, at most two running together:

~~~bash
sbatch --parsable --partition="$REUSE_PARTITION" --gres="$REUSE_GPU_REQUEST" --cpus-per-task="$REUSE_CPUS" --mem="$REUSE_MEM" --time="$REUSE_TIME" --array=0-3%2 --output="$REUSE_RUN_ROOT/logs/%A_%a.out" --error="$REUSE_RUN_ROOT/logs/%A_%a.err" scripts/slurm/run_gpu.sbatch capture --config configs/discovery.yaml --slurm-array --num-shards 4 --resume
~~~

Use afterok dependencies when appropriate, followed by a CPU merge/validation job. The merger must verify expected family/condition coverage, compatible hashes, and completion markers for every required shard. Scheduler success alone is insufficient evidence of complete data. Keep discovery, protocol locking, and confirmation as distinct gated stages; do not queue confirmation against an unfinished discovery configuration.

Install Python signal handlers for USR1 and TERM. The template requests advance USR1 notification to job steps. A handler should request a graceful stop; save at the next safe boundary rather than perform complex I/O inside the handler. Atomically commit results and progress at least every 25 families, and checkpoint more frequently when measured per-family time makes this necessary. On restart, skip only validated committed results; discard/recompute incomplete records.

A gracefully stopped but incomplete shard must exit nonzero (for example, 75), so downstream afterok jobs cannot consume partial data. Automatic requeue is site-dependent and must not be assumed. Resubmit failed indices with the same run/configuration hashes and original --num-shards; refuse concurrent duplicate writers. If a predecessor failed, recreate/update dependent submissions as needed rather than assuming they will recover automatically after a retry. Keep retry history and failure reasons.

### Monitoring, resource accounting, and collecting results

~~~bash
squeue -u "$USER"
sacct -j ACTUAL_JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
~~~

Use actual returned job IDs. Inspect logs and scheduler states for out-of-memory errors, timeouts, missing dependencies, and preemption. MaxRSS is host memory; additionally record PyTorch peak allocated/reserved GPU memory.

Count GPU-hours across all allocated GPU tasks, including smoke runs and retries, without double-counting job-step accounting rows. Queue wait is separate. Enforce the 30 GPU-hour project ceiling before submissions: include already consumed time and the maximum requested GPU time of outstanding jobs when checking available budget. Array concurrency reduces elapsed turnaround, not total GPU-hours. Keep an accounting ledger and an explicit remaining budget.

Download the small report package and sufficient raw scores/predictions to reproduce all reported statistics. For example, on the laptop, replacing the host/path placeholders:

~~~bash
rsync -av university-login:/absolute/shared/path/to/results/ACTUAL_RUN/report-package/ ./results/ACTUAL_RUN/report-package/
~~~

Do not require model weights or full activation dumps to rebuild figures. Keep larger captures on approved cluster storage and document retention/backup before scratch purge. Reporting must work on CPU without importing/loading the VLM.

### Acceptance checks and handoff

Before cluster access: syntax-check shell scripts, test submit.py with a fake sbatch, verify deterministic family sharding, and simulate interruption/resume to ensure no duplicated or omitted committed results. Verify an incomplete shard blocks merge. These are infrastructure checks, not model findings.

On the cluster: require a real GPU preflight, successful numerical smoke checks, measured pilot resources, and one interrupted/resumed small run matching an uninterrupted run within the declared numerical tolerance. Run CPU analysis through run_cpu.sbatch when it is too heavy for a laptop; that script requests no GPU and must not invoke the GPU preflight.

If Codex cannot access the cluster, it must finish CPU preparation and deliver the exact scripts/commands plus a handoff checklist. Mark GPU jobs and cluster-specific verification as pending. Do not claim submission or completion without scheduler IDs and saved outputs. No cluster password or private key belongs in the repository or chat.

Official operational references: [sbatch](https://slurm.schedmd.com/sbatch.html), [job arrays](https://slurm.schedmd.com/job_array.html), and [sacct](https://slurm.schedmd.com/sacct.html). Local university documentation takes precedence for resource names and usage policy.

## 14. Command and result contract

Implement a single CLI. Example commands after implementation:

~~~bash
python -m reuseinterp.cli audit --config configs/smoke.yaml
python -m reuseinterp.cli validate --config configs/smoke.yaml
python -m reuseinterp.cli data --config configs/pilot.yaml
python -m reuseinterp.cli baseline --config configs/pilot.yaml
python -m reuseinterp.cli capture --config configs/discovery.yaml
python -m reuseinterp.cli fit --config configs/discovery.yaml
python -m reuseinterp.cli discover-causal --config configs/discovery.yaml
python -m reuseinterp.cli lock --config configs/confirmation.yaml
python -m reuseinterp.cli confirm --config configs/confirmation.yaml --dry-run
python -m reuseinterp.cli confirm --config configs/confirmation.yaml
python -m reuseinterp.cli verify-results --run-dir results/ACTUAL_RUN
python -m reuseinterp.cli report --run-dir results/ACTUAL_RUN
~~~

ACTUAL_RUN must be replaced with a real returned run directory. The report command must consume saved data without silently launching experiments. Provide a CPU-only command to recompute tables/figures from shared results. Add the Slurm array, deterministic sharding, resume, and completion-validation interfaces specified in section 13A; configuration files must resolve run outputs under REUSE_RUN_ROOT. GPU commands shown here run through the Slurm wrapper when executing on the cluster.

Minimum result schema:

    run_id, config_hash, checkpoint_revision, split, family_id, example_id,
    task_family, module, invocation, physical_layer, hook_site, neuron_id,
    token_stratum, object_id, condition, dose, donor_id, target_answer,
    candidate_scores, generated_answer, validity, exclusion_reason

Use separate typed tables for encoding fits, probe predictions, and interventions where fields differ. Provide schema validation, not merely a CSV convention.

Every run manifest includes hardware, precision/backend, dependencies, seeds, code/model/data hashes, preprocessing, prompts, tokenization, trace conventions, calibration IDs, resource use, and failures.

## 15. Required final package

- README with exact installation and reproduction commands.
- Slurm submission scripts, environment/configuration examples, CLUSTER_RUNBOOK.md, scheduler job records, and resource-accounting ledger; clearly distinguish locally tested scripts from cluster-verified runs.
- Architecture audit with code references and verified weight identity.
- Data generator/adapter, split manifests, descriptors, masks, and validation report.
- Protocol lock and hash; neuron sample and candidate-selection records.
- Per-example clean/control/intervention scores and probe/encoding predictions.
- Numerical parity and synthetic-fixture reports.
- Figures:
  1. Task competence and image-dependence controls.
  2. Raw versus calibrated cross-pass semantic transfer.
  3. Representative-sample distributions, with unresolved units included.
  4. Representation-readout transfer matrices.
  5. Paired causal profiles and short-horizon comparisons.
- report.md, claim_ledger.csv, metrics.json, and a machine-readable limitations/status file.
- PROGRESS.md with completed milestones, unresolved blockers, and next commands.

Each headline number must have a numerator/denominator or an explicit estimand, confidence interval where meaningful, and source rows. A second person must be able to regenerate all figures from saved data.

Do not prefill results with expected conclusions. Example plots must be visibly labeled synthetic and excluded from the final findings.

## 16. Research positioning and sources

This plan studies population-level representational consistency and causal use under repeated application of fixed weights. It does not claim the first observation of state-dependent computation.

- [R1] LoopVL official runtime and checkpoint: https://github.com/Tier-Flow/LoopVL and https://huggingface.co/TierFlow/LoopVL
- [R2] Qian et al., LoopVL: Recurrent Visual Intelligence. https://arxiv.org/abs/2609.38426
- [R3] Park et al., Reason Through the Latent! Making Latent Visual Reasoning Necessary. https://arxiv.org/html/2609.06746v1 and https://github.com/dmis-lab/CVRR
- [R4] Johnson et al., CLEVR. https://cs.stanford.edu/people/jcjohns/clevr/
- [R5] Wu et al., Shared Weights, Selected Computations. https://arxiv.org/html/2609.39892v1
- [R6] Blayney et al., A Mechanistic Analysis of Looped Reasoning Language Models. https://arxiv.org/html/2604.11791v1
- [R7] Zhang and Nanda, Towards Best Practices of Activation Patching in Language Models. https://arxiv.org/abs/2309.16042
- [R8] Cammarata, Neural Networks Are Already Clean. https://already-clean.goodfire.com/

R3 already tests visual-evidence/state reliance. R5 already demonstrates state-controlled transitions and causal attention routing. R6 already studies recurrent stages and stabilization. R8 motivates testing native neurons but does not establish neuron consistency in looped VLMs. Verify current versions and record changes before execution; avoid universal novelty claims.

## 17. Paste this instruction into Codex

~~~text
Execute the attached CODEX_PLAN.md in the current research repository.

The scientific goal is to measure whether semantic descriptions and causal effects of the same physical LoopVL neurons transfer across recurrent invocations. Do not test the tautological claim that identical weights compute an identical function. Do not make a particular reasoning trace or anchor swap the project's central goal.

First read repository instructions, inspect available compute/data, and implement the numerical audit and synthetic analysis fixtures. There is no local GPU: complete CPU preparation locally and implement the Slurm workflow in section 13A for the university cluster. Keep site-specific settings configurable, start with a scheduled GPU smoke job, and use resumable independent shards only after validation. If cluster access is unavailable, deliver the executable handoff and mark GPU checks pending. Then build a broad task-family dataset, validate competence, capture a representative neuron sample, and implement discovery/confirmation separation.

Proceed through the milestones on existing available resources. Lock the protocol before confirmation, obey the resource cap, and stop expanding once the declared endpoints and controls are complete. Record negative and inconclusive results honestly. If GPU or data access is blocked, complete useful CPU work and provide exact pending commands without inventing model results.

The final package must let another researcher recompute every reported metric and figure from saved raw measurements. Every claim must identify its evidence, uncertainty, scope, and remaining alternative explanations.
~~~

