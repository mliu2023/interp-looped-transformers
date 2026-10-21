# LoopVL dynamics under weight reuse

**Planning revision: 2026-10-21. Status: implementation plan, not a completed experiment.**

This is the current scientific direction. It replaces semantic-description transfer
as the organizing question; `CODEX_PLAN.md` and its artifacts remain historical.
Reuse the validated infrastructure where suitable, not its old endpoints or gates
as substitutes for the new measurements. Do not center the study on an associative
memory analogy, a particular reasoning trace, or an image-anchor swap.

The question is: **what changes when the same physical computation acts repeatedly
on evolving states, and what consequences do those changes have?** Measure states,
attention routing, neuron populations, and effective feedback across invocations.
Do not assume later invocations are better, more abstract, sparse, or convergent.

GitHub main was fetched and the clean local checkout fast-forward checked before
writing this plan; it was already current at `6e8edc0`. No new model result was
available to the agent. The last supplied GPU evidence is job 7489067, which failed
CUDA initialization before loading the model. The CUDA 12.6 recovery is prepared;
successful recovery, loaded-model validation, and model feasibility remain pending.

## 1. Questions and scope

Study five connected questions without presuming their answers:

1. How do L and H change in magnitude, direction, variability, and relationship to
   each other, within and across outer cycles?
2. Do Q/K/V distributions and actual attention routing change for the same inputs,
   physical heads, and token positions?
3. Does the population of strongly active MLP channels change? Are changes due to
   rescaling, recruitment of different channels, sign changes, or cancellation?
4. Are latent updates concentrated in a small, reproducible subspace? Is that
   subspace shared across invocations, tasks, or the L/H boundary?
5. Can a neuron's write reduce its own activation at its next physical reuse, and
   how much of an outer-cycle change is attributable to visual reinjection?

Observations describe this checkpoint on the sampled input distribution. They do
not isolate the benefit of weight tying relative to separately trained untied
models. A pair of outer cycles does not establish asymptotic convergence. Behavioral
improvement, semantic interpretation, and task utility each require their own
evidence; activation drift alone establishes none of them.

## 2. Record the actual computation, not just a loop number

The pinned source has two distinct physical stacks, L and H, each with 16 blocks.
L is invoked three times per outer cycle, H once, for two outer cycles. Let
`h_before[c]` be the high state before reinjection, `r[c]` the image contribution,
and `h_cond[c] = h_before[c] + r[c]`. The schedule is:

```text
scaled multimodal embeddings; initialized low state
  reinjection 0 -> h_cond[0]
  L1(low + h_cond[0]), L2(low + h_cond[0]), L3(low + h_cond[0])
  H1(h_cond[0] + low_after_L3)
  reinjection 1 -> h_cond[1]
  L4(low + h_cond[1]), L5(low + h_cond[1]), L6(low + h_cond[1])
  H2(h_cond[1] + low_after_L6)
```

The low state carries over the outer-cycle boundary. The high conditioning state
is fixed during each group of three L invocations. Image reinjection occurs at
the start of **both** outer cycles. Its source is the same cloned image embedding;
gates depend on the current visual/instruction states and its learned scale is
cycle specific. It supplies renewed image evidence, not newly observed pixels.

Save an event ledger for initialization, before/after each reinjection, every stack
entry/exit, and selected block boundaries. At every stack entry preserve both
summands separately and their sum. Preserve outputs before and after final RMSNorm.
Ordinary stack hooks see the sum, not both local variables. Add an audited event
observer in a project-owned recurrence adapter that exposes the actual tensors;
do not reconstruct a summand by subtracting rounded sums. Preserve the pinned
upstream checkout. If implementing an observer-enabled forward mirror, hash its
source and compare against the independent official forward with that mirror and
all observers removed; comparing two paths that both use the mirror is insufficient.
Use these distinct differences:

- **Low-state update:** new low minus previous low.
- **High computation update:** new high minus the reinjected high conditioning
  state. Also record total new high minus pre-reinjection high.
- **Injection jump:** reinjected high minus pre-injection high.
- **Stack transformation:** stack output minus its summed input.
- **Matched-site drift:** one physical block/site's state at a later invocation
  minus its state at an earlier invocation.

These differences answer different questions and must not share one ambiguous
`delta_state` field. Stack final normalization can dominate a difference: retain
pre/post-normalization states and separate norm change from directional change.

Comparisons include L1→L2, L2→L3, L4→L5, L5→L6; matched inner phases L1↔L4,
L2↔L5, L3↔L6; the composite L3→L4 boundary; and H1↔H2. Never present the boundary
as a pure effect of reinjection: it also includes H computation and changed
conditioning. Never treat H as another invocation of L.

## 3. Inputs, matching, and independent units

Start with first-prediction forwards: no answer token or reference answer enters
the model. Use the official processor, precision, loop schedule, RoPE, mask and
`use_cache=False`. Each generated token restarts the recurrent computation in the
official generation path. Any later decoding study must use identical recorded
forced prefixes across conditions; it is not a continuing hidden-state trajectory.

Use two explicit data panels:

- **Available controlled panel:** the checksummed scene archive already staged by
  this project. Balance `direct_attributes`, `counting`, `comparison` and
  `relational_selection`. Default discovery is 32 entire families from the existing
  fit split and confirmation is 32 from the unopened confirmation split, with all
  four clean questions: 256 examples total. Select IDs by seeded, task/scene-balanced
  manifest rules before measurement. Reserve at most 16 discovery families for
  paired image-edit/question controls. All edits/questions/objects of one family
  stay together. This is a controlled 2D distribution, not a natural-image benchmark.
- **Natural-image panel:** if an accessible, licensed image/QA dataset with original
  images, provenance and reliable task labels can be staged, allocate a separate
  32 discovery + 32 confirmation independent image groups, balanced across available
  question types, at most four questions per group. Duplicates/near duplicates stay
  together. This is capped at 256 additional examples and must fit the same budget.
  Do not fabricate its availability or labels. Freeze its source/version/checksums
  before any capture. If blocked, complete and label the controlled-panel study;
  natural-image generalization remains untested, rather than delaying all work.

The 32-family defaults are bounded starting designs, not a power claim. The timing
pilot may require fewer token/site diagnostics or omission of the natural-image
panel. Final family IDs, panel weights and limits must be locked before confirmation.
Do not shrink a sample after observing confirmation results.

Always pair the exact same example, prompt, token positions, grid and prefix across
invocations. Separate visual tokens, instruction tokens and last-prompt position;
record their overlap rather than counting overlapping roles as independent data.
Object masks support secondary comparisons with equal object weight, not extra
independent samples. Include task, object count, question length, grid size and
correct/incorrect behavior as strata. Use equal family weight, then equal question
and declared token-role weights. Do not let image size or sequence length silently
determine the dataset distribution.

Measure competence with clean, neutral-image and shuffled-image behavior. Basic
dynamics can be reported even for tasks the model fails; claims about useful
reasoning require competence/image dependence and appropriate outcome measurements.
Correctness stratification is descriptive, not an intervention on competence.

## 4. Capture contract and measurement tiers

Every record includes dataset/family/example/prompt hashes, forward/decode IDs,
event order, L/H kind, outer/inner phase, physical layer/head/channel, real token
indices and roles, image grid and masks, tensor shape/dtype/projection stage,
position IDs, exact RoPE inputs/scaling, attention mask, backend and source identity.

Use three tiers, fixed without looking at confirmation activations:

| Tier | Coverage | Saved measurements |
|---|---|---|
| Broad | All 16 physical layers in both stacks; all their invocations and heads | Per-example/head/query routing records; all-channel activity summaries and last-prompt products; event-state summaries; behavior |
| Vector panel | Physical layers 0, 5, 10, 15 in each stack; seeded real-token panel reused across visits | Full-dimensional state/update vectors; sampled-channel gate/up/product values; Q/K/V at matched token positions |
| Detailed audit | At most 32 examples total, initially 16 discovery + 16 independently fixed confirmation examples; heads 0, 4, 8 at the vector-panel layers | Actual Q for selected queries and K/V for **all eligible keys**, mask/RoPE/gates, exact intermediate states and intervention outcomes |

The primary query is the last prompt position. A fixed instruction-token query
panel and seeded visual-token queries are secondary; some queries have different
eligible-key sets. Detailed comparisons always retain that full set. If keys are
subsampled to meet resources, label the result as routing over a sampled key panel;
do not report its maximum or entropy as full attention's maximum or entropy.

At each attention site observe the actual normalized input; raw Q/K/V projections;
Q/K after hybrid RoPE; attention-output sigmoid gate; ungated weighted V, gated
weighted V, and projected residual write when feasible. This checkpoint uses
12-head MHA with head dimension 128, no per-head Q/K normalization, and a sigmoid
output gate. Validate those facts against the loaded model. Hybrid RoPE includes
YaRN scaling; do not assume it preserves norms. PrefixLM is not an ordinary
triangular causal mask. Save the actual allowed keys.

Keep SDPA unchanged. Passive hooks on projections plus independently reconstructed
selected attention rows must not replace the model's attention implementation.
Validate reconstructed rows/output against a tiny audited reference calculation;
do not silently switch to eager attention just to obtain weights.
Define reconstructed routing rows using declared FP32 score/softmax arithmetic
from the actual projected tensors, masks and scale. These are reference routing
measurements; fused SDPA may round differently. Save the reference-versus-backend
output error and its calibrated tolerance. Do not claim access to bit-identical
internal SDPA probabilities or accept a failed reconstruction audit.

At MLP sites save gate projection, up projection, their actual SwiGLU product,
and down-projected output. The product is the channel activation. A positive gate
alone is not evidence that its product is large. For state geometry preserve
residual input, post-attention residual, post-MLP residual and final stack norm at
the declared sites, with exact semantic labels.

## 5. State dynamics and low-dimensional updates

For every declared update and token role, measure:

- Norm and relative update norm; cosine/angle to the previous state; norms of the
  two recurrent summands and their alignment; mean drift and variability.
- Alignment between successive updates; alignment with initial/current image
  representations and reinjection vectors. These are geometric associations,
  not evidence of semantic improvement.
- Changes in distances between examples, anisotropy, and paired representation
  similarity at a matched physical site. Show task/role-stratified distributions
  as well as means. Near-zero norms get explicit undefined-cosine flags.
- Delta spectra: uncentered and mean-centered singular values, stable rank
  `sum(s²)/max(s²)`, participation ratio `(sum(s²))²/sum(s⁴)`, and ranks explaining
  50/90/95% of observed energy. State ambient dimension, matrix row count and number
  of independent families. Constant/zero updates have defined flags, not NaNs.

Three different rank questions must be reported separately: within-example rank
across real tokens; rank across examples at a matched token role; and whether a
shared discovery-fitted subspace predicts updates of unseen families.

The strongest test uses a discovery mean `mu` and orthonormal update PCs `U_k`.
For held-out full-dimensional updates `d_e`, report

```text
R_k = 1 - sum_e w_e ||(d_e-mu) - U_k U_k^T(d_e-mu)||²
            / sum_e w_e ||d_e-mu||² .
```

This is held-out centered subspace reconstruction, not prediction of an update
from its input: the test update supplies its own projection coordinates. Do not
confuse it with the fraction of energy explained by the mean itself. Use ranks
1, 4, 8, 16, 32, 64 where supported;
rank 16 is the planned primary contrast, conditional on adequate discovery rows.
Report mean-only reconstruction separately. Evaluate transition-specific bases
and a shared basis, centering each transition by its own discovery mean before
pooling; form a train-transition × test-transition reconstruction matrix. Report
paired differences in held-out rank-16 capture versus random and state-PC bases.
With only 32 discovery families, correlated token/question rows do not establish
basis stability. Prespecify a discovery split-family/bootstrap stability and row
adequacy check; failure makes the rank-16 endpoint inconclusive rather than
switching to whichever rank looks best.

Controls: equal-rank seeded random subspaces; PCs of input states; shuffled
source/target family pairs matched on task/role/length; an explicitly regularized,
discovery-fitted state-whitened sensitivity analysis; and repeated-forward numerical
noise. Do not infer low rank from an in-sample SVD or a projection with few retained
coordinates. A random sketch only establishes properties of that sketch unless
full-space claims have a validated error bound. A low-rank injection matrix can
reflect repeated image vectors or pooling; compare injection and computation updates.

## 6. Attention geometry, distributions, and routing

For each matched physical head and query position, retain raw norms and angular
geometry separately. Compute paired Q/K/V cosine drift, norm ratios, per-coordinate
moments/covariance spectra on the declared vector panel, and between-example versus
within-example distances. Report distributions over independent examples, not a
single attractive heatmap.

For each actual allowed query/key row compute `s_ij = q_i·k_j / sqrt(head_dim)`
using actual post-RoPE Q/K and the loaded scale. Record:

- Mean, standard deviation, fixed quantiles, maximum and top-two gap of scores;
  cosine-score counterparts; Q and key norms; and the eligible-key count.
- Softmax entropy, entropy divided by `log(number_of_eligible_keys)` when greater
  than one, effective key count `exp(entropy)`, maximum probability and top-k mass.
- Top-key identity and its role, top-k overlap across visits, visual/instruction/
  self/position-band mass, and object-region mass when masks are available.
- Paired Jensen–Shannon divergence between attention rows over identical keys,
  using base-2 logs: `JS(p,q) = (KL2(p||m)+KL2(q||m))/2`, `m=(p+q)/2`, bounded
  between zero and one (not its square root);
  attention-output gate distribution; weighted-V and projected write norms.
  Higher concentration need not mean a larger or more useful write.

Use full eligible-key statistics for what the model did. For comparisons across
different examples/lengths also use a fixed-count, role/position-balanced key panel
chosen independently of scores. Report panel and full-set results separately:
maximum similarity grows with opportunities to find a match. Average per-query
quantiles are not quantiles of all pooled query-key scores. Save histogram counts
and edges if pooled distributions will be plotted.

Use four cross-visit combinations on the detailed panel: `(Q_s,K_s)`, `(Q_t,K_s)`,
`(Q_s,K_t)`, `(Q_t,K_t)`, with identical token identities/mask/scale. This distinguishes
query-side, key-side and joint geometric change. Similarly, holding routing fixed
while exchanging V and attention gates describes changes in the write. Such hybrid
calculations are offline diagnostics, not evidence that the model causally used
that hybrid. Only a separately validated forward patch can establish that effect.

Controls: identical-forward replay; same-position/role vectors from other families;
key-content permutations with positional rules explicitly preserved; self-key
included/excluded reports; and fixed role/length strata. Do not pool raw Q from
different physical heads as if they shared a coordinate system. Do not infer joint
QK behavior from unchanged Q and K marginal means alone.

## 7. Neuron population dynamics

For signed products `a_i`, retain mean, RMS, mean absolute magnitude, tails and
near-zero/sign frequencies for the declared populations. Preserve raw all-channel
last-prompt vectors, plus exact per-example/channel reductions for visual and
instruction roles; keep a seeded uniform channel sample for full token-level
distributions. Broad coverage must not become selection of only the most active units.

Use explicit definitions:

- Continuous strength: `abs(a_i)` and write magnitude `abs(a_i)*||down_column_i||`.
- Strong set: top 32 channels by absolute product, deterministic index tie-breaks;
  also show 8/64 as discovery sensitivity analyses.
- Threshold activity: `abs(a_i)>tau_i`, with thresholds fixed from discovery
  scale/numerical noise. Report positive and negative products separately.

Exact threshold prevalence requires retained raw activations or a threshold frozen
before streaming reduction. Use the raw channel/token panels for thresholds learned
after discovery capture. Broad all-channel reduced histograms require pilot-frozen
edges/thresholds, or explicitly report bin bounds; moments alone cannot reconstruct
arbitrary later thresholds or exact quantiles.

Measure strong-set turnover `1 - |top32_s ∩ top32_t|/32`, rank/activation correlation,
per-channel changes in activation probability, sign transitions, newly recruited
and persistently strong channels, and population participation. Report both
within-example turnover and changes in which channels dominate across the dataset.
Different top units in population means need not mean different units within a
matched example. Distinguish multiplicative rescaling from rank/recruitment changes.

Measure MLP write cancellation: total write norm versus the sum of individual
channel-write norms. A channel's large activation or write norm need not imply a
large net update or causal effect. Split pooled versus real-token summaries so
opposite signed token activations cannot disappear silently through averaging.

## 8. Does a neuron change its input so it later turns off?

Separate three hypotheses:

1. A channel is strong now and weaker at its next invocation (observation).
2. Its own write immediately reduces its readout under a hypothetical reapplication
   of the same MLP (local geometric diagnostic).
3. Increasing its write now lowers its activation at its **actual next physical
   reuse**, after intervening computation (causal effective negative feedback).

“Same neuron in the next layer” is ambiguous: index 17 in physical layers 5 and 6
is not the same neuron. Primary feedback follows one channel of one physical layer
to its next invocation. Adjacent-layer channel-to-channel effects are a separately
named cross-layer feedback analysis and require an explicit pair-selection rule.

Do not use `E[a_next-a_now | a_now is high] < 0` as causal evidence; selecting a high
initial value generates regression to the mean. Do not equate a negative raw
key/down-column dot product with feedback: SwiGLU has two input projections,
normalization is nonlinear, and attention/other blocks intervene.

Screen observation and local susceptibility on discovery only. Select at most four
candidate channels, plus four matched control channels, by a frozen rule with
minimum activation variability, repeated contexts, and nondegenerate write size.
Controls match physical layer, write norm, activation scale/prevalence and source
phase as closely as possible; save matching distances. Report the selection funnel.

At a fixed real-token position apply `+epsilon` and `-epsilon` to the actual
post-product activation at source invocation s, leaving the target invocation t
unmodified. Record all intervening state changes and the same channel's target
activation. The return response is

```text
J_ii(s→t) = (a_i_at_t[+epsilon] - a_i_at_t[-epsilon]) / (2*epsilon).
```

This expression uses the nominal dose for clarity; analysis divides by the actual
realized source-activation difference after dtype rounding.

This signed response alone does not mean the channel turns off: making a negative
product more negative increases its magnitude. Also run the **same paired edits**
with labels aligned to the baseline source sign: excitation increases source
`abs(a_i)` and attenuation decreases it. Record the finite contrast in target
`abs(a_i)`, divided by the realized source-magnitude contrast, and the change in
`I(abs(a_i_at_t) <= tau_i)`. A negative magnitude-return response supports reduced
activity; increased below-threshold probability supports operational “turning off.”
Zero/sign-ambiguous source units are ineligible under a discovery-frozen rule.
Report sign flips and both signed/magnitude results; do not rename a sign flip as
inactivity. These calculations reuse the ±dose forwards rather than double the budget.

Use discovery-calibrated doses, initially 0.25 times a robust activation scale;
report actual injected residual norm and reject out-of-range edits. Audit a smaller
dose on discovery for sign stability/nonlinearity. Norm-match control-channel edits,
which generally requires different activation doses. A zero-dose/self-replacement
sham must reproduce the clean trajectory. Both dose signs must be realized at the
model dtype, not round away. Preserve source token/mask and exact changed indices.

Test L1→L2 and L4→L5 separately within fixed high conditioning. Other next-L pairs
are secondary; L3→L4 crosses an H update and injection, and H1→H2 traverses the low
cycle. Label those longer paths explicitly. Plan at most 32 confirmation families
with one preselected balanced question each for the perturbation cohort. Freeze
the family/site/dose budget from measured runtime before confirmation.

For specificity, compare target channel i both after perturbing itself and after
perturbing matched control channel j. Measuring only j's own return is insufficient.
Record the full candidate/control source→target response matrix in every intervention
forward; this adds measurements, not forwards. Normalize cross-source specificity
comparisons by actual injected residual-write norm, with the rule locked in advance.
Determine dose eligibility from the clean source state, never the target response;
save every exclusion and denominator. Re-run replay/sham and matched-channel
controls on confirmation, not only discovery.

Also record other-channel responses, next-read gate/up components, and final
logits/behavior. Negative `J_ii` establishes a negative signed return response;
magnitude suppression and operational turning-off require their additional contrasts.
None of these establishes a neuron intending to suppress itself, a direct synapse,
or semantic inhibition. Shams, matched control neurons,
and dose sensitivity are mandatory for a feedback claim. If the causal executor
or budget is unavailable, report only observation/local geometry; feedback stays pending.

## 9. Visual reinjection as one explanatory control

First measure, without edits, cycle-specific alpha, per-token gates and injection
norms; alignment with pre-injection high, prior H update and subsequent L updates;
and changes in the metrics above immediately around the event. Record image and
instruction components used to calculate gates.

Then, on a bounded matched subset, compare clean, self-replaced injection, and
second-cycle-only zero injection. Keep the first injection, source image, question,
loop schedule, precision and all other operations unchanged. Add a norm-matched
within-image token-permutation injection control only if its extra forwards fit
the locked budget; label its distribution shift. The second-cycle delta is a total
effect of that intervention through subsequent computation, not a decomposition
of all L3→L4 differences. Gate/scale changes, H updating, and elapsed computation
remain separate alternative explanations for natural boundary drift.

An observation that changes persist without reinjection is not proof that H
conditioning is irrelevant. A zero-injection effect does not prove extra independent
visual information was obtained. This control explains one boundary; it does not
become the central project or expand into unrestricted image/anchor swapping.

## 10. Endpoints, discovery, confirmation and stopping

Broad exploration is allowed on discovery data, with every metric family registered
and labeled exploratory. Before opening confirmation activations, freeze the code,
data IDs, cohorts, site/token sampling, normalization, candidate selection, controls,
effect sizes of interest, output schemas, missingness rules and analysis script hash.
The new protocol is independent of the old semantic-transfer lock implementation.

Planned confirmation summaries, with exact transition/panel weights locked after
the timing/discovery pilot, are:

| Endpoint | Summary | Essential companion/control |
|---|---|---|
| Routing change | Family-weighted paired attention-row JS divergence at the last prompt, equal physical-layer/head weight | Replay floor, entropy/key-count/role decomposition |
| Recruitment change | Family-weighted top-32 absolute-product turnover at the same token/site | Activation scale, noise/ties, fixed physical channels |
| Update structure | Held-out centered update energy captured by rank-16 discovery PCs | Mean drift; equal-rank random and state-PC bases |
| Effective feedback | Sign-aligned source-magnitude → next-magnitude response for locked candidates and matched controls | Signed return response, off-threshold probability, both doses, shams and source-write norm |

Report the within-cycle transition profile and the outer-boundary profile separately;
do not average all phases into a single conclusion. H1↔H2 has its own label and
uncertainty. Rank and feedback endpoints can be inconclusive or unmeasurable; do
not replace them post hoc with a more favorable metric. The reinjection contrast
is a prespecified explanatory secondary analysis.

Use paired bootstrap of **whole families**, preserving all within-family questions,
tokens, objects and conditions; 2,000 seeded resamples and 95% intervals by default.
Fit means, PCs, thresholds, whitening and candidate rules on discovery only. Report
both aggregate and task-stratum effects. For formal tests across the locked four
endpoint families use Holm adjustment; exploratory per-head/neuron maps use an
explicit FDR family and remain exploratory. Candidate claims need simultaneous or
multiplicity-adjusted intervals, not only the most negative unit's unadjusted CI.
Near-zero noise-scale findings, wide intervals, and conflicting strata are honest
negative/inconclusive outcomes. Bootstrap intervals quantify sampled-family
uncertainty, not variability across trained checkpoints or unseen domains.

The declared inferential population is the sampled family design for this checkpoint.
Question weights are equal within family; site/head weights are equal within their
declared group. Keep transition profiles separate. Candidate weights and each
panel's weight are frozen in the lock. Report panels separately, or resample families
within panels under fixed panel weights; never treat the 128 questions from 32
families as 128 independent observations.

JS divergence and turnover are nonnegative: an ordinary percentile interval above
zero is not sufficient evidence of meaningful change. Lock practical margins above
replay/numerical variability in discovery. Formal change tests must address paired
excess over those margins with an appropriately centered family bootstrap/permutation
procedure. For rank, test paired differences from equal-rank random/state-PC controls;
for feedback, lock directional magnitude and specificity contrasts. Specify nulls,
resampling and the four multiplicity families in the protocol before confirmation.

Stop after one locked confirmation analysis and its declared controls. No extra
neurons, task families, loop schedules or dose searches because confirmation is
negative. Unsupported claims remain unsupported. Extra iterations beyond the
checkpoint's trained schedule and comparison with retrained untied models are
separate future studies, not automatic scope expansion here.

## 11. Raw artifacts, recomputation and limits

Use a new study/run ID, `loop-dynamics-v1` plus its frozen runtime variant, while
keeping the existing project-wide scheduler ledger and output root. Do not overwrite
or reinterpret historical raw measurements. Preserve model/source/checkpoint,
dependency lock, backend signature, dataset and protocol checksums.

Save per-example primitive measurements, exact token/vector panels, event manifests,
intervention doses and outcomes, numerical failures, and candidate selection tables.
Final metrics/figures must be recomputed by a CPU command from those saved artifacts
without loading the VLM. Publish a claim ledger with metric, evidence path/hash,
uncertainty, panel/task/site scope and remaining alternative explanations.

Broad exact reductions (counts, sums, squared sums, histograms, extrema, top-index
records) are raw measurements for their declared estimands; they do not preserve
every original tensor. Document which summaries can be recomputed from reduced
records and which can also be independently audited from complete tensor panels.
Do not claim arbitrary new post hoc distribution metrics are possible from moments.
Retain raw samples for exact quantile/distribution plots and raw full-dimensional
vectors for full-space low-rank claims. Store BF16 as lossless bits plus dtype tags;
preserve actual FP32 values where used. No silent FP16 quantization or model casts.

Use chunked, checksummed per-family artifacts with atomic commit and single-writer
locks. A checksum mismatch or missing event is a failure, not a skipped family.
Resume only with identical snapshot, study/config hashes, family assignment and
shard topology. Merge rejects duplicates, missing families/conditions, and stale fits.

## 12. Resources and Slurm execution design

The ceiling remains **30 GPU-hours total for this project**, including old jobs,
failed allocations, validation, interventions and retries. Recovery/new run IDs
do not reset it. Refresh actual sacct accounting before planning submissions.
There is no local GPU and no CPU fallback for model experiments.

Illustrative incremental allocation ceilings, subject to the actual remaining cap:

| Stage | Maximum GPU-hours |
|---|---:|
| Loaded numerical + new instrumentation validation | 2 |
| 32-example timing/competence pilot | 2 |
| Discovery observation | 4 |
| Locked confirmation observation | 4 |
| Feedback pilot and locked perturbations | 8 |
| Bounded reinjection controls | 2 |
| Reserved failure/retry margin | 3 |
| Total planned new ceiling | 25 |

These are scheduling ceilings, not measured runtime estimates or entitlements.
If prior consumption leaves less, reduce the design **before** its lock or complete
an explicitly observational study. Do not omit required controls yet claim causality.
Use measured full forward, instrumentation, transfer and serialization time, plus
model-load overhead, to bound every task. Reserve full requested walltime for every
array element regardless of concurrency. Unused headroom stays unused when done.

Count perturbation forwards explicitly: four candidates + four controls × two
source phases × two signs ×32 families =1,024 perturbed forwards, before discovery
dose checks, shams, clean baselines or behavioral continuation. A budget is not
feasible merely because there are few selected neurons. Reduce candidates/phases
or the confirmation cohort before locking if measured runtime requires it.

Research outputs remain at most **20 GiB across the shared project output root**,
including old outputs, temporary files, raw/merged copies and reports. An initial
new-stage target is ≤14 GiB final plus ≤4 GiB transient headroom, subject to existing
usage. Example category ceilings are 4 GiB broad measurements, 4 GiB vector panels,
4 GiB detailed QKV/state audits, 2 GiB perturbations/reports. Replace these with a
measured byte estimate after the 32-example pilot. If necessary reduce diagnostic
tokens/heads/examples before locking, not precision or quietly missing families.

For scale: 512 examples ×128 block visits ×8 tokens ×1536 coordinates ×2 bytes is
1.5 GiB for **one** BF16-width signal, before metadata and intermediate copies.
Full QK matrices scale quadratically with tokens and are excluded from broad
capture. Native dtypes, full key rows and channel widths must enter the byte model.
GPU kernels should reduce large tensors while resident; do not copy full tensors
to CPU just to discard them. Add a race-safe global byte-reservation ledger; the
current independent directory-size checks are insufficient for concurrent writers.

Recheck Brown's live associations/QoS/GRES before scheduling. The user reported
`gpu-debug/default/gpu-debug`, one `gpu:nvidia_rtx_a6000:1`, 8 CPUs, 32 GiB host RAM
and a one-hour QoS cap on 2026-10-16. Keep site settings configurable. Start with a
single scheduled GPU smoke. After validation and measured feasibility, use up to
four independent family shards, initially concurrency one; use two only within
current per-user CPU/GPU/memory limits. Keep one GPU per shard. CPU analysis/merge
runs on scheduled CPU nodes, not the login node. Under the supplied eight-CPU
per-user debug limit, two concurrent jobs require at most four CPUs each; jobs
retaining the eight-CPU default stay at concurrency one. Never submit dependent arrays
before gates exist; afterok is not a substitute for checking scientific validity.

After the timing pilot, external partition/time/CPU/memory/concurrency settings may
change within the validated backend and live limits. Changes to scientific cohorts,
token/site panels, precision or instrumentation require a new frozen snapshot/identity
and relevant validation. Discovery selection artifacts and the final protocol are
hashed runtime outputs consumed by frozen configs; do not edit queued snapshots
to install candidates or a lock. Same-backend selections allowed by the predeclared
policy are distinct from changing that policy.

## 13. Implementation work and acceptance gates

**None of the `dynamics-*` commands below exists yet.** This document and its JSON
specification are the requested plan. Existing scripts cannot collect the proposed
QKV/H/state panels simply by changing a config. Implement the following milestones
in order before presenting the future command block as runnable.

| Milestone | Concrete changes | Acceptance evidence |
|---|---|---|
| M0 CPU metrics/fixtures | New `src/reuseinterp/dynamics/` package: event schemas, QKV/routing, population, state/rank, feedback analyses and raw package/reporter | Known constant/rotating/rescaled distributions; low/full-rank and mean-only updates; routing changes with fixed marginals; no-op/negative/positive feedback; missingness/corruption/resume tests |
| M1 instrumentation | Event observer for initialization/reinjection/L/H; projection/block hooks; lossless panel writer; exact selected-row attention reconstruction | CPU toy contract plus allocated-GPU hook/no-hook and self-replacement parity, event counts, dtype/device/position/mask checks, bounded memory |
| M2 executable workflow | Flat `configs/dynamics-*.yaml`; new study fingerprint covering measurement policy; fresh snapshot using existing environment; scheduler/CLI stages and gates; race-safe output reservations | Dry-runs, full reservation accounting, negative gate tests, SIGUSR1 interruption/resume equal to uninterrupted output, no confirmation IDs in discovery |
| M3 timed observation | 32-example timed job; raw-storage estimate; competence summary; bounded discovery capture and CPU report | Real timing/VRAM/bytes and all planned event/panel validity checks; no invented feasibility |
| M4 targeted pilot + lock | Validated product ±dose and second-injection-only edits; discovery candidate/threshold/subspace fits; dynamics protocol lock | Exact changed sites; shams; realized/natural-range doses; planned sample and controls fit remaining resource cap |
| M5 confirmation + package | Locked observation/intervention collection, CPU estimates/intervals/figures and claim ledger | No fitting/selection on confirmation; every reported metric/figure reproduces; scope/negative findings and failures recorded |

Implement CPU fixtures that specifically catch: maxima changing only because key
count changes; Q/K marginals unchanged while paired routing changes; activation
rescaling versus neuron turnover; signed SwiGLU cancellation; normalization-only
state differences; rank limited by sample count; discovery-low-rank/test-full-rank
updates; regression-to-mean without feedback; and genuine signed effective feedback.
New hooks must be observational by default and removable even on an exception.

Current reusable pieces: pinned official loader/processor, mixed precision, checksum
verification, scene archive, basic product hooks, family shards, snapshot verifier,
Slurm GPU preflight and accounting. Current limitations: product capture samples
only L channels, state capture is only two L physical layers, H/QKV/reinjection
capture is absent, no new dynamics protocol/reporter exists, and causal collection
is not implemented. `discover-causal` fails closed; the old `capture`/`fit`/`lock`
stages must not be relabeled as completion of this plan.

Add these future interfaces to `reuseinterp.cli`, submitter stage sets/array rules,
GPU/CPU wrappers, configuration freezing and runtime identity together:

- GPU: `dynamics-validate`, `dynamics-pilot`, `dynamics-capture`, `dynamics-intervene`.
- CPU: `dynamics-merge`, `dynamics-analyze`, `dynamics-lock`, `dynamics-report`,
  `dynamics-verify`.
- Snapshot preparation: `cluster_setup.py snapshot-study --from-profile ...`
  creates new preparation/snapshot/env files under the same base, reuses the
  validated immutable venv and staged weights, preserves the canonical ledgers,
  and rejects queued/running mutations or incompatible runtime dependencies.

The protocol is a separate signed/hash-verified dynamics lock, not the historical
semantic-transfer draft. Confirmation configurations must require that exact lock;
stages must reject wrong panel/split/measurement/site/fingerprint hashes. Analysis
cohorts and stage operation are part of stage keys so different conditions cannot
silently collide or bypass retry guards.

## 14. Cluster commands: current versus future

**Available now**, on Oscar in the existing checkout, to inspect and repair setup:

```bash
export REUSE_BASE="$HOME/reuseinterp-loopvl"
cd "$REUSE_BASE/source"
git pull --ff-only origin main
python3 scripts/cluster_setup.py status --base-dir "$REUSE_BASE"
# If recovery already exists, inspect its separate state as well.
python3 scripts/cluster_setup.py status --base-dir "$REUSE_BASE" --profile cu126
```

Follow [CUDA_RECOVERY.md](docs/CUDA_RECOVERY.md) if still at job 7489067's failure.
If the `cu126` profile is already frozen, inspect it rather than running recovery
again. Activate the actual selected profile and reconcile the shared ledger:

```bash
source "$REUSE_BASE/cluster-cu126.env"
source "$REUSE_ENV_DIR/bin/activate"
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
sacctmgr -nP show qos where name=gpu-debug format=Name%24,MaxWall,MaxTRESPU%80
sinfo -p gpu-debug -N -o "%N %G %f"
```

**Future interface contract, NOT runnable until M0–M2 are implemented and tested:**

```bash
# Fresh source snapshot, same validated runtime and project ledgers.
python scripts/cluster_setup.py snapshot-study --base-dir "$REUSE_BASE" \
  --from-profile cu126 --name dynamics-v1
source "$REUSE_BASE/cluster-dynamics-v1.env"
source "$REUSE_ENV_DIR/bin/activate"

python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-validate \
  --config configs/dynamics-smoke.yaml --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-validate \
  --config configs/dynamics-smoke.yaml

# WAIT: successful numerical + instrumentation gates, then submit one timed pilot.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-pilot \
  --config configs/dynamics-pilot.yaml

# WAIT: valid 32-example timing/byte report and gate; configure measured requests.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-capture \
  --config configs/dynamics-discovery.yaml --num-shards 4 --concurrency 1 --resume --dry-run
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-capture \
  --config configs/dynamics-discovery.yaml --num-shards 4 --concurrency 1 --resume

# WAIT: all discovery shards complete; merge/analyze on scheduled CPU nodes.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-merge \
  --config configs/dynamics-discovery.yaml --num-shards 4
# WAIT for merge success.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-analyze \
  --config configs/dynamics-discovery.yaml

# WAIT for discovery analysis; run bounded causal pilot, then lock on CPU.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-intervene \
  --config configs/dynamics-feedback-pilot.yaml
# WAIT for pilot completion; merge singleton feedback and analyze on CPU.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-merge \
  --config configs/dynamics-feedback-pilot.yaml --num-shards 1
# WAIT for merge.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-analyze \
  --config configs/dynamics-feedback-pilot.yaml
# WAIT for pilot analysis, acceptance and budget checks.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-lock \
  --config configs/dynamics-lock.yaml

# WAIT for verified lock; refresh accounting before every further GPU submission.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" account --refresh
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-capture \
  --config configs/dynamics-confirmation.yaml --num-shards 4 --concurrency 1 --resume
# WAIT for confirmed capture before independent locked perturbations.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-intervene \
  --config configs/dynamics-feedback-confirmation.yaml
# WAIT; independently merge and analyze singleton perturbation families on CPU.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-merge \
  --config configs/dynamics-feedback-confirmation.yaml --num-shards 1
# WAIT for intervention merge.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-analyze \
  --config configs/dynamics-feedback-confirmation.yaml
# WAIT; merge/analyze observational confirmation with frozen discovery fits on CPU.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-merge \
  --config configs/dynamics-confirmation.yaml --num-shards 4
# WAIT for merge.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-analyze \
  --config configs/dynamics-confirmation.yaml
# WAIT for final analysis, then build and independently verify the CPU package.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-report \
  --config configs/dynamics-report.yaml
# WAIT for report.
python "$REUSE_PROJECT_DIR/scripts/slurm/submit.py" dynamics-verify \
  --config configs/dynamics-report.yaml
```

Configs for feedback must include the bounded reinjection cohort and every mandatory
sham/control; the implementation must expose their completed counts in analysis.
Final reporting must consume and checksum both observational and intervention
committed/merged manifests, not just the observational confirmation directory.
The implementation handoff must replace this contract with tested, copyable commands,
exact config filenames, expected artifact paths and exit-code/gate checks. Do not
paste the whole sequence as one asynchronous submission script. No fresh GPU
measurement or result figure is asserted by this plan.

## Source evidence

Architecture statements above come from the inspected pinned implementation, not
from a memory analogy: [recurrence](https://github.com/Tier-Flow/LoopVL/blob/6c14bed1d62fbecb559d5902459c6093c38da405/runtime/hrm_penguin/recurrent/hrm_multimodal.py),
[visual anchor](https://github.com/Tier-Flow/LoopVL/blob/6c14bed1d62fbecb559d5902459c6093c38da405/runtime/hrm_penguin/recurrent/visual_anchor.py),
[visual gate](https://github.com/Tier-Flow/LoopVL/blob/6c14bed1d62fbecb559d5902459c6093c38da405/runtime/hrm_penguin/recurrent/visual_gate.py),
and the locally preserved [HRM attention/MLP source](evidence/modeling_hrm_text.py)
and [checkpoint configuration](evidence/checkpoint_config.json). Loaded checks
override assumptions and fail closed if these contracts differ.
