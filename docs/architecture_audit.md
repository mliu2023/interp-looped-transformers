# LoopVL architecture audit — source verified, loaded checks pending

Audit date: 2026-10-16. No model weights were downloaded or loaded locally.
There is no local CUDA GPU. The repository initially contained only CODEX_PLAN.md;
no applicable AGENTS.md or existing experiments were found.

## Evidence and pinning

- [Official source](https://github.com/Tier-Flow/LoopVL/tree/6c14bed1d62fbecb559d5902459c6093c38da405),
  locally checked out at the exact revision specified in the plan. Public commit
  metadata is saved in `evidence/upstream_commit.json`.
- [Checkpoint](https://huggingface.co/TierFlow/LoopVL/tree/92aa28bc678f523f8641fdb2f9e5159f614d1c02),
  revision read from the public API. `evidence/checkpoint_metadata.json`,
  `checkpoint_files.json`, and `checkpoint/` preserve metadata and small inputs.
  Expected model.safetensors SHA-256 is
  `4d34dac46592a2ea273556fa29a540de9af8d8e3cff39f6e3e0ba54cd799f72f`.
  This comes from Hub LFS metadata; verification of downloaded bytes is pending.
- [Transformers 5.15.1 language implementation](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/hrm_text/modeling_hrm_text.py)
  is preserved in `evidence/modeling_hrm_text.py` with its upstream license header.
- The CPU static audit saves SHA-256 hashes of the actual source files inspected.
  This is distinct from the loaded-model audit produced by the scheduled smoke.

## Neuron and recurrence contract

`HrmTextMLP.forward` (saved source lines 70–83) uses
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The primary neuron is that actual
product channel. The pre-hook on `down_proj` captures/intervenes without replacing
the MLP implementation. Gate/up branches serve reconstruction diagnostics.
`down_proj` is audited as a linear module after loading; its bias is included.

`HrmTextDecoderLayer` (238–278) RMS-normalizes before attention and before the
MLP and adds residuals after both sublayers. `HrmTextRMSNorm` computes RMS in FP32
and restores input dtype; it has no learned affine weight. `HrmTextStack` (281–310)
contains 16 physical blocks and a final RMS normalization.

`runtime/hrm_penguin/recurrent/hrm_multimodal.py` (315–391) calls the **same**
`L_module` three times per outer cycle and `H_module` once, for two outer cycles:
L1,L2,L3,H1,L4,L5,L6,H2. L and H are different physical stacks. Primary comparisons
L1/L4 match positions within the outer cycle. The mutable cycle offset is passed
to attention/cache bookkeeping; this is not a separate MLP parameter set.

The high state begins at scaled multimodal embeddings; the low state begins at
the frozen `z_L_init`. Each low update receives low+high; each high update receives
high+low. A cloned initial visual anchor is fixed for that forward. The current
high visual tokens and instruction state feed a visual gate; cycle-indexed anchor
scales reinject the fixed anchor. These are recorded context/recurrence facts,
not proposed primary experimental manipulations.

## Masking, positions, preprocessing, and generation

PrefixLM mask creation uses `token_type_ids` for the visual/instruction prefix.
Hybrid 2D RoPE uses compact visual grid positions and separate response positions.
The official batch constructor emits one frame and a validated merged grid.
The checkpoint selects the dynamic processor: full-image resize without padding,
merge size 1, 256–2048 visual tokens. Mask projection uses measured grid dimensions
and row-major patch order, rejects crop/letterbox/EXIF changes or other merge sizes,
and saves geometric occupancy plus normalized per-object weights. RGB bicubic
mixing and contextualization mean this does not prove object-local features.
Runtime mask/grid alignment remains a GPU acceptance check.

`runtime/hrm_penguin/generation.py` recomputes the entire recurrent sequence for
every generated token (`use_cache=False`). The adapter uses the official loader
and batch constructor, reproduces this path, and captures its first output-head
logits independently for parity. Capture counters reset at every recurrent model
forward. Primary activations have decode_step 0, before any answer tokens. Full
conditional candidate-string scoring occurs in separate forwards. The reference
answer is never supplied to preparation/capture or free generation.

The official loader preserves BF16 core vision/language and FP32 projector,
visual gate, and visual anchor. It checks stored tensor dtype/shape before loading.
No whole-model dtype conversion, quantization, altered loop schedule, or training
is introduced. Every model stage requires Slurm CUDA availability.

## Required loaded-model acceptance

Pending: verify all loaded sites, parameter objects/storage/hashes at each matched
invocation; reconstruct gate×up and product×down(+bias); 16 varied official smoke
examples; actual invocation counts on every forward; independent official logits
and generated-answer parity for capture/self/zero; exact local changed indices;
donor alignment; actual model device placement; peak memory and runtime; real
interruption/resumption parity. Synthetic toy-model checks cannot establish any
of these for LoopVL.

The smoke freezes tolerance from repeated official forwards before interventions,
with absolute/margin floors of 1e-6 and caps 1e-4/1e-5. It saves absolute/relative
errors, margin token IDs/errors, raw logits, output tokens, hook counts and exact
changed coordinates. Any answer change fails. Backend changes require new parity.
Candidate-specific donors and short-horizon causal collection remain pending;
no donor/causal gate is claimed by basic numerical smoke.
