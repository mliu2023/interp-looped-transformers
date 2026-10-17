# LoopVL architecture audit

Source audit performed October 16, 2026; loaded model evidence reviewed October 17. [PROGRESS.md](../PROGRESS.md) records current workflow status, and the [cluster runbook](CLUSTER_RUNBOOK.md) contains operating commands.

The [loaded architecture audit](../../runs/loopvl-transfer-v1/loaded_architecture_audit.json) verifies structure and records parameters for all 16 L and 16 H MLP sites. The [numerical smoke](../../runs/loopvl-transfer-v1/parity_report.json) passed 64 checks on 16 examples at diagnostic layer 0, channel 0. Broader site validation, object alignment, and candidate specific causal checks remain separate requirements.

## Evidence and pinning

- [Official source](https://github.com/Tier-Flow/LoopVL/tree/6c14bed1d62fbecb559d5902459c6093c38da405),
  locally checked out at the exact revision specified in the plan. Public commit
  metadata is saved in `evidence/upstream_commit.json`.
- [Checkpoint](https://huggingface.co/TierFlow/LoopVL/tree/92aa28bc678f523f8641fdb2f9e5159f614d1c02),
  revision read from the public API. `evidence/checkpoint_metadata.json`,
  `checkpoint_files.json`, and `checkpoint/` preserve metadata and small inputs.
  Expected model.safetensors SHA-256 is
  `4d34dac46592a2ea273556fa29a540de9af8d8e3cff39f6e3e0ba54cd799f72f`.
  This comes from Hub LFS metadata; recovery staging records downloaded-byte
  verification, and the loaded audit records the checkpoint hashes.
- [Transformers 5.15.1 language implementation](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/hrm_text/modeling_hrm_text.py)
  is preserved in `evidence/modeling_hrm_text.py` with its upstream license header.
- The CPU static audit saves SHA-256 hashes of the actual source files inspected.
  This is distinct from the loaded-model audit produced by the scheduled smoke.

## Neuron and recurrence contract

`HrmTextMLP.forward` (saved source lines 70–83) uses
`down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The primary neuron is that actual
product channel. The pre-hook on `down_proj` captures/intervenes without replacing
the MLP implementation. Gate/up branches serve reconstruction diagnostics.
The loaded audit verifies `down_proj` as a linear module at every MLP site; reconstruction includes its bias if present.

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

## Loaded model evidence and remaining checks

The saved smoke establishes official first prediction and greedy answer parity for capture, self replacement, and zero dose on 16 varied examples. It reconstructs gate times up and product times down projection, checks shared parameter identity across L invocations, and verifies one local change at L4 for the diagnostic layer 0/channel 0. Capture/self/zero logit errors were zero in that run, within the frozen 1e-6 absolute and margin tolerances.

The loaded structure audit covers all 32 physical MLP sites, but the numerical hook/intervention test covers only that diagnostic site. Extend real checks as needed for selected layers, channels, token roles, and intervention conditions. A changed backend requires a new parity check.

Still required: actual object mask/grid alignment, candidate donor/recipient alignment, full pilot runtime and peak memory, real interruption/resume parity, and the candidate specific short horizon causal collector. The basic numerical gate does not authorize donor or causal conclusions. Synthetic tests and source inspection remain supporting engineering evidence.
