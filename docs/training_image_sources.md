# LoopVL image-source audit and bounded pilot pool

The new pilot uses original images embedded in LoopVL's reported training
corpora. It does not reuse the generated-shape dataset. It also does not treat
public source membership as verified consumption by the released checkpoint.

The [LoopVL paper, sections 3–4](https://arxiv.org/html/2609.38426#S3), describes
LLaVA-559K alignment, OneVision-1.5 Mid-Training-85M, and subsequent supervised
fine-tuning on OneVision-1.5 Instruct-22M, a spatial mixture, LLaVA Mix665K, and
Vision-R1-Cold, followed by Vision-R1 reinforcement learning. The recurrent
backbone is frozen during alignment and trainable during mid-training. The paper
also describes filtering and deduplication, so a public source row need not be a
retained training row.

The local pinned repository/model card and weight-packing manifest contain no
retained image-ID list or record-level optimizer log. They also do not establish
whether hub revision `92aa28bc678f523f8641fdb2f9e5159f614d1c02` corresponds exactly
to the base, SFT, or final post-RL stage. The current pool therefore uses the
status **`reported_corpus_member`**, with
`checkpoint_retained_record_membership_verified: false`. It supports a
training-source activation pilot; it cannot establish historical memorization
of an optimizer-consumed record. No unrelated evaluation images substitute for
the requested training sources.

The staging adapter pins the original
[OneVision-1.5 Instruct release](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data/tree/0efb7ad47f8ba36e1d262545cadec4cd8960dc5f)
at revision `0efb7ad47f8ba36e1d262545cadec4cd8960dc5f`. Natural-photograph
components are selected before inference. The first pinned shard's bounded row
groups are shuffled with seed `20261017`; decoded-image and conservative
perceptual-hash checks reject repeated families across components. This is a
small convenience sample, not a representative estimate of the training mixture.
The completed manifest records the actual component quotas and all skipped rows.

Staging runs in a scheduled CPU job with optional PyArrow 21.0.0 isolated from
the validated GPU environment. `scripts/stage_training_images.py --inspect-only`
fetches at most 16 MiB of Parquet footers and prints compressed row-group costs
before image retrieval. The staging command uses validated HTTP Range responses,
a bounded 2 MiB footer cache, a shared 500 MiB payload cap, and no full-shard
fallback. Its manifest records actual received bytes, selected row groups, the
immutable shard identity, row index and original source ID, byte/RGB hashes,
native dimensions, and source-paper links. A shard's published full hash is
recorded as metadata; it is explicitly not claimed to have been verified when
only ranges were fetched.

Original image bytes and source conversation metadata are retained separately.
The common experimental prompt receives the image, never a stored reference
answer. The adapter validates every original's hash, dimensions, membership label,
source pins, family uniqueness, and manifest fingerprint before capture. Failure
leaves an `incomplete_audit.json` and no completed manifest. Exact sampled images,
counts, transfer costs, and capture results belong in the experiment report after
the scheduled jobs finish.
