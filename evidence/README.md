# Public source evidence

Fetched 2026-10-16 from official GitHub/Hugging Face endpoints. No credentials,
private files, model weights, or published measurements are included.

`upstream_commit.json` pins the official commit. `checkpoint_metadata.json` and
`checkpoint_files.json` pin the actual checkpoint revision and Hub LFS byte hash.
`checkpoint/` contains the pinned small model/tokenizer/processor metadata.
`modeling_hrm_text.py` is the upstream Transformers 5.15.1 language implementation,
retaining its Apache license header. `checkpoint_config.json` duplicates the
readable configuration used for the source audit.

These artifacts verify source/metadata, not fresh model behavior. The staged weight checks and loaded graph evidence are recorded separately in
`../recovery-cu128-20261016/` and `../runs/` relative to the editable checkout.
See [current progress](../PROGRESS.md) for their exact scope and remaining gates.
