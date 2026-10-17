# New cluster installation reference

The current study is already installed at `/oscar/home/mliu171/reuseinterp-loopvl`. Continue with the [cluster operating guide](CLUSTER_RUNBOOK.md). Do not clone, download, or rebuild the environment to resume normal work.

Use this reference only when creating a different installation or replacing a failed environment deliberately.

1. Select an approved base with enough storage, inspect site instructions, and configure actual scheduler associations. The examples are `configs/cluster.brown.env.example` and `configs/cluster.env.example`.
2. Inspect `scripts/brown_setup.sh` and `scripts/cluster_setup.py`. Their phases are configure, install, stage, and freeze; `--help` documents the current arguments. Installation, downloads, substantial data checks, and tests must run on a node permitted by site policy, through a reviewed scheduled CPU job where required.
3. Stage the exact model/code revisions and packaged scene dataset, retain byte checks, and create a new dependency lock and immutable snapshot. Existing inputs and snapshots are not overwritten.
4. Use the generated environment file and editable submitter. A new environment/backend requires a scheduled CUDA preflight and numerical validation before research collection.

The current validated environment uses torch 2.8.0+cu128 and torchvision 0.23.0+cu128 on the assigned A6000. A different machine still needs its own runtime validation. Source, checkpoint, precision, and data requirements are in the [architecture audit](architecture_audit.md) and [scientific protocol](../CODEX_PLAN.md).

The [archived setup guide](archive/20261017/BROWN_QUICKSTART.md) preserves the original bootstrap recipe for reference. Its login, transfer, status, and submission sequence are historical; the operating guide is the current workflow.
