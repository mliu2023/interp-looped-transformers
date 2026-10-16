#!/usr/bin/env bash
# Sourced by scheduled jobs. Keep site configuration outside the immutable code.
set -euo pipefail
for name in REUSE_PROJECT_DIR REUSE_ENV_DIR REUSE_RUN_ROOT REUSE_CACHE_ROOT; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" != /* || "$value" == *REPLACE* ]]; then
    echo "Invalid absolute runtime path: $name" >&2
    return 2
  fi
done
if [[ -n "${REUSE_SITE_ENV:-}" ]]; then
  [[ "$REUSE_SITE_ENV" = /* && -f "$REUSE_SITE_ENV" ]] || { echo "Invalid REUSE_SITE_ENV" >&2; return 2; }
  source "$REUSE_SITE_ENV"
fi
[[ -f "$REUSE_ENV_DIR/bin/activate" ]] || { echo "Missing cluster environment" >&2; return 2; }
source "$REUSE_ENV_DIR/bin/activate"
export PYTHONPATH="$REUSE_PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REUSE_CACHE_ROOT"
export PYTHONUNBUFFERED=1
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
if [[ "${REUSE_CACHE_READY:-0}" == 1 ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$REUSE_RUN_ROOT"
python "$REUSE_PROJECT_DIR/scripts/slurm/freeze.py" --verify "$REUSE_PROJECT_DIR" --verify-environment --expected-sha256 "${REUSE_EXPECTED_SNAPSHOT_SHA256:?Submit through the snapshot-aware wrapper}"
