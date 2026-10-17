#!/usr/bin/env bash
#SBATCH --job-name=reuseinterp-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Environment recovery must run in a scheduled CPU allocation}"
: "${REUSE_RECOVERY_BASE:?Set the existing absolute study base directory}"
: "${REUSE_RECOVERY_NAME:=recovery-cu128-20261016}"
[[ "$REUSE_RECOVERY_BASE" = /* ]] || exit 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$REUSE_RECOVERY_BASE/pip-cache"
export UV_CACHE_DIR="$REUSE_RECOVERY_BASE/uv-cache"
export CUDA_VISIBLE_DEVICES=""
cd "$REUSE_RECOVERY_BASE/source"
exec python3 -u scripts/recover_cuda12.py --base-dir "$REUSE_RECOVERY_BASE" --name "$REUSE_RECOVERY_NAME"
