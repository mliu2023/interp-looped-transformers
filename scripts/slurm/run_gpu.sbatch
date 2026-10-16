#!/usr/bin/env bash
#SBATCH --job-name=reuseinterp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
set -euo pipefail
: "${SLURM_JOB_ID:?GPU experiments must run in a Slurm allocation}"
cd "${REUSE_PROJECT_DIR:?}"
source scripts/slurm/environment.sh
[[ "${REUSE_CACHE_READY:-0}" == 1 ]] || { echo "Complete and verify the offline model cache first" >&2; exit 2; }
srun --ntasks=1 python -u scripts/cluster_preflight.py
srun --ntasks=1 python -u -m reuseinterp.cli "$@"
