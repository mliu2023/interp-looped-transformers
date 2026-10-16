#!/usr/bin/env bash
#SBATCH --job-name=reuseinterp-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120
set -euo pipefail
: "${SLURM_JOB_ID:?Use a scheduled CPU allocation}"
cd "${REUSE_PROJECT_DIR:?}"
source scripts/slurm/environment.sh
srun --ntasks=1 python -u -m reuseinterp.cli "$@"
