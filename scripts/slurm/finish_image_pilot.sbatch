#!/usr/bin/env bash
# Either cost/submit after timing, or verify/export after full capture.
#SBATCH --job-name=loopvl-image-pipeline
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use a scheduled CPU allocation}"
cd "${REUSE_PROJECT_DIR:?Set the frozen project snapshot}"
source scripts/slurm/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --ntasks=1 python -u scripts/finish_image_pilot.py "$@"
