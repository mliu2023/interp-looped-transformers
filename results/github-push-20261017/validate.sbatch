#!/usr/bin/env bash
#SBATCH --job-name=reuse-push-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --account=default
set -euo pipefail
source /oscar/home/mliu171/reuseinterp-loopvl/recovery-cu128-20261016/cluster.env
export REUSE_PROJECT_DIR=/oscar/home/mliu171/reuseinterp-loopvl/experiments/github-push-20261017/snapshot
export REUSE_EXPECTED_SNAPSHOT_SHA256=ac4675ea32bb93c5d12e8ff6cac8bac4dc4593dbcc0e8e20288542d3a522d4f9
cd "$REUSE_PROJECT_DIR"
source scripts/slurm/environment.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --ntasks=1 python -m unittest discover -s tests -v
srun --ntasks=1 python -m reuseinterp.poles --run-dir /oscar/home/mliu171/reuseinterp-loopvl/source/results/poles-pilot-20261017
