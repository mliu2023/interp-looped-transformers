#!/usr/bin/env bash
# CPU-only session for an already exported, verified activity bundle.
# Submit with the frozen project/environment variables from its experiment.
#SBATCH --job-name=loopvl-viewer
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Start the viewer in a scheduled CPU allocation}"
: "${REUSE_PROJECT_DIR:?Set the frozen project snapshot}"
: "${LOOPVL_VIEWER_BUNDLE:?Set the exported bundle directory}"
cd "$REUSE_PROJECT_DIR"
source scripts/slurm/environment.sh
mkdir -p "$REUSE_RUN_ROOT/viewer-sessions"
srun --ntasks=1 python -u scripts/serve_viewer.py serve \
  --bundle-dir "$LOOPVL_VIEWER_BUNDLE" --host 0.0.0.0 \
  --port "${LOOPVL_VIEWER_PORT:-8765}" \
  --ready-file "$REUSE_RUN_ROOT/viewer-sessions/$SLURM_JOB_ID.json"
