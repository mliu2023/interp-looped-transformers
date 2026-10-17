#!/usr/bin/env bash
# Run once on a Brown node permitted to download/install user-space packages.
# Does not submit jobs, run inference, or modify system software/drivers.
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "Usage: bash scripts/brown_setup.sh /absolute/approved/shared/study-directory" >&2
  exit 2
fi
[[ "$(uname -s)" == Linux ]] || { echo "Run this setup on the Linux cluster." >&2; exit 2; }
REUSE_SETUP_BASE="$1"
REUSE_SETUP_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mkdir -p "$REUSE_SETUP_BASE"
REUSE_SETUP_BASE="$(cd "$REUSE_SETUP_BASE" && pwd -P)"
export PIP_CACHE_DIR="$REUSE_SETUP_BASE/pip-cache"
export UV_CACHE_DIR="$REUSE_SETUP_BASE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$REUSE_SETUP_BASE/python"

# A managed interpreter avoids guessing Brown module names or carrying login
# module state into jobs. Use the university environment instead if site policy
# requires it; the Python helper supports --python and --site-env directly.
if [[ ! -x "$REUSE_SETUP_BASE/bootstrap/bin/python" ]]; then
  python3 -m venv "$REUSE_SETUP_BASE/bootstrap"
fi
"$REUSE_SETUP_BASE/bootstrap/bin/python" -m pip install 'uv==0.13.1'
"$REUSE_SETUP_BASE/bootstrap/bin/uv" python install 3.12
REUSE_SETUP_PYTHON="$("$REUSE_SETUP_BASE/bootstrap/bin/uv" python find --managed-python 3.12)"

"$REUSE_SETUP_PYTHON" "$REUSE_SETUP_SOURCE/scripts/cluster_setup.py" configure \
  --base-dir "$REUSE_SETUP_BASE" --source-dir "$REUSE_SETUP_SOURCE" --python "$REUSE_SETUP_PYTHON"
"$REUSE_SETUP_PYTHON" "$REUSE_SETUP_SOURCE/scripts/cluster_setup.py" install \
  --base-dir "$REUSE_SETUP_BASE" \
  --torch-index-url "${REUSE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  --torch-version "${REUSE_TORCH_VERSION:-2.8.0+cu128}" \
  --torchvision-version "${REUSE_TORCHVISION_VERSION:-0.23.0+cu128}"
"$REUSE_SETUP_BASE/venv/bin/python" "$REUSE_SETUP_SOURCE/scripts/cluster_setup.py" stage \
  --base-dir "$REUSE_SETUP_BASE"
"$REUSE_SETUP_BASE/venv/bin/python" "$REUSE_SETUP_SOURCE/scripts/cluster_setup.py" freeze \
  --base-dir "$REUSE_SETUP_BASE"
"$REUSE_SETUP_BASE/venv/bin/python" "$REUSE_SETUP_SOURCE/scripts/cluster_setup.py" status \
  --base-dir "$REUSE_SETUP_BASE"
echo "Setup finished. Source $REUSE_SETUP_BASE/cluster.env, activate its venv, and submit the singleton validate job."
