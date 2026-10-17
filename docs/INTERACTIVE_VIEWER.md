# Explore saved LoopVL image responses

The activity explorer runs on a scheduled CPU node and opens in your local browser through an SSH tunnel. It reads saved images and activations; browsing never loads the model or uses a GPU. All JavaScript, styles, and plots are served locally, with no CDN dependency.

Choose a physical L layer, one of its sampled channels, a captured stream, and the invocation used to rank images. Click a high- or low-response image to see its signed spatial responses at L1–L6. The trajectory shows that image's spatial mean at each invocation; the scatter compares every image's response rank at L1 with another invocation. Scatter points are also clickable. Changing the stream distinguishes the pre-activation gate, effective gate, value branch, and gated product entering `down_proj`.

The blue/red overlay has one symmetric scale across **every image and all six invocations** for the selected channel and stream. Opacity is adjustable. Native aspect ratio and EXIF orientation are preserved. The overlay interpolates the saved row-major visual-token grid over the displayed image; it is not pixel attribution. Original source bytes and raw per-image captures can be downloaded from the selected-image panel.

Source evidence, original image hashes, the exact prompt, capture-validation evidence, and limitations are visible in the interface. High/low tails can have the same sign. Signs and rankings alone do not establish semantic poles, circuits, memorization, or superposition. A run labeled `training-source` can contain reported-corpus members whose exact consumption by this checkpoint remains unverified; inspect each image's membership evidence. The older shapes run is explicitly labeled historical.

## Create a portable bundle

Use the existing environment and run this export in a CPU allocation after collection finishes. It requires a complete `capture_index.json`, `protocol.json`, `image_rows.json`, and raw captures. It verifies recorded image/raw hashes and recomputes all four streams' spatial means directly from the raw arrays. It rejects incompatible token grids, cropped/padded layouts, missing records, or nonfinite activations.

```bash
cd /oscar/home/mliu171/reuseinterp-loopvl
source recovery-cu128-20261016/cluster.env
source "$REUSE_ENV_DIR/bin/activate"
python "$REUSE_SOURCE_DIR/scripts/serve_viewer.py" export \
  --run-dir /absolute/path/to/completed-real-image-run \
  --dataset-dir /absolute/path/to/root/of/image_path/values \
  --output-dir /absolute/path/to/new-viewer-bundle \
  --kind training-source
```

If original images are already under the run directory, omit `--dataset-dir`. The exporter never overwrites an existing bundle. Keep the bundle in retained experiment storage, outside Git: it contains original images, source metadata, and raw activations. A new export is required after a new experiment. Dataset-wide scales are computed during export rather than rescaled independently per panel.

To explore the historical shapes pilot, export its complete raw run with `--kind historical-shapes` and its original image root. The compact report committed to Git does not contain the raw activations or all original images and cannot by itself supply the spatial explorer.

## Start a scheduled CPU server

Use the reusable [CPU viewer script](../scripts/slurm/serve_viewer.sbatch). It requests one CPU, 4 GiB of memory, and four hours on `batch`, verifies the frozen source and environment, then serves the exported bundle. A viewer session needs no GPU reservation.

On Oscar, source the environment first, then set `REUSE_PROJECT_DIR` to the experiment's **full frozen source snapshot** and `REUSE_EXPECTED_SNAPSHOT_SHA256` to its recorded snapshot-manifest SHA256. Copy both values from the experiment's freeze record. Also set the completed bundle's absolute path:

```bash
source /oscar/home/mliu171/reuseinterp-loopvl/recovery-cu128-20261016/cluster.env
export REUSE_PROJECT_DIR=/absolute/path/to/full-experiment-snapshot
export REUSE_EXPECTED_SNAPSHOT_SHA256=RECORDED_SNAPSHOT_MANIFEST_SHA256
export LOOPVL_VIEWER_BUNDLE=/absolute/path/to/viewer-bundle
umask 077
mkdir -p "$REUSE_RUN_ROOT/viewer-sessions"
chmod 700 "$REUSE_RUN_ROOT/viewer-sessions"
cat "$REUSE_PROJECT_DIR/scripts/slurm/serve_viewer.sbatch"
sbatch --export=ALL \
  --output="$REUSE_RUN_ROOT/viewer-sessions/%j.log" \
  "$REUSE_PROJECT_DIR/scripts/slurm/serve_viewer.sbatch"
```

Save the returned numeric job ID. The script automatically writes `$REUSE_RUN_ROOT/viewer-sessions/JOB_ID.json` after startup verifies every manifest-listed artifact. Read that file using the actual ID:

```bash
cat "$REUSE_RUN_ROOT/viewer-sessions/JOB_ID.json"
```

The ready file records `compute_ip`, `node`, `port`, `local_url`, and the random session token. It is created with mode `0600`; keep it and the server log private and out of Git. The token is mandatory even when the server binds to the compute interface. The browser's first request sets an HttpOnly session cookie and removes the token from the address bar. Server request logs omit query strings. Set `LOOPVL_VIEWER_PORT` before submission to choose a compute-node port other than 8765.

## Connect from your computer

If the experiment is running through the queued completion pipeline, run this lightweight helper **on Oscar** with the experiment's directory:

```bash
python /oscar/home/mliu171/reuseinterp-loopvl/source/scripts/viewer_connection.py \
  --experiment-dir /absolute/path/to/experiment-directory
```

It reads `pipeline_state.json` once. While jobs are queued or running, it prints the recorded stage, job IDs, and connection-file location when available. After the viewer writes its ready file, it prints the exact tunnel command to run on your computer and the authenticated browser URL. Run the helper again when needed; it does not poll Slurm, submit work, or start a server. Its output includes the private session token, so keep it out of shared logs and Git. `--local-port 8766` changes only your computer's port if 8765 is occupied.

If you already know the private ready file, use the shortcut:

```bash
python /oscar/home/mliu171/reuseinterp-loopvl/source/scripts/viewer_connection.py \
  --ready-file "$REUSE_RUN_ROOT/viewer-sessions/JOB_ID.json"
```

These are recorded startup details; the helper does not check whether an old session has reached its wall-time limit. For manual forwarding, follow the steps below.

Read the private ready file on Oscar, then run this command in a terminal **on your computer**, using its actual `compute_ip`:

```bash
ssh -N -L 8765:COMPUTE_IP:8765 mliu171@ssh.ccv.brown.edu
```

Keep that terminal open and open the ready file's `local_url` in your browser. Oscar's [SSH-tunnel instructions](https://docs.ccv.brown.edu/oscar/jupyter-notebooks/jupyter-notebooks-on-oscar-1) use the same compute-IP forwarding pattern. If local port 8765 is occupied, forward `8766:COMPUTE_IP:8765` instead and change only the browser URL's local port to 8766.

The server binds to `127.0.0.1` by default. If compute-node SSH is available for your allocation, you can retain that default and use a jump instead:

```bash
ssh -N -J mliu171@ssh.ccv.brown.edu \
  -L 8765:127.0.0.1:8765 mliu171@COMPUTE_NODE
```

The session ends at the CPU job's wall-time limit. To restart, submit the CPU script again with the same frozen source and bundle, then read the new job ID's ready file and reconnect using its new node and token. To stop early, cancel that specific viewer job with `scancel JOB_ID`.

## Run a downloaded bundle locally

Copy the bundle and use this source checkout on your computer. A local Python environment needs only the package's existing NumPy and Pillow dependencies; PyTorch is unnecessary.

```bash
scp -r mliu171@ssh.ccv.brown.edu:/absolute/path/to/viewer-bundle ./viewer-bundle
python -m venv .viewer-venv
source .viewer-venv/bin/activate
python -m pip install -e .
python scripts/serve_viewer.py serve --bundle-dir ./viewer-bundle
```

Open the printed session URL. The local server uses loopback by default. It serves only exact manifest-listed artifacts, rejects path traversal and symlinks, and verifies bundle hashes before startup. It cannot browse arbitrary cluster or laptop files.

## Validation

Small fixture tests cover variable grids, physical-channel order, all four streams, EXIF orientation, source/raw integrity, authentication, and filesystem restrictions. Run them in a scheduled CPU allocation:

```bash
PYTHONPATH="$REUSE_SOURCE_DIR/src" python -m unittest discover \
  -s "$REUSE_SOURCE_DIR/tests" -p test_viewer.py -v
```

`scripts/smoke_viewer_browser.py` optionally exercises the real interface with Playwright, saving a screenshot and a compact result. Playwright is a validation-only dependency. Run it on the same CPU node as the server, with an installed Chromium browser:

```bash
python "$REUSE_SOURCE_DIR/scripts/smoke_viewer_browser.py" \
  --ready-file "$REUSE_RUN_ROOT/viewer-sessions/JOB_ID.json" \
  --screenshot /absolute/path/to/viewer-smoke.png
```
