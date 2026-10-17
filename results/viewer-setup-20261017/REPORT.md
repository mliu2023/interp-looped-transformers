# Interactive viewer setup and real-image rerun

The viewer is implemented and tested. The real-image rerun is submitted; it has no completed activation results yet.

- Final scheduled CPU check: **164 tests passed**, job `7522075`. See [validation summary](validation_summary.json) and [full test log](tests.log).
- Chromium validation: job `7521735` exercised selectors, galleries, six overlays, image selection, opacity, and charts with no JavaScript errors. It used historical saved captures to test the interface, not new natural-image measurements. See [browser evidence](browser_validation.json).
- Data: **64 original OneVision instruction images**, 22 COCO, 21 GQA, and 21 Visual Genome. Staging received 296,602,377 shard bytes under a 500 MiB cap. [Source provenance](source_provenance.json) records immutable source IDs and original image hashes; image bytes remain on the cluster.
- Membership: every image is a `reported_corpus_member`. The released artifacts do not establish which records this checkpoint retained or processed. See the [source audit](../../docs/training_image_sources.md).

Initial timing job `7521681` passed backend preflight but stopped before capture because the new validator assumed a 1,024-token limit. The pinned checkpoint specifies 256–2,048. The corrected collector derives the actual limits from the model and processor; regression tests cover the mismatch. No input image or native preprocessing was changed to work around the guard.

Corrected timing job **`7522110`** and dependent pipeline **`7522111`** are submitted against snapshot SHA256 `04d96aecdc5928bcaed3f2ec53d72f2fb169e1a30003a2864519e55e1777dad4`. The [submission receipt](queued_state.json) is a historical receipt; live saved state is on the cluster. The pipeline must pass native-grid official-path and full-MLP checks, compute a measured resource allowance, and respect the shared ledger before submitting the 64-image experiment. Successful capture triggers independent recomputation, browser validation, bundle export, and a four-hour CPU viewer. Failure stops the chain and saves its reason.

The full experiment fixes 48 channels in physical L layers 0, 7, and 15, all six L invocations, one common prompt, and eight paraphrase controls. It measures signed coverage and image-ranking transfer. It does not claim semantic poles, memorization, circuits, or superposition from signs alone.

Use the [viewer guide](../../docs/INTERACTIVE_VIEWER.md#current-rerun) to connect locally. On Oscar:

```bash
python /oscar/home/mliu171/reuseinterp-loopvl/source/scripts/viewer_connection.py \
  --experiment-dir /oscar/home/mliu171/reuseinterp-loopvl/experiments/loopvl-training-images-20261017
```

Once the server starts, this prints the exact local SSH tunnel and browser URL. Before startup it prints the saved pipeline state and job IDs.

The scientific report will be written to `/oscar/home/mliu171/reuseinterp-loopvl/runs/loopvl-training-source-photos-20261017-v1/report.md`; raw activations and images are retained with that run. Verification, the portable viewer bundle, and a browser preview belong under `/oscar/home/mliu171/reuseinterp-loopvl/experiments/loopvl-training-images-20261017/analysis-v1/`. No native-image numerical result is claimed by this setup report.
