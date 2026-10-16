# Exact controlled-scene dataset

`scenes-v1.tar.gz` contains the independently validated dataset prepared on CPU:
1,160 families, 2,320 scene versions, 9,280 questions and 64,952 object descriptor
rows, with exact labeled masks and all fixed splits. This is a deterministic 2D
distribution, not CLEVR. There are no model measurements in this archive.

The SHA-256 sidecar verifies the compressed bytes. The archived manifest verifies
every extracted image, mask and table. Setup extracts this archive rather than
rendering again under a different Pillow version. It refuses links, special files,
path traversal and incompatible existing data. Model competence and actual
processor-grid alignment remain unmeasured.

Archive creation sorted paths, zeroed timestamps, normalized ownership and
permissions, and used a zero-timestamp gzip header. The generator and its scene
properties are in `src/reuseinterp/data/`; images were rendered with Pillow 9.2.0.
