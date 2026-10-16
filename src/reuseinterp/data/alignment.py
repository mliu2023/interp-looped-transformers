"""Exact object-mask area weights for audited no-padding Penguin image grids.

Audited source: LoopVL runtime/hrm_penguin/vision/preprocessing.py's
PenguinDynamicResolutionProcessor uses full-image resize, one frame, merge=1.
runtime/penguin_encoder/image_processing_penguinvl.py::_preprocess patchifies
those images in row-major order when merge=1. The encoder's final reshape and
same-size interpolation preserve that ordering. Letterbox/crop/merged paths
require their own validation and are rejected here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ObjectTokenWeights(dict[int, np.ndarray]):
    """Mask-label -> normalized weights, with JSON-serializable raw coverage.

    ``coverage['objects'][str(label)]['patch_occupancy']`` stores the unnormalized
    per-patch covered fraction. A token can overlap several objects; these masks
    do not establish that contextualized token embeddings are object-local.
    """

    def __init__(self, weights: dict[int, np.ndarray], coverage: dict[str, Any]):
        super().__init__(weights)
        self.coverage = coverage


def _interval_overlap(length: int, bins: int) -> np.ndarray:
    edges = np.linspace(0.0, float(length), bins + 1)
    pixel_starts = np.arange(length, dtype=np.float64)
    return np.maximum(0.0, np.minimum(edges[1:, None], pixel_starts[None, :] + 1.0)
                      - np.maximum(edges[:-1, None], pixel_starts[None, :]))


def object_token_weights(mask_path: str | Path, grid_height: int, grid_width: int,
                         merge_size: int = 1,
                         source_image_size: tuple[int, int] | list[int] | None = None,
                         *, transform: str = "dynamic_resize_no_padding") -> ObjectTokenWeights:
    """Project exact labeled source masks onto a full-image visual-token grid.

    ``source_image_size`` uses PIL's (width, height) convention. Supply dimensions
    from the actual source image at capture time; disagreement is an error. The
    actual processor-emitted grid dimensions must be supplied, never inferred
    from an assumed token count. The caller must verify one frame and no padding
    with the audited dynamic processor. Non-dynamic transforms are refused.

    Areas are integrated geometrically over each patch's source-image footprint,
    including fractional boundary pixels. Normalization gives every visible
    object total weight one. It does not simulate bicubic RGB mixing or prove
    locality after contextualization. Capture real tokens before this pooling.
    """
    if transform != "dynamic_resize_no_padding":
        raise ValueError("unvalidated image crop/letterbox/transform: only dynamic_resize_no_padding is audited")
    if merge_size != 1:
        raise ValueError("unvalidated token merging: merge_size must equal 1")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
           for value in (grid_height, grid_width)):
        raise ValueError("actual grid dimensions must be positive integers")
    with Image.open(mask_path) as handle:
        if handle.mode != "L":
            raise ValueError("object masks must be PNG L label images; do not interpolate label IDs")
        if handle.getexif().get(274, 1) != 1:
            raise ValueError("unvalidated mask EXIF orientation")
        mask = np.asarray(handle).copy()
        image_size = handle.size
    if source_image_size is not None and tuple(source_image_size) != image_size:
        raise ValueError("mask/source dimensions differ; crop, letterbox, or orientation alignment is unvalidated")
    height, width = mask.shape
    labels = [int(label) for label in np.unique(mask) if label != 0]
    if not labels:
        raise ValueError("mask contains no visible objects")
    vertical = _interval_overlap(height, int(grid_height))
    horizontal = _interval_overlap(width, int(grid_width))
    source_patch_area = (height / grid_height) * (width / grid_width)
    weights, object_coverage = {}, {}
    for label in labels:
        binary = (mask == label).astype(np.float64)
        source_area = int(binary.sum())
        occupancy = ((vertical @ binary) @ horizontal.T / source_patch_area).ravel(order="C")
        occupancy = np.clip(occupancy, 0.0, 1.0)
        total = float(occupancy.sum())
        expected_total = source_area / source_patch_area
        if total <= 0 or not np.isclose(total, expected_total, rtol=1e-12, atol=1e-12):
            raise ValueError("object area was lost during token-grid projection")
        weights[label] = occupancy / total
        object_coverage[str(label)] = {
            "source_mask_area_pixels": source_area,
            "source_image_area_fraction": source_area / (height * width),
            "projected_area_patch_units": total,
            "nonzero_token_count": int(np.count_nonzero(occupancy > 0)),
            "patch_occupancy": occupancy.tolist(),
            "normalization": total,
        }
    coverage = {
        "schema_version": "reuseinterp.mask_alignment.v1",
        "source_image_size_width_height": [width, height],
        "grid_height": int(grid_height), "grid_width": int(grid_width),
        "merge_size": 1, "frame_count": 1, "flatten_order": "row_major",
        "transform": transform, "objects": object_coverage,
        "coverage_method": "exact area of labeled source pixels in full-image patch footprints",
        "scope": "geometric object pooling; bicubic boundary mixing and contextualization can spread information",
        "runtime_processor_validation": "caller must verify actual dynamic processor outputs and one-frame grid",
    }
    return ObjectTokenWeights(weights, coverage)
