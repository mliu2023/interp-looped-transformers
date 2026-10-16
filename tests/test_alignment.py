"""Geometry checks plus an upstream-source patch-order contract test."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

from reuseinterp.data import generate_dataset
from reuseinterp.data.alignment import object_token_weights


class MaskAlignmentTests(unittest.TestCase):
    def test_exact_fractional_coverage_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            # Object 1 covers the left column; object 2 covers two right columns.
            Image.fromarray(np.array([[1, 2, 2], [1, 2, 2]], dtype=np.uint8)).save(path)
            result = object_token_weights(path, 1, 2, source_image_size=(3, 2))
            np.testing.assert_allclose(result[1], [1., 0.])
            np.testing.assert_allclose(result[2], [.25, .75])
            np.testing.assert_allclose(result.coverage["objects"]["2"]["patch_occupancy"], [1 / 3, 1])
            self.assertEqual(result.coverage["objects"]["1"]["source_mask_area_pixels"], 2)
            json.dumps(result.coverage)  # Raw area record is portable JSON.

    def test_all_masks_have_equal_total_weight_and_edit_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            generate_dataset(root, {"smoke": 2}, image_size=96)
            scenes = [json.loads(line) for line in (root / "scenes.jsonl").read_text().splitlines()]
            for base, edited in zip(scenes[::2], scenes[1::2]):
                a = object_token_weights(root / base["mask_path"], 16, 16, source_image_size=(96, 96))
                b = object_token_weights(root / edited["mask_path"], 16, 16, source_image_size=(96, 96))
                self.assertEqual(set(a), {obj["mask_label"] for obj in base["objects"]})
                self.assertEqual(a.coverage, b.coverage)
                for label in a:
                    self.assertEqual(a[label].shape, (256,))
                    np.testing.assert_array_equal(a[label], b[label])
                    self.assertAlmostEqual(float(a[label].sum()), 1.)

    def test_upstream_patchify_row_major_with_merge_one(self):
        # Execute the actual vendored _preprocess method in isolation. Tiny
        # adapters replace library utilities, avoiding transformers/model loads.
        source = Path(__file__).resolve().parents[1] / "vendor/LoopVL/runtime/penguin_encoder/image_processing_penguinvl.py"
        if not source.exists():
            self.skipTest("upstream checkout unavailable; source contract pending")
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PenguinVLImageProcessor")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_preprocess")
        isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
        ast.fix_missing_locations(isolated)
        channel = SimpleNamespace(FIRST="channels_first", LAST="channels_last")
        scope = {
            "np": np, "ChannelDimension": channel,
            "make_list_of_images": lambda images: images,
            "to_numpy_array": np.asarray,
            "is_scaled_image": lambda image: False,
            "infer_channel_dimension_format": lambda image: channel.LAST,
            "get_image_size": lambda image, channel_dim: image.shape[:2],
            "to_channel_dimension_format": lambda image, data_format, input_channel_dim: image.transpose(2, 0, 1),
        }
        exec(compile(isolated, str(source), "exec"), scope)
        labels = np.repeat(np.repeat(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8), 2, axis=0), 2, axis=1)
        pixels = np.repeat(labels[:, :, None], 3, axis=2)
        patches, grid = scope["_preprocess"](
            SimpleNamespace(patch_size=2), [pixels], target_size=[4, 6],
            merge_size=1, do_resize=False, do_rescale=False, do_normalize=False, do_convert_rgb=False,
        )
        self.assertEqual(grid, (1, 2, 3))
        np.testing.assert_array_equal(patches[:, 0], np.arange(1, 7))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.fromarray(labels).save(path)
            result = object_token_weights(path, 2, 3, source_image_size=(6, 4))
            for label in range(1, 7):
                np.testing.assert_array_equal(result[label], np.eye(6)[label - 1])

    def test_unsupported_transforms_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            Image.new("L", (6, 4), 1).save(path)
            for kwargs in ({"merge_size": 2}, {"transform": "letterbox"}, {"transform": "center_crop"},
                           {"source_image_size": (8, 4)}):
                with self.assertRaises(ValueError):
                    object_token_weights(path, 2, 3, **kwargs)
            with self.assertRaises(ValueError):
                object_token_weights(path, 0, 3)
            Image.new("L", (6, 4), 0).save(path)
            with self.assertRaises(ValueError):
                object_token_weights(path, 2, 3)


if __name__ == "__main__":
    unittest.main()
