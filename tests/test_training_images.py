"""Small integrity checks for original-image provenance and bounded HTTP access."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from reuseinterp.data.images import (
    CHECKPOINT_REVISION, DATASET_ID, DATASET_REVISION, SCHEMA_VERSION,
    DownloadBudget, HTTPRangeReader, canonical_json, first_instruction,
    image_fingerprints, load_image_manifest,
)


class _Response(io.BytesIO):
    def __init__(self, payload, status=206, content_range="bytes 0-2/8"):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


class HTTPBoundTests(unittest.TestCase):
    def test_ranges_and_small_block_cache_obey_shared_byte_cap(self):
        requests = []
        data = b"abcdefgh"

        def opener(request, timeout):
            value = request.get_header("Range")
            requests.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            return _Response(data[start:end + 1], content_range=f"bytes {start}-{end}/8")

        budget = DownloadBudget(5)
        with HTTPRangeReader("https://example.invalid/shard", 8, budget, opener=opener) as reader:
            self.assertEqual(reader.read(3), b"abc")
            reader.seek(1)
            self.assertEqual(reader.read(2), b"bc")
            self.assertEqual(len(requests), 1)
            reader.seek(-2, io.SEEK_END)
            self.assertEqual(reader.read(), b"gh")
            reader.seek(3)
            with self.assertRaisesRegex(ValueError, "byte cap"):
                reader.read(1)
        self.assertEqual(budget.bytes_received, 5)
        self.assertEqual(requests, ["bytes=0-2", "bytes=6-7"])

    def test_full_download_fallback_and_wrong_range_are_rejected(self):
        for status, content_range in [(200, "bytes 0-2/8"), (206, "bytes 0-2/9")]:
            budget = DownloadBudget(8)
            response = _Response(b"abc", status=status, content_range=content_range)
            with HTTPRangeReader("https://example.invalid/shard", 8, budget,
                                 opener=lambda *args, **kwargs: response) as reader:
                with self.assertRaises(ValueError):
                    reader.read(3)
            self.assertEqual(budget.bytes_received, 0)

    def test_short_response_is_not_accepted_as_complete(self):
        budget = DownloadBudget(8)
        with HTTPRangeReader("https://example.invalid/shard", 8, budget,
                             opener=lambda *args, **kwargs: _Response(b"ab")) as reader:
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                reader.read(3)
            self.assertEqual(reader.tell(), 0)
        self.assertEqual(budget.bytes_received, 2)


class OriginalImageManifestTests(unittest.TestCase):
    def _manifest(self, directory):
        image_path = directory / "images" / "one.png"
        image_path.parent.mkdir()
        image = Image.new("RGB", (40, 32), (100, 150, 200))
        image.putpixel((10, 10), (0, 0, 0))
        image.save(image_path)
        identity = image_fingerprints(image_path.read_bytes())
        row = {"image_id": "one", "scene_id": "one", "family_id": "family_one",
               "image_path": "images/one.png",
               **{key: value for key, value in identity.items() if key != "extension"},
               "provenance": {"dataset_id": DATASET_ID, "revision": DATASET_REVISION,
                              "checkpoint_revision": CHECKPOINT_REVISION,
                              "training_membership": "reported_corpus_member"}}
        return {"schema_version": SCHEMA_VERSION, "complete": True, "images": [row]}

    def _write(self, directory, manifest):
        manifest.pop("dataset_fingerprint", None)
        manifest["dataset_fingerprint"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_integrity_resolution_and_family_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            path = self._write(directory, manifest)
            loaded = load_image_manifest(path)
            self.assertEqual(Path(loaded["images"][0]["image_path"]), directory / "images/one.png")
            duplicate = deepcopy(manifest["images"][0])
            duplicate["image_id"] = "different_question_same_image"
            manifest["images"].append(duplicate)
            self._write(directory, manifest)
            with self.assertRaisesRegex(ValueError, "repeats an image family"):
                load_image_manifest(path)

    def test_modified_pixels_overstated_membership_and_mutable_revisions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._manifest(directory)
            path = self._write(directory, manifest)
            for key, value in [("training_membership", "confirmed_training_record"), ("revision", "main")]:
                changed = deepcopy(manifest)
                changed["images"][0]["provenance"][key] = value
                self._write(directory, changed)
                with self.assertRaises(ValueError):
                    load_image_manifest(path)
            self._write(directory, manifest)
            Image.new("RGB", (40, 32), (255, 0, 0)).save(directory / "images" / "one.png")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                load_image_manifest(path)

    def test_source_answers_are_never_selected_as_instructions(self):
        row = {"conversations": [{"from": "gpt", "value": "secret answer"},
                                  {"from": "human", "value": "<image>\nWhat is visible?"},
                                  {"from": "gpt", "value": "another answer"}]}
        self.assertEqual(first_instruction(row), "What is visible?")
        self.assertIsNone(first_instruction({"conversations": row["conversations"][:1]}))


if __name__ == "__main__":
    unittest.main()
