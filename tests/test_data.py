"""CPU checks for controlled scene labels, grouping and edited masks."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reuseinterp.data import (
    DESCRIPTOR_WHITELIST, evaluate_program, generate_dataset,
    independent_answer, validate_dataset,
)
from reuseinterp.data.scenes import _make_family, _question


class SceneDatasetTests(unittest.TestCase):
    def test_independent_program_labels_for_many_scenes(self):
        # Covers all attributes, count ranges, comparison outcomes and edit roles.
        answers = set()
        for ordinal in range(180):
            base, edit, specs, _ = _make_family(ordinal, 97, 96)
            for scene in (base, edit):
                for spec in specs:
                    _, program, candidates = _question(spec)
                    answer = evaluate_program(scene, program)
                    self.assertEqual(answer, independent_answer(scene, spec))
                    self.assertIn(answer, candidates)
                    answers.add(answer)
        self.assertTrue({"more", "fewer", "equal", "0", "1", "6"}.issubset(answers))

    def test_family_grouping_exact_masks_and_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_dataset(Path(tmp) / "one", {"smoke": 6, "confirmation": 2, "transfer": 1}, seed=14, image_size=96)
            second = generate_dataset(Path(tmp) / "two", {"smoke": 6, "confirmation": 2, "transfer": 1}, seed=14, image_size=96)
            self.assertEqual(first, second)
            report = validate_dataset(Path(tmp) / "one")
            self.assertEqual(report["num_families"], 9)
            self.assertEqual(report["num_examples"], 72)
            self.assertEqual(report["coverage"]["split_families"], {"smoke": 6, "confirmation": 2, "transfer": 1})
            self.assertEqual(report["status"], "passed")
            self.assertEqual(generate_dataset(Path(tmp) / "one", {"smoke": 6, "confirmation": 2, "transfer": 1}, seed=14, image_size=96), first)
            with self.assertRaises(FileExistsError):
                generate_dataset(Path(tmp) / "one", {"smoke": 1}, seed=14, image_size=96)

    def test_modified_image_or_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            generate_dataset(root, {"smoke": 1}, image_size=96)
            path = root / "examples.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]["target_answer"] = "invented"
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_dataset(root)

    def test_reduced_preparation_cannot_move_families_to_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = generate_dataset(Path(tmp) / "small", {"smoke": 1, "pilot": 1}, image_size=96)
            larger = generate_dataset(Path(tmp) / "large", {"smoke": 2, "pilot": 3, "confirmation": 2}, image_size=96)
            large_families = {row["family_id"]: row for row in larger["families"]}
            for family in small["families"]:
                self.assertEqual(family, large_families[family["family_id"]])
                for scene_id in family["scene_ids"]:
                    filename = f"images/{scene_id}.png"
                    self.assertEqual(small["files"][filename], larger["files"][filename])

    def test_localization_validation_is_independent_of_checksums(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            manifest = generate_dataset(root, {"smoke": 1}, image_size=96)
            edited = next(name for name in manifest["files"] if "color_edit.png" in name)
            with Image.open(root / edited) as file:
                image = file.copy()
            image.putpixel((0, 0), (0, 0, 0))
            image.save(root / edited)
            manifest["files"][edited] = hashlib.sha256((root / edited).read_bytes()).hexdigest()
            canonical = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            manifest["dataset_hash"] = hashlib.sha256(canonical).hexdigest()
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "changed pixels"):
                validate_dataset(root)

    def test_independent_path_detects_wrong_program(self):
        scene, _, specs, _ = _make_family(5, 1, 96)
        spec = deepcopy(specs[0])
        spec["attribute"] = "color"
        _, program, _ = _question(spec)
        program["attribute"] = "shape"
        self.assertNotEqual(evaluate_program(scene, program), independent_answer(scene, spec))

    def test_descriptor_whitelist_excludes_answer_and_counts(self):
        for forbidden in ("target_answer", "final_answer", "count", "count_a", "count_b", "model_confidence", "candidate_scores", "version", "is_edited_object"):
            self.assertNotIn(forbidden, DESCRIPTOR_WHITELIST)

    def test_bad_configs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for counts in ({}, {"unknown": 1}, {"smoke": -1}, {"smoke": True}, {"smoke": 33}):
                with self.assertRaises(ValueError):
                    generate_dataset(Path(tmp) / "bad", counts)
            with self.assertRaises(ValueError):
                generate_dataset(Path(tmp) / "bad", {"smoke": 1}, image_size=97)


if __name__ == "__main__":
    unittest.main()
