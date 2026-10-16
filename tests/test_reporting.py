"""Behavioral controls and CPU recomputation must not launch inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from reuseinterp.io import file_hash
from reuseinterp.schemas import BehavioralMeasurement
from reuseinterp.report import build_report, competence_summary, verify_results


class ReportingTests(unittest.TestCase):
    def test_paired_competence_gate_reports_numerators_and_controls(self):
        rows = []
        for i in range(12):
            for task in ("direct_attributes", "relational_selection", "counting", "comparison"):
                for condition in ("clean", "shuffled_image", "neutral_image"):
                    rows.append(BehavioralMeasurement(
                        "synthetic", "config", "checkpoint", "pilot", "f%d" % i,
                        "e%d-%s" % (i, task), task, condition, "red", {"red": 1., "blue": 0.},
                        "red" if condition == "clean" else "blue", True).validate())
        result = competence_summary(rows, n_resamples=100)
        self.assertEqual(len(result["supported_families"]), 4)
        for value in result["families"].values():
            self.assertEqual(value["clean_correct"], 12)
            self.assertEqual(value["independent_families"], 12)
            self.assertEqual(value["paired_95_ci"], [1., 1.])
        with self.assertRaisesRegex(ValueError, "paired image"):
            competence_summary(rows[1:], n_resamples=100)

    def test_weak_competence_not_renamed_as_supported(self):
        rows = []
        for i in range(3):
            for condition in ("clean", "shuffled_image", "neutral_image"):
                rows.append(BehavioralMeasurement("synthetic", "config", "checkpoint", "pilot", "f%d" % i,
                            "e%d" % i, "counting", condition, "1", {"1": 0., "2": 1.}, "2", True).validate())
        self.assertEqual(competence_summary(rows, n_resamples=100)["supported_families"], [])

    def _verify_fixture_metrics(self, expected, recomputed):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "synthetic-fixtures"
            fixture.mkdir()
            (fixture / "metrics.json").write_text(json.dumps(expected))
            with patch("reuseinterp.fixtures.recompute_fixtures", return_value=recomputed):
                return verify_results(directory)

    def test_metric_reproduction_accepts_only_finite_float_roundoff(self):
        expected = {"schema": "synthetic.v1", "n": 16, "passed": True,
                    "scores": [0.0, 2.0], "undefined": None, "sha256": "a" * 64}
        recomputed = copy.deepcopy(expected)
        recomputed["n"] = np.int64(16)
        recomputed["scores"] = np.array([5e-11, 2.0 + 5e-10])
        result = self._verify_fixture_metrics(expected, recomputed)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reproduction_policy"]["finite_float_rtol"], 1e-9)
        self.assertEqual(result["reproduction_policy"]["finite_float_atol"], 1e-10)

    def test_metric_reproduction_rejects_changed_values_types_and_schema(self):
        expected = {"schema": "synthetic.v1", "n": 16, "passed": True,
                    "scores": [0.0, 2.0], "sha256": "a" * 64}
        changes = [
            {"scores": [0.0, 2.00001]},
            {"scores": [2e-10, 2.0]},
            {"scores": [0.0]},
            {"schema": "synthetic.v2"},
            {"n": 17},
            {"n": 16.0},
            {"passed": 1},
            {"passed": False},
            {"sha256": "b" * 64},
            {"unexpected": None},
        ]
        for change in changes:
            with self.subTest(change=change):
                recomputed = dict(expected, **change)
                with self.assertRaisesRegex(ValueError, "do not reproduce"):
                    self._verify_fixture_metrics(expected, recomputed)
        without_schema = {key: value for key, value in expected.items() if key != "schema"}
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            self._verify_fixture_metrics(expected, without_schema)

    def test_metric_reproduction_never_accepts_nonfinite_values(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for expected in (None, 0.0, value):
                with self.subTest(expected=expected, value=value):
                    with self.assertRaisesRegex(ValueError, "do not reproduce"):
                        self._verify_fixture_metrics({"value": expected}, {"value": value})

    def test_report_rebuild_preserves_hashed_original_figure_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "synthetic-fixtures"
            fixture.mkdir()
            metrics = {"synthetic": True, "score": 0.25}
            original = {"metrics.json": json.dumps(metrics).encode(),
                        "synthetic_fixture_checks.svg": b"<svg>historical evidence</svg>"}
            for name, content in original.items():
                (fixture / name).write_bytes(content)
            evidence = {"synthetic-fixtures/" + name: file_hash(fixture / name) for name in original}
            (root / "manifest.json").write_text(json.dumps({"evidence_files": evidence}))

            def render(fixture_dir, output):
                self.assertEqual(fixture_dir, fixture)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("<svg>rebuilt with platform fonts</svg>")

            with patch("reuseinterp.fixtures.recompute_fixtures", return_value=metrics), \
                    patch("reuseinterp.plotting.plot_fixtures", side_effect=render):
                for _ in range(2):
                    self.assertTrue(build_report(root)["verification"]["passed"])
                    self.assertTrue(verify_results(root)["passed"])
            for name, content in original.items():
                self.assertEqual((fixture / name).read_bytes(), content)
                self.assertEqual((root / "report-package/synthetic-fixtures" / name).read_bytes(), content)
            rebuilt = root / "report-package/figures/synthetic_fixture_checks.svg"
            self.assertEqual(rebuilt.read_text(), "<svg>rebuilt with platform fonts</svg>")


if __name__ == "__main__":
    unittest.main()
