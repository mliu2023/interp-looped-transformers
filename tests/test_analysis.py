"""Scientific regression checks: these fail for common invalid pipelines."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reuseinterp.encoding import (DescriptorSpec, evaluate_semantic_transfer,
                                 family_weights, fit_positive_affine, fit_ridge,
                                 regression_metrics, validate_split_families,
                                 validate_unique_ids)
from reuseinterp.fixtures import recompute_fixtures, run_fixtures
from reuseinterp.discovery import (assemble_capture_package, fit_capture_package,
                                   confirm_capture_package, recompute_confirmation)
from reuseinterp.controls import (balanced_semantic_gap, fit_context_balance,
                                 make_context_labels, matched_semantic_gaps,
                                 paired_edit_differences, paired_selectivity_profiles)
from reuseinterp.io import atomic_json, digest, file_hash
from reuseinterp.probes import (balanced_accuracy, coherent_family_permutation,
                               fit_linear_readout)
from reuseinterp.statistics import (causal_profile_interaction, centered_max_t,
                                    family_bootstrap, fit_causal_gain,
                                    group_profiles, layer_balanced_mean,
                                    semantic_gap_interval, validate_condition_rows)


class EncodingTests(unittest.TestCase):
    def test_descriptor_whitelist_and_unknown_levels(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            DescriptorSpec({"final_answer": ["yes", "no"]})
        spec = DescriptorSpec({"color": ["red", "blue"], "role": ["target", "other"]},
                              ["x"], [("color", "role")])
        values, names = spec.transform([{"color": "red", "role": "target", "x": 1},
                                       {"color": "blue", "role": "other", "x": -1}])
        self.assertEqual(values.shape, (2, 9))
        self.assertEqual(len(names), 9)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            spec.transform([{"color": "green", "role": "target", "x": 0}])

    def test_fit_preprocessing_is_frozen_and_intercept_unpenalized(self):
        xf = np.arange(20.)[:, None]
        model = fit_ridge(xf, 3 * xf + 7, xf + 20, 3 * (xf + 20) + 7, [.00001])
        np.testing.assert_allclose(model.predict([[1000]]), [[3007]], atol=.01)
        np.testing.assert_allclose(model.feature_mean, [9.5])
        self.assertAlmostEqual(float(model.target_mean[0]), 35.5)
        with self.assertRaisesRegex(ValueError, "development"):
            fit_ridge(xf, xf, xf, xf, development_split="confirmation")

    def test_positive_affine_recovers_gain_and_never_sign_flip(self):
        x = np.linspace(-2, 2, 20)
        fit = fit_positive_affine(x, 4 * x + 9)
        np.testing.assert_allclose(fit.alpha, [4])
        np.testing.assert_allclose(fit.predict(x), (4 * x + 9)[:, None])
        inverted = fit_positive_affine(x, -4 * x + 9)
        np.testing.assert_allclose(inverted.alpha, [0])
        with self.assertRaisesRegex(ValueError, "calibration"):
            fit_positive_affine(x, x, split="confirmation")

    def test_constant_source_calibrates_to_target_mean(self):
        fit = fit_positive_affine(np.ones(4), np.arange(4.))
        np.testing.assert_array_equal(fit.alpha, [0])
        np.testing.assert_allclose(fit.predict([1, 1]), [[1.5], [1.5]])

    def test_nmse_uses_frozen_target_variance_and_flags_inactive(self):
        metric = regression_metrics([[0, 0], [10, 0]], [[0, 1], [8, 1]], [2, 0], 1e-8)
        self.assertEqual(metric["mse"][0], 2)
        self.assertEqual(metric["nmse"][0], 1)
        self.assertAlmostEqual(metric["r2"][0], .92)
        self.assertTrue(np.isnan(metric["nmse"][1]))
        self.assertTrue(metric["low_variance"][1])

    def test_family_weights_not_object_counts(self):
        ids = ["a", "a", "a", "b"]
        self.assertAlmostEqual(float(np.average([0, 0, 0, 10], weights=family_weights(ids))), 5)

    def test_family_leakage_and_duplicate_confirmation_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split_families({"fit": ["base-a"], "confirmation": ["base-a"]})
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_unique_ids(["e1", "e1"])

    def test_development_eligibility_does_not_select_on_confirmation(self):
        rng = np.random.default_rng(4)
        splits = ("fit", "development", "calibration", "confirmation")
        x = {split: rng.normal(size=(40, 3)) for split in splits}
        y = {split: {p: x[split][:, :1].copy() for p in ("L1", "L4")} for split in splits}
        original = evaluate_semantic_transfer(x, y, penalties=[.001])
        y["confirmation"] = {p: rng.normal(size=(40, 1)) * 100 for p in ("L1", "L4")}
        changed = evaluate_semantic_transfer(x, y, penalties=[.001])
        np.testing.assert_array_equal(original["eligible_development"], changed["eligible_development"])
        for key in original["calibrations"]:
            np.testing.assert_array_equal(original["calibrations"][key].alpha, changed["calibrations"][key].alpha)


class ProbeTests(unittest.TestCase):
    def test_balanced_accuracy_is_mean_class_recall(self):
        self.assertAlmostEqual(balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]), .5)
        self.assertTrue(np.isnan(balanced_accuracy([0, 0], [0, 0], classes=[0, 1])))

    def test_frozen_readout_transfer_can_fail_with_information_intact(self):
        x = np.array([[-2., 0], [-1., 1], [1., -1], [2., 0]])
        y = np.array([0, 0, 1, 1])
        readout = fit_linear_readout(x, y, x, y, penalties=[.1])
        np.testing.assert_array_equal(readout.predict(x), y)
        self.assertEqual(balanced_accuracy(y, readout.predict(-x)), 0)
        within_late = fit_linear_readout(-x, y, -x, y, penalties=[.1])
        self.assertEqual(balanced_accuracy(y, within_late.predict(-x)), 1)

    def test_coherent_family_permutation_preserves_cross_pass_rows(self):
        values = np.arange(24).reshape(12, 2)
        ids = np.repeat(np.arange(6), 2)
        permuted = coherent_family_permutation(values, ids, seed=9)
        np.testing.assert_array_equal(permuted[:, 1] - permuted[:, 0], np.ones(12))
        self.assertEqual(sorted(map(tuple, values)), sorted(map(tuple, permuted)))
        for i in range(0, 12, 2):
            np.testing.assert_array_equal(permuted[i + 1] - permuted[i], [2, 2])
        with self.assertRaisesRegex(ValueError, "equal-shape"):
            coherent_family_permutation(values, np.array([0] * 3 + [1] * 9))


class StatisticsTests(unittest.TestCase):
    def test_layer_balancing_and_missing_stratum(self):
        self.assertEqual(layer_balanced_mean([0, 0, 0, 8], [0, 0, 0, 1]), 4)
        with self.assertRaisesRegex(ValueError, "layer"):
            layer_balanced_mean([0, np.nan], [0, 1])

    def test_whole_family_bootstrap_preserves_condition_pairs(self):
        values = np.tile([0., 1.], 12)
        family_ids = np.repeat(np.arange(12), 2)
        def estimator(rows, ids):
            # This assertion would fail under an ordinary row bootstrap.
            for family in np.unique(ids):
                self.assertEqual(sorted(rows[ids == family].tolist()), [0, 1])
            return rows.mean()
        result = family_bootstrap(values, family_ids, estimator, 100, seed=2)
        self.assertEqual(result["independent_families"], 12)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["lower"], .5)

    def test_repeated_family_draws_keep_their_multiplicity(self):
        values = np.array([0., 0., 10.])
        ids = np.array(["a", "a", "b"])
        def estimator(rows, new_ids):
            return np.average(rows, weights=family_weights(new_ids))
        result = family_bootstrap(values, ids, estimator, 200, seed=2)
        self.assertEqual(result["estimate"], 5)
        self.assertTrue(set(result["draws"]).issubset({0., 5., 10.}))

    def test_p1_fixed_models_layer_and_family_balanced(self):
        ids = np.repeat(np.arange(16), 2)
        gaps = np.tile([0., 0., 0., .2], (32, 1))
        result = semantic_gap_interval(gaps, ids, [0, 0, 0, 1], n_resamples=100)
        self.assertAlmostEqual(float(result["estimate"]), .1)
        self.assertAlmostEqual(float(result["lower"]), .1)

    def test_gain_uses_group_means_with_equal_group_weight(self):
        early = np.array([1., 1., 1., 3., 3.])
        late = np.array([2., 2., 2., 6., 6.])
        groups = np.array(["a", "a", "a", "b", "b"])
        ids = np.array([0, 1, 2, 0, 1])
        result = fit_causal_gain(early, late, groups, ids, ["a", "b"])
        np.testing.assert_allclose(result["gain"], [2])
        with self.assertRaisesRegex(ValueError, "calibration"):
            fit_causal_gain(early, late, groups, ids, ["a", "b"], split="confirmation")

    def test_global_gain_is_not_context_interaction(self):
        ids = np.repeat(np.arange(32), 2)
        groups = np.tile(["a", "b"], 32)
        early = np.tile([.2, .7], 32)
        result = causal_profile_interaction(early, early * 3, [3], groups, ids, ["a", "b"], "a", 100)
        np.testing.assert_allclose(result["estimate"], [[0]], atol=1e-14)
        self.assertEqual(result["centered_max_t_p"], 1)

    def test_selective_residual_interaction_and_centered_test(self):
        rng = np.random.default_rng(7)
        ids = np.repeat(np.arange(40), 2)
        groups = np.tile(["a", "b"], 40)
        early = .3 + rng.normal(0, .02, 80)
        late = early + np.where(groups == "b", .4, 0) + rng.normal(0, .005, 80)
        result = causal_profile_interaction(early, late, [1], groups, ids, ["a", "b"], "a", 200)
        self.assertGreater(float(result["lower"][0, 0]), .35)
        self.assertLess(result["centered_max_t_p"], .02)

    def test_centered_max_t_handles_exact_zero_variance(self):
        result = centered_max_t(np.array([0., .5]), np.tile([0., .5], (100, 1)))
        np.testing.assert_array_equal(result["zero_variance"], [True, True])
        np.testing.assert_allclose(result["lower"], [0, .5])
        self.assertEqual(result["centered_max_t_p"], 1 / 101)

    def test_missing_or_underpopulated_context_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "only"):
            group_profiles([1, 1], ["a", "b"], [0, 1], ["a", "b"])
        with self.assertRaisesRegex(ValueError, "only"):
            group_profiles([1, 1], ["a", "a"], [0, 1], ["a", "b"])

    def test_missing_and_duplicate_condition_rows_rejected(self):
        row = {"example_id": "e1", "physical_layer": 0, "neuron_id": 12, "invocation": "L1", "condition": "plus"}
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_condition_rows([row], ["plus", "minus"])
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_condition_rows([row, row], ["plus", "minus"])
        other = dict(row, condition="minus")
        self.assertEqual(validate_condition_rows([row, other], ["plus", "minus"]), 1)


class SyntheticFixtureTests(unittest.TestCase):
    def test_fixtures_recompute_from_raw_without_model_results(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_fixtures(directory, n_resamples=100)
            self.assertTrue(first["all_passed"])
            self.assertTrue(first["synthetic"])
            self.assertEqual(len(first["fixtures"]), 6)
            reproduced = recompute_fixtures(directory)
            self.assertEqual(first, reproduced)
            stored = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(first, stored)
            self.assertTrue((Path(directory) / "raw_synthetic.npz").is_file())
            self.assertTrue((Path(directory) / "synthetic_predictions.npz").is_file())
            with self.assertRaises(FileExistsError):
                run_fixtures(directory, n_resamples=100)


class DiscoverySeparationTests(unittest.TestCase):
    def test_assembler_joins_by_recorded_mask_labels_not_object_position(self):
        from reuseinterp.data import generate_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            generate_dataset(dataset, {"fit": 1, "development": 1, "calibration": 1})
            examples = [json.loads(line) for line in (dataset / "examples.jsonl").read_text().splitlines()]
            scenes = {r["scene_id"]: r for r in (json.loads(line) for line in (dataset / "scenes.jsonl").read_text().splitlines())}
            sample = {"channels": {"0": [3]}, "seed": 0}
            rows = []
            for example in examples:
                labels = [obj["mask_label"] for obj in scenes[example["scene_id"]]["objects"]][::-1]
                product = np.tile(np.array(labels + [99.])[None, :, None], (6, 1, 1))
                path = root / (example["example_id"] + ".npz")
                np.savez(path, products_layer_0=product, mask_labels=np.array(labels),
                         states_layer_0_input=np.tile(product, (1, 1, 2)),
                         states_layer_0_block=np.tile(product, (1, 1, 2)))
                rows.append({"example_id": example["example_id"], "family_id": example["family_id"], "split": example["split"],
                             "artifact": str(path), "sha256": file_hash(path), "object_mask_labels": labels,
                             "channel_selection_hash": digest(sample["channels"])})
            config = {"checkpoint_revision": "SYNTHETIC", "synthetic": True, "state_layers": [0]}
            analysis_config = assemble_capture_package(rows, dataset, sample, root / "aligned.npz", config)
            self.assertIn("descriptor_spec_hash", analysis_config)
            with np.load(root / "aligned.npz", allow_pickle=False) as aligned:
                for row_id, activations in zip(aligned["fit__row_ids"], aligned["fit__activations"]):
                    object_id = row_id.split("::")[1]
                    expected_label = int(object_id.split("_")[1]) + 1
                    np.testing.assert_array_equal(activations[:, 0], np.repeat(expected_label, 6))
                self.assertTrue(json.loads(str(aligned["manifest_json"].item()))["synthetic"])
            with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
                assemble_capture_package(rows[:-1], dataset, sample, root / "incomplete.npz", config)

    def _package(self, path, splits):
        rng = np.random.default_rng(22)
        arrays = {"passes": np.array(["L1", "L4"]), "physical_layers": np.array([0]),
                  "neuron_ids": np.array([2]), "feature_names": np.array(["color_signal", "x"]),
                  "manifest_json": np.asarray(json.dumps({"checkpoint_revision": "SYNTHETIC",
                      "dataset_hash": "synthetic-test", "token_stratum": "object_pooled",
                      "descriptor_spec_hash": "frozen-test-spec", "synthetic": True}))}
        for split in splits:
            x = rng.normal(size=(40, 2))
            arrays[split + "__descriptors"] = x
            arrays[split + "__activations"] = np.stack([x[:, :1], 2 * x[:, :1] + 1], axis=1)
            arrays[split + "__family_ids"] = np.repeat([split + str(i) for i in range(20)], 2)
            arrays[split + "__row_ids"] = np.array([split + "-row-" + str(i) for i in range(40)])
            arrays[split + "__color_labels"] = np.where(x[:, 0] >= 0, "red", "blue")
            arrays["probe__layer_0_input__" + split] = np.stack([x, x], axis=1)
        np.savez(path, **arrays)
        return arrays

    def _config(self):
        return {"descriptor_feature_names": ["color_signal", "x"],
                "descriptor_spec_hash": "frozen-test-spec", "penalties": [.001, .1]}

    def test_discovery_refuses_opening_confirmation_even_if_unused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root / "bad.npz", ["fit", "development", "calibration", "confirmation"])
            with self.assertRaisesRegex(ValueError, "must not open"):
                fit_capture_package(root / "bad.npz", root / "output", self._config())

    def test_confirmation_loads_frozen_models_and_checks_lock_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._package(root / "discovery.npz", ["fit", "development", "calibration"])
            record = fit_capture_package(root / "discovery.npz", root / "fits", self._config())
            self.assertEqual(record["p2_site"], "layer_0_input")
            confirmation = self._package(root / "confirmation.npz", ["confirmation"])
            model_path = root / "fits" / "discovery_models.json"
            protocol = {"discovery_artifact_hashes": {"discovery_models.json": file_hash(model_path)},
                        "p2_site": record["p2_site"], "confirmation_family_ids": sorted(set(confirmation["confirmation__family_ids"].tolist())),
                        "statistics": {"resamples": 100, "seed": 1729}}
            # Unit-test boundary token; create_lock's full scientific gate tests
            # live in protocol tests, not in this synthetic array adapter test.
            atomic_json(root / "lock.json", {"status": "locked", "protocol": protocol, "protocol_hash": digest(protocol)})
            with patch("reuseinterp.discovery.fit_ridge", side_effect=AssertionError("Confirmation refitted an encoder")), \
                    patch("reuseinterp.discovery.fit_positive_affine", side_effect=AssertionError("Confirmation refitted calibration")):
                result = confirm_capture_package(root / "confirmation.npz", root / "fits", root / "confirmation-out", root / "lock.json", 100)
            self.assertTrue(result["synthetic"])
            self.assertAlmostEqual(float(result["p1"]["estimate"]), 0, places=5)
            self.assertEqual(float(result["p2"]["estimate"]), 0)
            self.assertEqual(result, recompute_confirmation(root / "confirmation-out"))
            model_path.write_text(model_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "differ from protocol"):
                confirm_capture_package(root / "confirmation.npz", root / "fits", root / "another-out", root / "lock.json", 100)


class AlternativeExplanationControlTests(unittest.TestCase):
    def test_role_contexts_preserve_overlapping_reference_and_target_flags(self):
        labels = make_context_labels(["selection", "selection"],
                                     {"is_target": [1, 0], "is_reference": [1, 1]})
        self.assertIn("is_reference=1;is_target=1", labels[0])
        self.assertNotEqual(labels[0], labels[1])
        with self.assertRaisesRegex(ValueError, "boolean"):
            make_context_labels(["selection"], {"is_target": [2]})

    def test_matched_contexts_use_identical_pass_population_and_report_missing(self):
        ids = np.repeat(np.arange(12), 2)
        groups = np.tile(["attribute_target", "counting_member"], 12)
        gaps = np.where(groups == "attribute_target", .2, -.2)[:, None]
        results = matched_semantic_gaps(gaps, ids, [0], groups,
                                       ["attribute_target", "counting_member", "reference_missing"], n_resamples=100)
        self.assertAlmostEqual(float(results["attribute_target"]["interval"]["estimate"]), .2)
        self.assertAlmostEqual(float(results["counting_member"]["interval"]["estimate"]), -.2)
        self.assertEqual(results["reference_missing"]["status"], "unavailable")
        self.assertEqual(results["reference_missing"]["independent_families"], 0)
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            matched_semantic_gaps(gaps, ids, [0], groups, ["attribute_target"], n_resamples=100)

    def test_discovery_balance_ignores_clutter_and_is_frozen_on_confirmation(self):
        # Each of two families has a different number of repeated object rows.
        discovery_groups = ["a"] * 10 + ["b"] * 2
        discovery_families = ["f0"] * 10 + ["f1"] * 2
        balanced_families = fit_context_balance(discovery_groups, discovery_families)
        np.testing.assert_allclose(balanced_families.discovery_probabilities, [.5, .5])
        groups = np.tile(["a"] * 9 + ["b"], 20)
        ids = np.repeat(np.arange(20), 10)
        fitted = fit_context_balance(groups, ids, clip=(.25, 2))
        self.assertEqual(fitted.weights[1], 2)
        self.assertGreater(fitted.discovery_clipped_fraction, 0)
        self.assertEqual(float(fitted.transform(["b", "b", "a"])[0]), 2)
        with self.assertRaisesRegex(ValueError, "discovery"):
            fit_context_balance(groups, ids, split="confirmation")
        with self.assertRaisesRegex(ValueError, "support"):
            fitted.transform(["new_context"])

    def test_balanced_estimate_uses_locked_weights_and_reports_clipped_distribution(self):
        groups = np.tile(["a"] * 3 + ["b"], 16)
        ids = np.repeat(np.arange(16), 4)
        fitted = fit_context_balance(groups, ids, clip=(.25, 4))
        gaps = np.where(groups == "a", 0., 1.)[:, None]
        result = balanced_semantic_gap(gaps, ids, [0], groups, fitted, n_resamples=100)
        self.assertAlmostEqual(float(result["estimate"]), .5)
        self.assertAlmostEqual(result["context_distribution_after"]["a"], .5)
        self.assertAlmostEqual(result["context_distribution_before"]["a"], .75)

    def test_paired_differences_join_question_and_object_ids_with_unordered_rows(self):
        activations = np.array([[1., 2.], [13., 14.], [3., 4.], [11., 12.]])
        paired = paired_edit_differences(activations,
            ["q0", "q0", "q0", "q0"], ["o0", "o1", "o1", "o0"],
            ["base", "color_edit", "base", "color_edit"], ["f"] * 4)
        np.testing.assert_allclose(paired["delta"], [[10., 10.], [10., 10.]])
        self.assertEqual(paired["object_ids"].tolist(), ["o0", "o1"])
        with self.assertRaisesRegex(ValueError, "Missing"):
            paired_edit_differences(activations[:3], ["q0"] * 3, ["o0", "o1", "o1"],
                                    ["base", "color_edit", "base"], ["f"] * 3)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_edit_differences(activations[:2], ["q0"] * 2, ["o0"] * 2, ["base"] * 2, ["f"] * 2)

    def test_paired_profiles_resample_whole_families_and_do_not_drop_irrelevant(self):
        ids = np.repeat(np.arange(12), 2)
        groups = np.tile(["target", "irrelevant"], 12)
        deltas = np.where(groups == "target", .5, .1)[:, None, None] * np.ones((24, 2, 1))
        profiles = paired_selectivity_profiles(deltas, ids, groups,
                                              ["target", "irrelevant"], n_resamples=100)
        np.testing.assert_allclose(profiles["target"]["interval"]["estimate"], [[.5], [.5]])
        np.testing.assert_allclose(profiles["irrelevant"]["interval"]["estimate"], [[.1], [.1]])
        self.assertEqual(profiles["irrelevant"]["paired_objects"], 12)


if __name__ == "__main__":
    unittest.main()
