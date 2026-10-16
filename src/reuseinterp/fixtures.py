"""Visibly synthetic pipeline checks, stored separately from model evidence.

run_fixtures creates raw arrays first. recompute_fixtures reads only those arrays
to regenerate every fixture statistic and prediction without any model access.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .encoding import (evaluate_semantic_transfer, fit_positive_affine,
                       validate_split_families, validate_unique_ids)
from .probes import evaluate_readout_transfer
from .statistics import (causal_profile_interaction, fit_causal_gain,
                         probe_deficit_interval, semantic_gap_interval,
                         validate_condition_rows)


FIXTURE_NAMES = (
    "stable_latent_meaning", "positive_gain_offset", "sign_reversal",
    "changed_upstream_fixed_weights", "changed_downstream_reader",
    "different_continuation_lengths",
)


def json_safe(value):
    """Convert NumPy output to strict JSON; undefined metrics become null."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_npz(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _strip_bootstrap(result):
    return {key: value for key, value in result.items() if key != "draws"}


def _negative_controls():
    """These deliberate pipeline mistakes must each raise a clear error."""
    checks = {}
    cases = {
        "family_leakage_detected": lambda: validate_split_families({"fit": ["scene-a"], "confirmation": ["scene-a"]}),
        "confirmation_calibration_detected": lambda: fit_positive_affine([[1], [2]], [[2], [4]], split="confirmation"),
        "duplicated_confirmation_ids_detected": lambda: validate_unique_ids(["confirm-1", "confirm-1"]),
        "missing_condition_detected": lambda: validate_condition_rows([
            {"example_id": "e", "physical_layer": 0, "neuron_id": 1, "invocation": "L1", "condition": "plus"}], ["plus", "minus"]),
        "duplicate_condition_detected": lambda: validate_condition_rows([
            {"example_id": "e", "physical_layer": 0, "neuron_id": 1, "invocation": "L1", "condition": "plus"}] * 2, ["plus", "minus"]),
    }
    for name, action in cases.items():
        try:
            action()
        except ValueError as error:
            checks[name] = {"passed": True, "rejection": str(error)}
        else:
            checks[name] = {"passed": False, "rejection": None}
    return checks


def run_fixtures(output_dir, seed=314159, n_resamples=2000):
    """Generate reproducible CPU fixtures and save raw data before analysis.

    All quantities are synthetic. Existing outputs are never overwritten. The
    returned result is the same object produced by recompute_fixtures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Fixture destination is nonempty; choose a new output directory")
    rng = np.random.default_rng(seed)
    raw = {}
    split_sizes = {"fit": 128, "development": 64, "calibration": 64, "confirmation": 128}
    weights = np.array([1., .4, 0., 0.])
    raw["fixed_neuron_weights"] = weights
    for split, count in split_sizes.items():
        n = count * 2
        x = rng.normal(size=(n, 4))
        raw[split + "__descriptors"] = x
        raw[split + "__family_ids"] = np.repeat(["%s-%04d" % (split, i) for i in range(count)], 2)
        raw[split + "__example_ids"] = np.array(["%s-%04d" % (split, i) for i in range(n)])
        raw[split + "__groups"] = np.tile(np.array(["attribute", "counting"]), count)
        for index, name in enumerate(FIXTURE_NAMES):
            early_input = x.copy()
            late_input = x.copy()
            if name == "positive_gain_offset":
                late_input *= 2.5
                late_input[:, 0] += 1.7
            elif name == "sign_reversal":
                late_input *= -1
            elif name == "changed_upstream_fixed_weights":
                late_input = x[:, [2, 1, 0, 3]]
            raw[name + "__" + split + "__input_L1"] = early_input
            raw[name + "__" + split + "__input_L4"] = late_input
            raw[name + "__" + split + "__L1"] = (early_input @ weights + rng.normal(0, .025, n))[:, None]
            raw[name + "__" + split + "__L4"] = (late_input @ weights + rng.normal(0, .025, n))[:, None]
        # Same semantic labels, all objects, with either portable or changed coordinates.
        labels = np.where(x[:, 0] >= 0, "red", "blue")
        state = np.column_stack([np.where(labels == "red", 1., -1.) + rng.normal(0, .05, n), x[:, 1]])
        raw[split + "__color_labels"] = labels
        raw[split + "__state_L1"] = state
        raw[split + "__state_L4_stable"] = state + rng.normal(0, .01, state.shape)
        raw[split + "__state_L4_changed"] = state * np.array([-1., 1.])
        if split in {"calibration", "confirmation"}:
            groups = raw[split + "__groups"]
            early = .3 + rng.normal(0, .025, n)
            raw[split + "__causal_early"] = early[:, None]
            raw[split + "__causal_changed_reader"] = (early * np.where(groups == "attribute", 1.0, -1.0) + rng.normal(0, .015, n))[:, None]
            # The next-block readout is unchanged; extra subsequent computation
            # alone accounts for different final-answer effects across passes.
            raw[split + "__local_horizon_L1"] = early[:, None]
            raw[split + "__local_horizon_L4"] = (early + rng.normal(0, .005, n))[:, None]
            raw[split + "__final_continuation_L1"] = (early * np.where(groups == "attribute", .2, 2.0))[:, None]
            raw[split + "__final_continuation_L4"] = early[:, None]
    _save_npz(output_dir / "raw_synthetic.npz", raw)
    _write_json(output_dir / "fixture_manifest.json", {
        "synthetic": True, "not_model_results": True, "seed": seed,
        "n_resamples": n_resamples, "split_independent_families": split_sizes,
        "raw_file": "raw_synthetic.npz", "fixed_weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "noise_sd": .025, "variance_floor": 1e-8, "primary_passes": ["L1", "L4"],
        "penalties": [.01, .1, 1., 10.], "fixture_names": list(FIXTURE_NAMES),
        "confidence": 1 - .05 / 3,
        "scope": "Software validation on synthetic fixed neurons; no empirical claim about LoopVL.",
    })
    return recompute_fixtures(output_dir, write=True)


def recompute_fixtures(output_dir, write=False):
    """Recompute metrics/predictions from saved raw synthetic measurements only.

    write=True atomically replaces derived files, never raw measurements. Useful
    for a report-reproduction command; no checkpoint, images or GPU are needed.
    """
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "fixture_manifest.json").read_text())
    if manifest.get("synthetic") is not True or manifest.get("not_model_results") is not True:
        raise ValueError("Refusing to treat an unlabeled result as a synthetic fixture")
    with np.load(output_dir / manifest["raw_file"], allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    weight_hash = hashlib.sha256(raw["fixed_neuron_weights"].tobytes()).hexdigest()
    if weight_hash != manifest["fixed_weight_sha256"]:
        raise ValueError("Fixture fixed-neuron weight checksum changed")
    seed, n_resamples = manifest["seed"], manifest["n_resamples"]
    splits = tuple(manifest["split_independent_families"])
    x = {split: raw[split + "__descriptors"] for split in splits}
    families = {split: raw[split + "__family_ids"] for split in splits}
    validate_split_families(families)
    for split in splits:
        validate_unique_ids(raw[split + "__example_ids"].tolist(), split)
    result = {"synthetic": True, "not_model_results": True, "fixtures": {}, "probe_fixtures": {},
              "negative_controls": _negative_controls(), "raw_source": manifest["raw_file"],
              "scope": manifest["scope"], "fixed_weight_sha256": weight_hash}
    predictions, fit_records = {}, {}
    for index, name in enumerate(FIXTURE_NAMES):
        y = {split: {p: raw[name + "__" + split + "__" + p] for p in ("L1", "L4")} for split in splits}
        evaluation = evaluate_semantic_transfer(x, y, families, penalties=manifest["penalties"], variance_floor=manifest["variance_floor"])
        interval = semantic_gap_interval(evaluation["gap_rows"], families["confirmation"], [0], n_resamples, seed + index)
        metrics = {key: {kind: {m: values for m, values in values_by_metric.items()}
                        for kind, values_by_metric in kinds.items()} for key, kinds in evaluation["metrics"].items()}
        raw_gap = .5 * (metrics["L1_to_L4"]["raw"]["nmse"][0] - metrics["L4_to_L4"]["raw"]["nmse"][0]
                        + metrics["L4_to_L1"]["raw"]["nmse"][0] - metrics["L1_to_L1"]["raw"]["nmse"][0])
        item = {"synthetic": True, "p1_gap": _strip_bootstrap(interval), "raw_gap": float(raw_gap),
                "transfer_matrix": metrics, "eligible_development": evaluation["eligible_development"],
                "calibrations": {key: {"alpha": model.alpha, "beta": model.beta} for key, model in evaluation["calibrations"].items()},
                "independent_confirmation_families": len(np.unique(families["confirmation"])),
                "fixed_neuron_weights_identical": True}
        result["fixtures"][name] = item
        fit_records[name] = {p: model.to_dict() for p, model in evaluation["models"].items()}
        predictions[name + "__gap_rows"] = evaluation["gap_rows"]
        for key, kinds in evaluation["predictions"].items():
            for kind, values in kinds.items():
                predictions[name + "__" + key + "__" + kind] = values
    # Probe transfer fixtures separately distinguish fixed readout portability
    # from the availability of the same linearly decodable property.
    for state_kind in ("stable", "changed"):
        states = {split: {"L1": raw[split + "__state_L1"], "L4": raw[split + "__state_L4_" + state_kind]}
                  for split in ("fit", "development", "confirmation")}
        labels = {split: raw[split + "__color_labels"] for split in states}
        evaluation = evaluate_readout_transfer(states, labels, ["blue", "red"], families, manifest["penalties"])
        interval = probe_deficit_interval(evaluation["labels"], evaluation["predictions"], evaluation["classes"], families["confirmation"],
                                          n_resamples=n_resamples, seed=seed + 10)
        result["probe_fixtures"][state_kind] = {"synthetic": True, "p2": _strip_bootstrap(interval), "transfer_matrix": evaluation["metrics"]}
        fit_records["probe_" + state_kind] = {p: model.to_dict() for p, model in evaluation["models"].items()}
        for key, values in evaluation["predictions"].items():
            predictions["probe_" + state_kind + "__" + key] = values
    locked_groups = ["attribute", "counting"]
    calibration_groups = raw["calibration__groups"]
    test_groups = raw["confirmation__groups"]
    causal_records = {}
    for name, early_key, late_key in (
        ("changed_downstream_reader", "causal_early", "causal_changed_reader"),
        ("different_continuation_lengths", "final_continuation_L1", "final_continuation_L4"),
        ("short_horizon_continuation_control", "local_horizon_L1", "local_horizon_L4"),
    ):
        calibration = fit_causal_gain(raw["calibration__" + early_key], raw["calibration__" + late_key],
                                     calibration_groups, families["calibration"], locked_groups)
        interaction = causal_profile_interaction(raw["confirmation__" + early_key], raw["confirmation__" + late_key],
            calibration["gain"], test_groups, families["confirmation"], locked_groups, "attribute", n_resamples, seed + 20)
        causal_records[name] = {"synthetic": True, "calibration": calibration, "interaction": _strip_bootstrap(interaction)}
        predictions[name + "__causal_bootstrap_contrasts"] = interaction["draws"]
    result["causal_fixtures"] = causal_records
    semantic = result["fixtures"]
    checks = {
        "stable_description_transfers": abs(float(semantic["stable_latent_meaning"]["p1_gap"]["estimate"])) < .01,
        "positive_gain_needs_raw_calibration": semantic["positive_gain_offset"]["raw_gap"] > .2,
        "positive_affine_recovers_gain": abs(float(semantic["positive_gain_offset"]["p1_gap"]["estimate"])) < .01,
        "sign_reversal_not_gain": float(semantic["sign_reversal"]["p1_gap"]["estimate"]) > .5 and float(semantic["sign_reversal"]["calibrations"]["L1_to_L4"]["alpha"][0]) == 0,
        "fixed_weights_dont_imply_semantic_transfer": float(semantic["changed_upstream_fixed_weights"]["p1_gap"]["estimate"]) > .2,
        "stable_selectivity_can_have_changed_causal_profile": abs(float(semantic["changed_downstream_reader"]["p1_gap"]["estimate"])) < .01 and
            abs(float(causal_records["changed_downstream_reader"]["interaction"]["estimate"][0, 0])) > .3,
        "continuation_alternative_detected": abs(float(causal_records["different_continuation_lengths"]["interaction"]["estimate"][0, 0])) > .1 and
            abs(float(causal_records["short_horizon_continuation_control"]["interaction"]["estimate"][0, 0])) < .01,
        "probe_stable_format_transfers": abs(float(result["probe_fixtures"]["stable"]["p2"]["estimate"])) < .02,
        "probe_changed_format_retains_within_pass_information": float(result["probe_fixtures"]["changed"]["p2"]["estimate"]) > .9,
    }
    checks.update({name: value["passed"] for name, value in result["negative_controls"].items()})
    result["checks"] = checks
    result["all_passed"] = bool(all(checks.values()))
    result["interpretation_constraints"] = [
        "All fixture outputs are synthetic software checks, excluded from model findings.",
        "Transfer failure with fixed neuron weights does not imply a change in its mathematical function.",
        "Final-answer effect differences with equal short-horizon effects can arise from remaining continuation.",
        "Changed final effects alone establish pass-dependent outcome sensitivity, not semantic reassignment.",
        "A near-zero fitted causal gain or late profile is weak/absent measured influence or unresolved, not causal equivalence.",
    ]
    if write:
        _save_npz(output_dir / "synthetic_predictions.npz", predictions)
        _write_json(output_dir / "synthetic_fits.json", fit_records)
        _write_json(output_dir / "metrics.json", result)
    if not result["all_passed"]:
        raise AssertionError("Synthetic fixture checks failed: %s" % [name for name, passed in checks.items() if not passed])
    return json_safe(result)
