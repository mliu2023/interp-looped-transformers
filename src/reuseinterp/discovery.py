"""CPU analysis handoff for aligned, independently split activation packages.

The discovery function refuses confirmation/transfer arrays. The confirmation
function loads frozen fits and requires a hash-verified protocol lock. An NPZ is
an interchange format, not evidence that captures or model gates actually ran.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .encoding import (FORBIDDEN_PRIMARY_DESCRIPTORS, DescriptorSpec, PositiveAffine, RidgeEncoder,
                       family_weights, fit_positive_affine, fit_ridge,
                       regression_metrics, validate_split_families, validate_unique_ids)
from .fixtures import _save_npz, json_safe
from .io import atomic_json, digest, file_hash, read_json
from .probes import LinearReadout, balanced_accuracy, fit_linear_readout
from .protocol import verify_lock
from .statistics import probe_deficit_interval, semantic_gap_interval


def default_descriptor_spec():
    """Frozen primary 2D descriptor specification; no answer-derived fields."""
    from .data.scenes import COLORS, SHAPES, SIZES, POSITIONS, TASK_FAMILIES
    return DescriptorSpec(
        {"color": list(COLORS), "shape": list(SHAPES), "size": list(SIZES),
         "position": list(POSITIONS), "question_family": list(TASK_FAMILIES)},
        ["center_x", "center_y", "visible_fraction", "mask_area_fraction", "is_reference", "is_target",
         "count_set_member", "compare_set_a_member", "compare_set_b_member", "background_luminance", "clutter_level"],
        [("color", "is_target"), ("color", "is_reference"), ("color", "question_family"),
         ("shape", "question_family"), ("size", "question_family")])


def assemble_capture_package(capture_rows, dataset_dir, neuron_sample, output_npz,
                              config, splits=("fit", "development", "calibration")):
    """Join committed physical captures to independently validated object labels.

    capture_rows must come from a successful shard merge. This function checks
    every artifact hash and exact example/object coverage, uses recorded mask
    labels to align scene object IDs, and preserves all objects/questions/edits.
    It never invokes a model. Only the object-pooled stratum is supported here;
    last-prompt fitting needs a separately frozen question-level descriptor set.
    Returns an analysis config including the frozen expanded feature vocabulary.
    """
    from .data.scenes import validate_dataset
    if config.get("token_stratum", "object_pooled") != "object_pooled":
        raise ValueError("This assembler handles object_pooled only")
    output_npz = Path(output_npz)
    if output_npz.exists():
        raise FileExistsError("Never overwrite an aligned capture package")
    splits = tuple(splits)
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("Declare unique requested splits")
    if (set(splits) & {"confirmation", "transfer"}) and len(splits) != 1:
        raise ValueError("Confirmation and transfer packages must remain isolated")
    dataset_dir = Path(dataset_dir)
    validation = validate_dataset(dataset_dir)
    manifest = read_json(dataset_dir / "manifest.json")
    def jsonl(name):
        with (dataset_dir / manifest["paths"][name]).open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    examples = {row["example_id"]: row for row in jsonl("examples") if row["split"] in splits}
    scenes = {row["scene_id"]: row for row in jsonl("scenes") if row["split"] in splits}
    descriptors = {(row["example_id"], row["object_id"]): row for row in jsonl("descriptors") if row["split"] in splits}
    rows = list(capture_rows)
    validate_unique_ids([row["example_id"] for row in rows], "captured example")
    if set(examples) != {row["example_id"] for row in rows}:
        raise ValueError("Missing or unexpected captured examples for requested splits")
    if isinstance(neuron_sample, (str, Path)):
        neuron_sample = read_json(neuron_sample)
    channels = neuron_sample["channels"]
    ordered_layers = sorted(channels, key=int)
    layer_ids = [int(layer) for layer in ordered_layers for _ in channels[layer]]
    neuron_ids = [int(index) for layer in ordered_layers for index in channels[layer]]
    spec_config = config.get("descriptor_spec")
    spec = default_descriptor_spec() if spec_config is None else DescriptorSpec(**spec_config)
    spec_hash = digest(asdict(spec))
    data = {split: {"features": [], "activations": [], "family_ids": [], "row_ids": [], "color_labels": [], "states": {}}
            for split in splits}
    state_layers = config.get("state_layers", [0, 5, 10, 15])
    for row in sorted(rows, key=lambda r: r["example_id"]):
        example = examples[row["example_id"]]
        if row["split"] != example["split"] or row["family_id"] != example["family_id"]:
            raise ValueError("Capture trace differs from scene-family assignment")
        if row.get("channel_selection_hash") != digest(channels):
            raise ValueError("Capture representative sample changed")
        path = Path(row["artifact"])
        if file_hash(path) != row["sha256"]:
            raise ValueError("Capture tensor checksum mismatch")
        with np.load(path, allow_pickle=False) as saved:
            tensors = {key: saved[key] for key in saved.files}
        scene = scenes[example["scene_id"]]
        by_mask = {obj["mask_label"]: obj["object_id"] for obj in scene["objects"]}
        labels = tensors["mask_labels"].tolist()
        if labels != row["object_mask_labels"] or set(labels) != set(by_mask) or len(labels) != len(set(labels)):
            raise ValueError("Captured mask/object labels do not align with scene graph")
        products = []
        for layer in ordered_layers:
            values = tensors["products_layer_" + layer]
            if values.shape != (6, len(labels) + 1, len(channels[layer])):
                raise ValueError("Physical product tensor has incomplete pass/object/neuron coverage")
            products.append(values[:, :-1, :])
        # objects first, invocation second, then fixed physical neuron sample.
        product = np.concatenate(products, axis=2).transpose(1, 0, 2)
        bucket = data[example["split"]]
        for oi, label in enumerate(labels):
            descriptor = descriptors[(example["example_id"], by_mask[label])]
            bucket["features"].append(descriptor["features"])
            bucket["activations"].append(product[oi])
            bucket["family_ids"].append(example["family_id"])
            bucket["row_ids"].append(example["example_id"] + "::" + by_mask[label])
            bucket["color_labels"].append(descriptor["features"]["color"])
        for layer in state_layers:
            for kind in ("input", "block"):
                tensor_key = "states_layer_%d_%s" % (layer, kind)
                if tensor_key not in tensors:
                    raise ValueError("Missing prespecified representation state %s" % tensor_key)
                state = tensors[tensor_key]
                if state.ndim != 3 or state.shape[:2] != (6, len(labels) + 1):
                    raise ValueError("Representation state/object/pass alignment failed")
                site = "layer_%d_%s" % (layer, kind)
                bucket["states"].setdefault(site, []).extend(state[:, :-1].transpose(1, 0, 2))
    arrays = {"passes": np.asarray(["L%d" % i for i in range(1, 7)]),
              "physical_layers": np.asarray(layer_ids), "neuron_ids": np.asarray(neuron_ids)}
    feature_names = None
    for split in splits:
        bucket = data[split]
        values, names = spec.transform(bucket["features"])
        if feature_names is not None and names != feature_names:
            raise ValueError("Descriptor vocabulary changed across splits")
        feature_names = names
        arrays[split + "__descriptors"] = values
        for field in ("activations", "family_ids", "row_ids", "color_labels"):
            arrays[split + "__" + field] = np.asarray(bucket[field])
        for site, states in bucket["states"].items():
            arrays["probe__" + site + "__" + split] = np.asarray(states)
    arrays["feature_names"] = np.asarray(feature_names)
    arrays["manifest_json"] = np.asarray(json.dumps({
        "checkpoint_revision": config["checkpoint_revision"], "dataset_hash": manifest["dataset_hash"],
        "token_stratum": "object_pooled", "descriptor_spec_hash": spec_hash,
        "capture_row_hash": digest(rows), "neuron_sample_hash": digest(neuron_sample),
        "dataset_validation_hash": digest(validation), "synthetic": bool(config.get("synthetic", False)),
        "distribution": "deterministic_2d_not_clevr", "decode_step": 0,
        "aggregation": "validated renderer-mask weights normalized within every visible object"}, sort_keys=True))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    _save_npz(output_npz, arrays)
    return dict(config, descriptor_feature_names=feature_names, descriptor_spec_hash=spec_hash)


def _load_package(path, stage):
    with np.load(path, allow_pickle=False) as saved:
        # Inspect keys before reading any held-out numerical array into memory.
        if stage == "discovery" and any(key.startswith(("confirmation__", "transfer__")) or key.endswith(("__confirmation", "__transfer")) for key in saved.files):
            raise ValueError("Discovery must not open confirmation or transfer arrays")
        if stage == "confirmation" and any(key.startswith(("fit__", "development__", "calibration__", "transfer__")) or key.endswith(("__fit", "__development", "__calibration", "__transfer")) for key in saved.files):
            raise ValueError("Confirmation package must contain only confirmation rows")
        arrays = {key: saved[key] for key in saved.files}
    for name in ("passes", "physical_layers", "neuron_ids", "feature_names", "manifest_json"):
        if name not in arrays:
            raise ValueError("Aligned capture package is missing %s" % name)
    arrays["manifest"] = json.loads(str(arrays["manifest_json"].item()))
    manifest = arrays["manifest"]
    for name in ("checkpoint_revision", "dataset_hash", "token_stratum", "descriptor_spec_hash"):
        if not manifest.get(name):
            raise ValueError("Capture provenance is missing %s" % name)
    layers, neurons = arrays["physical_layers"], arrays["neuron_ids"]
    if layers.ndim != 1 or neurons.shape != layers.shape or len(layers) == 0:
        raise ValueError("Physical site labels disagree")
    validate_unique_ids(list(zip(layers.tolist(), neurons.tolist())), "physical neuron")
    feature_names = arrays["feature_names"].tolist()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate descriptor feature names")
    # Expanded categorical names are checked against their base field as well.
    for name in feature_names:
        terms = str(name).replace("*", " ").split()
        if any(term.split("=")[0].split("[")[0] in FORBIDDEN_PRIMARY_DESCRIPTORS for term in terms):
            raise ValueError("Answer-derived feature in primary encoding: %s" % name)
    if len(set(arrays["passes"].tolist())) != len(arrays["passes"]):
        raise ValueError("Duplicate invocation labels")
    return arrays


def _check_split(arrays, split):
    required = ("descriptors", "activations", "family_ids", "row_ids")
    for field in required:
        if split + "__" + field not in arrays:
            raise ValueError("Missing %s capture field %s" % (split, field))
    x, y = arrays[split + "__descriptors"], arrays[split + "__activations"]
    n = len(x)
    if x.ndim != 2 or x.shape[1] != len(arrays["feature_names"]):
        raise ValueError("Descriptor vocabulary/dimensions disagree")
    if y.shape != (n, len(arrays["passes"]), len(arrays["neuron_ids"])):
        raise ValueError("Expected activations [matched row, invocation, physical neuron]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Captures contain nonfinite values; record exclusions explicitly")
    ids, row_ids = arrays[split + "__family_ids"], arrays[split + "__row_ids"]
    if ids.shape != (n,) or row_ids.shape != (n,) or n == 0:
        raise ValueError("Trace/family coverage disagrees")
    validate_unique_ids(row_ids.tolist(), split)
    return x, y, ids, row_ids


def _new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError("Analysis output directory is nonempty")
    return path


def fit_capture_package(input_npz, output_dir, config):
    """Freeze observational discovery models from saved CPU arrays.

    Required global fields: passes, physical_layers, neuron_ids, feature_names,
    manifest_json. For fit/development/calibration, include SPLIT__descriptors,
    SPLIT__activations [row,pass,unit], SPLIT__family_ids, SPLIT__row_ids.
    Optional color readout arrays are probe__SITE__SPLIT [row,pass,feature] and
    SPLIT__color_labels. All objects and passes must share the same row order.
    See this module's discovery-package tests for an executable small example.

    config must include descriptor_feature_names and descriptor_spec_hash from
    the frozen whitelist, and may specify penalties, variance_floor, p2_site.
    Calibration families fit gains only; site selection uses development only.
    """
    arrays = _load_package(input_npz, "discovery")
    if arrays["feature_names"].tolist() != config["descriptor_feature_names"]:
        raise ValueError("Capture features differ from frozen descriptor whitelist")
    if arrays["manifest"]["descriptor_spec_hash"] != config["descriptor_spec_hash"]:
        raise ValueError("Descriptor specification hash mismatch")
    split_data = {split: _check_split(arrays, split) for split in ("fit", "development", "calibration")}
    families = {split: values[2].tolist() for split, values in split_data.items()}
    validate_split_families(families)
    validate_unique_ids([row for values in split_data.values() for row in values[3].tolist()], "discovery")
    passes = arrays["passes"].tolist()
    primary = config.get("primary_passes", ["L1", "L4"])
    if len(primary) != 2 or any(p not in passes for p in primary):
        raise ValueError("Primary pass coverage is incomplete")
    floor = float(config.get("variance_floor", 1e-8))
    if floor <= 0:
        raise ValueError("Variance floor must be fixed above zero")
    penalties = config.get("penalties", [.01, .1, 1., 10., 100.])
    xf, yf, ff, _ = split_data["fit"]
    xd, yd, fd, _ = split_data["development"]
    xc, yc, fc, _ = split_data["calibration"]
    models, calibration, variance, dev_r2, dev_variance = {}, {}, {}, {}, {}
    predictions = {}
    for pi, p in enumerate(passes):
        models[p] = fit_ridge(xf, yf[:, pi], xd, yd[:, pi], penalties, family_weights(ff), family_weights(fd))
        current = yc[:, pi]
        variance[p] = np.average((current - np.average(current, axis=0, weights=family_weights(fc))) ** 2,
                                 axis=0, weights=family_weights(fc))
        dev_variance[p] = np.average((yd[:, pi] - np.average(yd[:, pi], axis=0, weights=family_weights(fd))) ** 2,
                                    axis=0, weights=family_weights(fd))
        dev_prediction = models[p].predict(xd)
        dev_r2[p] = regression_metrics(yd[:, pi], dev_prediction, variance[p], floor, family_weights(fd))["r2"]
        predictions["development__" + p + "__within_prediction"] = dev_prediction
    for p in passes:
        source_cal = models[p].predict(xc)
        for qi, q in enumerate(passes):
            affine = fit_positive_affine(source_cal, yc[:, qi], family_weights(fc))
            calibration[p + "_to_" + q] = {"alpha": affine.alpha, "beta": affine.beta}
    p, q = primary
    eligible = (dev_variance[p] >= floor) & (dev_variance[q] >= floor) & (dev_r2[p] >= .1) & (dev_r2[q] >= .1)
    # Probe candidates are evaluated on development; calibration never ranks sites.
    probe_sites = sorted({key.split("__")[1] for key in arrays if key.startswith("probe__")})
    probes, probe_development = {}, {}
    if probe_sites:
        labels_fit = arrays["fit__color_labels"]
        labels_dev = arrays["development__color_labels"]
        classes = config.get("color_classes", np.unique(labels_fit).tolist())
        for site in probe_sites:
            probes[site], probe_development[site] = {}, {}
            for pi, invocation in enumerate(passes):
                sf = arrays["probe__" + site + "__fit"]
                sd = arrays["probe__" + site + "__development"]
                if sf.shape[:2] != yf.shape[:2] or sd.shape[:2] != yd.shape[:2]:
                    raise ValueError("Probe object/trace rows do not match encoding rows")
                probe = fit_linear_readout(sf[:, pi], labels_fit, sd[:, pi], labels_dev, classes, penalties,
                                           family_weights(ff), family_weights(fd))
                probes[site][invocation] = probe.to_dict()
                probe_development[site][invocation] = balanced_accuracy(labels_dev, probe.predict(sd[:, pi]), classes, family_weights(fd))
    selected_site = config.get("p2_site")
    if selected_site is not None and selected_site not in probes:
        raise ValueError("Locked P2 site is absent from saved discovery states")
    if selected_site is None and probes:
        adequate = [site for site in probe_sites if min(probe_development[site][p], probe_development[site][q]) >= config.get("probe_min_development_ba", .65)]
        if adequate:
            selected_site = max(adequate, key=lambda site: .5 * (probe_development[site][p] + probe_development[site][q]))
    position_columns, position_baseline = [], None
    if probe_sites:
        position_columns = [i for i, name in enumerate(arrays["feature_names"].tolist())
                            if name.startswith(("position=", "question_family=")) or name in {"center_x", "center_y"}]
        if position_columns:
            baseline = fit_linear_readout(xf[:, position_columns], labels_fit, xd[:, position_columns], labels_dev,
                                          classes, penalties, family_weights(ff), family_weights(fd))
            position_baseline = baseline.to_dict()
    output_dir = _new_output(output_dir)
    record = {"schema_version": 1, "status": "frozen_observational_discovery", "synthetic": bool(arrays["manifest"].get("synthetic", False)),
              "input_sha256": file_hash(input_npz), "config_hash": digest(config), "manifest": arrays["manifest"],
              "descriptor_feature_names": arrays["feature_names"], "descriptor_spec_hash": config["descriptor_spec_hash"],
              "passes": passes, "primary_passes": primary, "physical_layers": arrays["physical_layers"], "neuron_ids": arrays["neuron_ids"],
              "split_family_ids": {split: sorted(set(ids)) for split, ids in families.items()},
              "split_row_ids": {split: values[3] for split, values in split_data.items()},
              "models": {p: model.to_dict() for p, model in models.items()}, "calibrations": calibration,
              "calibration_variance": variance, "variance_floor": floor, "eligible_development": eligible,
              "development_r2": dev_r2, "penalties": penalties, "probe_models": probes,
              "probe_development_balanced_accuracy": probe_development, "p2_site": selected_site,
              "position_question_baseline": position_baseline, "position_question_columns": position_columns,
              "p2_status": "selected_on_development" if selected_site is not None else "pending_no_adequate_saved_color_probe",
              "causal_status": "pending_separate_validated_intervention_package",
              "control_status": {
                  "matched_role_task_family": "standalone_callable_pending_integration",
                  "discovery_fitted_clipped_reweighting": "standalone_callable_pending_integration",
                  "paired_edit_selectivity": "standalone_callable_pending_integration",
                  "coherent_family_label_permutation": "standalone_callable_pending_integration",
                  "question_position_probe_baseline": "fitted_discovery_only" if position_baseline else "pending_no_question_position_features",
                  "calibration_diagnostic": "frozen_positive_affine_with_raw_comparison"},
              "uncertainty_scope": "Confirmation bootstrap conditions on these frozen fitted models and calibration parameters."}
    atomic_json(output_dir / "discovery_models.json", json_safe(record))
    _save_npz(output_dir / "discovery_predictions.npz", predictions)
    return json_safe(record)


def _restore_encoder(value):
    return RidgeEncoder(**{key: np.asarray(value[key]) for key in
                           ("feature_mean", "feature_scale", "target_mean", "coefficients", "penalties")})


def confirm_capture_package(input_npz, discovery_dir, output_dir, protocol_lock,
                            n_resamples=2000, seed=1729):
    """Apply frozen fits to saved confirmation arrays after immutable lock.

    No fitting occurs. The protocol's discovery_artifact_hashes must contain
    discovery_models.json with its exact SHA-256. Every confirmation family is
    checked against discovery/calibration families before predictions are made.
    Returns saved P1/P2 metrics; P3 requires a separate intervention package.
    """
    lock = verify_lock(protocol_lock)
    protocol = lock["protocol"]
    statistics = protocol.get("statistics", {})
    if statistics.get("resamples") != n_resamples or statistics.get("seed") != seed:
        raise ValueError("Bootstrap resample count and seed must match the protocol lock")
    if statistics.get("confidence", 1 - .05 / 3) != 1 - .05 / 3:
        raise ValueError("Keep the three-endpoint confidence allocation")
    model_path = Path(discovery_dir) / "discovery_models.json"
    expected = protocol["discovery_artifact_hashes"].get("discovery_models.json")
    if expected != file_hash(model_path):
        raise ValueError("Frozen discovery fits are absent from or differ from protocol lock")
    record = read_json(model_path)
    arrays = _load_package(input_npz, "confirmation")
    for key in ("checkpoint_revision", "dataset_hash", "token_stratum", "descriptor_spec_hash"):
        if arrays["manifest"][key] != record["manifest"][key]:
            raise ValueError("Confirmation provenance changed: %s" % key)
    for key, record_key in (("passes", "passes"), ("physical_layers", "physical_layers"), ("neuron_ids", "neuron_ids"),
                            ("feature_names", "descriptor_feature_names")):
        if arrays[key].tolist() != record[record_key]:
            raise ValueError("Confirmation site/feature vocabulary changed: %s" % key)
    x, y, families, row_ids = _check_split(arrays, "confirmation")
    validate_split_families(dict(record["split_family_ids"], confirmation=families.tolist()))
    validate_unique_ids([row for ids in record["split_row_ids"].values() for row in ids] + row_ids.tolist(), "all split rows")
    declared_ids = protocol.get("confirmation_family_ids")
    if declared_ids is None or sorted(set(families.tolist())) != sorted(declared_ids):
        raise ValueError("Exact confirmation_family_ids must be locked before evaluation")
    if protocol["p2_site"] != record["p2_site"]:
        raise ValueError("P2 site changed since discovery")
    predictions, metric_matrix, errors = {}, {}, {}
    passes = record["passes"]
    primary = record["primary_passes"]
    floor = record["variance_floor"]
    weights = family_weights(families)
    predictions.update({"family_ids": families, "row_ids": row_ids,
                        "physical_layers": np.asarray(record["physical_layers"]),
                        "neuron_ids": np.asarray(record["neuron_ids"]),
                        "passes": np.asarray(passes), "encoding_targets": y,
                        "eligible_development": np.asarray(record["eligible_development"])})
    for p in passes:
        raw = _restore_encoder(record["models"][p]).predict(x)
        # Save each source prediction once. All target calibrations are scalar
        # and reproducible from the copied analysis constants, so materializing
        # 36 almost-duplicate prediction tables would waste the output budget.
        predictions["encoding_source_prediction__" + p] = raw
        for qi, q in enumerate(passes):
            key = p + "_to_" + q
            affine = record["calibrations"][key]
            calibrated = PositiveAffine(np.asarray(affine["alpha"]), np.asarray(affine["beta"])).predict(raw)
            metric_matrix[key] = {kind: regression_metrics(y[:, qi], values, record["calibration_variance"][q], floor, weights)
                                 for kind, values in (("raw", raw), ("calibrated", calibrated))}
            if p in primary and q in primary:
                errors[key] = (calibrated - y[:, qi]) ** 2
    p, q = record["primary_passes"]
    variance = {r: np.asarray(record["calibration_variance"][r]) for r in (p, q)}
    usable = (variance[p] >= floor) & (variance[q] >= floor)
    gap_rows = np.full_like(errors[p + "_to_" + q], np.nan)
    gap_rows[:, usable] = .5 * (
        (errors[p + "_to_" + q] - errors[q + "_to_" + q])[:, usable] / variance[q][usable]
        + (errors[q + "_to_" + p] - errors[p + "_to_" + p])[:, usable] / variance[p][usable])
    predictions["semantic_gap_rows"] = gap_rows
    p1 = semantic_gap_interval(gap_rows, families, record["physical_layers"], n_resamples, seed)
    p1_descriptive = semantic_gap_interval(gap_rows, families, record["physical_layers"], n_resamples, seed, confidence=.95)
    result = {"schema_version": 1, "synthetic": record["synthetic"], "protocol_hash": lock["protocol_hash"],
              "input_sha256": file_hash(input_npz), "discovery_fit_sha256": expected,
              "confirmation_independent_families": len(set(families.tolist())), "confirmation_rows": len(row_ids),
              "encoding_transfer_matrix": metric_matrix,
              "p1": {k: v for k, v in p1.items() if k != "draws"},
              "p1_descriptive_95": {k: v for k, v in p1_descriptive.items() if k != "draws"},
              "eligible_development": record["eligible_development"],
              "control_status": record["control_status"],
              "per_neuron_gap": np.average(gap_rows[:, usable], axis=0, weights=weights),
              "usable_neuron_indices": np.flatnonzero(usable),
              "numerically_usable_neurons": int(usable.sum()), "sampled_neurons": len(usable),
              "p2": {"status": "pending_no_adequate_saved_color_probe"},
              "p3": {"status": "pending_separate_validated_intervention_package"},
              "scope": "Frozen observational encoding/readout transfer; no causal interpretation."}
    site = record["p2_site"]
    if site is not None:
        states = arrays["probe__" + site + "__confirmation"]
        labels = arrays["confirmation__color_labels"]
        if states.shape[:2] != y.shape[:2] or len(labels) != len(y):
            raise ValueError("Confirmation probe rows differ from encoding objects")
        probe_predictions, classes = {}, None
        for source in (p, q):
            saved = record["probe_models"][site][source]
            model = LinearReadout(_restore_encoder(saved["encoder"]), np.asarray(saved["classes"]), saved["selected_penalty"])
            classes = model.classes
            for target in (p, q):
                key = source + "_to_" + target
                probe_predictions[key] = model.predict(states[:, passes.index(target)])
                predictions["probe__" + key] = probe_predictions[key]
        p2 = probe_deficit_interval(labels, probe_predictions, classes, families, (p, q), n_resamples, seed)
        result["p2"] = {k: v for k, v in p2.items() if k != "draws"}
        p2_descriptive = probe_deficit_interval(labels, probe_predictions, classes, families, (p, q), n_resamples, seed, confidence=.95)
        result["p2_descriptive_95"] = {k: v for k, v in p2_descriptive.items() if k != "draws"}
        result["p2"]["site"] = site
        result["p2"]["transfer_balanced_accuracy"] = {key: balanced_accuracy(labels, values, classes, weights)
                                                       for key, values in probe_predictions.items()}
        predictions["probe_labels"] = labels
        predictions["probe_classes"] = classes
        if record["position_question_baseline"] is not None:
            saved_baseline = record["position_question_baseline"]
            baseline = LinearReadout(_restore_encoder(saved_baseline["encoder"]), np.asarray(saved_baseline["classes"]), saved_baseline["selected_penalty"])
            baseline_prediction = baseline.predict(x[:, record["position_question_columns"]])
            predictions["question_position_baseline"] = baseline_prediction
            result["p2"]["question_position_baseline_balanced_accuracy"] = balanced_accuracy(labels, baseline_prediction, classes, weights)
            result["control_status"]["question_position_probe_baseline"] = "evaluated_frozen_baseline"
    output_dir = _new_output(output_dir)
    _save_npz(output_dir / "confirmation_predictions.npz", predictions)
    atomic_json(output_dir / "analysis_constants.json", json_safe({
        "schema_version": 1, "protocol_hash": lock["protocol_hash"], "synthetic": record["synthetic"],
        "calibrations": record["calibrations"], "calibration_variance": record["calibration_variance"],
        "variance_floor": floor, "primary_passes": record["primary_passes"],
        "n_resamples": n_resamples, "seed": seed, "p2_site": site,
        "input_sha256": file_hash(input_npz), "discovery_fit_sha256": expected,
        "prediction_sha256": file_hash(output_dir / "confirmation_predictions.npz")}))
    atomic_json(output_dir / "metrics.json", json_safe(result))
    return json_safe(result)


def recompute_confirmation(output_dir, write=False):
    """Rebuild P1/P2 and all transfer matrix metrics from the small saved package.

    Requires no model weights, source capture files or descriptor arrays. Only
    raw target activations, frozen source predictions, calibration constants,
    probe predictions and complete family IDs are read. A saved checksum guards
    accidental edits. Returns the recomputed metrics, never launches inference.
    """
    output_dir = Path(output_dir)
    constants = read_json(output_dir / "analysis_constants.json")
    prediction_path = output_dir / "confirmation_predictions.npz"
    if file_hash(prediction_path) != constants["prediction_sha256"]:
        raise ValueError("Saved confirmation prediction checksum mismatch")
    with np.load(prediction_path, allow_pickle=False) as saved:
        raw = {key: saved[key] for key in saved.files}
    result = read_json(output_dir / "metrics.json")
    passes = raw["passes"].tolist()
    families, weights = raw["family_ids"], family_weights(raw["family_ids"])
    matrices = {}
    for p in passes:
        source = raw["encoding_source_prediction__" + p]
        for qi, q in enumerate(passes):
            key = p + "_to_" + q
            affine = constants["calibrations"][key]
            calibrated = PositiveAffine(np.asarray(affine["alpha"]), np.asarray(affine["beta"])).predict(source)
            matrices[key] = {kind: regression_metrics(raw["encoding_targets"][:, qi], values,
                constants["calibration_variance"][q], constants["variance_floor"], weights)
                for kind, values in (("raw", source), ("calibrated", calibrated))}
    result["encoding_transfer_matrix"] = matrices
    p, q = constants["primary_passes"]
    rows = {}
    for source, target in ((p, q), (q, q), (q, p), (p, p)):
        key = source + "_to_" + target
        affine = constants["calibrations"][key]
        calibrated = PositiveAffine(np.asarray(affine["alpha"]), np.asarray(affine["beta"])).predict(raw["encoding_source_prediction__" + source])
        rows[key] = (calibrated - raw["encoding_targets"][:, passes.index(target)]) ** 2
    vp, vq = np.asarray(constants["calibration_variance"][p]), np.asarray(constants["calibration_variance"][q])
    valid = (vp >= constants["variance_floor"]) & (vq >= constants["variance_floor"])
    gaps = np.full_like(rows[p + "_to_" + q], np.nan)
    gaps[:, valid] = .5 * ((rows[p + "_to_" + q] - rows[q + "_to_" + q])[:, valid] / vq[valid]
                         + (rows[q + "_to_" + p] - rows[p + "_to_" + p])[:, valid] / vp[valid])
    if not np.allclose(gaps, raw["semantic_gap_rows"], rtol=0, atol=0, equal_nan=True):
        raise ValueError("Saved gap rows disagree with recomputed frozen predictions")
    for label, confidence in (("p1", 1 - .05 / 3), ("p1_descriptive_95", .95)):
        interval = semantic_gap_interval(gaps, families, raw["physical_layers"], constants["n_resamples"], constants["seed"], confidence)
        result[label] = {key: value for key, value in interval.items() if key != "draws"}
    if constants["p2_site"] is not None:
        probe_predictions = {source + "_to_" + target: raw["probe__" + source + "_to_" + target]
                             for source in (p, q) for target in (p, q)}
        interval = probe_deficit_interval(raw["probe_labels"], probe_predictions, raw["probe_classes"], families,
                                          (p, q), constants["n_resamples"], constants["seed"])
        result["p2"] = {key: value for key, value in interval.items() if key != "draws"}
        descriptive = probe_deficit_interval(raw["probe_labels"], probe_predictions, raw["probe_classes"], families,
                                              (p, q), constants["n_resamples"], constants["seed"], confidence=.95)
        result["p2_descriptive_95"] = {key: value for key, value in descriptive.items() if key != "draws"}
        result["p2"].update({"site": constants["p2_site"], "transfer_balanced_accuracy": {
            key: balanced_accuracy(raw["probe_labels"], values, raw["probe_classes"], weights)
            for key, values in probe_predictions.items()}})
        if "question_position_baseline" in raw:
            result["p2"]["question_position_baseline_balanced_accuracy"] = balanced_accuracy(
                raw["probe_labels"], raw["question_position_baseline"], raw["probe_classes"], weights)
    if write:
        atomic_json(output_dir / "metrics.json", json_safe(result))
    return json_safe(result)
