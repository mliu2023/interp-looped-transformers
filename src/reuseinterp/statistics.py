"""Family-clustered uncertainty for frozen analysis models and calibrations.

Bootstrap draws preserve complete scene-family blocks across versions, objects,
questions, passes and conditions. None of these routines refits an encoder,
calibrator, readout, candidate list or site selection.
"""
from typing import Callable

import numpy as np

from .encoding import _weights, family_weights, matrix


def _family_groups(family_ids, n):
    ids = np.asarray(family_ids)
    if ids.shape != (n,):
        raise ValueError("One family ID is required per measurement row")
    families = np.unique(ids)
    if len(families) < 2:
        raise ValueError("Uncertainty requires at least two independent families")
    return families, [np.flatnonzero(ids == family) for family in families]


def family_bootstrap(values, family_ids, estimator: Callable,
                     n_resamples=2000, seed=0, confidence=.95):
    """Resample whole families and evaluate an estimator of saved measurements.

    estimator(values, resampled_family_ids) must keep all fitted quantities
    frozen. Repeated sampled families receive distinct bootstrap copy IDs so
    equal-family weighting retains the sampled family's multiplicity.
    """
    values = np.asarray(values)
    families, groups = _family_groups(family_ids, len(values))
    if n_resamples < 100 or not 0 < confidence < 1:
        raise ValueError("At least 100 bootstrap draws and a valid confidence are required")
    estimate = np.asarray(estimator(values, np.asarray(family_ids)), float)
    if not np.isfinite(estimate).all():
        raise ValueError("Observed estimand is undefined; report missing/underpopulated groups")
    generator = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        selected = generator.integers(0, len(families), len(families))
        indices = np.concatenate([groups[index] for index in selected])
        copy_ids = np.concatenate([np.full(len(groups[index]), copy) for copy, index in enumerate(selected)])
        draws.append(estimator(values[indices], copy_ids))
    draws = np.asarray(draws, float)
    # Missing locked groups are never replaced with a zero effect.
    valid = np.isfinite(draws).reshape(n_resamples, -1).all(axis=1)
    if valid.sum() < .95 * n_resamples:
        raise ValueError("More than 5% of bootstrap draws lack estimable locked groups")
    draws = draws[valid]
    alpha = 1.0 - confidence
    return {"estimate": estimate, "lower": np.quantile(draws, alpha / 2, axis=0),
            "upper": np.quantile(draws, 1 - alpha / 2, axis=0), "draws": draws,
            "confidence": confidence, "independent_families": len(families),
            "requested_resamples": n_resamples, "valid_resamples": int(valid.sum()),
            "uncertainty_scope": "Conditional on frozen fitted models, calibration parameters and selections."}


def layer_balanced_mean(per_neuron, physical_layers):
    """Equal mean weight per physical layer, equal weight per valid sampled unit.

    NaN flags a failed/low-variance unit. Missing layers are an error: dropping
    an entire stratum silently would change the declared estimand.
    """
    values, layers = np.asarray(per_neuron, float), np.asarray(physical_layers)
    if values.ndim != 1 or layers.shape != values.shape:
        raise ValueError("Layer labels must identify every sampled neuron")
    averages = []
    for layer in np.unique(layers):
        selected = values[layers == layer]
        valid = np.isfinite(selected)
        if not valid.any():
            raise ValueError("No numerically usable neurons in layer %s" % layer)
        averages.append(np.mean(selected[valid]))
    return float(np.mean(averages))


def semantic_gap_interval(gap_rows, family_ids, physical_layers, n_resamples=2000,
                          seed=0, confidence=1 - .05 / 3):
    """P1 with whole-family bootstrap and fixed layer/neuron/calibration weights."""
    gap_rows = np.asarray(gap_rows, float)
    if gap_rows.ndim != 2 or gap_rows.shape[1] != len(physical_layers):
        raise ValueError("Expected [row, sampled neuron] transfer gaps")
    # Exclude only units globally flagged undefined, never individual high errors.
    usable = np.isfinite(gap_rows).all(axis=0)
    if np.any(np.isfinite(gap_rows).any(axis=0) & ~usable):
        raise ValueError("Partial missing neuron rows require explicit exclusions before analysis")
    def estimator(rows, ids):
        means = np.full(rows.shape[1], np.nan)
        means[usable] = np.average(rows[:, usable], axis=0, weights=family_weights(ids))
        return layer_balanced_mean(means, physical_layers)
    # The estimator understands all-NaN unusable columns; bootstrap validates output.
    result = family_bootstrap(gap_rows, family_ids, estimator, n_resamples, seed, confidence)
    result.update({"sampled_neurons": gap_rows.shape[1], "usable_neurons": int(usable.sum()),
                   "failed_or_low_variance_neurons": int((~usable).sum())})
    return result


def neuron_stratified_sensitivity(per_neuron, physical_layers, n_resamples=2000,
                                 seed=0, confidence=.95):
    """Secondary sample sensitivity, resampling neurons within fixed layer strata.

    This conditions on observed family measurements and does not estimate
    uncertainty across model pretraining runs or architectures.
    """
    values, layers = np.asarray(per_neuron, float), np.asarray(physical_layers)
    observed = layer_balanced_mean(values, layers)
    generator, draws = np.random.default_rng(seed), []
    strata = [values[(layers == layer) & np.isfinite(values)] for layer in np.unique(layers)]
    for _ in range(n_resamples):
        draws.append(np.mean([generator.choice(group, size=len(group), replace=True).mean() for group in strata]))
    alpha = 1 - confidence
    return {"estimate": observed, "lower": float(np.quantile(draws, alpha / 2)),
            "upper": float(np.quantile(draws, 1 - alpha / 2)),
            "scope": "Neuron-sample sensitivity within fixed physical layer strata; no training-seed uncertainty."}


def probe_deficit_interval(labels, predictions, classes, family_ids, primary_passes=("L1", "L4"),
                           n_resamples=2000, seed=0, confidence=1 - .05 / 3):
    """P2 paired symmetric balanced-accuracy deficit on the same held-out objects."""
    from .probes import balanced_accuracy
    p, q = primary_passes
    keys = ("%s_to_%s" % (q, q), "%s_to_%s" % (p, q), "%s_to_%s" % (p, p), "%s_to_%s" % (q, p))
    labels = np.asarray(labels)
    rows = np.column_stack([labels] + [np.asarray(predictions[key]) for key in keys])
    def estimator(sample, ids):
        scores = [balanced_accuracy(sample[:, 0], sample[:, k], classes, family_weights(ids)) for k in range(1, 5)]
        return .5 * (scores[0] - scores[1] + scores[2] - scores[3])
    return family_bootstrap(rows, family_ids, estimator, n_resamples, seed, confidence)


def fit_causal_gain(early, late, groups, family_ids, locked_groups, split="calibration"):
    """Fit one nonnegative gain per candidate on equal-weight group profiles.

    Calibration effects must use the same physical dose and population across
    passes. No offset is fitted, so this is a gain adjustment, not a new model.
    """
    if split != "calibration":
        raise ValueError("Causal gain must be fitted on calibration only")
    early, late = matrix(early), matrix(late)
    if early.shape != late.shape:
        raise ValueError("Causal pass populations differ")
    e = group_profiles(early, groups, family_ids, locked_groups)
    l = group_profiles(late, groups, family_ids, locked_groups)
    denominator = np.sum(e ** 2, axis=0)
    gain = np.maximum(np.divide(np.sum(e * l, axis=0), denominator,
                               out=np.zeros_like(denominator), where=denominator > 1e-15), 0)
    return {"gain": gain, "early_profile": e, "late_profile": l,
            "weak_calibration_early_profile": denominator <= 1e-15}


def group_profiles(effects, groups, family_ids, locked_groups, min_families=2):
    """Group means with equal independent-family weight within each group."""
    effects, groups, ids = matrix(effects), np.asarray(groups), np.asarray(family_ids)
    if len(groups) != len(effects) or len(ids) != len(effects):
        raise ValueError("Group or family coverage differs from effect rows")
    if len(locked_groups) != len(set(locked_groups)) or set(groups) - set(locked_groups):
        raise ValueError("Unregistered or duplicate context groups")
    profiles = []
    for group in locked_groups:
        chosen = groups == group
        count = len(np.unique(ids[chosen]))
        if count < min_families:
            raise ValueError("Context group %s has only %d independent families" % (group, count))
        profiles.append(np.average(effects[chosen], axis=0, weights=family_weights(ids[chosen])))
    return np.asarray(profiles)


def centered_max_t(estimate, bootstrap_draws, confidence=1 - .05 / 3):
    """Simultaneous band and centered max-t test of zero locked contrasts.

    SE is the bootstrap SD of each linear contrast. Exactly zero SE with zero
    estimate is assigned t=0; zero SE with nonzero estimate has observed t=inf
    and a degenerate band. No bootstrap norm of uncentered contrasts is used.
    """
    estimate, draws = np.asarray(estimate, float), np.asarray(bootstrap_draws, float)
    if draws.ndim != estimate.ndim + 1 or draws.shape[1:] != estimate.shape or len(draws) < 100:
        raise ValueError("Max-t bootstrap shapes disagree")
    if not np.isfinite(estimate).all() or not np.isfinite(draws).all() or not 0 < confidence < 1:
        raise ValueError("Max-t requires finite estimable contrasts")
    se = np.std(draws, axis=0, ddof=1)
    numerical_floor = np.finfo(float).eps * 64 * np.maximum(1, np.max(np.abs(draws), axis=0))
    zero = se <= numerical_floor
    se[zero] = 0.0
    centered = draws - estimate
    standardized = np.divide(centered, se, out=np.zeros_like(centered), where=~zero)
    max_t = np.max(np.abs(standardized).reshape(len(draws), -1), axis=1)
    critical = float(np.quantile(max_t, confidence))
    observed_t = np.divide(np.abs(estimate), se, out=np.zeros_like(estimate), where=~zero)
    observed_t[zero & (np.abs(estimate) > numerical_floor)] = np.inf
    observed_max = float(np.max(observed_t))
    return {"lower": estimate - critical * se, "upper": estimate + critical * se,
            "standard_error": se, "zero_variance": zero,
            "critical_max_t": critical, "observed_max_t": observed_max,
            "centered_max_t_p": float((1 + np.sum(max_t >= observed_max)) / (len(draws) + 1)),
            "confidence": confidence,
            "method": "Centered bootstrap max-t over locked linear group contrasts; bootstrap SD studentization."}


def causal_profile_interaction(early, late, gain, groups, family_ids, locked_groups,
                               reference_group, n_resamples=2000, seed=0,
                               confidence=1 - .05 / 3, min_families=2):
    """P3 residual-by-context contrasts with simultaneous family-bootstrap bands.

    For candidate i, R[g]=mean(E_late-lambda[i]*E_early | g), and
    C[g]=R[g]-R[reference]. Gain, context list and reference are frozen before
    confirmation. Missing/underpopulated groups fail explicitly, never imputed.
    """
    e, l, gain = matrix(early), matrix(late), np.asarray(gain, float)
    if e.shape != l.shape or gain.shape != (e.shape[1],) or not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("Invalid matched effects or frozen nonnegative gains")
    if len(locked_groups) < 2 or reference_group not in locked_groups:
        raise ValueError("Lock at least two contexts and a reference on development")
    group_profiles(e, groups, family_ids, locked_groups, min_families)
    groups = np.asarray(groups)
    residual = l - gain * e
    # Encode context in a numeric column without mixing effect and class scales.
    group_codes = np.array([list(locked_groups).index(g) for g in groups])
    rows = np.column_stack([group_codes, residual])
    reference = list(locked_groups).index(reference_group)
    other = [index for index in range(len(locked_groups)) if index != reference]
    def estimator(sample, ids):
        try:
            profile = group_profiles(sample[:, 1:], sample[:, 0].astype(int), ids,
                                     list(range(len(locked_groups))), min_families)
            return (profile[other] - profile[reference]).T
        except ValueError:
            return np.full((e.shape[1], len(other)), np.nan)
    bootstrap = family_bootstrap(rows, family_ids, estimator, n_resamples, seed, confidence)
    bands = centered_max_t(bootstrap["estimate"], bootstrap["draws"], confidence)
    bootstrap.update(bands)
    bootstrap.update({"gain": gain, "contrasts": ["%s - %s" % (locked_groups[k], reference_group) for k in other],
                      "early_profile": group_profiles(e, groups, family_ids, locked_groups, min_families),
                      "late_profile": group_profiles(l, groups, family_ids, locked_groups, min_families),
                      "residual_profile": group_profiles(residual, groups, family_ids, locked_groups, min_families),
                      "groups": list(locked_groups), "reference_group": reference_group})
    return bootstrap


def validate_condition_rows(rows, required_conditions, key_fields=("example_id", "physical_layer", "neuron_id", "invocation")):
    """Require exactly one saved row per key/condition before intervention contrasts."""
    seen = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        condition = row["condition"]
        if condition not in required_conditions:
            raise ValueError("Unexpected condition %s" % condition)
        if condition in seen.setdefault(key, set()):
            raise ValueError("Duplicated confirmation key/condition: %s %s" % (key, condition))
        seen[key].add(condition)
    if not seen:
        raise ValueError("No intervention measurement rows")
    for key, conditions in seen.items():
        missing = set(required_conditions) - conditions
        if missing:
            raise ValueError("Missing condition rows for %s: %s" % (key, sorted(missing)))
    return len(seen)
