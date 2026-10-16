"""Frozen descriptor encoders and positive-affine cross-invocation evaluation.

These functions operate on saved CPU arrays. They neither import nor run a VLM.
All fitted quantities must come from the named discovery/calibration splits.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


FORBIDDEN_PRIMARY_DESCRIPTORS = frozenset({
    "answer", "final_answer", "target_answer", "model_answer", "predicted_answer",
    "candidate_scores", "confidence", "model_confidence", "answer_score",
})


def matrix(values, name="array"):
    """Return a finite float64 matrix, promoting a vector to one column."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
        raise ValueError("%s must be a nonempty finite matrix" % name)
    return result


def family_weights(family_ids):
    """Equal total weight per independent family, equal weight within family."""
    ids = np.asarray(family_ids)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def validate_split_families(split_families: Mapping[str, Sequence[str]]):
    """Reject descendants of the same scene family in different splits."""
    seen = {}
    for split, families in split_families.items():
        for family in set(families):
            if family in seen and seen[family] != split:
                raise ValueError("Family leakage: %s occurs in %s and %s" %
                                 (family, seen[family], split))
            seen[family] = split


def validate_unique_ids(ids, label="confirmation"):
    """Reject duplicated measurement keys before any analysis or merge."""
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicated %s IDs" % label)


@dataclass(frozen=True)
class DescriptorSpec:
    """Explicit frozen vocabulary, continuous fields and pairwise interactions.

    An interaction ("color", "role") expands all products of those term columns.
    Categories must be declared before confirmation; unknown levels are errors.
    This whitelist intentionally excludes answer-derived primary descriptors.
    """
    categorical: Mapping[str, Sequence[str]]
    continuous: Sequence[str] = ()
    interactions: Sequence[Sequence[str]] = ()

    def __post_init__(self):
        names = list(self.categorical) + list(self.continuous)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate descriptor names")
        forbidden = set(names) & FORBIDDEN_PRIMARY_DESCRIPTORS
        if forbidden:
            raise ValueError("Answer leakage in primary descriptors: %s" % sorted(forbidden))
        for name, levels in self.categorical.items():
            if not levels or len(levels) != len(set(levels)):
                raise ValueError("Invalid vocabulary for %s" % name)
        for interaction in self.interactions:
            if len(interaction) != 2 or any(x not in names for x in interaction):
                raise ValueError("Interactions must name two declared descriptors")

    def transform(self, rows):
        """Encode rows without learning new categories or preprocessing statistics."""
        if not rows:
            raise ValueError("Cannot encode an empty population")
        terms, labels = {}, []
        for name, levels in self.categorical.items():
            values = [row[name] for row in rows]
            unknown = set(values) - set(levels)
            if unknown:
                raise ValueError("Unknown %s levels: %s" % (name, sorted(unknown)))
            terms[name] = np.array([[v == level for level in levels] for v in values], float)
            labels.extend(["%s=%s" % (name, level) for level in levels])
        for name in self.continuous:
            terms[name] = matrix([row[name] for row in rows], name)
            labels.append(name)
        columns = list(terms.values())
        for left, right in self.interactions:
            product = (terms[left][:, :, None] * terms[right][:, None, :]).reshape(len(rows), -1)
            columns.append(product)
            labels.extend(["%s[%d]*%s[%d]" % (left, i, right, j)
                           for i in range(terms[left].shape[1])
                           for j in range(terms[right].shape[1])])
        if not columns:
            raise ValueError("Declare at least one descriptor")
        return matrix(np.column_stack(columns), "descriptors"), labels


@dataclass
class RidgeEncoder:
    """Regularized linear encoder; scaling and intercept are fitted on fit only."""
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray
    penalties: np.ndarray

    def predict(self, x):
        x = matrix(x, "predictor")
        if x.shape[1] != len(self.feature_mean):
            raise ValueError("Predictor vocabulary/dimension changed")
        return ((x - self.feature_mean) / self.feature_scale) @ self.coefficients + self.target_mean

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in (
            "feature_mean", "feature_scale", "target_mean", "coefficients", "penalties")}


def fit_ridge(x_fit, y_fit, x_development, y_development,
              penalties=(0.01, 0.1, 1.0, 10.0, 100.0), fit_weights=None,
              development_weights=None, development_split="development"):
    """Choose each output's penalty on development, never confirmation.

    The intercept is unpenalized. Optional weights allow equal-family weighting.
    The same fixed search grid must be used for every compared invocation.
    """
    if development_split not in {"development", "dev"}:
        raise ValueError("Hyperparameter selection requires development families")
    xf, yf = matrix(x_fit), matrix(y_fit)
    xd, yd = matrix(x_development), matrix(y_development)
    if len(xf) != len(yf) or len(xd) != len(yd) or xf.shape[1] != xd.shape[1] or yf.shape[1] != yd.shape[1]:
        raise ValueError("Encoding array shapes disagree")
    penalties = tuple(float(p) for p in penalties)
    if not penalties or any(p <= 0 or not np.isfinite(p) for p in penalties):
        raise ValueError("Ridge penalties must be finite and strictly positive")
    wf = _weights(fit_weights, len(xf))
    wd = _weights(development_weights, len(xd))
    mean = np.average(xf, axis=0, weights=wf)
    scale = np.sqrt(np.average((xf - mean) ** 2, axis=0, weights=wf))
    scale[scale < 1e-12] = 1.0
    target_mean = np.average(yf, axis=0, weights=wf)
    z = (xf - mean) / scale
    zd = (xd - mean) / scale
    gram = z.T @ (wf[:, None] * z)
    rhs = z.T @ (wf[:, None] * (yf - target_mean))
    fits, losses = [], []
    for penalty in penalties:
        coefficients = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), rhs)
        fits.append(coefficients)
        losses.append(np.average((zd @ coefficients + target_mean - yd) ** 2, axis=0, weights=wd))
    best = np.argmin(losses, axis=0)
    coefficients = np.column_stack([fits[index][:, unit] for unit, index in enumerate(best)])
    return RidgeEncoder(mean, scale, target_mean, coefficients, np.asarray(penalties)[best])


def _weights(weights, n):
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid row weights")
    return weights


@dataclass
class PositiveAffine:
    """One nonnegative scalar gain and one offset per neuron; no rotations."""
    alpha: np.ndarray
    beta: np.ndarray

    def predict(self, prediction):
        return matrix(prediction) * self.alpha + self.beta


def fit_positive_affine(prediction, target, weights=None, split="calibration"):
    """Fit nonnegative gain/offset using calibration families only."""
    if split != "calibration":
        raise ValueError("Affine calibration requires calibration families, not %s" % split)
    x, y = matrix(prediction), matrix(target)
    if x.shape != y.shape:
        raise ValueError("Calibration shapes disagree")
    w = _weights(weights, len(x))
    xm, ym = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
    variance = np.average((x - xm) ** 2, axis=0, weights=w)
    covariance = np.average((x - xm) * (y - ym), axis=0, weights=w)
    alpha = np.maximum(np.divide(covariance, variance, out=np.zeros_like(variance), where=variance > 1e-15), 0.0)
    return PositiveAffine(alpha, ym - alpha * xm)


def regression_metrics(target, prediction, calibration_variance, variance_floor, weights=None):
    """Return MSE, held-out R², and NMSE using frozen calibration variance.

    Low-variance units get undefined (NaN) NMSE, not an inflated denominator.
    Their status and raw errors remain available for honest denominators.
    """
    y, p = matrix(target), matrix(prediction)
    if y.shape != p.shape or variance_floor <= 0:
        raise ValueError("Invalid metric inputs or numerical variance floor")
    w = _weights(weights, len(y))
    var = np.asarray(calibration_variance, float)
    if var.shape != (y.shape[1],) or not np.isfinite(var).all() or (var < 0).any():
        raise ValueError("Invalid frozen calibration variance")
    mse = np.average((y - p) ** 2, axis=0, weights=w)
    heldout_var = np.average((y - np.average(y, axis=0, weights=w)) ** 2, axis=0, weights=w)
    r2 = 1.0 - np.divide(mse, heldout_var, out=np.full_like(mse, np.nan), where=heldout_var >= variance_floor)
    nmse = np.divide(mse, var, out=np.full_like(mse, np.nan), where=var >= variance_floor)
    return {"mse": mse, "r2": r2, "nmse": nmse, "low_variance": var < variance_floor}


def evaluate_semantic_transfer(x_by_split, y_by_split, family_ids_by_split=None,
                               penalties=(0.01, 0.1, 1.0, 10.0, 100.0),
                               variance_floor=1e-8, primary_passes=("L1", "L4")):
    """Fit on fit/dev, calibrate on calibration and evaluate frozen confirmation.

    y_by_split maps split -> pass -> [row,neuron] activations. Each split's rows
    must be aligned over passes. Returned predictions and per-row gap components
    can be saved and bootstrapped without refitting models or calibrations.
    """
    required = ("fit", "development", "calibration", "confirmation")
    if any(split not in x_by_split or split not in y_by_split for split in required):
        raise ValueError("Fit, development, calibration and confirmation are required")
    if family_ids_by_split is not None:
        validate_split_families(family_ids_by_split)
    passes = tuple(y_by_split["fit"])
    if any(set(y_by_split[s]) != set(passes) for s in required):
        raise ValueError("Invocation coverage differs across splits")
    if any(p not in passes for p in primary_passes):
        raise ValueError("Primary pass missing")
    for split in required:
        dimensions = {matrix(y_by_split[split][p]).shape for p in passes}
        if len(dimensions) != 1 or next(iter(dimensions))[0] != len(x_by_split[split]):
            raise ValueError("Matched invocation rows or neuron dimensions disagree")
    weights = {s: None if family_ids_by_split is None else family_weights(family_ids_by_split[s]) for s in required}
    models = {p: fit_ridge(x_by_split["fit"], y_by_split["fit"][p], x_by_split["development"],
                           y_by_split["development"][p], penalties, weights["fit"], weights["development"])
              for p in passes}
    variances = {q: np.average((matrix(y_by_split["calibration"][q]) - np.average(
        matrix(y_by_split["calibration"][q]), axis=0, weights=weights["calibration"])) ** 2,
        axis=0, weights=weights["calibration"]) for q in passes}
    predictions, calibrations, metrics, squared_errors = {}, {}, {}, {}
    for p in passes:
        source_cal = models[p].predict(x_by_split["calibration"])
        source_test = models[p].predict(x_by_split["confirmation"])
        for q in passes:
            key = "%s_to_%s" % (p, q)
            calibrator = fit_positive_affine(source_cal, y_by_split["calibration"][q], weights["calibration"])
            calibrated = calibrator.predict(source_test)
            calibrations[key] = calibrator
            predictions[key] = {"raw": source_test, "calibrated": calibrated}
            metrics[key] = {kind: regression_metrics(y_by_split["confirmation"][q], prediction,
                variances[q], variance_floor, weights["confirmation"])
                for kind, prediction in predictions[key].items()}
            squared_errors[key] = (calibrated - matrix(y_by_split["confirmation"][q])) ** 2
    p, q = primary_passes
    gap_rows = np.full_like(squared_errors["%s_to_%s" % (p, q)], np.nan)
    valid = (variances[p] >= variance_floor) & (variances[q] >= variance_floor)
    gap_rows[:, valid] = 0.5 * (
        (squared_errors["%s_to_%s" % (p, q)] - squared_errors["%s_to_%s" % (q, q)])[:, valid] / variances[q][valid]
        + (squared_errors["%s_to_%s" % (q, p)] - squared_errors["%s_to_%s" % (p, p)])[:, valid] / variances[p][valid])
    dev_r2 = {r: regression_metrics(y_by_split["development"][r], models[r].predict(x_by_split["development"]),
                variances[r], variance_floor, weights["development"])["r2"] for r in primary_passes}
    # Eligibility depends exclusively on development, never confirmation outcomes.
    dev_variances = {r: np.average((matrix(y_by_split["development"][r]) - np.average(
        matrix(y_by_split["development"][r]), axis=0, weights=weights["development"])) ** 2,
        axis=0, weights=weights["development"]) for r in primary_passes}
    eligible = (dev_variances[p] >= variance_floor) & (dev_variances[q] >= variance_floor) & (dev_r2[p] >= .10) & (dev_r2[q] >= .10)
    return {"models": models, "calibrations": calibrations, "predictions": predictions,
            "metrics": metrics, "calibration_variance": variances, "gap_rows": gap_rows,
            "eligible_development": eligible, "development_r2": dev_r2,
            "uncertainty_scope": "Conditional on frozen fitted models and calibration; bootstrap whole scene families."}
