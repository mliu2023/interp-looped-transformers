"""CPU alternative-explanation controls using frozen saved measurements.

These standalone functions do not establish that a model experiment passed its
controls. Their outputs must be integrated, recorded and checked before claims.
All contexts are external annotations, not inferred internal model selection.
"""
from dataclasses import dataclass

import numpy as np

from .encoding import family_weights
from .statistics import family_bootstrap, layer_balanced_mean, semantic_gap_interval


def _labels(values, n=None):
    result = np.asarray(values)
    if result.ndim != 1 or (n is not None and len(result) != n) or not len(result):
        raise ValueError("One nonempty context label is required per row")
    if not all(isinstance(v, (str, np.str_)) and len(v) for v in result):
        raise ValueError("Context labels must be explicit nonempty strings")
    return result


def make_context_labels(question_families, role_flags):
    """Create matched contexts without merging overlapping annotated roles.

    role_flags is a fixed mapping, such as is_target/is_reference/set_member,
    to boolean arrays. Multiple true flags are preserved in the label; no
    post-hoc priority rule silently collapses reference and target objects.
    """
    questions = _labels(question_families)
    if not role_flags:
        raise ValueError("Declare external role flags before analysis")
    flags = {}
    for name, values in sorted(role_flags.items()):
        values = np.asarray(values)
        if values.shape != questions.shape or not np.isin(values, [0, 1, False, True]).all():
            raise ValueError("Role flags must be matched boolean arrays")
        flags[name] = values.astype(bool)
    return np.asarray([str(question) + "|" + ";".join("%s=%d" % (name, values[i]) for name, values in flags.items())
                       for i, question in enumerate(questions)])


@dataclass(frozen=True)
class FrozenBalance:
    """Discovery-fitted inverse-frequency weights, clipped at locked bounds."""
    groups: tuple
    weights: tuple
    discovery_probabilities: tuple
    clip: tuple
    discovery_independent_families: int
    discovery_clipped_fraction: float

    def transform(self, context_labels):
        """Apply discovery weights without learning confirmation frequencies."""
        labels = _labels(context_labels)
        lookup = dict(zip(self.groups, self.weights))
        missing = set(labels) - set(lookup)
        if missing:
            raise ValueError("No discovery support for locked contexts: %s" % sorted(missing))
        return np.asarray([lookup[label] for label in labels], float)

    def to_dict(self):
        return {"groups": list(self.groups), "weights": list(self.weights),
                "discovery_probabilities": list(self.discovery_probabilities), "clip": list(self.clip),
                "discovery_independent_families": self.discovery_independent_families,
                "discovery_clipped_fraction": self.discovery_clipped_fraction,
                "scope": "Discovery-fitted inverse-frequency reweighting toward equal context groups; clipping can prevent exact balance."}


def fit_context_balance(context_labels, family_ids, clip=(.25, 4.), split="development"):
    """Fit weights from fit/development only, targeting equal context prevalence.

    Frequencies use equal family mass, so clutter and more repeated questions do
    not dominate fitting. Weights are clipped without renormalizing past the
    locked bounds. Report actual weighted distributions alongside the bounds.
    """
    if split not in {"fit", "development", "dev"}:
        raise ValueError("Balance fitting requires discovery families, never calibration/confirmation")
    labels = _labels(context_labels)
    family_ids = np.asarray(family_ids)
    if family_ids.shape != labels.shape or len(set(family_ids.tolist())) < 2:
        raise ValueError("At least two independent discovery families are required")
    if len(clip) != 2 or not 0 < clip[0] <= 1 <= clip[1] or not np.isfinite(clip).all():
        raise ValueError("Clip bounds must be finite, positive and enclose one")
    base = family_weights(family_ids)
    groups = tuple(np.unique(labels).tolist())
    probabilities = np.asarray([base[labels == group].sum() / base.sum() for group in groups])
    raw = 1. / (len(groups) * probabilities)
    clipped = np.clip(raw, clip[0], clip[1])
    clipped_fraction = float(np.sum(probabilities[raw != clipped]))
    return FrozenBalance(groups, tuple(clipped.tolist()), tuple(probabilities.tolist()), tuple(clip),
                         len(set(family_ids.tolist())), clipped_fraction)


def balanced_semantic_gap(gap_rows, family_ids, physical_layers, context_labels,
                           fitted_balance, n_resamples=2000, seed=0, confidence=.95):
    """Secondary frozen-weight P1 estimate with complete-family uncertainty."""
    gaps = np.asarray(gap_rows, float)
    if gaps.ndim != 2 or gaps.shape[1] != len(physical_layers):
        raise ValueError("Expected matched per-row/per-neuron gap measurements")
    labels = _labels(context_labels, len(gaps))
    balance = fitted_balance.transform(labels)
    usable = np.isfinite(gaps).all(axis=0)
    if np.any(np.isfinite(gaps).any(axis=0) & ~usable):
        raise ValueError("Partial missing unit rows require explicit exclusions")
    rows = np.column_stack([balance, gaps])
    def estimator(sample, ids):
        means = np.full(gaps.shape[1], np.nan)
        weights = family_weights(ids) * sample[:, 0]
        means[usable] = np.average(sample[:, 1:][:, usable], axis=0, weights=weights)
        return layer_balanced_mean(means, physical_layers)
    result = family_bootstrap(rows, family_ids, estimator, n_resamples, seed, confidence)
    before = family_weights(family_ids)
    after = before * balance
    result.update({"balance_fit": fitted_balance.to_dict(),
                   "applied_weight_range": [float(balance.min()), float(balance.max())],
                   "context_distribution_before": {g: float(before[labels == g].sum() / before.sum()) for g in fitted_balance.groups},
                   "context_distribution_after": {g: float(after[labels == g].sum() / after.sum()) for g in fitted_balance.groups},
                   "interpretation": "Secondary discovery-weighted population; report alongside unweighted and matched-context estimates."})
    return result


def matched_semantic_gaps(gap_rows, family_ids, physical_layers, context_labels,
                          locked_contexts, n_resamples=2000, seed=0, confidence=.95,
                          min_families=2):
    """Repeat frozen-model transfer comparisons within locked task/role cells.

    Underpopulated cells are explicit unavailable results. The same precomputed
    per-row L1/L4 contrasts enter every comparison; source and target populations
    can never differ. Context selection must be frozen before confirmation.
    """
    gaps, families = np.asarray(gap_rows, float), np.asarray(family_ids)
    labels = _labels(context_labels, len(gaps))
    if families.shape != labels.shape or len(locked_contexts) != len(set(locked_contexts)):
        raise ValueError("Context/family coverage or locked context list is invalid")
    if set(labels) - set(locked_contexts):
        raise ValueError("Unregistered confirmation context")
    result = {}
    for index, group in enumerate(locked_contexts):
        chosen = labels == group
        count = len(np.unique(families[chosen]))
        row = {"rows": int(chosen.sum()), "independent_families": count}
        if count < min_families:
            row.update({"status": "unavailable", "reason": "missing_or_underpopulated_context"})
        else:
            try:
                interval = semantic_gap_interval(gaps[chosen], families[chosen], physical_layers,
                                                 n_resamples, seed + index, confidence)
                row.update({"status": "estimated", "interval": {k: v for k, v in interval.items() if k != "draws"}})
            except ValueError as error:
                row.update({"status": "unavailable", "reason": str(error)})
        result[group] = row
    return result


def paired_edit_differences(activations, pair_ids, object_ids, versions, family_ids,
                             base_version="base", edited_version="color_edit"):
    """Return exact aligned color-edit minus base activations for every object.

    pair_ids must include the question/paraphrase identity. Every (pair,object)
    has exactly one base and one edit and remains in one family. This measures
    activation selectivity; it does not fabricate a two-answer causal margin
    for answer-preserving scenes or irrelevant objects.
    """
    values = np.asarray(activations, float)
    if values.ndim < 2 or not np.isfinite(values).all():
        raise ValueError("Paired activations must have finite [row,...] shape")
    n = len(values)
    pairs = _labels(pair_ids, n)
    objects = _labels(object_ids, n)
    versions = _labels(versions, n)
    families = _labels(family_ids, n)
    if set(versions) - {base_version, edited_version} or base_version == edited_version:
        raise ValueError("Unexpected or identical scene edit version names")
    joined = {}
    for index, key in enumerate(zip(pairs, objects)):
        group = joined.setdefault(key, {})
        version = versions[index]
        if version in group:
            raise ValueError("Duplicate paired edit row for %s %s" % (key, version))
        group[version] = index
    base_indices, edit_indices, keys = [], [], []
    for key, rows in sorted(joined.items()):
        if set(rows) != {base_version, edited_version}:
            raise ValueError("Missing base/edit condition for %s" % (key,))
        base, edit = rows[base_version], rows[edited_version]
        if families[base] != families[edit]:
            raise ValueError("Paired scene versions leaked across families")
        base_indices.append(base)
        edit_indices.append(edit)
        keys.append(key)
    base_indices, edit_indices = np.asarray(base_indices), np.asarray(edit_indices)
    return {"delta": values[edit_indices] - values[base_indices],
            "base_activations": values[base_indices], "edited_activations": values[edit_indices],
            "pair_ids": np.asarray([key[0] for key in keys]), "object_ids": np.asarray([key[1] for key in keys]),
            "family_ids": families[base_indices], "base_indices": base_indices, "edited_indices": edit_indices,
            "direction": "%s - %s" % (edited_version, base_version)}


def paired_selectivity_profiles(deltas, family_ids, context_labels, locked_contexts,
                                 n_resamples=2000, seed=0, confidence=.95, min_families=2):
    """Raw paired-edit selectivity profiles, with equal-family uncertainty.

    The context for an edit pair (for example its base role) is declared by the
    caller; both versions' role annotations should also be saved if they differ.
    """
    deltas, families = np.asarray(deltas, float), np.asarray(family_ids)
    labels = _labels(context_labels, len(deltas))
    if deltas.ndim < 2 or not np.isfinite(deltas).all() or families.shape != labels.shape:
        raise ValueError("Paired delta/context coverage disagrees")
    if len(set(locked_contexts)) != len(locked_contexts) or set(labels) - set(locked_contexts):
        raise ValueError("Unregistered or duplicate locked paired-edit contexts")
    results = {}
    for index, group in enumerate(locked_contexts):
        chosen = labels == group
        count = len(np.unique(families[chosen]))
        result = {"independent_families": count, "paired_objects": int(chosen.sum())}
        if count < min_families:
            result.update({"status": "unavailable", "reason": "missing_or_underpopulated_context"})
        else:
            def estimator(values, ids):
                return np.average(values, axis=0, weights=family_weights(ids))
            interval = family_bootstrap(deltas[chosen], families[chosen], estimator, n_resamples, seed + index, confidence)
            result.update({"status": "estimated", "interval": {k: v for k, v in interval.items() if k != "draws"}})
        results[group] = result
    return results
