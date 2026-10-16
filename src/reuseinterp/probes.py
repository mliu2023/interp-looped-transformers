"""Frozen regularized linear readouts on matched held-out object populations."""
from dataclasses import dataclass

import numpy as np

from .encoding import RidgeEncoder, _weights, fit_ridge, matrix, validate_split_families


def balanced_accuracy(truth, prediction, classes=None, weights=None):
    """Mean class recall; undefined if a locked class has no held-out examples."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("Classification rows disagree")
    classes = list(np.unique(truth)) if classes is None else list(classes)
    if not classes or set(truth) - set(classes):
        raise ValueError("Truth includes a class outside the locked vocabulary")
    w = _weights(weights, len(truth))
    recalls = []
    for label in classes:
        chosen = truth == label
        if not np.any(chosen) or w[chosen].sum() == 0:
            return float("nan")
        recalls.append(np.average(prediction[chosen] == truth[chosen], weights=w[chosen]))
    return float(np.mean(recalls))


@dataclass
class LinearReadout:
    """One-vs-rest ridge classifier; labels and preprocessing are frozen."""
    encoder: RidgeEncoder
    classes: np.ndarray
    selected_penalty: float

    def scores(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        return self.classes[np.argmax(self.scores(x), axis=1)]

    def to_dict(self):
        return {"encoder": self.encoder.to_dict(), "classes": self.classes.tolist(),
                "selected_penalty": self.selected_penalty,
                "score_kind": "one-vs-rest ridge scores, not calibrated probabilities"}


def fit_linear_readout(x_fit, labels_fit, x_development, labels_development,
                       classes=None, penalties=(0.01, 0.1, 1.0, 10.0, 100.0),
                       fit_weights=None, development_weights=None,
                       development_split="development"):
    """Select a common classifier penalty by development balanced accuracy.

    Fit-only scaling/intercepts and locked labels transfer unchanged across
    invocations. Ties select the first declared penalty deterministically.
    """
    if development_split not in {"development", "dev"}:
        raise ValueError("Probe selection requires development families")
    yf, yd = np.asarray(labels_fit), np.asarray(labels_development)
    classes = np.unique(yf) if classes is None else np.asarray(classes)
    if yf.ndim != 1 or yd.ndim != 1 or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Readout needs at least two unique classes and vector labels")
    if set(yf) != set(classes) or set(yd) - set(classes):
        raise ValueError("Fit must cover every locked class; development cannot add classes")
    target_fit = (yf[:, None] == classes[None, :]).astype(float)
    target_dev = (yd[:, None] == classes[None, :]).astype(float)
    best, best_accuracy = None, -float("inf")
    for penalty in penalties:
        encoder = fit_ridge(x_fit, target_fit, x_development, target_dev, (penalty,), fit_weights, development_weights)
        predicted = classes[np.argmax(encoder.predict(x_development), axis=1)]
        accuracy = balanced_accuracy(yd, predicted, classes, development_weights)
        if not np.isfinite(accuracy):
            raise ValueError("Development must cover every locked class")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = LinearReadout(encoder, classes, float(penalty))
    if best is None:
        raise ValueError("Declare a nonempty penalty grid")
    return best


def evaluate_readout_transfer(states_by_split, labels_by_split, classes=None,
                              family_ids_by_split=None, penalties=(0.01, 0.1, 1.0, 10.0, 100.0),
                              primary_passes=("L1", "L4")):
    """Return matched frozen-readout transfer and raw per-object predictions.

    Labels are independent scene properties, shared over passes; all objects
    enter this interface. P2 is the symmetric within-minus-cross balanced
    accuracy deficit on precisely the same confirmation objects.
    """
    required = ("fit", "development", "confirmation")
    if any(s not in states_by_split or s not in labels_by_split for s in required):
        raise ValueError("Probe fitting requires fit, development and confirmation")
    if family_ids_by_split is not None:
        validate_split_families(family_ids_by_split)
    from .encoding import family_weights
    weights = {s: None if family_ids_by_split is None else family_weights(family_ids_by_split[s]) for s in required}
    passes = tuple(states_by_split["fit"])
    for split in required:
        if set(states_by_split[split]) != set(passes):
            raise ValueError("Missing probe invocation")
        for p in passes:
            if len(states_by_split[split][p]) != len(labels_by_split[split]):
                raise ValueError("Probe populations differ over passes")
    models = {p: fit_linear_readout(states_by_split["fit"][p], labels_by_split["fit"],
              states_by_split["development"][p], labels_by_split["development"], classes,
              penalties, weights["fit"], weights["development"]) for p in passes}
    predictions, scores, metrics = {}, {}, {}
    for p in passes:
        for q in passes:
            key = "%s_to_%s" % (p, q)
            scores[key] = models[p].scores(states_by_split["confirmation"][q])
            predictions[key] = models[p].predict(states_by_split["confirmation"][q])
            metrics[key] = {
                "balanced_accuracy": balanced_accuracy(labels_by_split["confirmation"], predictions[key], models[p].classes, weights["confirmation"]),
                "accuracy": float(np.average(predictions[key] == labels_by_split["confirmation"], weights=weights["confirmation"])),
                "objects": len(labels_by_split["confirmation"])}
    p, q = primary_passes
    if p not in passes or q not in passes:
        raise ValueError("Missing primary probe pass")
    deficit = .5 * (metrics["%s_to_%s" % (q, q)]["balanced_accuracy"] - metrics["%s_to_%s" % (p, q)]["balanced_accuracy"]
                    + metrics["%s_to_%s" % (p, p)]["balanced_accuracy"] - metrics["%s_to_%s" % (q, p)]["balanced_accuracy"])
    return {"models": models, "predictions": predictions, "scores": scores,
            "metrics": metrics, "p2_deficit": float(deficit),
            "labels": np.asarray(labels_by_split["confirmation"]),
            "classes": models[p].classes,
            "uncertainty_scope": "Conditional on frozen readout weights and development site selection."}


def coherent_family_permutation(values, family_ids, seed=0):
    """Permute complete equal-shape families, keeping all passes together.

    Call once on an array whose trailing axes include all paired invocations;
    never draw separate label permutations per pass. Unequal family sizes need
    an explicitly prespecified matched-family permutation design.
    """
    values, ids = np.asarray(values), np.asarray(family_ids)
    if len(values) != len(ids):
        raise ValueError("Permutation family coverage disagrees")
    groups = [np.flatnonzero(ids == f) for f in np.unique(ids)]
    if len({len(g) for g in groups}) != 1:
        raise ValueError("Coherent permutation requires equal-shape family blocks")
    permutation = np.random.default_rng(seed).permutation(len(groups))
    output = np.empty_like(values)
    for target, source in enumerate(permutation):
        output[groups[target]] = values[groups[source]]
    return output
