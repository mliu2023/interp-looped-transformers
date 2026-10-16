"""Invocation-local real-token edits and precisely oriented outcome units."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ProductIntervention:
    layer: int
    invocation: int
    neuron: int
    tokens: tuple
    dose: float = 0.0
    donor: object = None
    self_replace: bool = False

    def matches(self, layer, invocation):
        return self.layer == layer and self.invocation == invocation

    def apply(self, activation):
        import torch
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Tokens must be distinct real positions")
        if min(self.tokens) < 0 or max(self.tokens) >= activation.shape[1] or not 0 <= self.neuron < activation.shape[2]:
            raise ValueError("Intervention site out of range")
        replacement = activation.clone()
        original = activation[0, list(self.tokens), self.neuron]
        if self.donor is not None:
            donor = torch.as_tensor(self.donor, device=activation.device, dtype=activation.dtype)
            if donor.shape != original.shape:
                raise ValueError("Donor token alignment/shape mismatch")
            value = donor
        elif self.self_replace:
            value = original
        else:
            value = original + self.dose
        replacement[0, list(self.tokens), self.neuron] = value
        changed = (replacement != activation).nonzero().detach().cpu().tolist()
        allowed = {(0, t, self.neuron) for t in self.tokens}
        if any(tuple(i) not in allowed for i in changed):
            raise ValueError("Intervention escaped selected indices")
        return replacement, changed


def validate_alignment(recipient, donor):
    for name in ("token_indices", "grid", "mask_hash", "question", "shape"):
        if recipient[name] != donor[name]:
            raise ValueError("Donor/recipient mismatch: " + name)


def fixed_dose(discovery_l1, discovery_l4):
    values = np.concatenate([discovery_l1, discovery_l4])
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("Invalid discovery activation population")
    return float(0.25 * np.std(values, ddof=1))


def half_contrast(plus_scores, minus_scores, base_answer, edit_answer):
    if base_answer == edit_answer:
        return None  # use separate specificity measures; no nonexistent margin
    return ((plus_scores[edit_answer] - plus_scores[base_answer]) -
            (minus_scores[edit_answer] - minus_scores[base_answer])) / 2.0
