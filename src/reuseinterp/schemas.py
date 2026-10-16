"""Typed result contracts with explicit missing values, exclusions and keys."""
from dataclasses import dataclass, asdict
import math


@dataclass
class BehavioralMeasurement:
    run_id: str
    config_hash: str
    checkpoint_revision: str
    split: str
    family_id: str
    example_id: str
    task_family: str
    condition: str
    target_answer: str
    candidate_scores: dict
    generated_answer: str
    validity: bool
    exclusion_reason: object = None
    module: object = None
    invocation: object = None
    physical_layer: object = None
    hook_site: object = None
    neuron_id: object = None
    token_stratum: object = None
    object_id: object = None
    dose: object = None
    donor_id: object = None
    seed: int = 1729

    def validate(self):
        for name in ("run_id", "config_hash", "checkpoint_revision", "split", "family_id", "example_id", "task_family", "condition"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError("Invalid " + name)
        if not isinstance(self.validity, bool) or (not self.validity and not self.exclusion_reason):
            raise ValueError("Invalid measurements require exclusion reasons")
        if not isinstance(self.candidate_scores, dict) or any(not math.isfinite(float(v)) for v in self.candidate_scores.values()):
            raise ValueError("Invalid candidate scores")
        if self.validity and not self.candidate_scores:
            raise ValueError("Valid behavioral rows require answer scores")
        return asdict(self)


def validate_rows(rows):
    seen = set()
    for row in rows:
        value = BehavioralMeasurement(**row).validate()
        key = tuple(str(value[k]) for k in ("example_id", "module", "invocation", "physical_layer", "hook_site", "neuron_id", "condition", "seed"))
        if key in seen:
            raise ValueError("Duplicated result key")
        seen.add(key)
    return len(seen)
