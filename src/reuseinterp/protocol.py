"""Fail-closed, immutable confirmation boundary; drafting is never a lock."""
from pathlib import Path
from .io import atomic_json, digest, read_json, file_hash

REQUIRED_FIELDS = ("model", "runtime", "dataset", "neuron_sample", "causal_candidates",
                   "primary_comparisons", "descriptor_spec", "hyperparameters", "calibration_ids",
                   "numerical_tolerances", "dose_rules", "eligibility", "margins", "weighting",
                   "statistics", "multiplicity", "resource_cap", "exclusion_rules", "p2_site",
                   "causal_family_ids", "discovery_artifact_hashes", "control_status", "confirmation_family_ids")


def create_lock(path, protocol, gates):
    path = Path(path)
    if path.exists():
        raise ValueError("Protocol lock already exists; never overwrite it")
    for gate in ("parity", "pilot", "discovery"):
        if gates.get(gate, {}).get("passed") is not True:
            raise ValueError("Cannot lock before gate: " + gate)
    missing = [field for field in REQUIRED_FIELDS if field not in protocol or protocol[field] is None]
    if missing:
        raise ValueError("Incomplete protocol: " + ", ".join(missing))
    if protocol.get("confirmation_opened") is not False:
        raise ValueError("Discovery/confirmation boundary not established")
    required_controls = ("numerical_parity", "actual_mask_alignment", "matched_roles_and_task_families",
                         "discovery_fitted_clipped_reweighting", "paired_edit_selectivity",
                         "coherent_family_permutation", "question_position_probe_baseline")
    pending_controls = [name for name in required_controls if protocol["control_status"].get(name) is not True]
    if pending_controls:
        raise ValueError("Mandatory controls not complete: " + ", ".join(pending_controls))
    if not protocol["causal_candidates"] and not protocol.get("p3_skip_reason"):
        raise ValueError("P3 skip needs a discovery-supported reason")
    if not protocol["confirmation_family_ids"] or len(protocol["confirmation_family_ids"]) != len(set(protocol["confirmation_family_ids"])):
        raise ValueError("Freeze unique complete confirmation family IDs")
    if protocol["resource_cap"]["gpu_hours"] > 30 or protocol["resource_cap"]["output_gib"] > 20:
        raise ValueError("Protocol exceeds resource ceiling")
    if len(protocol["causal_candidates"]) > 8:
        raise ValueError("At most eight initial causal candidates")
    if protocol["multiplicity"].get("endpoint_families") != 3:
        raise ValueError("Keep three-way alpha allocation even if P3 is skipped")
    result = {"status": "locked", "protocol": protocol, "protocol_hash": digest(protocol)}
    atomic_json(path, result)
    return result


def verify_lock(path, expected_hash=None):
    value = read_json(path)
    if value.get("status") != "locked" or value.get("protocol_hash") != digest(value["protocol"]):
        raise ValueError("Invalid/tampered protocol lock")
    if expected_hash and file_hash(path) != expected_hash:
        raise ValueError("Protocol lock differs from confirmation manifest")
    return value
