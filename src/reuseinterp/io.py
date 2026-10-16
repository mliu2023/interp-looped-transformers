"""Canonical hashes, atomic writes, and immutable run identity."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def config_load(path):
    # Shipped .yaml files use JSON, a YAML subset: no implicit YAML coercions.
    config = read_json(path)
    if config.get("gpu_hour_cap", 30) > 30:
        raise ValueError("Project ceiling is 30 GPU-hours")
    if config.get("output_gib_cap", 20) > 20:
        raise ValueError("Project ceiling is 20 GiB of research outputs")
    if config.get("study_fingerprint") != study_fingerprint(config):
        raise ValueError("Study identity changed; regenerate fingerprint and rerun required gates in a new study")
    return config


def study_fingerprint(config):
    keys = ("code_revision", "checkpoint_revision", "checkpoint_hashes", "dataset_hash", "seed",
            "generation_budget", "neurons_per_layer", "state_layers", "primary_invocations", "numerical", "descriptor_policy")
    identity = {key: config[key] for key in keys}
    identity["runtime"] = "official mixed BF16-FP32; SDPA; batch1; H2L3; no dtype override"
    return digest(identity)


def run_directory(config):
    root = Path(os.environ.get("REUSE_RUN_ROOT", "results"))
    run_id = config["run_id"]
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a single directory name")
    return root / run_id
