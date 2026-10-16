"""Scheduled behavioral/capture stages. Entire scene families commit atomically."""
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .io import read_json, atomic_json, file_hash, digest
from .model_adapter import LoopVLAdapter
from .schemas import BehavioralMeasurement
from .shards import FamilyShard, StopRequested, shard_for
from .capture import ProductHooks, sample_neurons, pool_tokens


def load_examples(dataset_dir, splits, expected_hash=None):
    root = Path(dataset_dir)
    manifest = read_json(root / "manifest.json")
    if expected_hash is not None and manifest["dataset_hash"] != expected_hash:
        raise ValueError("Dataset differs from the pinned split/content manifest")
    examples = []
    with open(root / manifest["paths"]["examples"]) as f:
        for line in f:
            item = __import__("json").loads(line)
            if item["split"] in splits:
                examples.append(item)
    if len({r["example_id"] for r in examples}) != len(examples):
        raise ValueError("Duplicate example IDs")
    return manifest, examples


def preflight_record():
    root = Path(os.environ["REUSE_RUN_ROOT"]) / "preflight"
    return read_json(root / (os.environ["SLURM_JOB_ID"] + "_" + os.environ.get("SLURM_ARRAY_TASK_ID", "single") + ".json"))


def check_gates(run_dir, config, names):
    gates = read_json(Path(run_dir) / "gates.json")
    fingerprint = config["study_fingerprint"]
    if gates.get("study_fingerprint") != fingerprint:
        raise ValueError("Study fingerprint differs from numerical validation")
    current = preflight_record()
    for name in names:
        gate = gates.get(name, {})
        if gate.get("passed") is not True or gate.get("study_fingerprint") != fingerprint:
            raise ValueError("Unpassed/stale gate: " + name)
    if gates.get("parity", {}).get("backend_signature") != current["backend_signature"]:
        raise ValueError("Backend changed; rerun unmodified numerical parity")
    return gates


def write_gate(run_dir, config, name, **fields):
    path = Path(run_dir) / "gates.json"
    gates = read_json(path) if path.exists() else {"study_fingerprint": config["study_fingerprint"]}
    if gates["study_fingerprint"] != config["study_fingerprint"]:
        raise ValueError("Cannot mix study gates")
    gates[name] = dict(fields, study_fingerprint=config["study_fingerprint"])
    atomic_json(path, gates)


def output_limit(root, config):
    root = Path(os.environ.get("REUSE_RUN_ROOT", str(Path(root).parent)))
    cap = int(config.get("output_gib_cap", 20) * 1024 ** 3)
    size = sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
    if size >= cap:
        raise RuntimeError("Research output ceiling reached; stop and report partial results")
    return size


def _atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_baseline(config, run_dir, index=0, num_shards=1, resume=False, pilot=False):
    import torch
    from PIL import Image
    from .data import validate_dataset
    from .report import competence_summary
    check_gates(run_dir, config, ["parity"] if pilot else ["parity", "pilot"])
    dataset = Path(config["dataset_dir"])
    validation = validate_dataset(dataset)
    manifest, examples = load_examples(dataset, config.get("splits", ["pilot"]), config.get("dataset_hash"))
    if pilot:
        # Exactly 32 forward examples, all controls/scoring additionally timed.
        examples = examples[:32]
    families = sorted({e["family_id"] for e in examples})
    adapter = LoopVLAdapter(config)
    identity = {"config_hash": digest(config), "dataset_hash": manifest["dataset_hash"],
                "study_fingerprint": config["study_fingerprint"], "stage": "timing-pilot" if pilot else "baseline"}
    stage = Path(run_dir) / identity["stage"] / "shards"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    completed = []
    with FamilyShard(stage, identity, families, index, num_shards, resume) as shard, StopRequested() as stop:
        for family in shard.pending():
            family_started = time.monotonic()
            rows = []
            for example in [e for e in examples if e["family_id"] == family]:
                position = examples.index(example)
                donor = next((examples[(position + j) % len(examples)] for j in range(1, len(examples))
                              if examples[(position + j) % len(examples)]["family_id"] != family), None)
                if donor is None:
                    raise ValueError("Shuffled-image control needs another independent family")
                for condition in ("clean", "shuffled_image", "neutral_image"):
                    image_path = dataset / (donor if condition == "shuffled_image" else example)["image_path"]
                    if condition == "neutral_image":
                        image_path = shard.path / "neutral.png"
                        if not image_path.exists():
                            with Image.open(dataset / example["image_path"]) as img:
                                Image.new("RGB", img.size, (128, 128, 128)).save(image_path)
                    batch = adapter.prepare(image_path, example["prompt"])
                    generated = adapter.generate(batch, config.get("generation_budget", 8))
                    scores = adapter.score(batch, example["candidate_answers"])
                    row = BehavioralMeasurement(
                        config["run_id"], digest(config), config["checkpoint_revision"], example["split"],
                        family, example["example_id"], example["task_family"], condition, example["target_answer"],
                        scores, adapter.tokenizer.decode(generated, skip_special_tokens=True).strip(), True,
                        seed=config.get("seed", 1729)).validate()
                    rows.append(row)
            shard.commit(family, rows, metadata={
                "elapsed_seconds": time.monotonic() - family_started,
                "measured_examples": len(rows) // 3,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()})
            completed.extend(rows)
            output_limit(run_dir, config)
            if stop.requested:
                atomic_json(shard.path / "STOPPED.json", {"reason": "signal", "safe_boundary": family})
                return 75
        shard.finish()
    elapsed = time.monotonic() - started
    committed = [read_json(stage / ("shard-%04d" % index) / (digest(f) + ".json"))["metadata"]
                 for f in families if shard_for(f, num_shards) == index]
    measured = sum(r["measured_examples"] for r in committed)
    measured_elapsed = sum(r["elapsed_seconds"] for r in committed)
    resources = {"elapsed_seconds": measured_elapsed, "this_process_elapsed_seconds": elapsed,
                 "measured_examples": measured,
                 "peak_allocated_vram_bytes": max((r["peak_allocated_vram_bytes"] for r in committed), default=0),
                 "peak_reserved_vram_bytes": max((r["peak_reserved_vram_bytes"] for r in committed), default=0),
                 "seconds_per_example_including_controls_scores": measured_elapsed / measured if measured else None,
                 "timing_scope": "sum of committed family inference/control/scoring times across retries; excludes model loading; ledger counts entire allocation"}
    atomic_json(stage.parent / ("resources-%04d.json" % index), resources)
    if pilot:
        write_gate(run_dir, config, "pilot", passed=True, **resources)
    if num_shards == 1:
        # Re-read on resume to include already committed rows, not just this process's work.
        from .shards import merge_shards
        all_rows = merge_shards(stage, families, 1, ("clean", "shuffled_image", "neutral_image"))
        summary = competence_summary(all_rows, seed=config.get("seed", 1729))
        atomic_json(stage.parent / "competence.json", summary)
        if not pilot:
            write_gate(run_dir, config, "capability", passed=True,
                       behavioral_breadth_passed=len(summary["supported_families"]) >= 3 and
                       "direct_attributes" in summary["supported_families"],
                       observational_scope="all families; behavioral claims restricted to supported families", **summary)
    return 0


def capture_example(adapter, example, dataset, channels, state_layers, artifact, save_raw=False,
                    budget_root=None, output_gib_cap=20):
    import torch
    from PIL import Image
    from .data.alignment import object_token_weights
    batch = adapter.prepare(Path(dataset) / example["image_path"], example["prompt"])
    h, w = int(batch["grid_heights"][0]), int(batch["grid_widths"][0])
    if not getattr(adapter.model.config.vision, "dynamic_resolution", False):
        raise ValueError("Object pooling requires audited dynamic resize without padding; letterbox alignment is unvalidated")
    with Image.open(Path(dataset) / example["image_path"]) as source_image:
        if source_image.getexif().get(274, 1) != 1:
            raise ValueError("Source-image EXIF orientation requires separate mask alignment validation")
        source_image_size = source_image.size
    weights = object_token_weights(Path(dataset) / example["mask_path"], h, w,
                                   merge_size=int(batch["merge_sizes"][0]),
                                   source_image_size=source_image_size)
    labels = sorted(weights)
    header, count = int(batch["header_length"]), int(batch["visual_token_counts"][0])
    if count != h * w:
        raise ValueError("Token grid disagrees with actual visual token count")
    last = batch["input_ids"].shape[1] - 1
    tokens = list(range(header, header + count)) + [last]
    arrays, handles, state_visits = {}, [], {}
    def state_hook(layer, kind):
        def hook(module, args, output=None):
            tensor = args[0] if kind == "input" else output
            values = tensor[0, tokens].detach().float().cpu().numpy()
            pooled = np.stack([pool_tokens(values[:-1], weights[label]) for label in labels] + [values[-1]])
            state_visits.setdefault((layer, kind), []).append(pooled)
        return hook
    for layer in state_layers:
        block = adapter.recurrent.L_module.layers[layer]
        handles.append(block.mlp.register_forward_pre_hook(state_hook(layer, "input")))
        handles.append(block.register_forward_hook(state_hook(layer, "block")))
    try:
        with ProductHooks(adapter.recurrent, channels, example["example_id"], tokens) as hooks:
            logits = adapter.forward(batch).cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    for layer in channels:
        rows = [r for r in hooks.rows if r["physical_layer"] == int(layer)]
        if [r["module_invocation"] for r in rows] != list(range(1, 7)):
            raise ValueError("Incomplete product traces")
        actual = np.asarray([r["activations"] for r in rows], dtype=np.float32)
        arrays["products_layer_" + layer] = np.stack([
            np.stack([pool_tokens(v[:-1], weights[label]) for label in labels] + [v[-1]]) for v in actual])
        if save_raw:
            arrays["diagnostic_real_tokens_layer_" + layer] = actual
    for (layer, kind), values in state_visits.items():
        if len(values) != 6:
            raise ValueError("Incomplete state traces")
        arrays["states_layer_%d_%s" % (layer, kind)] = np.asarray(values, np.float32)
    arrays["first_prediction_logits"] = logits
    arrays["mask_labels"] = np.asarray(labels)
    arrays["token_indices"] = np.asarray(tokens)
    if budget_root is not None:
        current = sum(p.stat().st_size for p in Path(budget_root).rglob("*") if p.is_file())
        maximum_new = sum(v.nbytes for v in arrays.values()) + 65536
        if current + maximum_new > output_gib_cap * 1024 ** 3:
            raise RuntimeError("Insufficient output budget for next atomic capture")
    _atomic_npz(artifact, arrays)
    return {"schema": "reuseinterp.capture.v1", "family_id": example["family_id"],
            "example_id": example["example_id"], "split": example["split"], "task_family": example["task_family"],
            "artifact": str(artifact), "sha256": file_hash(artifact), "object_mask_labels": labels,
            "mask_alignment": weights.coverage,
            "grid": [h, w], "token_strata": ["object_pooled", "last_prompt"],
            "question_summary": "secondary stratum pending", "decode_step": 0,
            "forward_id": 0, "module_kind": "L", "invocations": list(range(1, 7)),
            "outer_cycle_by_invocation": [0, 0, 0, 1, 1, 1], "token_indices": tokens,
            "hook_name": "mlp.down_proj.input", "channel_selection_hash": digest(channels)}


def run_capture(config, run_dir, index=0, num_shards=1, resume=False, confirmation=False):
    check_gates(run_dir, config, ("parity", "pilot", "capability"))
    from .data import validate_dataset
    validate_dataset(config["dataset_dir"])
    manifest, examples = load_examples(config["dataset_dir"], config["splits"], config.get("dataset_hash"))
    if not confirmation and set(config["splits"]) & {"confirmation", "transfer"}:
        raise ValueError("Discovery capture cannot open confirmation/transfer; use a locked confirmation stage")
    stage = "confirmation" if confirmation else "capture"
    if confirmation:
        from .protocol import verify_lock
        gates = check_gates(run_dir, config, ("parity", "pilot", "capability", "discovery", "protocol_lock"))
        lock = verify_lock(Path(run_dir) / "protocol_lock.json", gates["protocol_lock"]["sha256"])
        protocol = lock["protocol"]
        if config["splits"] != ["confirmation"]:
            raise ValueError("Locked observational confirmation opens only confirmation")
        if protocol["dataset"]["dataset_hash"] != manifest["dataset_hash"]:
            raise ValueError("Locked dataset differs from captures")
        actual = sorted({e["family_id"] for e in examples})
        if actual != sorted(protocol["confirmation_family_ids"]):
            raise ValueError("Confirmation family list differs from lock")
        if protocol["causal_candidates"]:
            raise ValueError("Candidate-specific causal collection requires validated donor/horizon executor; observational-only executor cannot satisfy this protocol")
        if not protocol.get("p3_skip_reason"):
            raise ValueError("Skipping causal collection needs an explicit discovery-supported reason in the lock")
    adapter = LoopVLAdapter(config)
    channels = sample_neurons([block.mlp.down_proj.in_features for block in adapter.recurrent.L_module.layers],
                              config.get("seed", 1729), config.get("neurons_per_layer", 32))
    sample_path = Path(run_dir) / "neuron_sample.json"
    sample = {"selection": "uniform_without_replacement_per_physical_layer", "seed": config.get("seed", 1729), "channels": channels}
    if sample_path.exists() and read_json(sample_path) != sample:
        raise ValueError("Representative neuron sample changed")
    atomic_json(sample_path, sample)
    families = sorted({e["family_id"] for e in examples})
    if confirmation and protocol["neuron_sample"] != sample:
        raise ValueError("Representative sample differs from protocol")
    identity = {"stage": stage, "config_hash": digest(config), "dataset_hash": manifest["dataset_hash"],
                "study_fingerprint": config["study_fingerprint"], "neuron_sample_hash": digest(sample)}
    with FamilyShard(Path(run_dir) / stage / "shards", identity, families, index, num_shards, resume) as shard, StopRequested() as stop:
        for family in shard.pending():
            rows = []
            for example in [e for e in examples if e["family_id"] == family]:
                path = shard.path / "arrays" / (example["example_id"] + ".npz")
                row = capture_example(adapter, example, config["dataset_dir"], channels,
                                      config.get("state_layers", [0, 5]), path,
                                      budget_root=os.environ.get("REUSE_RUN_ROOT", str(Path(run_dir).parent)),
                                      output_gib_cap=config.get("output_gib_cap", 20))
                row["artifact"] = str(path.resolve())
                rows.append(row)
            shard.commit(family, rows)
            output_limit(run_dir, config)
            if stop.requested:
                return 75
        shard.finish()
    return 0
