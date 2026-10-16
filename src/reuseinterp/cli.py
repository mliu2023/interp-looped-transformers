"""Single entry point. Model stages require real scheduled GPU validation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from .io import config_load, digest, atomic_json, read_json, file_hash, run_directory

GPU_STAGES = {"validate", "baseline", "capture", "discover-causal", "confirm"}


def _shard_args(args):
    index = args.shard_index
    if args.slurm_array:
        if "SLURM_ARRAY_TASK_ID" not in os.environ:
            raise ValueError("--slurm-array requires SLURM_ARRAY_TASK_ID")
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < args.num_shards:
        raise ValueError("Shard outside original --num-shards")
    return index


def _manifest(config, directory):
    identity = {"run_id": config["run_id"], "study_fingerprint": config["study_fingerprint"],
                "code_revision": config["code_revision"], "checkpoint_revision": config["checkpoint_revision"]}
    path = directory / "study.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError("Incompatible existing study identity")
    atomic_json(path, identity)
    atomic_json(directory / "configs" / (digest(config) + ".json"), config)


def static_audit(config, directory):
    from .model_adapter import CODE_REVISION, CHECKPOINT_REVISION
    source = Path(os.environ.get("REUSE_LOOPVL_DIR", config.get("loopvl_dir", "vendor/LoopVL")))
    files = ["runtime/modeling_loopvl.py", "runtime/hrm_penguin/recurrent/hrm_multimodal.py",
             "runtime/hrm_penguin/inference.py", "runtime/hrm_penguin/generation.py",
             "runtime/hrm_penguin/vision/preprocessing.py", "runtime/requirements.txt"]
    missing = [p for p in files if not (source / p).exists()]
    if missing:
        raise FileNotFoundError("Stage official source before static audit: " + ", ".join(missing))
    report = {"status": "static_source_only", "model_loaded": False, "code_revision": CODE_REVISION,
              "checkpoint_revision": CHECKPOINT_REVISION, "files": {p: file_hash(source / p) for p in files},
              "physical_site": "L_module.layers[l].mlp.down_proj input",
              "loaded_weight_identity": "pending", "parity": "pending", "schedule": ["L", "L", "L", "H", "L", "L", "L", "H"]}
    atomic_json(directory / "static_architecture_audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "validate", "data", "baseline", "capture", "fit", "discover-causal", "lock", "confirm", "verify-results", "report", "fixtures", "merge"])
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--slurm-array", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args(argv)
    try:
        index = _shard_args(args)
        if args.stage in {"report", "verify-results"}:
            if not args.run_dir and args.config:
                args.run_dir = str(run_directory(config_load(args.config)))
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            from .report import build_report, verify_results
            result = (build_report if args.stage == "report" else verify_results)(args.run_dir)
            print(json.dumps(result, indent=2))
            return 0 if result.get("passed", result.get("verification", {}).get("passed", True)) else 2
        if not args.config:
            raise ValueError("--config is required")
        config = config_load(args.config)
        directory = run_directory(config)
        if args.dry_run:
            print(json.dumps({"stage": args.stage, "run_dir": str(directory), "config_hash": digest(config),
                              "shard_index": index, "num_shards": args.num_shards,
                              "resource_estimates": config.get("resource_estimates", {}).get(args.stage, "pending measured pilot"),
                              "gpu_hours_cap": config.get("gpu_hour_cap", 30), "will_execute": False}, indent=2))
            return 0
        directory.mkdir(parents=True, exist_ok=True)
        _manifest(config, directory)
        if args.stage in GPU_STAGES and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU model stages must run inside Slurm; use scripts/slurm/submit.py")
        if args.stage == "audit":
            result = static_audit(config, directory)
        elif args.stage == "data":
            from .data import generate_dataset, validate_dataset
            generate_dataset(config["dataset_dir"], config.get("split_counts"), config.get("seed", 1729))
            result = validate_dataset(config["dataset_dir"])
            atomic_json(directory / "data_validation.json", result)
        elif args.stage == "fixtures":
            from .fixtures import run_fixtures
            result = run_fixtures(directory / "synthetic-fixtures", n_resamples=config.get("bootstrap_resamples", 2000))
        elif args.stage == "validate":
            if args.num_shards != 1:
                raise ValueError("Numerical smoke is a single scheduled GPU job")
            from .model_adapter import LoopVLAdapter
            from .pipeline import load_examples, preflight_record, write_gate
            from .numerical import validate_model
            adapter = LoopVLAdapter(config)
            atomic_json(directory / "loaded_architecture_audit.json", adapter.audit)
            from .data import validate_dataset
            validate_dataset(config["dataset_dir"])
            _, examples = load_examples(config["dataset_dir"], ["smoke"], config.get("dataset_hash"))
            result = validate_model(adapter, examples[:16], config["dataset_dir"], directory, config)
            write_gate(directory, config, "parity", passed=result["passed"], backend_signature=preflight_record()["backend_signature"],
                       config_hash=digest(config), report_sha256=file_hash(directory / "parity_report.json"))
            if not result["passed"]:
                return 2
        elif args.stage in {"baseline", "capture"}:
            from .pipeline import run_baseline, run_capture
            if args.stage == "baseline":
                return run_baseline(config, directory, index, args.num_shards, args.resume, args.pilot)
            return run_capture(config, directory, index, args.num_shards, args.resume)
        elif args.stage == "lock":
            from .protocol import create_lock
            from .pipeline import write_gate
            protocol = read_json(config.get("protocol_draft", "docs/protocol_lock.json"))
            result = create_lock(directory / "protocol_lock.json", protocol, read_json(directory / "gates.json"))
            write_gate(directory, config, "protocol_lock", passed=True, sha256=file_hash(directory / "protocol_lock.json"))
        elif args.stage == "merge":
            from .pipeline import load_examples
            from .shards import merge_shards
            _, examples = load_examples(config["dataset_dir"], config["splits"])
            stage = config.get("merge_stage", "capture")
            families = sorted({e["family_id"] for e in examples})
            rows = merge_shards(directory / stage / "shards", families, args.num_shards,
                                ("clean", "shuffled_image", "neutral_image") if stage == "baseline" else None)
            for row in rows:
                if "artifact" in row and file_hash(row["artifact"]) != row["sha256"]:
                    raise ValueError("Capture artifact checksum mismatch")
            atomic_json(directory / stage / "merged.json", {"rows": rows, "num_families": len(families), "num_shards": args.num_shards})
            result = {"merged_rows": len(rows), "independent_families": len(families)}
        elif args.stage == "fit":
            from .discovery import fit_capture_package, assemble_capture_package, confirm_capture_package
            from .pipeline import write_gate
            confirmation = config.get("analysis_stage") == "confirmation"
            stage = "confirmation" if confirmation else "capture"
            merged = read_json(directory / stage / "merged.json")
            package = directory / ("confirmation_input.npz" if confirmation else "discovery_input.npz")
            analysis_config_path = directory / ("confirmation_analysis_config.json" if confirmation else "discovery_analysis_config.json")
            if package.exists():
                analysis_config = read_json(analysis_config_path)
            else:
                analysis_config = assemble_capture_package(merged["rows"], config["dataset_dir"], directory / "neuron_sample.json", package,
                                                          config, splits=("confirmation",) if confirmation else ("fit", "development", "calibration"))
                atomic_json(analysis_config_path, analysis_config)
            if confirmation:
                result = confirm_capture_package(package, directory / "discovery", directory / "confirmation-analysis", directory / "protocol_lock.json",
                                                 n_resamples=config.get("bootstrap_resamples", 2000), seed=config.get("seed", 1729))
            else:
                result = fit_capture_package(package, directory / "discovery", analysis_config)
                write_gate(directory, config, "discovery", passed=True,
                           discovery_models_sha256=file_hash(directory / "discovery/discovery_models.json"),
                           scope="frozen observational fits; causal discovery pending")
        elif args.stage == "confirm":
            from .pipeline import run_capture
            return run_capture(config, directory, index, args.num_shards, args.resume, confirmation=True)
        else:
            # No fabricated causal selection or confirmation results when gates are absent.
            from .pipeline import check_gates
            required = ["parity", "pilot", "capability", "discovery"]
            if args.stage == "confirm":
                required.append("protocol_lock")
            check_gates(directory, config, required)
            raise ValueError("Locked causal/confirmation execution requires the audited discovery package and candidate-specific controls; see PROGRESS.md")
        from .fixtures import json_safe
        print(json.dumps(json_safe(result), indent=2))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, ImportError, OSError) as error:
        print("reuseinterp: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
