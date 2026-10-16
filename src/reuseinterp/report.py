"""Recompute reports using saved measurements only; never import the model."""
import csv
import math
from pathlib import Path
import numpy as np
from .io import read_json, atomic_json, file_hash
from .statistics import family_bootstrap


REPRODUCTION_RTOL = 1e-9
REPRODUCTION_ATOL = 1e-10


def _reproduced_metric_values(expected, actual):
    """Allow finite-float BLAS roundoff; preserve every JSON type and structure.

    Hashes, labels, counts, booleans, nulls and schemas are exact. Integer/float
    substitutions are rejected even when Python considers their values equal.
    Nonfinite values never count as reproducible, including matching NaNs/Infs.
    """
    if isinstance(actual, np.ndarray):
        actual = actual.tolist()
    elif isinstance(actual, np.generic):
        actual = actual.item()
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _reproduced_metric_values(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            _reproduced_metric_values(left, right) for left, right in zip(expected, actual))
    if isinstance(expected, float):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=REPRODUCTION_RTOL, abs_tol=REPRODUCTION_ATOL)
    return (expected is None or isinstance(expected, (str, bool, int))) and expected == actual


def _accuracy(text, target):
    # Strict normalization is declared rather than forgiving free-form prose.
    return str(text).strip().lower().strip(".") == str(target).strip().lower()


def competence_summary(rows, seed=1729, n_resamples=2000):
    from .schemas import validate_rows
    validate_rows(rows)
    by_example = {}
    for row in rows:
        by_example.setdefault(row["example_id"], {})[row["condition"]] = row
    if any(set(v) != {"clean", "shuffled_image", "neutral_image"} for v in by_example.values()):
        raise ValueError("Capability requires complete paired image controls")
    result, supported = {}, []
    for task in sorted({r["task_family"] for r in rows}):
        records = [r for r in by_example.values() if r["clean"]["task_family"] == task]
        families = [r["clean"]["family_id"] for r in records]
        clean = np.asarray([_accuracy(r["clean"]["generated_answer"], r["clean"]["target_answer"]) for r in records], float)
        shuffled = np.asarray([_accuracy(r["shuffled_image"]["generated_answer"], r["clean"]["target_answer"]) for r in records], float)
        neutral = np.asarray([_accuracy(r["neutral_image"]["generated_answer"], r["clean"]["target_answer"]) for r in records], float)
        if len(set(families)) >= 2:
            interval = family_bootstrap((clean - shuffled)[:, None], families, lambda x, ids: np.mean(x, axis=0),
                                        n_resamples=n_resamples, seed=seed, confidence=.95)
            ci = [float(interval["lower"][0]), float(interval["upper"][0])]
        else:
            ci = [None, None]
        supported_task = clean.mean() >= .65 and (clean - shuffled).mean() >= .10 and ci[0] is not None and ci[0] > 0
        if supported_task:
            supported.append(task)
        result[task] = {"examples": len(records), "independent_families": len(set(families)),
                        "clean_correct": int(clean.sum()), "clean_accuracy": float(clean.mean()),
                        "shuffled_correct": int(shuffled.sum()), "shuffled_accuracy": float(shuffled.mean()),
                        "neutral_correct": int(neutral.sum()), "neutral_accuracy": float(neutral.mean()),
                        "clean_minus_shuffled": float((clean - shuffled).mean()), "paired_95_ci": ci,
                        "candidate_accuracy": float(np.mean([max(r["clean"]["candidate_scores"], key=r["clean"]["candidate_scores"].get) == r["clean"]["target_answer"] for r in records])),
                        "malformed_clean": sum(r["clean"]["generated_answer"].strip().lower().strip(".") not in r["clean"]["candidate_scores"] for r in records),
                        "supported_by_screen": bool(supported_task)}
    return {"families": result, "supported_families": supported,
            "scope": "deterministic 2D distribution; free generation and conditional candidate scoring reported separately",
            "normalization": "strip whitespace, lowercase, strip terminal period; no semantic parsing"}


def verify_results(run_dir):
    root = Path(run_dir)
    checks = []
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for relative, expected in manifest.get("evidence_files", {}).items():
            if file_hash(root / relative) != expected:
                raise ValueError("Evidence checksum mismatch: " + relative)
            checks.append(relative)
    fixture = root / "synthetic-fixtures"
    if fixture.exists():
        from .fixtures import recompute_fixtures
        recomputed = recompute_fixtures(fixture, write=False)
        if not _reproduced_metric_values(read_json(fixture / "metrics.json"), recomputed):
            raise ValueError("Synthetic metrics do not reproduce from saved raw measurements")
        checks.append("synthetic metrics reproduce from raw_synthetic.npz")
    for complete in root.glob("**/shards/shard-*/COMPLETE.json"):
        value = read_json(complete)
        for name, expected in value["files"].items():
            if file_hash(complete.parent / name) != expected:
                raise ValueError("Committed shard checksum mismatch")
            if name != "manifest.json":
                record = read_json(complete.parent / name)
                if "rows" in record:
                    from .io import digest
                    if digest(record["rows"]) != record["rows_hash"]:
                        raise ValueError("Committed row checksum mismatch")
                    for row in record["rows"]:
                        if "artifact" in row and file_hash(row["artifact"]) != row["sha256"]:
                            raise ValueError("Captured raw tensor checksum mismatch")
        checks.append(str(complete.relative_to(root)))
    # Any started shard without a marker is incomplete, even if Slurm succeeded.
    incomplete = [str(p.parent.relative_to(root)) for p in root.glob("**/shards/shard-*/manifest.json")
                  if not (p.parent / "COMPLETE.json").exists()]
    return {"passed": not incomplete, "checks": checks, "incomplete_shards": incomplete,
            "reproduction_policy": {"finite_float_rtol": REPRODUCTION_RTOL,
                                    "finite_float_atol": REPRODUCTION_ATOL,
                                    "exact": "JSON types, structure, counts, booleans, labels and evidence hashes",
                                    "nonfinite_values": "rejected"},
            "model_findings": "pending"}


def build_report(run_dir):
    root = Path(run_dir)
    verification = verify_results(root)
    package = root / "report-package"
    package.mkdir(parents=True, exist_ok=True)
    metrics = {"status": "partial_cpu_preparation", "model_endpoints": {"P1": None, "P2": None, "P3": None},
               "verification": verification}
    status = read_json(root / "status.json") if (root / "status.json").exists() else {}
    dataset_path = root / "data_validation.json"
    if dataset_path.exists():
        metrics["dataset_validation"] = read_json(dataset_path)
    fixture = root / "synthetic-fixtures"
    if fixture.exists():
        from .fixtures import recompute_fixtures, json_safe
        metrics["synthetic_only"] = json_safe(recompute_fixtures(fixture, write=False))
        if (fixture / "synthetic_fixture_checks.svg").exists():
            from .plotting import plot_fixtures
            # Preserve manifest-hashed historical evidence across font/OS changes.
            plot_fixtures(fixture, package / "figures" / "synthetic_fixture_checks.svg")
    confirmation = root / "confirmation-analysis"
    if confirmation.exists():
        from .discovery import recompute_confirmation
        metrics["observational_confirmation"] = recompute_confirmation(confirmation)
        metrics["model_endpoints"].update({key: metrics["observational_confirmation"].get(key.lower()) for key in ("P1", "P2")})
        metrics["status"] = "observational_confirmation_with_remaining_controls_as_recorded"
    atomic_json(package / "metrics.json", metrics)
    atomic_json(package / "limitations.json", {
        "status": "no_fresh_model_results", "completed": status.get("completed", []),
        "pending": ["GPU preflight", "loaded-model architecture/weight identity", "16-example official parity and hook localization",
                    "32-example timing and peak VRAM", "task competence/image controls", "model-grid mask validation",
                    "representative capture", "discovery fits/candidate selection", "immutable protocol lock",
                    "confirmation P1/P2 and P3 if feasible", "five model-result figures"],
        "cluster_access": status.get("cluster_access", "not verified"),
        "remaining_implementation": ["candidate-specific donor and short-horizon causal collection executor",
                                     "orchestration of matched-context, reweighted, paired-edit and permutation controls",
                                     "secondary question-token summaries and last-prompt encoding descriptor set",
                                     "five scientific figure renderers after measurement schema validation"],
        "uncertainty_scope": "Future family intervals condition on fitted models, calibration, and selected sites; no training-seed uncertainty.",
        "alternative_explanations": ["incomplete external descriptors", "upstream representation/distribution change",
                                     "different remaining computation", "limited causal candidate scope", "2D distribution may lack competence"],
        "no_claims": ["a fixed neuron's mathematical function changes", "weight sharing caused the measured organization", "all neurons are interpretable"]})
    with (package / "claim_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_id", "claim", "classification", "evidence", "estimand_or_denominator", "uncertainty", "scope", "alternatives", "recompute"])
        writer.writeheader()
        writer.writerow({"claim_id": "CPU-1", "claim": "CPU engineering checks only; model endpoints unmeasured",
                         "classification": "observed_status", "evidence": "../status.json; ../test-results.txt",
                         "estimand_or_denominator": "0 model examples; 0 GPU jobs", "uncertainty": "not applicable",
                         "scope": "this laptop preparation", "alternatives": "cluster GPU execution pending",
                         "recompute": "python -m unittest discover -s tests -v"})
        if dataset_path.exists():
            data = read_json(dataset_path)
            writer.writerow({"claim_id": "CPU-DATA-1", "claim": "Generated 2D dataset passed independent label/edit/mask/split validation",
                             "classification": "engineering_check", "evidence": "data_validation.json; datasets/scenes-v1/manifest.json",
                             "estimand_or_denominator": "%d families; %d questions" % (data["num_families"], data["num_examples"]),
                             "uncertainty": "exhaustive deterministic validation, not an inferential interval",
                             "scope": "renderer scenes only; no model competence measurement", "alternatives": "model may fail on this distribution",
                             "recompute": "python -c 'from reuseinterp.data import validate_dataset; print(validate_dataset(\"datasets/scenes-v1\"))'"})
        if fixture.exists():
            writer.writerow({"claim_id": "SYNTHETIC-1", "claim": "Synthetic analysis fixtures reproduce from saved arrays",
                             "classification": "engineering_check", "evidence": "../synthetic-fixtures/raw_synthetic.npz; metrics.json#synthetic_only",
                             "estimand_or_denominator": "6 synthetic mechanisms; no model neurons", "uncertainty": "synthetic family bootstrap only",
                             "scope": "constructed fixtures", "alternatives": "does not establish real-model validity",
                             "recompute": "python -m reuseinterp.cli report --run-dir " + str(root)})
    report = """# CPU preparation report — no model findings

The study asks whether externally defined semantic descriptions, frozen readouts,
and causal effect profiles of the same physical LoopVL neurons transfer between
recurrent invocations. Parameter identity is an instrumentation check.

No LoopVL inference or GPU experiment has run in this repository. P1, P2, and P3
are unmeasured. Numerical parity, competence, loaded weight identity, discovery,
and confirmation remain pending. No scientific protocol lock has been issued.

Saved synthetic fixtures check stable semantics, positive gain, sign reversal,
fixed weights with changed upstream representations, changed downstream readers,
and unequal continuation lengths. They are engineering evidence only and are
excluded from model findings. All fixture metrics are recomputed from saved raw
arrays by the report command; no VLM is imported. Their definitions and family
intervals are in metrics.json under synthetic_only.

The deterministic 2D dataset has explicit scene properties, programs, paired
color edits, masks, and immutable family splits. It is a different distribution
from CLEVR. Renderer validation does not establish model competence or visual
token alignment. The exact data counts are in dataset_validation in metrics.json.

The five planned model figures (competence, semantic transfer, neuron distribution,
readout transfer, causal profiles) are pending valid model measurements. No mock
model plots are substituted. Claim provenance is in claim_ledger.csv and remaining
limitations and alternatives are in limitations.json.

Cluster SSH reached Brown but was rejected for keyboard-interactive authentication.
No scheduler job IDs exist; local GPU-hours consumed are zero. Site resources,
quotas, purge policy, and environment remain unverified. Follow
docs/CLUSTER_RUNBOOK.md for the immutable snapshot and scheduled smoke job.

Future semantic transfer failure can reflect missing descriptors or upstream
distribution change. Probe failure establishes only failure of that readout.
Final-answer differences without reliable short-horizon corroboration indicate
pass-dependent outcome sensitivity. They cannot establish semantic reassignment.
"""
    (package / "report.md").write_text(report)
    # A compact portable package includes source raw fixture arrays and verification evidence.
    import shutil
    for name in ("status.json", "data_validation.json", "test-results.txt", "manifest.json", "static_architecture_audit.json", "availability.json"):
        if (root / name).exists():
            shutil.copy2(root / name, package / name)
    if fixture.exists():
        shutil.copytree(fixture, package / "synthetic-fixtures", dirs_exist_ok=True)
    if confirmation.exists():
        shutil.copytree(confirmation, package / "confirmation-analysis", dirs_exist_ok=True)
    if (root / "manifest.json").exists():
        for relative in read_json(root / "manifest.json").get("evidence_files", {}):
            target = Path(relative)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError("Evidence paths must remain within report package")
            destination = package / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / target, destination)
    return {"report_package": str(package), "model_results": "pending", "verification": verification}
