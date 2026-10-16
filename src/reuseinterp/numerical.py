"""Milestone A runner: no scientific claims until every check passes."""
from pathlib import Path
import numpy as np
from .io import atomic_json, digest
from .capture import ProductHooks
from .interventions import ProductIntervention


def logit_errors(reference, actual):
    a, b = np.asarray(reference, float).reshape(-1), np.asarray(actual, float).reshape(-1)
    difference = np.abs(a - b)
    top = np.argsort(a)[-2:]
    return {"max_abs": float(difference.max()),
            "max_relative": float((difference / np.maximum(np.abs(a), 1e-6)).max()),
            "margin_token_ids": top.tolist(),
            "margin_abs": float(abs((a[top[-1]] - a[top[-2]]) - (b[top[-1]] - b[top[-2]]))),
            "same_argmax": int(a.argmax()) == int(b.argmax())}


def validate_model(adapter, examples, dataset_dir, run_dir, config):
    import torch
    run_dir = Path(run_dir)
    if len(examples) < 16 or len({e["task_family"] for e in examples}) < 4:
        raise ValueError("Smoke requires at least 16 varied examples covering four task families")
    budget = config.get("generation_budget", 8)
    repeated, raw, checks = [], {}, []
    # Measure nondeterminism before hook comparisons. Freeze tolerance once.
    for example in examples[:4]:
        batch = adapter.prepare(Path(dataset_dir) / example["image_path"], example["prompt"])
        _, a = adapter.official_with_logits(batch, budget=1)
        _, b = adapter.official_with_logits(batch, budget=1)
        repeated.append(logit_errors(a, b))
        raw[example["example_id"] + "__repeat_a"] = a
        raw[example["example_id"] + "__repeat_b"] = b
    tolerance = max(config["numerical"]["absolute_floor"], 4 * max(r["max_abs"] for r in repeated))
    margin_tolerance = max(config["numerical"]["margin_floor"], 4 * max(r["margin_abs"] for r in repeated))
    if tolerance > config["numerical"]["absolute_cap"] or margin_tolerance > config["numerical"]["margin_cap"]:
        raise ValueError("Repeated official forwards exceed prespecified numerical caps")
    atomic_json(run_dir / "numerical_tolerances.json", {
        "status": "frozen_before_intervention_checks", "max_abs": tolerance,
        "margin_abs": margin_tolerance, "repeated_official": repeated})
    for example in examples:
        batch = adapter.prepare(Path(dataset_dir) / example["image_path"], example["prompt"])
        token = batch["input_ids"].shape[1] - 1
        official_answer, baseline = adapter.official_with_logits(batch, budget)
        raw[example["example_id"] + "__official"] = baseline
        local_dose = .125
        for condition in ("capture", "self", "zero", "local_dose"):
            intervention = None if condition == "capture" else ProductIntervention(
                layer=0, invocation=4, neuron=0, tokens=(token,),
                dose=local_dose if condition == "local_dose" else 0.0, self_replace=condition == "self")
            with ProductHooks(adapter.recurrent, {"0": [0]}, example["example_id"], [token],
                              intervention, reconstruct=condition == "capture") as hooks:
                logits = adapter.forward(batch).cpu().numpy()
            errors = logit_errors(baseline, logits)
            raw[example["example_id"] + "__" + condition] = logits
            answer = None
            if condition != "local_dose":
                with ProductHooks(adapter.recurrent, {"0": [0]}, example["example_id"], [token], intervention) as generation_hooks:
                    answer = adapter.generate(batch, budget)
                answer_equal = answer == official_answer
                valid = errors["max_abs"] <= tolerance and errors["margin_abs"] <= margin_tolerance and errors["same_argmax"] and answer_equal
            else:
                valid = len(hooks.changed) == 1 and hooks.changed[0]["invocation"] == 4
                # Localization is checked directly before projection; downstream effects may propagate.
                valid = valid and hooks.changed[0]["indices"] == [[0, token, 0]]
            if condition == "capture":
                valid = valid and all(v["max_abs"] <= tolerance for v in hooks.reconstruction)
                visits = hooks.parameter_visits
                valid = valid and len({(v["object_id"], v["storage_ptr"], v["hash"]) for v in visits}) == 1
                valid = valid and len({digest(v["mlp_parameters"]) for v in visits}) == 1
                diagnostic_value = next(r for r in hooks.rows if r["module_invocation"] == 4)["activations"][0][0]
                dtype = adapter.recurrent.L_module.layers[0].mlp.down_proj.weight.dtype
                local_dose = max(.125, 4 * torch.finfo(dtype).eps * abs(diagnostic_value))
            checks.append({"example_id": example["example_id"], "task_family": example["task_family"],
                           "condition": condition, "errors": errors, "official_answer_ids": official_answer,
                           "diagnostic_dose": intervention.dose if intervention is not None else None,
                           "answer_ids": answer, "valid": bool(valid), "counts": hooks.counts,
                           "changed": hooks.changed, "reconstruction": hooks.reconstruction,
                           "parameter_visits": hooks.parameter_visits})
        atomic_json(run_dir / "parity_report.json", {"status": "in_progress", "checks": checks, "max_abs_tolerance": tolerance})
    np.savez_compressed(run_dir / "parity_raw_logits.npz", **raw)
    passed = all(c["valid"] for c in checks)
    report = {"status": "passed" if passed else "failed", "passed": passed,
              "examples": len(examples), "checks": checks, "max_abs_tolerance": tolerance,
              "margin_abs_tolerance": margin_tolerance,
              "scope": "first prediction and greedy outputs; diagnostic physical layer 0/channel 0",
              "donor_object_alignment": "pending separately; no donor analysis authorized by this gate"}
    atomic_json(run_dir / "parity_report.json", report)
    return report
