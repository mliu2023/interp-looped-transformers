"""Deterministic 2D scene families with exact masks and executable labels.

One family contains the original scene, a single-object color edit, and four
questions asked of both versions. It is the indivisible split/resampling unit.
No model outputs or weights enter generation or validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

SCHEMA_VERSION = "reuseinterp.scenes.v1"
DEFAULT_SPLIT_COUNTS = {
    "smoke": 32, "pilot": 128, "fit": 400, "development": 100,
    "calibration": 100, "confirmation": 300, "transfer": 100,
}
COLORS = {
    "red": (205, 47, 47), "green": (43, 150, 66),
    "blue": (48, 91, 205), "yellow": (226, 190, 32),
    "purple": (147, 64, 185), "orange": (229, 122, 35),
}
SHAPES = ("circle", "square", "triangle")
SIZES = ("small", "large")
POSITIONS = (
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
)
TASK_FAMILIES = ("direct_attributes", "relational_selection", "counting", "comparison")

# Frozen primary external-property vocabulary. Final answers, total set counts,
# model scores/confidence, and whether an image is edited are deliberately absent.
# Membership/relevance variables are ground-truth annotations, not claims about
# the model's internal selection. Counts/answers remain in separate label tables.
DESCRIPTOR_WHITELIST = (
    "color", "shape", "size", "position", "center_x", "center_y",
    "visible_fraction", "mask_area_fraction", "question_family",
    "is_reference", "is_target", "count_set_member", "compare_set_a_member",
    "compare_set_b_member", "background_luminance", "clutter_level",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, records: list[dict]) -> None:
    with path.open("wb") as stream:
        for row in records:
            stream.write(_canonical(row) + b"\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _position_object(scene: dict, position: str) -> dict:
    selected = [o for o in scene["objects"] if o["position"] == position]
    if len(selected) != 1:
        raise ValueError(f"position {position!r} does not select exactly one object")
    return selected[0]


def evaluate_program(scene: dict, node: dict) -> Any:
    """Evaluate explicit labels without reference to rendered pixels or a model."""
    op = node["op"]
    if op == "at":
        return _position_object(scene, node["position"])
    if op == "relate":
        ref = evaluate_program(scene, node["reference"])
        row, col = divmod(POSITIONS.index(ref["position"]), 3)
        delta = {"left": -1, "right": 1}[node["relation"]]
        if not 0 <= col + delta < 3:
            raise ValueError("relation leaves scene grid")
        return _position_object(scene, POSITIONS[3 * row + col + delta])
    if op == "query":
        return evaluate_program(scene, node["object"])[node["attribute"]]
    if op == "filter":
        return [o for o in scene["objects"] if o[node["attribute"]] == node["value"]]
    if op == "count":
        return str(len(evaluate_program(scene, node["set"])))
    if op == "compare_counts":
        difference = len(evaluate_program(scene, node["left"])) - len(evaluate_program(scene, node["right"]))
        return "more" if difference > 0 else "fewer" if difference < 0 else "equal"
    raise ValueError(f"unknown program operation: {op!r}")


def independent_answer(scene: dict, specification: dict) -> str:
    """Second label implementation using question specs, not program traversal.

    The independent path detects incorrect program construction or stored labels.
    It intentionally does not call evaluate_program or _position_object.
    """
    by_position = {obj["position"]: obj for obj in scene["objects"]}
    task = specification["task_family"]
    if task == "direct_attributes":
        return str(by_position[specification["position"]][specification["attribute"]])
    if task == "relational_selection":
        ref_index = POSITIONS.index(specification["reference_position"])
        target_index = ref_index + (1 if specification["relation"] == "right" else -1)
        if ref_index // 3 != target_index // 3:
            raise ValueError("invalid independent relation")
        return str(by_position[POSITIONS[target_index]][specification["attribute"]])
    counts = Counter(obj["color"] for obj in scene["objects"])
    if task == "counting":
        return str(counts[specification["color"]])
    if task == "comparison":
        left, right = counts[specification["color_a"]], counts[specification["color_b"]]
        return ("more" if left > right else "fewer" if left < right else "equal")
    raise ValueError(f"unknown task family: {task!r}")


def _question(spec: dict, transfer: bool = False) -> tuple[str, dict, list[str]]:
    task = spec["task_family"]
    readable = lambda value: value.replace("_", " ")
    if task in ("direct_attributes", "relational_selection"):
        attribute = spec["attribute"]
        candidates = list(COLORS) if attribute == "color" else list(SHAPES if attribute == "shape" else SIZES)
        if task == "direct_attributes":
            location = readable(spec["position"])
            wording = (f"Give the {attribute} of the {location} object." if transfer else
                       f"What {attribute} is the object in the {location} position?")
            object_program = {"op": "at", "position": spec["position"]}
        else:
            location, relation = readable(spec["reference_position"]), spec["relation"]
            wording = (f"Find the object directly {relation} of the {location} object. Give its {attribute}." if transfer else
                       f"What {attribute} is the object immediately to the {relation} of the object in the {location} position?")
            object_program = {"op": "relate", "relation": relation,
                              "reference": {"op": "at", "position": spec["reference_position"]}}
        return wording, {"op": "query", "attribute": attribute, "object": object_program}, candidates
    if task == "counting":
        wording = (f"Give the total number of {spec['color']} objects." if transfer else
                   f"How many objects are {spec['color']}?")
        return wording, {"op": "count", "set": {"op": "filter", "attribute": "color", "value": spec["color"]}}, [str(i) for i in range(10)]
    wording = (f"Compare the number of {spec['color_a']} objects to {spec['color_b']} objects: more, fewer, or equal?" if transfer else
               f"Are there more, fewer, or an equal number of {spec['color_a']} objects compared with {spec['color_b']} objects? Answer more, fewer, or equal.")
    return wording, {"op": "compare_counts",
                     "left": {"op": "filter", "attribute": "color", "value": spec["color_a"]},
                     "right": {"op": "filter", "attribute": "color", "value": spec["color_b"]}}, ["more", "fewer", "equal"]


def _make_family(ordinal: int, seed: int, image_size: int) -> tuple[dict, dict, list[dict], str]:
    family_id = f"family_{ordinal:06d}"
    rng = random.Random(int(_hash([seed, family_id])[:16], 16))
    palette = list(COLORS)
    # Counts include zero and large sets; color, shape, position, nuisance, and
    # edit direction are varied independently by the recorded family seed.
    clutter = ordinal % 3
    positions = list(POSITIONS[:6]) + rng.sample(list(POSITIONS[6:]), clutter)
    count_color = palette[(ordinal // 3) % len(palette)]
    desired_count = (ordinal // 18) % (len(positions) + 1)
    colors = [count_color] * desired_count
    alternatives = [color for color in palette if color != count_color]
    colors += [rng.choice(alternatives) for _ in range(len(positions) - desired_count)]
    rng.shuffle(colors)
    objects = []
    for index, (position, color) in enumerate(zip(positions, colors)):
        row, column = divmod(POSITIONS.index(position), 3)
        objects.append({"object_id": f"object_{index:02d}", "mask_label": index + 1,
                        "color": color, "shape": SHAPES[(ordinal + index) % len(SHAPES)],
                        "size": SIZES[(ordinal // 3 + index) % len(SIZES)],
                        "position": position, "center_x": (column + .5) / 3,
                        "center_y": (row + .5) / 3, "visible_fraction": 1.0})
    scene = {"family_id": family_id, "image_size": image_size, "objects": objects,
             "nuisance": {"background_luminance": (225, 239, 250)[(ordinal // 3) % 3], "clutter_level": clutter}}
    # The designated object's role differs across questions in the same family.
    edited_index = rng.randrange(len(objects))
    edited = deepcopy(scene)
    old_color = scene["objects"][edited_index]["color"]
    new_color = palette[(palette.index(old_color) + 1 + (ordinal // 6) % 5) % len(palette)]
    edited["objects"][edited_index]["color"] = new_color
    direct_index = (edited_index if ordinal % 2 == 0 else (edited_index + 1) % len(objects))
    ref_index = rng.choice([0, 1, 2, 3, 4, 5])
    ref_col = ref_index % 3
    direction = "right" if ref_col == 0 else "left" if ref_col == 2 else rng.choice(["left", "right"])
    # Use an independent randomization stream: modulo schedules for attributes
    # can otherwise align with the six-color or nuisance cycles.
    attribute_rng = random.Random(int(_hash(["question_attributes", seed, family_id])[:16], 16))
    specs = [
        {"task_family": "direct_attributes", "attribute": attribute_rng.choice(("color", "shape", "size")), "position": objects[direct_index]["position"]},
        {"task_family": "relational_selection", "attribute": attribute_rng.choice(("color", "shape", "size")), "reference_position": POSITIONS[ref_index], "relation": direction},
        {"task_family": "counting", "color": count_color},
        {"task_family": "comparison", "color_a": palette[ordinal % 6], "color_b": palette[(ordinal % 6 + 1 + ordinal // 6 % 5) % 6]},
    ]
    return scene, edited, specs, objects[edited_index]["object_id"]


def _render(scene: dict, image_path: Path, mask_path: Path | None = None) -> dict[str, int]:
    from PIL import Image, ImageDraw

    size = scene["image_size"]
    background = scene["nuisance"]["background_luminance"]
    image = Image.new("RGB", (size, size), (background,) * 3)
    mask = Image.new("L", (size, size), 0)
    draw, mask_draw = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    for obj in scene["objects"]:
        cx, cy = obj["center_x"] * size, obj["center_y"] * size
        radius = size * (.060 if obj["size"] == "small" else .086)
        box = (round(cx - radius), round(cy - radius), round(cx + radius), round(cy + radius))
        kwargs = {"fill": COLORS[obj["color"]]}
        mask_kwargs = {"fill": obj["mask_label"]}
        if obj["shape"] == "circle":
            draw.ellipse(box, **kwargs)
            mask_draw.ellipse(box, **mask_kwargs)
        elif obj["shape"] == "square":
            draw.rectangle(box, **kwargs)
            mask_draw.rectangle(box, **mask_kwargs)
        else:
            triangle = ((round(cx), box[1]), (box[0], box[3]), (box[2], box[3]))
            draw.polygon(triangle, **kwargs)
            mask_draw.polygon(triangle, **mask_kwargs)
    image.save(image_path)
    if mask_path is not None:
        mask.save(mask_path)
    return {str(label): area for label, area in Counter(mask.getdata()).items() if label}


def _descriptors(scene: dict, example: dict, areas: dict) -> list[dict]:
    spec = example["question_spec"]
    task = spec["task_family"]
    target_position = spec.get("position")
    if task == "relational_selection":
        index = POSITIONS.index(spec["reference_position"])
        target_position = POSITIONS[index + (1 if spec["relation"] == "right" else -1)]
    rows = []
    for obj in scene["objects"]:
        row = {key: obj[key] for key in ("color", "shape", "size", "position", "center_x", "center_y", "visible_fraction")}
        row.update({"mask_area_fraction": areas[str(obj["mask_label"])] / scene["image_size"] ** 2,
                    "question_family": task,
                    "is_reference": obj["position"] == spec.get("reference_position"),
                    "is_target": obj["position"] == target_position,
                    "count_set_member": task == "counting" and obj["color"] == spec.get("color"),
                    "compare_set_a_member": task == "comparison" and obj["color"] == spec.get("color_a"),
                    "compare_set_b_member": task == "comparison" and obj["color"] == spec.get("color_b"),
                    **scene["nuisance"]})
        rows.append({"example_id": example["example_id"], "family_id": example["family_id"],
                     "split": example["split"], "object_id": obj["object_id"], "features": row})
    return rows


def generate_dataset(output_dir: str | Path, split_counts: dict[str, int] | None = None,
                     seed: int = 1729, image_size: int = 192) -> dict:
    """Write a new immutable dataset atomically; return its manifest.

    Passing a partial split_counts creates only those splits, useful for smoke
    preparation. Omitted split_counts creates the plan's full 1,160 families.
    Existing matching datasets are validated/reused; incompatible paths fail.
    """
    import PIL

    counts = dict(DEFAULT_SPLIT_COUNTS if split_counts is None else split_counts)
    if not counts or any(split not in DEFAULT_SPLIT_COUNTS for split in counts):
        raise ValueError("split_counts must contain known, nonempty split names")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("split counts must be nonnegative integers with positive total")
    if any(count > DEFAULT_SPLIT_COUNTS[split] for split, count in counts.items()):
        raise ValueError("requested split exceeds the fixed protocol universe; declare a new protocol before expansion")
    if image_size < 96 or image_size % 3:
        raise ValueError("image_size must be a multiple of three and at least 96")
    config = {"schema_version": SCHEMA_VERSION, "split_counts": counts, "seed": seed,
              "image_size": image_size, "renderer": "Pillow", "renderer_version": PIL.__version__,
              "generator_code_sha256": _file_hash(Path(__file__)),
              "split_universe": DEFAULT_SPLIT_COUNTS,
              "descriptor_whitelist": list(DESCRIPTOR_WHITELIST)}
    output = Path(output_dir)
    if output.exists():
        manifest_file = output / "manifest.json"
        if not manifest_file.exists():
            raise FileExistsError(f"refusing to overwrite existing path {output}")
        existing = json.loads(manifest_file.read_text())
        if existing["config_hash"] != _hash(config):
            raise FileExistsError(f"incompatible existing dataset at {output}")
        validate_dataset(output)
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (staging / "images").mkdir()
        (staging / "masks").mkdir()
        # Rank the complete, prespecified universe even for a small CPU subset.
        # Expanding smoke/pilot to the full manifest therefore never moves a
        # previously exposed family into confirmation or transfer.
        ordinals = sorted(range(sum(DEFAULT_SPLIT_COUNTS.values())), key=lambda i: _hash(["split", seed, i]))
        assignment = {}
        offset = 0
        for split, capacity in DEFAULT_SPLIT_COUNTS.items():
            for ordinal in ordinals[offset:offset + counts.get(split, 0)]:
                assignment[ordinal] = split
            offset += capacity
        scenes, examples, descriptors, families = [], [], [], []
        files = {}
        for ordinal in sorted(assignment):
            base, edited, specs, edited_object_id = _make_family(ordinal, seed, image_size)
            family_id, split = base["family_id"], assignment[ordinal]
            mask_file = f"masks/{family_id}.png"
            family_examples = []
            for version, scene in (("base", base), ("color_edit", edited)):
                scene_id = f"{family_id}__{version}"
                image_file = f"images/{scene_id}.png"
                areas = _render(scene, staging / image_file, staging / mask_file if version == "base" else None)
                scene.update({"scene_id": scene_id, "version": version, "split": split,
                              "image_path": image_file, "mask_path": mask_file,
                              "mask_areas": areas, "edited_object_id": edited_object_id})
                scenes.append(scene)
                files[image_file] = _file_hash(staging / image_file)
                for q_index, spec in enumerate(specs):
                    text, program, candidates = _question(spec, transfer=split == "transfer")
                    answer = str(evaluate_program(scene, program))
                    if answer != independent_answer(scene, spec):
                        raise ValueError("independent program label validation failed")
                    example_id = f"{scene_id}__q{q_index}"
                    example = {"example_id": example_id, "family_id": family_id,
                               "scene_id": scene_id, "split": split, "version": version,
                               "pair_id": f"{family_id}__q{q_index}", "task_family": spec["task_family"],
                               "question": text, "prompt": f"Answer with one word or one number.\n{text}",
                               "question_spec": spec, "program": program,
                               "target_answer": answer, "candidate_answers": candidates,
                               "image_path": image_file, "mask_path": mask_file,
                               "distribution": "deterministic_2d_not_clevr"}
                    examples.append(example)
                    family_examples.append(example_id)
                    descriptors.extend(_descriptors(scene, example, areas))
            files[mask_file] = _file_hash(staging / mask_file)
            families.append({"family_id": family_id, "split": split, "edited_object_id": edited_object_id,
                             "scene_ids": [base["scene_id"], edited["scene_id"]], "example_ids": family_examples})
        for path, records in (("scenes.jsonl", scenes), ("examples.jsonl", examples), ("descriptors.jsonl", descriptors)):
            _jsonl(staging / path, records)
            files[path] = _file_hash(staging / path)
        manifest = {**config, "config_hash": _hash(config), "dataset_hash": _hash(files),
                    "distribution": "deterministic_2d_not_clevr",
                    "split_method": "sha256 rank of [split,seed,ordinal] in fixed 1160-family universe; reduced subsets retain split membership",
                    "split_unit": "whole scene family, both versions and all questions",
                    "transfer_difference": "held-out question wording; same scene generator",
                    "paths": {"scenes": "scenes.jsonl", "examples": "examples.jsonl", "descriptors": "descriptors.jsonl"},
                    "num_families": len(families), "num_scenes": len(scenes), "num_examples": len(examples),
                    "families": families, "files": files,
                    "mask_convention": "PNG L: 0 background; object.mask_label identifies exact object pixels; original image grid",
                    "model_grid_alignment_status": "pending: processor resizing/cropping/merging must be audited on GPU",
                    "competence_status": "not_evaluated"}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report = validate_dataset(staging)
        (staging / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_dataset(output_dir: str | Path) -> dict:
    """Validate hashes, labels, grouping, descriptors, masks and local edits.

    Raises ValueError on any error. This validates renderer ground truth, never
    model competence or the model processor's mapping into visual tokens.
    """
    from PIL import Image

    root = Path(output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported data schema")
    if manifest["descriptor_whitelist"] != list(DESCRIPTOR_WHITELIST):
        raise ValueError("descriptor vocabulary changed")
    config = {key: manifest[key] for key in ("schema_version", "split_counts", "seed", "image_size", "renderer", "renderer_version", "generator_code_sha256", "split_universe", "descriptor_whitelist")}
    if _hash(config) != manifest["config_hash"] or manifest["split_universe"] != DEFAULT_SPLIT_COUNTS:
        raise ValueError("dataset configuration hash or split universe mismatch")
    for filename, expected in manifest["files"].items():
        if _file_hash(root / filename) != expected:
            raise ValueError(f"dataset checksum mismatch: {filename}")
    if _hash(manifest["files"]) != manifest["dataset_hash"]:
        raise ValueError("dataset hash mismatch")
    scenes = _read_jsonl(root / manifest["paths"]["scenes"])
    examples = _read_jsonl(root / manifest["paths"]["examples"])
    descriptors = _read_jsonl(root / manifest["paths"]["descriptors"])
    scene_map = {scene["scene_id"]: scene for scene in scenes}
    example_map = {row["example_id"]: row for row in examples}
    if len(scene_map) != len(scenes) or len(example_map) != len(examples):
        raise ValueError("duplicate scene or example IDs")
    if len(scenes) != manifest["num_scenes"] or len(examples) != manifest["num_examples"]:
        raise ValueError("manifest count mismatch")
    descriptor_map = defaultdict(list)
    for row in descriptors:
        if set(row["features"]) != set(DESCRIPTOR_WHITELIST):
            raise ValueError("descriptor whitelist violation")
        descriptor_map[row["example_id"]].append(row)
    coverage = {"task_family": Counter(), "split_examples": Counter(), "split_families": Counter(),
                "answers_by_task": defaultdict(Counter), "answers_by_question_type": defaultdict(Counter),
                "queried_positions": Counter(), "nuisance_families": Counter(), "edit_directions": Counter(),
                "changed_answers_by_task": Counter(), "edited_object_roles": Counter()}
    all_family_examples, all_family_scenes, family_ids = set(), set(), set()
    for family in manifest["families"]:
        if family["family_id"] in family_ids:
            raise ValueError("duplicate family ID")
        family_ids.add(family["family_id"])
        coverage["split_families"][family["split"]] += 1
        family_scenes = [scene_map[sid] for sid in family["scene_ids"]]
        if len(family_scenes) != 2 or {s["version"] for s in family_scenes} != {"base", "color_edit"}:
            raise ValueError("each family requires base and edited scenes")
        base, edited = sorted(family_scenes, key=lambda s: s["version"] != "base")
        if any(s["family_id"] != family["family_id"] or s["split"] != family["split"] for s in family_scenes):
            raise ValueError("scene-family split leakage")
        if len(set(family["example_ids"])) != 8:
            raise ValueError("each family requires four questions for both versions")
        if all_family_examples.intersection(family["example_ids"]) or all_family_scenes.intersection(family["scene_ids"]):
            raise ValueError("a descendant belongs to multiple families")
        all_family_examples.update(family["example_ids"])
        all_family_scenes.update(family["scene_ids"])
        if base["nuisance"] != edited["nuisance"] or base["image_size"] != edited["image_size"]:
            raise ValueError("color edit changes nuisance or geometry")
        coverage["nuisance_families"][f"background={base['nuisance']['background_luminance']},clutter={base['nuisance']['clutter_level']}"] += 1
        differences = []
        for original, changed in zip(base["objects"], edited["objects"]):
            diff = {key for key in original if original[key] != changed[key]}
            if diff:
                if diff != {"color"}:
                    raise ValueError("edit changed fields other than color")
                differences.append(original["object_id"])
        if differences != [family["edited_object_id"]] or len(base["objects"]) != len(edited["objects"]):
            raise ValueError("expected exactly one edited object")
        old = next(obj for obj in base["objects"] if obj["object_id"] == differences[0])
        new = next(obj for obj in edited["objects"] if obj["object_id"] == differences[0])
        coverage["edit_directions"][f"{old['color']}->{new['color']}"] += 1
        with Image.open(root / base["mask_path"]) as mask_file:
            mask = mask_file.copy()
        if mask.mode != "L" or mask.size != (base["image_size"],) * 2:
            raise ValueError("invalid mask encoding/geometry")
        mask_pixels = list(mask.getdata())
        areas = Counter(mask_pixels)
        if set(areas) != {0, *[obj["mask_label"] for obj in base["objects"]]}:
            raise ValueError("mask object coverage failure")
        if {str(label): area for label, area in areas.items() if label} != base["mask_areas"] or edited["mask_areas"] != base["mask_areas"]:
            raise ValueError("mask areas disagree with manifest")
        with Image.open(root / base["image_path"]) as image_file:
            if image_file.mode != "RGB" or image_file.size != mask.size:
                raise ValueError("image and mask dimensions/modes differ")
            before = list(image_file.convert("RGB").getdata())
        with Image.open(root / edited["image_path"]) as image_file:
            if image_file.mode != "RGB" or image_file.size != mask.size:
                raise ValueError("image and mask dimensions/modes differ")
            after = list(image_file.convert("RGB").getdata())
        if len(before) != len(mask_pixels) or len(after) != len(mask_pixels):
            raise ValueError("image and mask dimensions differ")
        for label, pixel_before, pixel_after in zip(mask_pixels, before, after):
            if (pixel_before != pixel_after) != (label == old["mask_label"]):
                raise ValueError("changed pixels do not match edited-object mask")
        for scene, pixels in ((base, before), (edited, after)):
            palette = {0: (scene["nuisance"]["background_luminance"],) * 3,
                       **{obj["mask_label"]: COLORS[obj["color"]] for obj in scene["objects"]}}
            if any(pixel != palette[label] for label, pixel in zip(mask_pixels, pixels)):
                raise ValueError("rendered pixels disagree with scene/mask color")
        pairs = defaultdict(dict)
        for eid in family["example_ids"]:
            example = example_map[eid]
            if example["family_id"] != family["family_id"] or example["split"] != family["split"]:
                raise ValueError("example-family split leakage")
            scene = scene_map[example["scene_id"]]
            text, program, candidates = _question(example["question_spec"], transfer=family["split"] == "transfer")
            if example["question"] != text or example["program"] != program or example["candidate_answers"] != candidates:
                raise ValueError("stored question, program, or candidates disagree with specification")
            if example["prompt"] != f"Answer with one word or one number.\n{text}":
                raise ValueError("model prompt must contain only fixed instructions and the question")
            answer = str(evaluate_program(scene, example["program"]))
            if answer != independent_answer(scene, example["question_spec"]) or answer != example["target_answer"]:
                raise ValueError("independent label disagreement")
            if answer not in example["candidate_answers"]:
                raise ValueError("answer absent from candidate vocabulary")
            expected_rows = _descriptors(scene, example, scene["mask_areas"])
            if sorted(expected_rows, key=lambda row: row["object_id"]) != sorted(descriptor_map[eid], key=lambda row: row["object_id"]):
                raise ValueError("object descriptor coverage or label disagreement")
            coverage["task_family"][example["task_family"]] += 1
            coverage["split_examples"][example["split"]] += 1
            coverage["answers_by_task"][example["task_family"]][answer] += 1
            question_type = f"{example['task_family']}:{example['question_spec'].get('attribute', 'set_counts')}"
            coverage["answers_by_question_type"][question_type][answer] += 1
            for key in ("position", "reference_position"):
                if key in example["question_spec"]:
                    coverage["queried_positions"][f"{question_type}:{key}={example['question_spec'][key]}"] += 1
            role = next(row["features"] for row in expected_rows if row["object_id"] == old["object_id"])
            role_name = "target" if role["is_target"] else "reference" if role["is_reference"] else "set_member" if any(role[k] for k in ("count_set_member", "compare_set_a_member", "compare_set_b_member")) else "irrelevant"
            coverage["edited_object_roles"][f"{example['task_family']}:{role_name}"] += 1
            pairs[example["pair_id"]][example["version"]] = example
        for pair in pairs.values():
            if set(pair) != {"base", "color_edit"} or pair["base"]["question"] != pair["color_edit"]["question"]:
                raise ValueError("paired edits must retain the same question")
            if pair["base"]["target_answer"] != pair["color_edit"]["target_answer"]:
                coverage["changed_answers_by_task"][pair["base"]["task_family"]] += 1
    if all_family_examples != set(example_map) or all_family_scenes != set(scene_map):
        raise ValueError("ungrouped descendants")
    if set(descriptor_map) != set(example_map):
        raise ValueError("ungrouped descriptors")
    if len(family_ids) != manifest["num_families"] or dict(coverage["split_families"]) != {k: v for k, v in manifest["split_counts"].items() if v}:
        raise ValueError("family split count mismatch")
    return {"schema_version": "reuseinterp.data_validation.v1", "status": "passed",
            "dataset_hash": manifest["dataset_hash"], "num_families": len(family_ids),
            "num_scenes": len(scenes), "num_examples": len(examples), "num_object_descriptors": len(descriptors),
            "checks": ["file checksums", "whole-family disjoint splits", "independent functional labels",
                       "frozen answer-free descriptor vocabulary", "complete visible-object masks",
                       "image colors match masks and scene labels", "single-color exact local edit",
                       "fixed paired question and geometry"],
            "coverage": json.loads(json.dumps(coverage)),
            "limitations": ["Synthetic 2D distribution; not CLEVR.",
                            "Model competence and image dependence have not been measured.",
                            "Mask-to-model-token alignment remains pending processor audit.",
                            "Transfer split tests held-out wording, not a new renderer or natural images.",
                            "Answer, role, and edit-direction balance is descriptive, not guaranteed exact."]}
