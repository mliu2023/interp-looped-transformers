"""Frozen-checkpoint signed responses on original LoopVL training-source images.

This is a distinct experiment from ``poles.py``: no synthetic edit pairs, fixed
patch grid, or claim that public source membership proves retained training use.
The old validated branch observer is reused without modifying its contract.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np

from .io import atomic_json, digest, file_hash, read_json
from .poles import BranchCapture, _median, safe, spearman_columns


STREAMS = ('gate_pre', 'gate_effective', 'value', 'product')
SCOPE = 'exploratory_training_source_signed_activity_no_causal_claims'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _npz(path, arrays):
    """Publish a complete archive atomically, including on resumed runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ordered_channels(specification):
    """JSON object sorting must never determine a saved tensor's unit axis."""
    channels = specification['channels']
    layers = [str(layer) for layer in specification['layer_order']]
    if len(set(layers)) != len(layers) or set(layers) != set(channels):
        raise ValueError('layer_order must identify every configured layer exactly once')
    if any(not channels[layer] or len(set(channels[layer])) != len(channels[layer])
           or any(not isinstance(unit, int) or unit < 0 for unit in channels[layer]) for layer in layers):
        raise ValueError('Each layer needs distinct nonnegative integer channels')
    return {layer: channels[layer] for layer in layers}


def load_manifest(specification):
    """Select original image families before model loading or activation access."""
    from .data.images import load_image_manifest
    manifest_path = Path(specification['manifest_path']).resolve()
    if file_hash(manifest_path) != specification['manifest_sha256']:
        raise ValueError('Training-source manifest changed after configuration freezing')
    # The source adapter authenticates completion, the dataset fingerprint,
    # immutable source/checkpoint pins, native dimensions, and decoded RGB hashes.
    manifest = load_image_manifest(manifest_path)
    limit = specification.get('image_limit', 64)
    if not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError('This pilot permits between 1 and 64 primary images')
    rows = manifest['images'][:limit]
    if len(rows) != limit:
        raise ValueError('Manifest contains fewer images than the frozen image limit')
    identifiers, families, hashes = set(), set(), set()
    result = []
    for original in rows:
        row = dict(original)
        image_id = row.get('image_id', row.get('scene_id'))
        if not isinstance(image_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}', image_id):
            raise ValueError('Image IDs must be safe single file-name components')
        row['image_id'] = row['scene_id'] = image_id
        if image_id in identifiers or row['family_id'] in families or row['image_sha256'] in hashes:
            raise ValueError('This pilot requires distinct image identities, families, and byte hashes')
        identifiers.add(image_id)
        families.add(row['family_id'])
        hashes.add(row['image_sha256'])
        provenance = row.get('provenance', {})
        if provenance.get('training_membership') not in ('confirmed_training_record', 'reported_corpus_member'):
            raise ValueError('Every primary image needs an explicit, supported training-membership label')
        if not provenance.get('dataset_id') or not provenance.get('revision'):
            raise ValueError('Every image needs a pinned dataset identity and revision')
        image_path = Path(row['image_path'])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if file_hash(image_path) != row['image_sha256']:
            raise ValueError('Original image bytes do not match the manifest: ' + image_id)
        row['source_image_path'] = str(image_path)
        row['split'] = 'pilot'
        result.append(row)
    controls = specification.get('paraphrase_image_ids', [])
    if len(set(controls)) != len(controls) or set(controls) - identifiers:
        raise ValueError('Paraphrase images must be a distinct subset of selected primary images')
    return result


def validate_layout(batch, adapter):
    """Check the official single-image resize/grid/prefix contract at runtime."""
    import torch
    from hrm_penguin.recurrent.positions import build_compact_positions
    grid = batch['grid_sizes'].detach().cpu().numpy()
    merge = batch['merge_sizes'].detach().cpu().numpy()
    if grid.shape != (1, 3) or merge.shape != (1,) or int(grid[0, 0]) != 1:
        raise ValueError('Only one native image frame is supported')
    # The pinned dynamic processor explicitly requires merge=1. Supporting a
    # different encoder merge/permutation needs a new spatial mapping audit.
    if int(merge[0]) != 1:
        raise ValueError('Pinned native dynamic preprocessing requires merge_size=1')
    h, w = int(batch['grid_heights'][0]), int(batch['grid_widths'][0])
    count, header = int(batch['visual_token_counts'][0]), int(batch['header_length'])
    instruction = int(batch['instruction_length'])
    if h < 1 or w < 1 or count != h*w or [h, w] != [int(grid[0, 1]), int(grid[0, 2])]:
        raise ValueError('Effective visual grid does not match the native encoder grid')
    if batch['input_ids'].shape != (1, header + count + instruction):
        raise ValueError('Input sequence does not match the header/image/instruction partition')
    if not 256 <= count <= 1024:
        raise ValueError('Native visual-token count is outside the pinned processor range')
    if type(adapter.processor).__name__ != 'PenguinDynamicResolutionProcessor':
        raise ValueError('This experiment requires the official dynamic-resolution processor')
    device = batch['input_ids'].device
    _, rope, visual, instruction_mask = build_compact_positions(
        torch.tensor([header], device=device), torch.tensor([instruction], device=device),
        torch.tensor([0], device=device), visual_token_counts=batch['visual_token_counts'],
        grid_heights=batch['grid_heights'], grid_widths=batch['grid_widths'],
        grid_h=adapter.model.config.vision.grid_height, grid_w=adapter.model.config.vision.grid_width,
        rope_mode=adapter.model.config.recurrent_visual.rope_mode)
    tokens = list(range(header, header+count))
    if torch.nonzero(visual[0], as_tuple=False).flatten().tolist() != tokens:
        raise ValueError('Actual visual mask differs from the captured sequence offsets')
    if int(instruction_mask.sum()) != instruction or bool((visual & instruction_mask).any()):
        raise ValueError('Instruction tokens overlap visual capture')
    if adapter.model.config.recurrent_visual.rope_mode == 'hybrid_2d':
        expected_rows = torch.arange(h, device=device).repeat_interleave(w) + header
        expected_cols = torch.arange(w, device=device).repeat(h) + header
        if not torch.equal(rope[0, 0, tokens], expected_rows) or not torch.equal(rope[1, 0, tokens], expected_cols):
            raise ValueError('Visual positions do not follow the audited row-major mapping')
    return tokens, {
        'grid': [h, w], 'visual_tokens': count, 'header_length': header,
        'instruction_length': instruction, 'token_indices': tokens,
        'spatial_layout': {'kind': 'native_resize_row_major', 'effective_grid': [h, w],
                           'encoder_grid': grid[0].tolist(), 'merge_size': int(merge[0]),
                           'exif_transposed': True, 'no_padding': True, 'no_crop': True},
    }


def summarize(means, minima, maxima, rows, channels, epsilon=1e-6, draws=500, seed=1729):
    """Family-cluster bootstrap of per-channel ranks; no synthetic pair endpoint."""
    means, minima, maxima = map(np.asarray, (means, minima, maxima))
    units = [(int(layer), neuron) for layer in channels for neuron in channels[layer]]
    expected_shape = (len(rows), 6, len(units))
    if means.shape != expected_shape or minima.shape != expected_shape or maxima.shape != expected_shape:
        raise ValueError('Image, invocation, or ordered neuron axis does not match pooled measurements')
    if not all(np.isfinite(value).all() for value in (means, minima, maxima)):
        raise ValueError('Nonfinite pooled measurements')
    if not rows or draws < 1:
        raise ValueError('Nonempty images and positive bootstrap count are required')
    correlations = np.stack([spearman_columns(means[:, 0], means[:, q]) for q in range(6)])
    families = sorted({row['family_id'] for row in rows})
    groups = [np.asarray([i for i, row in enumerate(rows) if row['family_id'] == family]) for family in families]
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        bootstrap.append(_median(spearman_columns(means[idx, 0], means[idx, 3])))
    signed, records = [], []
    pos_means, neg_means, pos_patches, neg_patches = [], [], [], []
    for q in range(6):
        positive = np.asarray([[(means[idx, q, u] > epsilon).any() for u in range(len(units))] for idx in groups]).sum(0)
        negative = np.asarray([[(means[idx, q, u] < -epsilon).any() for u in range(len(units))] for idx in groups]).sum(0)
        spatial_positive = np.asarray([[(maxima[idx, q, u] > epsilon).any() for u in range(len(units))] for idx in groups]).sum(0)
        spatial_negative = np.asarray([[(minima[idx, q, u] < -epsilon).any() for u in range(len(units))] for idx in groups]).sum(0)
        pos_means.append(positive)
        neg_means.append(negative)
        pos_patches.append(spatial_positive)
        neg_patches.append(spatial_negative)
        signed.append({'invocation': q+1, 'units_with_both_signed_image_means': int(((positive >= 4) & (negative >= 4)).sum()),
                       'units_with_both_signed_spatial_responses': int(((spatial_positive >= 4) & (spatial_negative >= 4)).sum()),
                       'denominator': len(units), 'min_distinct_families_per_sign': 4})
    for u, (layer, neuron) in enumerate(units):
        records.append({'physical_layer': layer, 'neuron': neuron, 'l1_l4_spearman': safe(correlations[3, u]),
                        'image_mean_std_by_pass': means[:, :, u].std(0).tolist(),
                        'positive_image_mean_families_by_pass': [int(v[u]) for v in pos_means],
                        'negative_image_mean_families_by_pass': [int(v[u]) for v in neg_means],
                        'positive_spatial_families_by_pass': [int(v[u]) for v in pos_patches],
                        'negative_spatial_families_by_pass': [int(v[u]) for v in neg_patches]})
    finite_bootstrap = np.asarray(bootstrap)[np.isfinite(bootstrap)]
    interval = np.percentile(finite_bootstrap, [2.5, 97.5]) if len(finite_bootstrap) else [None, None]
    return safe({'images': len(rows), 'independent_families': len(families), 'physical_units': len(units),
                 'invocations': 6, 'layer_order': [int(layer) for layer in channels], 'both_sign_coverage': signed,
                 'median_l1_to_pass_spearman': [_median(v) for v in correlations],
                 'median_l1_l4_spearman_by_layer': {layer: _median(correlations[3, [u for u, unit in enumerate(units) if unit[0] == int(layer)]]) for layer in channels},
                 'defined_l1_l4_correlations': int(np.isfinite(correlations[3]).sum()),
                 'median_l1_l4_spearman_family_bootstrap_95_interval': interval,
                 'bootstrap_draws': draws, 'finite_bootstrap_draws': len(finite_bootstrap), 'units': records})


def reduce_records(run, specification, rows, records, verify=True):
    """Reduce raw captures in frozen image order; used identically by verification."""
    run = Path(run)
    channels = ordered_channels(specification)
    primary, controls = {}, {}
    epsilon = specification['epsilon']
    for record in records:
        identity = record.get('image_id') or record['scene_id']
        target = Path(record['artifact'])
        if not target.is_absolute():
            target = run / target
        if verify and file_hash(target) != record['sha256']:
            raise ValueError('Capture archive changed: ' + str(target))
        if record['counts'] != {'L': 6, 'H': 2}:
            raise ValueError('Capture lacks the complete recurrent schedule')
        h, w = record['grid']
        if record['visual_tokens'] != h*w:
            raise ValueError('Capture grid does not match visual tokens')
        with np.load(target, allow_pickle=False) as data:
            expected_keys = {f'layer_{layer}__{stream}' for layer in channels for stream in STREAMS}
            if set(data.files) != expected_keys:
                raise ValueError('Raw capture streams differ from the frozen protocol')
            all_streams = {}
            for stream in STREAMS:
                values = []
                for layer in channels:
                    value = data[f'layer_{layer}__{stream}']
                    if value.shape != (6, h*w, len(channels[layer])) or not np.isfinite(value).all():
                        raise ValueError('Raw capture shape/finiteness failed')
                    values.append(value)
                all_streams[stream] = np.concatenate(values, axis=2)
        pooled = {}
        for stream, value in all_streams.items():
            for statistic, reduced in (('means', value.mean(1)), ('minima', value.min(1)), ('maxima', value.max(1))):
                pooled[f'{stream}_{statistic}'] = reduced
        product, gate, value = [all_streams[stream] for stream in ('product', 'gate_effective', 'value')]
        positive = product > epsilon
        pooled['sign_counts'] = np.stack([positive.sum(1),
            (positive & (gate < -epsilon) & (value < -epsilon)).sum(1)], axis=-1)
        collection = primary if record['condition'] == 'primary' else controls if record['condition'] == 'paraphrase' else None
        if collection is None or identity in collection:
            raise ValueError('Unknown or duplicated capture condition/image')
        collection[identity] = pooled
    image_ids = [row['image_id'] for row in rows]
    control_ids = [image_id for image_id in image_ids if image_id in specification.get('paraphrase_image_ids', [])]
    if set(primary) != set(image_ids) or set(controls) != set(control_ids):
        raise ValueError('Raw captures do not cover the exact planned primary/control images')
    result = {key: np.stack([primary[image_id][key] for image_id in image_ids]) for key in next(iter(primary.values()))}
    for key in ('means', 'minima', 'maxima'):
        result[key] = result['product_' + key]
    shape = result['means'].shape[1:]
    result['paraphrase_means'] = np.stack([controls[image_id]['product_means'] for image_id in control_ids]) if control_ids else np.empty((0,)+shape, dtype=np.float32)
    return result


def primary_statistics(run, specification, rows, pooled):
    channels = ordered_channels(specification)
    result = summarize(pooled['means'], pooled['minima'], pooled['maxima'], rows, channels,
                       epsilon=specification['epsilon'], draws=specification['bootstrap_draws'], seed=specification['sample_seed'])
    counts = pooled['sign_counts'].sum(0)
    fractions = np.divide(counts[:, :, 1], counts[:, :, 0], out=np.full(counts.shape[:2], np.nan), where=counts[:, :, 0] > 0)
    result['median_double_negative_fraction_of_positive_products_by_pass'] = [_median(v) for v in fractions]
    controls = specification.get('paraphrase_image_ids', [])
    indices = [i for i, row in enumerate(rows) if row['image_id'] in controls]
    if indices:
        differences = np.abs(pooled['paraphrase_means'] - pooled['means'][indices])
        std = pooled['means'].std(0)
        normalized = np.divide(differences, std[None], out=np.full(differences.shape, np.nan), where=std[None] > 1e-8)
        result['median_paraphrase_absolute_change_in_image_std_units'] = _median(normalized)
        result['prompt_control_median_absolute_change'] = float(np.median(differences))
        result['prompt_control_max_absolute_change'] = float(differences.max())
    else:
        result['median_paraphrase_absolute_change_in_image_std_units'] = None
        result['prompt_control_median_absolute_change'] = None
        result['prompt_control_max_absolute_change'] = None
    result['paraphrase_images'] = len(indices)
    return safe(result)


def verify(run):
    """Verify image/activation hashes, variable grids, pooling, axes, and statistics."""
    run = Path(run)
    protocol = read_json(run/'protocol.json')
    specification = protocol['specification']
    rows = read_json(run/'image_rows.json')
    index = read_json(run/'capture_index.json')
    summary = read_json(run/'summary.json')
    if not index['complete'] or not summary['capture_validation']['passed']:
        raise ValueError('A completed capture and independent official-path validation are required')
    if summary['layer_order'] != [int(layer) for layer in ordered_channels(specification)]:
        raise ValueError('Saved layer axis differs from the frozen ordered protocol')
    for row in rows:
        target = run / row['image_path']
        if file_hash(target) != row['image_sha256']:
            raise ValueError('Copied original image bytes changed')
    pooled = reduce_records(run, specification, rows, index['records'])
    with np.load(run/'pooled_measurements.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(pooled):
            raise ValueError('Pooled archive keys differ from raw reduction')
        for key, actual in pooled.items():
            if not np.array_equal(actual, saved[key]):
                raise ValueError('Pooled archive does not exactly reproduce raw captures: ' + key)
    statistics = primary_statistics(run, specification, rows, pooled)
    if digest(statistics) != digest({key: summary[key] for key in statistics}):
        raise ValueError('Saved primary statistics do not reproduce')
    result = {'passed': True, 'raw_capture_files_verified': len(index['records']),
              'original_image_files_verified': len(rows), 'raw_shapes_and_finiteness_passed': True,
              'pooled_arrays_exactly_reproduce_raw_captures': True,
              'primary_statistics_sha256': digest(statistics), 'layer_order': summary['layer_order'],
              'implementation_sha256': file_hash(__file__)}
    atomic_json(run/'verification.json', result)
    return result


def _report(run, summary):
    fmt = lambda value: 'undefined' if value is None else f'{value:.4f}'
    text = f'''# LoopVL original training-source image pilot

Status: completed observational signed-activity experiment on original images from reported LoopVL training sources.

- {summary['images']} original images from {summary['independent_families']} distinct image families.
- {summary['physical_units']} presampled MLP channels in physical layers {', '.join(map(str, summary['layer_order']))}, all six L invocations.
- {summary['total_model_forwards']} attempted model forwards including independent official-path validation and {summary['paraphrase_images']} prompt controls.
- Native EXIF-oriented, aspect-ratio-preserving preprocessing; per-image grids are recorded without padding or synthetic edit pairs.
- Official logits and full MLP reconstruction passed. The response is the unweighted mean over actual image-token products; spatial extrema are separate.

Median per-channel L1/L4 image-ranking Spearman: **{fmt(summary['median_l1_to_pass_spearman'][3])}**. Image-family bootstrap 95% interval: {summary['median_l1_l4_spearman_family_bootstrap_95_interval']}.

Per-layer median L1/L4 correlations: {summary['median_l1_l4_spearman_by_layer']}.

Channels with both positive and negative image means (at least four distinct families per sign), L1–L6: {[item['units_with_both_signed_image_means'] for item in summary['both_sign_coverage']]} / {summary['physical_units']}.

Prompt paraphrase median absolute change: {fmt(summary['median_paraphrase_absolute_change_in_image_std_units'])} image standard deviations.

Training-membership evidence: {summary['training_membership_counts']}. Public corpus membership does not prove retention or optimizer exposure in this exact checkpoint. The provenance manifest, exact image-byte hashes, source revisions, and stage evidence accompany every image.

The bounded, source-stratified convenience sample is not a representative estimate of LoopVL's training mixture. Its fixed selection procedure is recorded in `source_manifest_original.json`.

Use the interactive viewer for original images, signed overlays, four MLP streams, positive/negative response rankings, and all six invocations. Raw captures, pooling, and summary statistics can be independently verified using `scripts/verify_image_pilot.py`.

These measurements do not establish coherent semantic poles, memorization, circuits, or superposition. Bootstrap intervals describe image-family uncertainty conditional on these channels, this checkpoint, and this prompt. They do not cover all neurons, training seeds, or models. No original population-study gate changed.
'''
    (Path(run)/'report.md').write_text(text)


def run_pilot(config, run, resume=False):
    """Execute one bounded, hash-frozen native-image capture inside Slurm."""
    import torch
    from .model_adapter import LoopVLAdapter
    from .pipeline import output_limit
    run = Path(run)
    specification = config['image_pilot']
    if specification['scope'] != SCOPE or config.get('output_gib_cap', 20) > 20:
        raise ValueError('Explicit exploratory scope and project output limit are required')
    if config.get('dataset_hash') != specification['manifest_sha256']:
        raise ValueError('Study dataset_hash must equal the frozen training-source manifest SHA256')
    channels = ordered_channels(specification)
    selected = load_manifest(specification)
    for row in selected:
        checkpoint = row['provenance'].get('checkpoint_revision')
        if checkpoint is not None and checkpoint != config['checkpoint_revision']:
            raise ValueError('Image provenance targets a different checkpoint revision')
    parent = Path(specification['parent_parity_report'])
    gates = read_json(specification['parent_gates'])
    if not read_json(parent)['passed'] or file_hash(parent) != gates['parity']['report_sha256']:
        raise ValueError('Parent model-loader parity evidence is missing or corrupt')
    frozen = {'schema_version': 1, 'config_hash': digest(config), 'specification': specification,
              'checkpoint_revision': config['checkpoint_revision'], 'code_revision': config['code_revision'],
              'selected_manifest_rows_sha256': digest(selected), 'parent_parity_sha256': file_hash(parent),
              'implementation_sha256': file_hash(__file__),
              'branch_observer_source_sha256': file_hash(Path(__file__).with_name('poles.py')),
              'updates_original_study_gates': False}
    protocol_path = run/'protocol.json'
    if protocol_path.exists():
        if not resume:
            raise FileExistsError('Run already exists; use --resume only for the identical frozen protocol')
        if digest(read_json(protocol_path)) != digest(frozen):
            raise ValueError('Cannot resume after changing configuration, data selection, or collector source')
        if (run/'summary.json').exists() and read_json(run/'capture_index.json')['complete']:
            verify(run)
            return read_json(run/'summary.json')
    else:
        atomic_json(protocol_path, frozen)
    images_dir = run/'images'
    images_dir.mkdir(exist_ok=True)
    rows = []
    for original in selected:
        row = dict(original)
        source = Path(row.pop('source_image_path'))
        suffix = source.suffix.lower() or '.image'
        if not re.fullmatch(r'\.[a-z0-9]{1,10}', suffix):
            suffix = '.image'
        relative = Path('images') / (row['image_id'] + suffix)
        target = run / relative
        if target.exists():
            if file_hash(target) != row['image_sha256']:
                raise ValueError('Staged run image differs from the frozen source bytes')
        else:
            shutil.copyfile(source, target)
        row['image_path'] = relative.as_posix()
        rows.append(row)
    atomic_json(run/'image_rows.json', rows)
    atomic_json(run/'source_manifest.json', {'manifest_sha256': specification['manifest_sha256'], 'images': rows})
    original_manifest = run/'source_manifest_original.json'
    if original_manifest.exists() and file_hash(original_manifest) != specification['manifest_sha256']:
        raise ValueError('The run copy of the original source manifest changed')
    if not original_manifest.exists():
        shutil.copyfile(specification['manifest_path'], original_manifest)
    index_path = run/'capture_index.json'
    index = read_json(index_path) if index_path.exists() else {'records': [], 'complete': False}
    records = index['records']
    done = set()
    for record in records:
        target = run/record['artifact']
        if file_hash(target) != record['sha256']:
            raise ValueError('A resumed raw capture changed')
        key = (record['image_id'], record['condition'])
        if key in done:
            raise ValueError('Duplicate completed task in resumed index')
        done.add(key)
    tasks = [(row, 'primary', specification['prompt']) for row in rows]
    tasks += [(row, 'paraphrase', specification['paraphrase']) for row in rows
              if row['image_id'] in specification.get('paraphrase_image_ids', [])]
    if done - {(row['image_id'], condition) for row, condition, _ in tasks}:
        raise ValueError('Resumed captures include an unplanned image or condition')
    execution_path = run/'execution_state.json'
    execution = read_json(execution_path) if execution_path.exists() else {'attempts': []}
    attempt = {'started_utc': _utc(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               'attempted_model_forwards': 0, 'completed_captures': 0}
    execution['attempts'].append(attempt)
    atomic_json(execution_path, execution)
    started = time.monotonic()
    adapter = LoopVLAdapter(config)
    adapter.model.eval()
    atomic_json(run/'loaded_architecture_audit.json', adapter.audit)
    torch.cuda.reset_peak_memory_stats()
    validated_this_attempt = False
    for row, condition, prompt in tasks:
        if (row['image_id'], condition) in done:
            continue
        image_path = run/row['image_path']
        if file_hash(image_path) != row['image_sha256']:
            raise ValueError('Selected input image changed during capture')
        batch = adapter.prepare(image_path, prompt)
        tokens, layout = validate_layout(batch, adapter)
        validation = not validated_this_attempt
        task_start = time.monotonic()
        if validation:
            attempt['attempted_model_forwards'] += 1
            atomic_json(execution_path, execution)
            _, official = adapter.official_with_logits(batch, budget=1)
        attempt['attempted_model_forwards'] += 1
        atomic_json(execution_path, execution)
        with torch.no_grad():
            with BranchCapture(adapter.recurrent, channels, tokens, reconstruct=validation) as observed:
                logits = adapter.forward(batch).cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite output logits')
        if validation:
            error = float(np.max(np.abs(official-logits)))
            if not np.isfinite(official).all() or not np.isfinite(error) or error > 1e-6 or int(np.argmax(official)) != int(np.argmax(logits)):
                raise ValueError('Native-image capture changed official logits')
            if any(not np.isfinite(record['max_abs']) or record['max_abs'] > 1e-6 for record in observed.errors):
                raise ValueError('Native-image MLP reconstruction contains a nonfinite or excessive error')
            audit = {'passed': True, 'official_logit_max_abs_error': error,
                     'layers': [int(layer) for layer in channels], 'counts': observed.counts,
                     'reconstruction': observed.errors, 'scope': 'first image of this attempt, every selected layer and L invocation',
                     'image_id': row['image_id'], 'condition': condition, 'layout': layout,
                     'hook_site': 'actual mlp.down_proj input; native visual tokens only',
                     'parameter_identity_constant_within_image': True}
            atomic_json(run/'capture_validation.json', audit)
            validated_this_attempt = True
        arrays = observed.arrays()
        artifact = Path('arrays')/(row['image_id']+'__'+condition+'.npz')
        target = run/artifact
        _npz(target, arrays)
        record = {'image_id': row['image_id'], 'scene_id': row['image_id'], 'condition': condition,
                  **layout, 'artifact': artifact.as_posix(), 'sha256': file_hash(target),
                  'counts': observed.counts, 'elapsed_seconds': time.monotonic()-task_start,
                  'parameter_identity_constant_within_image': True}
        records.append(record)
        done.add((row['image_id'], condition))
        attempt['completed_captures'] += 1
        attempt['elapsed_seconds'] = time.monotonic()-started
        attempt['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated()
        atomic_json(index_path, {'records': records, 'complete': False})
        atomic_json(execution_path, execution)
        output_limit(run, config)
        print(json.dumps({'completed': len(done), 'planned': len(tasks), 'image_id': row['image_id'],
                          'condition': condition, 'grid': layout['grid'], 'visual_tokens': layout['visual_tokens']}), flush=True)
    pooled = reduce_records(run, specification, rows, records)
    _npz(run/'pooled_measurements.npz', pooled)
    summary = primary_statistics(run, specification, rows, pooled)
    attempt['elapsed_seconds'] = time.monotonic()-started
    attempt['peak_allocated_vram_bytes'] = torch.cuda.max_memory_allocated()
    attempt['finished_utc'] = _utc()
    atomic_json(execution_path, execution)
    summary.update({
        'status': 'completed_exploratory_training_source_image_pilot',
        'mode': specification.get('mode', 'pilot'), 'checkpoint_revision': config['checkpoint_revision'],
        'code_revision': config['code_revision'], 'manifest_sha256': specification['manifest_sha256'],
        'training_membership_counts': dict(Counter(row['provenance']['training_membership'] for row in rows)),
        'total_model_forwards': sum(item['attempted_model_forwards'] for item in execution['attempts']),
        'elapsed_seconds': sum(item.get('elapsed_seconds', 0) for item in execution['attempts']),
        'peak_allocated_vram_bytes': max(item.get('peak_allocated_vram_bytes', 0) for item in execution['attempts']),
        'prompt': specification['prompt'], 'capture_validation': read_json(run/'capture_validation.json'),
        'visual_token_counts': [record['visual_tokens'] for record in records if record['condition'] == 'primary'],
        'capture_compressed_bytes': sum((run/record['artifact']).stat().st_size for record in records),
        'token_summary': 'unweighted mean over actual native visual-token products; spatial extrema separately',
        'interpretation': 'Signed responses and rank transfer only; no established semantic poles, memorization, circuits, or superposition claim',
    })
    atomic_json(run/'summary.json', safe(summary))
    with (run/'per_neuron.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['units'][0]))
        writer.writeheader()
        writer.writerows(summary['units'])
    _report(run, summary)
    output_limit(run, config)
    atomic_json(index_path, {'records': records, 'complete': True})
    return {key: value for key, value in safe(summary).items() if key not in ('units', 'capture_validation')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_pilot(read_json(args.config), args.run_dir, resume=args.resume), indent=2))


if __name__ == '__main__':
    main()
