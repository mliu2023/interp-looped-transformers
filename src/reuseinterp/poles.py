"""Bounded exploratory signed-activity pilot; never a population/causal gate."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from .io import atomic_json, digest, file_hash, read_json


class BranchCapture:
    """Observe actual branches/product with exact recurrent event identities."""
    def __init__(self, recurrent, channels, tokens, reconstruct=False):
        self.r, self.channels, self.tokens = recurrent, channels, list(tokens)
        self.reconstruct = reconstruct
        self.handles, self.rows, self.cache = [], {k: [] for k in channels}, {}
        self.counts = {'L': 0, 'H': 0}
        self.errors, self.identities = [], {k: [] for k in channels}
        self.forward_count = 0

    def __enter__(self):
        import torch
        def start(module, args, kwargs):
            self.forward_count += 1
            if self.forward_count != 1:
                raise ValueError('One capture context must identify one prefill forward')
        self.handles.append(self.r.register_forward_pre_hook(start, with_kwargs=True))
        for kind in ('L', 'H'):
            def stack(module, args, kind=kind):
                self.counts[kind] += 1
            self.handles.append(getattr(self.r, kind + '_module').register_forward_pre_hook(stack))
        activation_ids = [id(self.r.L_module.layers[int(k)].mlp.act_fn) for k in self.channels]
        if len(set(activation_ids)) != len(activation_ids):
            raise ValueError('Shared activation module requires a different observation contract')
        for key, indices in self.channels.items():
            mlp = self.r.L_module.layers[int(key)].mlp
            self.cache[key] = {}
            for name, module in [('gate_pre', mlp.gate_proj), ('gate_effective', mlp.act_fn), ('value', mlp.up_proj)]:
                def branch(module, args, output, key=key, name=name, indices=indices):
                    selected = output[:, self.tokens][:, :, indices].detach()
                    self.cache[key][name] = selected
                    if self.reconstruct:
                        self.cache[key]['full_' + name] = output.detach()
                self.handles.append(module.register_forward_hook(branch))
            def product(module, args, key=key, indices=indices, mlp=mlp):
                native = args[0]
                selected = native[:, self.tokens][:, :, indices].detach()
                cache = self.cache[key]
                if set(('gate_pre', 'gate_effective', 'value')) - set(cache):
                    raise ValueError('Missing actual branch observations')
                expected = cache['gate_effective'] * cache['value']
                error = float((expected - selected).abs().max())
                self.errors.append({'layer': int(key), 'invocation': self.counts['L'],
                                    'kind': 'selected_product', 'max_abs': error})
                if not torch.isfinite(selected).all() or error > 1e-6:
                    raise ValueError('Native branch/product reconstruction failed')
                row = {name: cache[name][0].float().cpu().numpy().copy()
                       for name in ('gate_pre', 'gate_effective', 'value')}
                row['product'] = selected[0].float().cpu().numpy().copy()
                row['invocation'] = self.counts['L']
                self.rows[key].append(row)
                self.identities[key].append(tuple((id(p), p.untyped_storage().data_ptr()) for p in mlp.parameters()))
                if self.reconstruct:
                    full = cache['full_gate_effective'] * cache['full_value']
                    self.errors.append({'layer': int(key), 'invocation': self.counts['L'],
                                        'kind': 'full_product', 'max_abs': float((full-native).abs().max())})
                    cache['full_product'] = native.detach()
            self.handles.append(mlp.down_proj.register_forward_pre_hook(product))
            if self.reconstruct:
                def projected(module, args, output, key=key):
                    cache = self.cache[key]
                    expected = torch.nn.functional.linear(cache['full_product'], module.weight, module.bias)
                    self.errors.append({'layer': int(key), 'invocation': self.counts['L'],
                                        'kind': 'full_projection', 'max_abs': float((expected-output).abs().max())})
                    cache.clear()
                self.handles.append(mlp.down_proj.register_forward_hook(projected))
        return self

    def __exit__(self, typ, *args):
        try:
            if typ is None:
                if self.counts != {'L': 6, 'H': 2}:
                    raise ValueError('Incomplete recurrent schedule')
                for key in self.channels:
                    if [row['invocation'] for row in self.rows[key]] != list(range(1, 7)):
                        raise ValueError('Missing or overwritten invocation observations')
                    if len(set(self.identities[key])) != 1:
                        raise ValueError('Physical parameter identity changed across invocations')
                if any(row['max_abs'] > 1e-6 for row in self.errors):
                    raise ValueError('Full MLP reconstruction failed')
        finally:
            for handle in self.handles:
                handle.remove()

    def arrays(self):
        return {f'layer_{key}__{name}': np.stack([row[name] for row in self.rows[key]])
                for key in self.channels for name in ('gate_pre', 'gate_effective', 'value', 'product')}


def ranks(values):
    """Average ranks with ties; rows are observations and columns are units."""
    values = np.asarray(values)
    result = np.empty(values.shape, float)
    for col in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, col], return_inverse=True, return_counts=True)
        average = np.cumsum(counts) - (counts - 1) / 2.0
        result[:, col] = average[inverse]
    return result


def spearman_columns(a, b):
    a, b = ranks(a), ranks(b)
    a, b = a-a.mean(0), b-b.mean(0)
    denominator = np.sqrt((a*a).sum(0) * (b*b).sum(0))
    return np.divide((a*b).sum(0), denominator, out=np.full(a.shape[1], np.nan), where=denominator > 0)


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _median(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float('nan')


def summarize(means, minima, maxima, rows, channels, epsilon=1e-6, draws=500, seed=1729):
    layers = list(channels)
    units = [(int(layer), neuron) for layer in layers for neuron in channels[layer]]
    correlations = np.stack([spearman_columns(means[:, 0], means[:, q]) for q in range(6)])
    families = sorted({row['family_id'] for row in rows})
    groups = [np.flatnonzero(np.asarray([row['family_id'] for row in rows]) == f) for f in families]
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        bootstrap.append(_median(spearman_columns(means[idx, 0], means[idx, 3])))
    both = []
    for q in range(6):
        image_both = []
        patch_both = []
        for u in range(len(units)):
            npos = sum(any(means[idx, q, u] > epsilon) for idx in groups)
            nneg = sum(any(means[idx, q, u] < -epsilon) for idx in groups)
            image_both.append(npos >= 4 and nneg >= 4)
            npos = sum(any(maxima[idx, q, u] > epsilon) for idx in groups)
            nneg = sum(any(minima[idx, q, u] < -epsilon) for idx in groups)
            patch_both.append(npos >= 4 and nneg >= 4)
        both.append({'invocation': q+1, 'units_with_both_signed_image_means': sum(image_both),
                     'units_with_both_signed_spatial_responses': sum(patch_both),
                     'denominator': len(units), 'min_distinct_families_per_sign': 4})
    delta = []
    for family in families:
        by_version = {row['version']: i for i, row in enumerate(rows) if row['family_id'] == family}
        delta.append(means[by_version['color_edit']] - means[by_version['base']])
    delta = np.stack(delta)
    eligible = (np.abs(delta[:, 0]) > epsilon) & (np.abs(delta[:, 3]) > epsilon)
    consistent = ((delta[:, 0]*delta[:, 3]) > 0) & eligible
    agreement = np.divide(consistent.sum(0), eligible.sum(0), out=np.full(len(units), np.nan), where=eligible.sum(0)>0)
    records = []
    for u, (layer, neuron) in enumerate(units):
        records.append({'physical_layer': layer, 'neuron': neuron,
                        'l1_l4_spearman': safe(correlations[3, u]),
                        'paired_edit_sign_agreement_l1_l4': safe(agreement[u]),
                        'eligible_pairs': int(eligible[:, u].sum()),
                        'image_mean_std_by_pass': means[:, :, u].std(0).tolist()})
    return safe({'images': len(rows), 'independent_families': len(families), 'physical_units': len(units),
                 'invocations': 6, 'both_sign_coverage': both,
                 'median_l1_to_pass_spearman': [_median(x) for x in correlations],
                 'median_l1_l4_spearman_by_layer': {str(layer): _median(correlations[3, [u for u, unit in enumerate(units) if unit[0]==int(layer)]]) for layer in layers},
                 'defined_l1_l4_correlations': int(np.isfinite(correlations[3]).sum()),
                 'median_l1_l4_spearman_family_bootstrap_95_interval': np.nanpercentile(bootstrap, [2.5, 97.5]),
                 'bootstrap_draws': draws,
                 'median_paired_color_edit_sign_agreement_l1_l4': _median(agreement),
                 'units': records})


def gallery(run, means, rows, channels, dataset):
    from PIL import Image, ImageDraw
    layers = list(channels)
    unit_offset = 0
    cases = []
    width, height = 900, len(layers)*2*190
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    for li, layer in enumerate(layers):
        # First seeded draw, frozen before any activations were seen.
        neuron = channels[layer][0]
        for qi, q in enumerate((0, 3)):
            y = (li*2+qi)*190
            values = means[:, q, unit_offset]
            draw.text((5, y+3), f'L layer {layer}, channel {neuron}, invocation L{q+1}: image-mean signed responses', fill='black')
            order = np.argsort(values)
            selected = list(order[-3:][::-1]) + list(order[:3])
            for col, i in enumerate(selected):
                x = col*150
                with Image.open(dataset/rows[i]['image_path']) as original:
                    image = original.convert('RGB').resize((140, 140))
                canvas.paste(image, (x+3, y+20))
                label = ('top' if col<3 else 'bottom')+f' {values[i]:+.3g}'
                draw.text((x+3, y+162), label, fill='black')
                draw.text((x+3, y+175), rows[i]['scene_id'].replace('family_', ''), fill='black')
                cases.append({'layer': int(layer), 'neuron': neuron, 'invocation': q+1,
                              'tail': 'top' if col<3 else 'bottom', 'activation': float(values[i]),
                              'scene_id': rows[i]['scene_id'], 'image_path': rows[i]['image_path']})
        unit_offset += len(channels[layer])
    canvas.save(run/'signed_tail_gallery.png')
    atomic_json(run/'gallery_examples.json', cases)


def run_pilot(config, run):
    import torch
    from .model_adapter import LoopVLAdapter
    run = Path(run)
    specification = config['poles_pilot']
    if specification['scope'] != 'exploratory_signed_activity_no_behavioral_or_causal_claims':
        raise ValueError('Pilot scope must be explicit')
    if any(row['split'] != 'pilot' for row in specification['images']):
        raise ValueError('This pilot must not open fit, confirmation, or transfer images')
    if len(specification['images']) > 64:
        raise ValueError('This bounded pilot permits at most 64 primary images')
    parent = Path(specification['parent_parity_report'])
    parent_gates = read_json(specification['parent_gates'])
    if not read_json(parent)['passed'] or file_hash(parent) != parent_gates['parity']['report_sha256']:
        raise ValueError('Parent model parity evidence is missing or corrupt')
    if (run/'protocol.json').exists():
        raise FileExistsError('Never overwrite a pilot; review partial results before a new attempt')
    atomic_json(run/'protocol.json', {'config_hash': digest(config), 'specification': specification,
                                   'parent_parity_sha256': file_hash(parent),
                                   'updates_original_study_gates': False})
    started = time.monotonic()
    adapter = LoopVLAdapter(config)
    adapter.model.eval()
    atomic_json(run/'loaded_architecture_audit.json', adapter.audit)
    torch.cuda.reset_peak_memory_stats()
    channels = specification['channels']
    layers = list(channels)
    arrays_dir = run/'arrays'
    arrays_dir.mkdir()
    rows, means, minima, maxima, sign_counts, control_means = [], [], [], [], [], []
    records = []
    audit = None
    forward_count = 0
    tasks = [(row, 'primary', specification['prompt']) for row in specification['images']]
    tasks += [(row, 'paraphrase', specification['paraphrase']) for row in specification['images']
              if row['scene_id'] in specification['paraphrase_scene_ids']]
    for index, (row, condition, prompt) in enumerate(tasks):
        image_path = Path(config['dataset_dir'])/row['image_path']
        if file_hash(image_path) != row['image_sha256']:
            raise ValueError('A selected input image changed after protocol freezing')
        batch = adapter.prepare(image_path, prompt)
        h, w = int(batch['grid_heights'][0]), int(batch['grid_widths'][0])
        count, header = int(batch['visual_token_counts'][0]), int(batch['header_length'])
        if count != h*w or int(batch['merge_sizes'][0]) != 1:
            raise ValueError('Unvalidated visual token layout')
        tokens = list(range(header, header+count))
        if index == 0:
            _, official = adapter.official_with_logits(batch, budget=1)
            forward_count += 1
        with BranchCapture(adapter.recurrent, channels, tokens, reconstruct=index==0) as observed:
            logits = adapter.forward(batch).cpu().numpy()
        forward_count += 1
        if index == 0:
            error = float(np.max(np.abs(official-logits)))
            if error > 1e-6 or int(np.argmax(official)) != int(np.argmax(logits)):
                raise ValueError('New-site capture changed official logits')
            audit = {'passed': True, 'official_logit_max_abs_error': error,
                     'layers': [int(k) for k in layers], 'counts': observed.counts,
                     'reconstruction': observed.errors, 'scope': 'one image, all selected layers, all L invocations',
                     'hook_site': 'actual mlp.down_proj input; visual tokens only'}
            atomic_json(run/'capture_validation.json', audit)
        arrays = observed.arrays()
        target = arrays_dir/(row['scene_id']+'__'+condition+'.npz')
        np.savez_compressed(target, **arrays)
        product = np.concatenate([arrays['layer_'+k+'__product'] for k in layers], axis=2)
        ge = np.concatenate([arrays['layer_'+k+'__gate_effective'] for k in layers], axis=2)
        value = np.concatenate([arrays['layer_'+k+'__value'] for k in layers], axis=2)
        mean = product.mean(1)
        if condition == 'primary':
            means.append(mean)
            minima.append(product.min(1))
            maxima.append(product.max(1))
            positive = product > specification['epsilon']
            sign_counts.append(np.stack([positive.sum(1),
                               (positive & (ge < -specification['epsilon']) & (value < -specification['epsilon'])).sum(1)], axis=-1))
            rows.append(row)
        else:
            control_means.append((row['scene_id'], mean))
        records.append({'scene_id': row['scene_id'], 'condition': condition,
                        'grid': [h, w], 'visual_tokens': count, 'token_indices': tokens,
                        'artifact': str(target.resolve()), 'sha256': file_hash(target),
                        'counts': observed.counts})
        atomic_json(run/'capture_index.json', {'records': records, 'complete': False})
        from .pipeline import output_limit
        output_limit(run, config)
        print(json.dumps({'completed': index+1, 'planned': len(tasks), 'scene': row['scene_id'], 'condition': condition}), flush=True)
    means, minima, maxima = map(np.stack, (means, minima, maxima))
    np.savez_compressed(run/'pooled_measurements.npz', means=means, minima=minima, maxima=maxima,
                        sign_counts=np.stack(sign_counts),
                        paraphrase_means=np.stack([value for _,value in control_means]))
    atomic_json(run/'image_rows.json', rows)
    summary = summarize(means, minima, maxima, rows, channels, epsilon=specification['epsilon'],
                        draws=specification['bootstrap_draws'], seed=specification['sample_seed'])
    counts = np.stack(sign_counts).sum(0)
    fractions = np.divide(counts[:, :, 1], counts[:, :, 0], out=np.full(counts.shape[:2], np.nan), where=counts[:, :, 0]>0)
    summary['median_double_negative_fraction_of_positive_products_by_pass'] = [_median(v) for v in fractions]
    differences = []
    for scene_id, alternative in control_means:
        primary_index = next(i for i,row in enumerate(rows) if row['scene_id']==scene_id)
        differences.append(np.abs(alternative-means[primary_index]))
    std = means.std(0)
    normalized = np.divide(np.stack(differences), std[None], out=np.full((len(differences),)+std.shape, np.nan), where=std[None]>1e-8)
    summary.update(paraphrase_images=len(control_means),
                   median_paraphrase_absolute_change_in_image_std_units=_median(normalized),
                   total_model_forwards=forward_count,
                   elapsed_seconds=time.monotonic()-started,
                   peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                   checkpoint_revision=config['checkpoint_revision'],
                   prompt=specification['prompt'],
                   token_summary='unweighted mean of all actual visual-token products; spatial extrema separate',
                   capture_validation=audit,
                   status='completed_exploratory_pilot',
                   interpretation='Signed responses and rank transfer only; no established semantic pole, circuit, or superposition claim')
    gallery(run, means, rows, channels, Path(config['dataset_dir']))
    atomic_json(run/'summary.json', safe(summary))
    atomic_json(run/'capture_index.json', {'records': records, 'complete': True})
    with (run/'per_neuron.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['units'][0]))
        writer.writeheader()
        writer.writerows(summary['units'])
    fmt = lambda value: 'undefined' if value is None else f'{value:.4f}'
    report = f'''# Signed activity pilot on LoopVL

Status: completed exploratory pilot. This is one observational experiment on controlled 2D scenes, not an ImageNet reproduction.

- {summary['images']} images from {summary['independent_families']} independent paired scene families.
- {summary['physical_units']} sampled channels in layers {', '.join(layers)}, measured at all six L invocations.
- {forward_count} forwards including one official validation forward and {len(control_means)} prompt controls.
- Primary summary: mean product activation over actual visual tokens. No generated answer tokens enter capture.
- New-site capture/MLP reconstruction validation passed; official logit maximum absolute error {audit['official_logit_max_abs_error']}.

L1 to L4 median per-channel image ranking Spearman correlation: {fmt(summary['median_l1_to_pass_spearman'][3])}. Family bootstrap 95% interval: {summary['median_l1_l4_spearman_family_bootstrap_95_interval']}.

Per-layer median L1/L4 correlations: {summary['median_l1_l4_spearman_by_layer']}.

Median paired color-edit sign agreement between L1 and L4: {fmt(summary['median_paired_color_edit_sign_agreement_l1_l4'])}.

Prompt paraphrase median absolute response change: {fmt(summary['median_paraphrase_absolute_change_in_image_std_units'])} image standard deviations on {len(control_means)} images.

See summary.json for all signed-tail coverage and branch-regime measurements, per_neuron.csv for denominators, pooled_measurements.npz for recomputation, and signed_tail_gallery.png for three channels selected before capture.

High ranking correlation does not establish an invariant semantic explanation. Negative values or bottom tails do not establish a coherent negative semantic pole. Results condition on this prompt, synthetic image distribution, sampled sites, and first prediction. Patches and edited siblings are not independent families. Bootstrap intervals describe image-family uncertainty conditional on these channels; they are not uncertainty across models, training seeds, or all neurons. No original population-study gate was changed.
'''
    (run/'report.md').write_text(report)
    return {k:v for k,v in safe(summary).items() if k not in ('units','capture_validation')}


def verify_recomputation(run):
    """Recompute primary statistics from compact saved arrays without a model."""
    run = Path(run)
    specification = read_json(run/'protocol.json')['specification']
    rows = read_json(run/'image_rows.json')
    expected = read_json(run/'summary.json')
    # Canonical JSON sorts mapping keys (0, 15, 7), whereas the tensor axis
    # follows collection order (0, 7, 15). Restore the recorded list explicitly.
    layers = [str(layer) for layer in expected['capture_validation']['layers']]
    if len(layers) != len(set(layers)) or set(layers) != set(specification['channels']):
        raise ValueError('Recorded layer order does not match configured capture sites')
    channels = {layer: specification['channels'][layer] for layer in layers}
    units = [(int(layer), neuron) for layer in layers for neuron in channels[layer]]
    if units != [(row['physical_layer'], row['neuron']) for row in expected['units']]:
        raise ValueError('Recorded neuron axis does not match capture layer order')
    with np.load(run/'pooled_measurements.npz', allow_pickle=False) as data:
        actual = summarize(data['means'], data['minima'], data['maxima'], rows,
                           channels, epsilon=specification['epsilon'],
                           draws=specification['bootstrap_draws'], seed=specification['sample_seed'])
    # Canonical hashes compare every primary metric, including per-neuron rows.
    passed = digest(actual) == digest({key: expected[key] for key in actual})
    if not passed:
        raise ValueError('Saved pilot statistics do not reproduce from their raw arrays')
    return {'passed': True, 'primary_statistics_sha256': digest(actual), 'images': len(rows)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Recompute primary signed-activity pilot statistics without loading a model')
    parser.add_argument('--run-dir', required=True)
    print(json.dumps(verify_recomputation(parser.parse_args().run_dir), indent=2))
