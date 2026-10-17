#!/usr/bin/env python3
"""Verify one completed pilot against raw captures and report descriptive context."""
import argparse
import json
from pathlib import Path

import numpy as np

from reuseinterp.io import atomic_json, file_hash, read_json
from reuseinterp.poles import safe, verify_recomputation


def verify(run):
    run = Path(run)
    summary = read_json(run/'summary.json')
    specification = read_json(run/'protocol.json')['specification']
    index = read_json(run/'capture_index.json')
    rows = read_json(run/'image_rows.json')
    assert index['complete'] and len(index['records']) == 72
    assert len(rows) == 64 and summary['total_model_forwards'] == 73
    layers = [str(layer) for layer in summary['capture_validation']['layers']]
    channels = {layer: specification['channels'][layer] for layer in layers}
    means, minima, maxima, sign_counts, controls = [], [], [], [], []
    primary_ids, control_ids = [], []
    capture_bytes = 0
    for record in index['records']:
        assert record['counts'] == {'L': 6, 'H': 2}
        artifact = Path(record['artifact'])
        assert file_hash(artifact) == record['sha256']
        capture_bytes += artifact.stat().st_size
        with np.load(artifact, allow_pickle=False) as arrays:
            for layer in layers:
                for stream in ('gate_pre', 'gate_effective', 'value', 'product'):
                    value = arrays[f'layer_{layer}__{stream}']
                    assert value.shape == (6, record['visual_tokens'], len(channels[layer]))
                    assert np.isfinite(value).all()
            product, gate, value = [np.concatenate(
                [arrays[f'layer_{layer}__{stream}'] for layer in layers], axis=2)
                for stream in ('product', 'gate_effective', 'value')]
        if record['condition'] == 'primary':
            primary_ids.append(record['scene_id'])
            means.append(product.mean(1))
            minima.append(product.min(1))
            maxima.append(product.max(1))
            positive = product > specification['epsilon']
            sign_counts.append(np.stack([positive.sum(1),
                (positive & (gate < -specification['epsilon']) &
                 (value < -specification['epsilon'])).sum(1)], axis=-1))
        else:
            assert record['condition'] == 'paraphrase'
            control_ids.append(record['scene_id'])
            controls.append(product.mean(1))
    assert primary_ids == [row['scene_id'] for row in rows]
    assert control_ids == [row['scene_id'] for row in rows
                           if row['scene_id'] in specification['paraphrase_scene_ids']]
    recomputed = dict(means=np.stack(means), minima=np.stack(minima), maxima=np.stack(maxima),
                      sign_counts=np.stack(sign_counts), paraphrase_means=np.stack(controls))
    with np.load(run/'pooled_measurements.npz', allow_pickle=False) as pooled:
        for key, actual in recomputed.items():
            assert np.array_equal(actual, pooled[key]), key
    primary_verification = verify_recomputation(run)
    means = recomputed['means']
    families = sorted({row['family_id'] for row in rows})
    groups = [np.asarray([i for i, row in enumerate(rows) if row['family_id'] == family])
              for family in families]
    delta = []
    for family in families:
        by_version = {row['version']: i for i, row in enumerate(rows) if row['family_id'] == family}
        delta.append(means[by_version['color_edit']] - means[by_version['base']])
    delta = np.stack(delta)
    epsilon = specification['epsilon']
    eligible = (np.abs(delta[:, 0]) > epsilon) & (np.abs(delta[:, 3]) > epsilon)
    agreement = (delta[:, 0]*delta[:, 3] > 0) & eligible
    signed_by_layer = {}
    offset = 0
    for layer in layers:
        signed = []
        for invocation in range(6):
            count = 0
            for unit in range(offset, offset+len(channels[layer])):
                positive = sum(np.any(means[group, invocation, unit] > epsilon) for group in groups)
                negative = sum(np.any(means[group, invocation, unit] < -epsilon) for group in groups)
                count += positive >= 4 and negative >= 4
            signed.append(int(count))
        signed_by_layer[layer] = signed
        offset += len(channels[layer])
    differences = np.stack([np.abs(control-means[primary_ids.index(scene)])
                            for scene, control in zip(control_ids, controls)])
    context = safe({
        'layer_order': layers, 'signed_image_mean_counts_by_layer_and_invocation': signed_by_layer,
        'color_edit_eligible_pair_unit_comparisons': int(eligible.sum()),
        'color_edit_total_pair_unit_comparisons': int(eligible.size),
        'color_edit_sign_agreements_pooled': int(agreement.sum()),
        'color_edit_pooled_sign_agreement': float(agreement.sum()/eligible.sum()),
        'color_edit_median_absolute_delta_by_invocation': np.median(np.abs(delta), axis=(0, 2)),
        'color_edit_positive_delta_counts_by_invocation': (delta > epsilon).sum(axis=(0, 2)),
        'color_edit_negative_delta_counts_by_invocation': (delta < -epsilon).sum(axis=(0, 2)),
        'prompt_control_max_absolute_change': float(differences.max()),
        'prompt_control_median_absolute_change': float(np.median(differences)),
        'capture_compressed_bytes': capture_bytes,
        'interpretation': 'Descriptive context for the same pilot; no additional hypothesis test or semantic label.'})
    atomic_json(run/'descriptive_context.json', context)
    result = {'passed': True, 'raw_capture_files_verified': len(index['records']),
              'raw_shapes_and_finiteness_passed': True,
              'pooled_arrays_exactly_reproduce_raw_captures': True,
              'primary_statistics': primary_verification,
              'layer_order_restored_from_recorded_capture_list': layers,
              'verification_script_sha256': file_hash(__file__)}
    atomic_json(run/'verification.json', result)
    print(json.dumps({'verification': result, 'descriptive_context': context}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    verify(parser.parse_args().run_dir)
