"""Real-image safeguards: variable grids, family units, and immutable provenance."""
import json
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from reuseinterp.image_pilot import (STREAMS, load_manifest, ordered_channels,
                                     native_token_limits, primary_statistics, reduce_records, summarize, verify)
from reuseinterp.io import atomic_json, file_hash


class ImageSummaryTests(unittest.TestCase):
    def test_no_synthetic_pair_or_version_requirement(self):
        rows = [{'image_id': f'photo{i}', 'family_id': f'original{i}'} for i in range(8)]
        x = np.arange(8, dtype=float) - 3.5
        means = np.repeat(x[:, None, None], 6, axis=1)
        means[:, 3, 0] *= -1
        result = summarize(means, means-1, means+1, rows, {'7': [91]}, draws=20)
        self.assertAlmostEqual(result['median_l1_to_pass_spearman'][3], -1.)
        self.assertEqual(result['both_sign_coverage'][0]['units_with_both_signed_image_means'], 1)
        self.assertEqual(result['units'][0]['positive_image_mean_families_by_pass'], [4]*6)
        self.assertNotIn('median_paired_color_edit_sign_agreement_l1_l4', result)

    def test_duplicate_observations_do_not_inflate_signed_family_coverage(self):
        rows = [{'image_id': str(i), 'family_id': 'positive' if i < 8 else 'negative'} for i in range(16)]
        x = np.array([1.]*8 + [-1.]*8)
        means = np.repeat(x[:, None, None], 6, axis=1)
        result = summarize(means, means, means, rows, {'0': [4]}, draws=10)
        self.assertEqual(result['independent_families'], 2)
        self.assertEqual(result['both_sign_coverage'][0]['units_with_both_signed_image_means'], 0)

    def test_constant_units_have_explicit_undefined_statistics(self):
        rows = [{'image_id': str(i), 'family_id': str(i)} for i in range(4)]
        x = np.zeros((4, 6, 1), dtype=np.float32)
        result = summarize(x, x, x, rows, {'0': [4]}, draws=10)
        self.assertEqual(result['median_l1_to_pass_spearman'], [None]*6)
        self.assertEqual(result['median_l1_l4_spearman_family_bootstrap_95_interval'], [None, None])
        self.assertEqual(result['finite_bootstrap_draws'], 0)


class NativeArchiveTests(unittest.TestCase):
    def fixture(self, run):
        spec = {'channels': {'0': [2], '7': [4], '15': [6]}, 'layer_order': [0, 7, 15],
                'epsilon': 1e-6, 'bootstrap_draws': 10, 'sample_seed': 37,
                'paraphrase_image_ids': ['photo1']}
        rows = []
        records = []
        (run/'images').mkdir()
        (run/'arrays').mkdir()
        for i, (height, width) in enumerate(((2, 3), (3, 4), (4, 2), (2, 5))):
            image = run/'images'/f'photo{i}.jpg'
            image.write_bytes(b'original-image-' + str(i).encode())
            rows.append({'image_id': f'photo{i}', 'scene_id': f'photo{i}', 'family_id': f'family{i}',
                         'image_path': f'images/photo{i}.jpg', 'image_sha256': file_hash(image)})
            for condition in (('primary', 'paraphrase') if i == 1 else ('primary',)):
                arrays = {}
                for layer in spec['layer_order']:
                    for stream_index, stream in enumerate(STREAMS):
                        values = np.arange(6*height*width, dtype=np.float32).reshape(6, height*width, 1)
                        arrays[f'layer_{layer}__{stream}'] = values + i*10 + layer + stream_index
                artifact = f'arrays/photo{i}__{condition}.npz'
                np.savez_compressed(run/artifact, **arrays)
                records.append({'image_id': f'photo{i}', 'scene_id': f'photo{i}', 'condition': condition,
                                'grid': [height, width], 'visual_tokens': height*width,
                                'native_token_limits': {'min_visual_tokens': 1, 'max_visual_tokens': 12},
                                'artifact': artifact, 'sha256': file_hash(run/artifact), 'counts': {'L': 6, 'H': 2}})
        return spec, rows, records

    def test_variable_grid_reduction_preserves_explicit_axis_and_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            spec, rows, records = self.fixture(run)
            # Canonical JSON sorts physical-layer keys as 0,15,7.
            atomic_json(run/'spec.json', spec)
            spec = json.loads((run/'spec.json').read_text())
            self.assertEqual(list(spec['channels']), ['0', '15', '7'])
            self.assertEqual(list(ordered_channels(spec)), ['0', '7', '15'])
            pooled = reduce_records(run, spec, rows, records)
            self.assertEqual(pooled['means'].shape, (4, 6, 3))
            self.assertEqual(pooled['paraphrase_means'].shape, (1, 6, 3))
            np.testing.assert_array_equal(pooled['means'][0, 0], [5.5, 12.5, 20.5])
            np.testing.assert_array_equal(pooled['paraphrase_means'][0], pooled['means'][1])
            records[0]['grid'] = [2, 2]
            with self.assertRaisesRegex(ValueError, 'grid'):
                reduce_records(run, spec, rows, records)

    def test_full_verification_detects_tampered_pool_and_original_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            spec, rows, records = self.fixture(run)
            pooled = reduce_records(run, spec, rows, records)
            statistics = primary_statistics(run, spec, rows, pooled)
            limits = {'min_visual_tokens': 1, 'max_visual_tokens': 12}
            summary = dict(statistics, layer_order=[0, 7, 15], native_token_limits=limits,
                           capture_validation={'passed': True, 'layout': {'native_token_limits': limits}})
            atomic_json(run/'protocol.json', {'specification': spec})
            atomic_json(run/'image_rows.json', rows)
            atomic_json(run/'capture_index.json', {'records': records, 'complete': True})
            atomic_json(run/'summary.json', summary)
            np.savez_compressed(run/'pooled_measurements.npz', **pooled)
            self.assertTrue(verify(run)['passed'])
            pooled['means'] = pooled['means'].copy()
            pooled['means'][0, 0, 0] += 1
            np.savez_compressed(run/'pooled_measurements.npz', **pooled)
            with self.assertRaisesRegex(ValueError, 'exactly reproduce'):
                verify(run)
            (run/rows[0]['image_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'image bytes changed'):
                verify(run)


class NativePolicyTests(unittest.TestCase):
    def adapter(self):
        vision = SimpleNamespace(dynamic_resolution=True, min_visual_tokens=256, max_visual_tokens=2048)
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vision=vision)),
                               processor=SimpleNamespace(min_visual_tokens=256, max_visual_tokens=2048))

    def test_loaded_checkpoint_2048_limit_overrides_the_runtime_1024_default(self):
        adapter = self.adapter()
        self.assertEqual(native_token_limits(adapter, 2048),
                         {'min_visual_tokens': 256, 'max_visual_tokens': 2048})
        self.assertEqual(native_token_limits(adapter, 1472)['max_visual_tokens'], 2048)
        with self.assertRaisesRegex(ValueError, 'outside the checkpoint'):
            native_token_limits(adapter, 2049)

    def test_processor_checkpoint_policy_mismatch_is_rejected(self):
        adapter = self.adapter()
        adapter.processor.max_visual_tokens = 1024
        with self.assertRaisesRegex(ValueError, 'differ from the loaded checkpoint'):
            native_token_limits(adapter, 1000)


class ManifestTests(unittest.TestCase):
    def test_unknown_membership_or_repeated_family_cannot_enter_primary_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            from PIL import Image
            from reuseinterp.data.images import (CHECKPOINT_REVISION, DATASET_ID, DATASET_REVISION,
                                                 SCHEMA_VERSION, canonical_json, image_fingerprints)
            root = Path(temporary)
            images = []
            for i in range(2):
                image = Image.new('RGB', (32, 32), (i*127, 20, 60))
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                payload = buffer.getvalue()
                (root/f'image{i}.png').write_bytes(payload)
                images.append({'image_id': f'photo{i}', 'family_id': f'family{i}', 'image_path': f'image{i}.png',
                               **image_fingerprints(payload),
                               'provenance': {'dataset_id': DATASET_ID, 'revision': DATASET_REVISION,
                                              'checkpoint_revision': CHECKPOINT_REVISION,
                                              'training_membership': 'reported_corpus_member'}})
            path = root/'manifest.json'
            def publish():
                manifest = {'schema_version': SCHEMA_VERSION, 'complete': True, 'images': images}
                manifest['dataset_fingerprint'] = hashlib.sha256(canonical_json(manifest)).hexdigest()
                atomic_json(path, manifest)
            publish()
            spec = {'manifest_path': str(path), 'manifest_sha256': file_hash(path), 'image_limit': 2}
            self.assertEqual(len(load_manifest(spec)), 2)
            images[1]['family_id'] = images[0]['family_id']
            publish()
            spec['manifest_sha256'] = file_hash(path)
            with self.assertRaisesRegex(ValueError, 'repeats an image family'):
                load_manifest(spec)
            images[1]['family_id'] = 'newfamily'
            images[1]['provenance']['training_membership'] = 'unknown'
            publish()
            spec['manifest_sha256'] = file_hash(path)
            with self.assertRaisesRegex(ValueError, 'training membership'):
                load_manifest(spec)
            # Source metadata cannot silently change after the protocol is frozen.
            images[1]['provenance']['training_membership'] = 'reported_corpus_member'
            publish()
            with self.assertRaisesRegex(ValueError, 'changed after configuration freezing'):
                load_manifest(spec)


if __name__ == '__main__':
    unittest.main()
