"""Small synthetic checks for the exploratory signed-activity pilot."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
from reuseinterp.poles import ranks, spearman_columns, summarize


class RankTests(unittest.TestCase):
    def test_ties_and_constant_units(self):
        x = np.asarray([[1., 4.], [1., 4.], [3., 4.]])
        np.testing.assert_array_equal(ranks(x)[:, 0], [1.5, 1.5, 3.])
        self.assertAlmostEqual(spearman_columns(x, 3*x+7)[0], 1.)
        self.assertTrue(np.isnan(spearman_columns(x, x)[1]))
        self.assertAlmostEqual(spearman_columns(x, -x)[0], -1.)

    def test_signed_coverage_and_family_grouping(self):
        rows = [{'family_id': f'f{i}', 'version': v}
                for i in range(4) for v in ('base', 'color_edit')]
        x = np.asarray([-4., 4., -3., 3., -2., 2., -1., 1.])
        means = np.stack([x, x+1, x+2, 2*x, 3*x, 4*x], axis=1)[:, :, None]
        out = summarize(means, means-1, means+1, rows, {'0': [2]}, draws=20)
        self.assertEqual(out['independent_families'], 4)
        self.assertEqual(out['both_sign_coverage'][0]['units_with_both_signed_image_means'], 1)
        self.assertAlmostEqual(out['median_l1_to_pass_spearman'][3], 1.)
        self.assertAlmostEqual(out['median_paired_color_edit_sign_agreement_l1_l4'], 1.)
        self.assertEqual(out['units'][0]['eligible_pairs'], 4)

    def test_recomputation_preserves_tensor_axis_after_json_key_sorting(self):
        from reuseinterp.io import atomic_json
        from reuseinterp.poles import verify_recomputation
        rows = [{'family_id': f'f{i}', 'version': v}
                for i in range(4) for v in ('base', 'color_edit')]
        x = np.arange(8, dtype=float)-3.5
        columns = np.stack([x, np.roll(x, 1), np.roll(x, 2)], axis=1)
        means = np.repeat(columns[:, None, :], 6, axis=1)
        means[:, 3, 1] *= -1
        channels = {'0': [2], '7': [4], '15': [6]}
        summary = summarize(means, means-1, means+1, rows, channels, draws=20, seed=37)
        summary['capture_validation'] = {'layers': [0, 7, 15]}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            atomic_json(run/'protocol.json', {'specification': {
                'channels': channels, 'epsilon': 1e-6, 'bootstrap_draws': 20, 'sample_seed': 37}})
            atomic_json(run/'image_rows.json', rows)
            atomic_json(run/'summary.json', summary)
            np.savez_compressed(run/'pooled_measurements.npz', means=means,
                                minima=means-1, maxima=means+1)
            self.assertTrue(verify_recomputation(run)['passed'])
            summary['capture_validation']['layers'] = [0, 15, 7]
            atomic_json(run/'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'neuron axis'):
                verify_recomputation(run)


class ActualBranchTests(unittest.TestCase):
    def test_native_observation_reconstructs_and_keeps_all_invocations(self):
        import torch
        from test_hooks import SyntheticRecurrent
        from reuseinterp.poles import BranchCapture
        torch.set_num_threads(1)
        torch.manual_seed(91)
        model = SyntheticRecurrent().eval()
        x = torch.randn(1, 5, 4, dtype=torch.float64)
        with torch.no_grad():
            original = model(x)
            with BranchCapture(model, {'0': [4, 0], '1': [3, 1]}, [0, 2], reconstruct=True) as capture:
                actual = model(x)
        torch.testing.assert_close(original, actual, atol=0, rtol=0)
        arrays = capture.arrays()
        self.assertEqual(arrays['layer_0__product'].shape, (6, 2, 2))
        self.assertEqual(capture.counts, {'L': 6, 'H': 2})
        self.assertTrue(all(row['max_abs']==0 for row in capture.errors))
        self.assertEqual(len(model.L_module.layers[0].mlp.down_proj._forward_pre_hooks), 0)


if __name__ == '__main__':
    unittest.main()
