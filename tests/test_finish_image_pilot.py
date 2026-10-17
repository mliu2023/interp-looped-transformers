"""Measured allocation limits for the single automatic real-image experiment."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'finish_image_pilot', Path(__file__).resolve().parents[1]/'scripts/finish_image_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AllocationBoundTests(unittest.TestCase):
    def evidence(self):
        return ({'elapsed_seconds': 12., 'peak_allocated_vram_bytes': 5_000_000_000,
                 'native_token_limits': {'min_visual_tokens': 256, 'max_visual_tokens': 1024}},
                [{'condition': 'primary', 'elapsed_seconds': 2., 'visual_tokens': 1024,
                  'includes_official_validation': True},
                 {'condition': 'primary', 'elapsed_seconds': 3., 'visual_tokens': 1024,
                  'includes_official_validation': False}])

    def test_native_maximum_grid_cost_rounds_up_to_five_minutes(self):
        summary, records = self.evidence()
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['load_and_preparation_overhead_seconds'], 7.)
        self.assertEqual(bound['wall_seconds'], 900)
        self.assertEqual(bound['reserved_gpu_hours'], .25)
        self.assertEqual(bound['maximum_normalized_forward_seconds'], 3.)
        self.assertEqual(bound['unique_official_validation_slack_seconds'], 1.)

    def test_smaller_timing_grids_require_quadratic_full_grid_allowance(self):
        summary, records = self.evidence()
        records[0]['visual_tokens'] = 512
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['quadratic_token_multiplier'], 4.)
        self.assertEqual(bound['wall_seconds'], 2100)

    def test_invalid_measurements_and_more_than_one_hour_are_rejected(self):
        summary, records = self.evidence()
        records[0]['elapsed_seconds'] = 80.
        with self.assertRaisesRegex(ValueError, 'one-hour'):
            module.timing_bound(summary, records)
        records[0]['elapsed_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid elapsed'):
            module.timing_bound(summary, records)
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            module.timing_bound(summary, records[:1])

    def test_checkpoint_2048_policy_controls_full_run_allocation(self):
        summary, records = self.evidence()
        summary['native_token_limits']['max_visual_tokens'] = 2048
        records[0]['visual_tokens'] = 1536
        records[1]['visual_tokens'] = 2048
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['maximum_full_visual_tokens'], 2048)
        self.assertAlmostEqual(bound['quadratic_token_multiplier'], (2048/1536)**2)
        self.assertEqual(bound['wall_seconds'], 1200)

    def test_unique_validation_forward_does_not_repeat_across_all_72_tasks(self):
        summary, records = self.evidence()
        records[0]['elapsed_seconds'] = 6.
        summary['elapsed_seconds'] = 16.
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['maximum_timing_record_seconds'], 6.)
        self.assertEqual(bound['maximum_normalized_forward_seconds'], 3.)
        self.assertEqual(bound['unique_official_validation_slack_seconds'], 3.)
        self.assertEqual(bound['wall_seconds'], 900)


if __name__ == '__main__':
    unittest.main()
