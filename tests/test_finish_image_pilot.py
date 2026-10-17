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
        return ({'elapsed_seconds': 12., 'peak_allocated_vram_bytes': 5_000_000_000},
                [{'condition': 'primary', 'elapsed_seconds': 2., 'visual_tokens': 1024},
                 {'condition': 'primary', 'elapsed_seconds': 3., 'visual_tokens': 1024}])

    def test_native_maximum_grid_cost_rounds_up_to_five_minutes(self):
        summary, records = self.evidence()
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['load_and_preparation_overhead_seconds'], 7.)
        self.assertEqual(bound['wall_seconds'], 900)
        self.assertEqual(bound['reserved_gpu_hours'], .25)

    def test_smaller_timing_grids_require_quadratic_full_grid_allowance(self):
        summary, records = self.evidence()
        records[0]['visual_tokens'] = 512
        bound = module.timing_bound(summary, records)
        self.assertEqual(bound['quadratic_token_multiplier'], 4.)
        self.assertEqual(bound['wall_seconds'], 2100)

    def test_invalid_measurements_and_more_than_one_hour_are_rejected(self):
        summary, records = self.evidence()
        records[0]['elapsed_seconds'] = 40.
        with self.assertRaisesRegex(ValueError, 'one-hour'):
            module.timing_bound(summary, records)
        records[0]['elapsed_seconds'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid elapsed'):
            module.timing_bound(summary, records)
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            module.timing_bound(summary, records[:1])


if __name__ == '__main__':
    unittest.main()
