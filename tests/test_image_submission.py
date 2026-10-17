"""A real-image diagnostic must authenticate parent parity, not inherit gates."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_spec = importlib.util.spec_from_file_location(
    'image_submission_wrapper', Path(__file__).parents[1] / 'scripts/slurm/submit.py')
submission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(submission)


class ImageSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.report = root / 'parity_report.json'
        self.report.write_text(json.dumps({'passed': True, 'scope': 'parent backend'}))
        self.gates = root / 'gates.json'
        self.evidence = {
            'study_fingerprint': 'parent-study',
            'parity': {'passed': True, 'study_fingerprint': 'parent-study',
                       'report_sha256': hashlib.sha256(self.report.read_bytes()).hexdigest()}}
        self.gates.write_text(json.dumps(self.evidence))
        self.config = {'study_fingerprint': 'real-image-study', 'image_pilot': {
            'parent_gates': str(self.gates), 'parent_parity_report': str(self.report),
            'parent_study_fingerprint': 'parent-study'}}

    def errors(self):
        return submission.gate_errors('image-pilot', self.config, self.gates)

    def test_separate_identity_can_reuse_authenticated_backend_evidence(self):
        self.assertEqual([], self.errors())
        self.assertIn('image-pilot', submission.GPU_STAGES)

    def test_rejects_changed_parent_report(self):
        self.report.write_text(json.dumps({'passed': True, 'scope': 'different report'}))
        self.assertTrue(self.errors())

    def test_rejects_unrelated_or_failed_parent(self):
        self.config['image_pilot']['parent_study_fingerprint'] = 'unrelated-study'
        self.assertTrue(self.errors())
        self.config['image_pilot']['parent_study_fingerprint'] = 'parent-study'
        self.evidence['parity']['passed'] = False
        self.gates.write_text(json.dumps(self.evidence))
        self.assertTrue(self.errors())

    def test_rejects_reusing_synthetic_study_identity(self):
        self.config['study_fingerprint'] = 'parent-study'
        self.assertTrue(self.errors())

    def test_rejects_gate_path_substitution(self):
        other = self.gates.with_name('other.json')
        other.write_bytes(self.gates.read_bytes())
        self.assertTrue(submission.gate_errors('image-pilot', self.config, other))


if __name__ == '__main__':
    unittest.main()
