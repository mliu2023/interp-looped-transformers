"""Recovery scheduling and failure isolation, without installations or compute."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('recover_cuda12', ROOT / 'scripts/recover_cuda12.py')
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
sys.path.pop(0)


class CudaRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.original = self.base / 'cluster.env'
        self.original.write_text('original environment, unchanged\n')
        self.parent = {'base_dir': str(self.base), 'python': sys.executable, 'frozen': True,
                       'snapshot_sha256': 'a' * 64,
                       'environment': {'REUSE_CACHE_READY': '1', 'REUSE_ENV_DIR': str(self.base / 'venv'),
                           'REUSE_PROJECT_DIR': str(self.base / 'snapshot'), 'REUSE_RUN_ROOT': str(self.base / 'runs'),
                           'REUSE_LOOPVL_DIR': str(self.base / 'LoopVL'), 'REUSE_MODEL_DIR': str(self.base / 'model'),
                           'REUSE_DATASET_DIR': str(self.base / 'datasets/scenes-v1')}}

    def test_installation_refuses_login_node(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'scheduled CPU'):
                recovery.recover(str(self.base), 'recovery-test')
        self.assertEqual(list(self.base.iterdir()), [self.original])

    def test_paths_reject_escape_existing_directory_and_incomplete_inputs(self):
        with patch.object(recovery.setup, 'load_state', return_value=self.parent):
            for name in ('../venv', 'snapshot', 'recovery-../other', 'recovery-/tmp'):
                with self.assertRaises(ValueError):
                    recovery.recovery_paths(str(self.base), name)
            (self.base / 'recovery-test').mkdir()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                recovery.recovery_paths(str(self.base), 'recovery-test')
            self.parent['environment']['REUSE_CACHE_READY'] = '0'
            with self.assertRaisesRegex(ValueError, 'fully staged'):
                recovery.recovery_paths(str(self.base), 'recovery-new')

    def test_failed_install_keeps_original_and_never_publishes_environment(self):
        freezer = SimpleNamespace(verify=lambda path: 'a' * 64)
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic-cpu-job'}), \
                patch.object(recovery.setup, 'load_state', return_value=self.parent), \
                patch.object(recovery.setup, 'freeze_module', return_value=freezer), \
                patch.object(recovery.setup, 'run', side_effect=subprocess.CalledProcessError(2, ['pip'])):
            with self.assertRaises(subprocess.CalledProcessError):
                recovery.recover(str(self.base), 'recovery-test')
        destination = self.base / 'recovery-test'
        record = json.loads((destination / 'recovery_state.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertFalse((destination / 'cluster.env').exists())
        self.assertEqual(self.original.read_text(), 'original environment, unchanged\n')
        self.assertEqual(self.parent['environment']['REUSE_ENV_DIR'], str(self.base / 'venv'))

    def test_success_publishes_new_environment_after_tests_inputs_and_freeze(self):
        commands = []
        pair = {'torch': recovery.TORCH, 'torchvision': recovery.TORCHVISION, 'cuda_runtime': '12.8'}
        def runner(state, command, **kwargs):
            command = list(map(str, command))
            commands.append(command)
            self.assertFalse((self.base / 'recovery-test/cluster.env').exists())
            if '-c' in command:
                return json.dumps(pair)
            if 'freeze' in command:
                return 'torch==' + recovery.TORCH + '\n'
            if 'rev-parse' in command:
                return 'synthetic-source-commit\n'
            return ''
        def verify_inputs(state):
            self.assertEqual(state['environment']['REUSE_CACHE_READY'], '0')
            self.assertEqual(state['environment']['REUSE_MODEL_DIR'], self.parent['environment']['REUSE_MODEL_DIR'])
            return {'synthetic_test': True}
        freezer = SimpleNamespace(verify=lambda path: 'a' * 64, files=lambda root: iter(()))
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'synthetic-cpu-job'}), \
                patch.object(recovery.setup, 'load_state', return_value=self.parent), \
                patch.object(recovery.setup, 'freeze_module', return_value=freezer), \
                patch.object(recovery.setup, 'run', side_effect=runner), \
                patch.object(recovery.setup, 'verify_staged', side_effect=verify_inputs), \
                patch.object(recovery.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as suite:
            recovery.recover(str(self.base), 'recovery-test')
        destination = self.base / 'recovery-test'
        record = json.loads((destination / 'recovery_state.json').read_text())
        self.assertEqual(record['status'], 'ready_for_numerical_smoke')
        self.assertTrue(record['tests']['passed'])
        self.assertIn('REUSE_CACHE_READY=1', (destination / 'cluster.env').read_text())
        self.assertIn(str(destination / 'venv'), (destination / 'cluster.env').read_text())
        self.assertEqual(self.original.read_text(), 'original environment, unchanged\n')
        self.assertEqual(suite.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertFalse(any(command[0] in ('sbatch', 'srun') for command in commands))


if __name__ == '__main__':
    unittest.main()
