"""CPU infrastructure checks. No real scheduler, SSH connection, or model is used."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('slurm_submit', ROOT / 'scripts/slurm/submit.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


class SlurmTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.project = self.root / 'snapshot'
        self.results = self.root / 'results'
        (self.source / 'scripts/slurm').mkdir(parents=True)
        (self.source / 'configs').mkdir()
        (self.source / 'src/reuseinterp/data').mkdir(parents=True)
        (self.source / 'src/reuseinterp/data/__init__.py').write_text('# preserve source data package\n')
        for path in (ROOT / 'scripts/slurm').glob('*'):
            if path.is_file():
                shutil.copy2(path, self.source / 'scripts/slurm' / path.name)
        self.config = {'run_id': 'test-run', 'study_fingerprint': 'study-v1',
                       'resource_estimates': {'capture': {'forwards': 100, 'storage_bytes': 4096}}}
        (self.source / 'configs/test.yaml').write_text(json.dumps(self.config))
        self.lock = self.root / 'deps.lock'
        self.lock.write_text('example-package==1.0\n')
        freeze = subprocess.run([sys.executable, str(ROOT / 'scripts/slurm/freeze.py'),
                                 '--source', str(self.source), '--destination', str(self.project),
                                 '--dependency-lock', str(self.lock)], capture_output=True, text=True)
        self.assertEqual(freeze.returncode, 0, freeze.stderr)
        self.fake = self.root / 'fake-sbatch'
        self.fake.write_text('#!/usr/bin/env python3\nprint("12345;cluster")\n')
        self.fake.chmod(0o755)
        self.environment = {'REUSE_PROJECT_DIR': str(self.project), 'REUSE_ENV_DIR': str(self.root / 'env'),
                            'REUSE_RUN_ROOT': str(self.results), 'REUSE_CACHE_ROOT': str(self.root / 'cache'),
                            'REUSE_LOOPVL_DIR': str(self.root / 'LoopVL'), 'REUSE_MODEL_DIR': str(self.root / 'weights'),
                            'REUSE_PARTITION': 'verified-gpu', 'REUSE_CPU_PARTITION': 'verified-cpu',
                            'REUSE_ACCOUNT': '', 'REUSE_QOS': '',
                            'REUSE_CPU_ACCOUNT': '', 'REUSE_CPU_QOS': '', 'REUSE_SITE_ENV': '',
                            'REUSE_GPU_FLAG': '--gres', 'REUSE_GPU_REQUEST': 'gpu:1',
                            'REUSE_CPUS': '2', 'REUSE_MEM': '8G', 'REUSE_TIME': '00:30:00',
                            'REUSE_CACHE_READY': '1', 'REUSE_GPU_HOUR_CAP': '30',
                            'REUSE_DTYPE': 'auto', 'REUSE_MIN_VRAM_GIB': '0'}
        self.env_patch = patch.dict(os.environ, self.environment)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.gates = self.results / 'test-run/gates.json'
        self.gates.parent.mkdir(parents=True)
        value = {'study_fingerprint': 'study-v1'}
        for name in ['parity', 'pilot', 'capability', 'discovery', 'protocol_lock']:
            value[name] = {'passed': True, 'study_fingerprint': 'study-v1',
                           'measured_examples': 32, 'sha256': 'a' * 64, 'backend_signature': 'b' * 64}
        self.gates.write_text(json.dumps(value))

    def args(self, stage='validate', **overrides):
        values = dict(stage=stage, config='configs/test.yaml', num_shards=1, concurrency=1,
                      indices=None, resume=False, pilot=False, dry_run=False, gates=None,
                      dependency=None, ledger=None, sbatch=str(self.fake), sacct='sacct',
                      refresh=False, sacct_file=None)
        values.update(overrides)
        return type('Args', (), values)()

    def run_submit(self, args):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            result = submit.submit(args)
        return result, stream.getvalue()

    def test_snapshot_preserves_source_data_and_detects_mutation(self):
        self.assertTrue((self.project / 'src/reuseinterp/data/__init__.py').exists())
        (self.project / 'configs/test.yaml').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            submit.build_submission(self.args())

    def test_fake_sbatch_records_real_returned_id_and_budget(self):
        self.run_submit(self.args())
        ledger = json.loads((self.results / 'scheduler_ledger.json').read_text())
        self.assertEqual(ledger['jobs'][0]['job_id'], '12345')
        self.assertEqual(submit.budget(ledger)['outstanding_max_gpu_hours'], 0.5)
        self.assertTrue((self.results / 'logs').is_dir())

    def test_dry_run_does_not_submit_or_write_ledger(self):
        _, output = self.run_submit(self.args(dry_run=True))
        self.assertIn('"dry_run": true', output)
        self.assertFalse((self.results / 'scheduler_ledger.json').exists())

    def test_optional_flags_and_exports_are_exact_arguments(self):
        with patch.dict(os.environ, {'REUSE_ACCOUNT': 'account-name', 'REUSE_QOS': 'qos-name'}):
            record, errors = submit.build_submission(self.args())
        self.assertFalse(errors)
        self.assertIn('--account=account-name', record['command'])
        self.assertIn('--qos=qos-name', record['command'])
        exported = next(item for item in record['command'] if item.startswith('--export='))
        self.assertTrue(exported.startswith('--export=REUSE_PROJECT_DIR='))
        self.assertNotIn('NONE,', exported)
        self.assertNotIn('TOKEN', exported)

    def test_inherited_scheduler_defaults_cannot_expand_allocation(self):
        evidence = self.root / 'sbatch-env.json'
        self.fake.write_text('#!/usr/bin/env python3\nimport os, json\n'
                             + f'open({str(evidence)!r}, "w").write(json.dumps(dict(os.environ)))\n'
                             + 'print("12345")\n')
        with patch.dict(os.environ, {'SBATCH_GPUS': '8', 'SBATCH_ARRAY_INX': '0-999'}):
            self.run_submit(self.args())
        child = json.loads(evidence.read_text())
        self.assertNotIn('SBATCH_GPUS', child)
        self.assertNotIn('SBATCH_ARRAY_INX', child)

    def test_cpu_and_gpu_use_separate_account_and_qos(self):
        settings = {'REUSE_ACCOUNT': 'gpu-account', 'REUSE_QOS': 'gpu-debug',
                    'REUSE_CPU_ACCOUNT': 'cpu-account', 'REUSE_CPU_QOS': 'normal'}
        with patch.dict(os.environ, settings):
            gpu, gpu_errors = submit.build_submission(self.args('validate'))
            cpu, cpu_errors = submit.build_submission(self.args('fit'))
        self.assertFalse(gpu_errors)
        self.assertFalse(cpu_errors)
        self.assertIn('--account=gpu-account', gpu['command'])
        self.assertIn('--qos=gpu-debug', gpu['command'])
        self.assertNotIn('--account=cpu-account', gpu['command'])
        self.assertNotIn('--qos=normal', gpu['command'])
        self.assertIn('--account=cpu-account', cpu['command'])
        self.assertIn('--qos=normal', cpu['command'])
        self.assertNotIn('--account=gpu-account', cpu['command'])
        self.assertNotIn('--qos=gpu-debug', cpu['command'])

    def test_cpu_account_and_qos_fall_back_only_when_unset(self):
        with patch.dict(os.environ, {'REUSE_ACCOUNT': 'legacy-account', 'REUSE_QOS': 'legacy-qos'}):
            os.environ.pop('REUSE_CPU_ACCOUNT', None)
            os.environ.pop('REUSE_CPU_QOS', None)
            record, errors = submit.build_submission(self.args('fit'))
        self.assertFalse(errors)
        self.assertIn('--account=legacy-account', record['command'])
        self.assertIn('--qos=legacy-qos', record['command'])

    def test_explicit_empty_cpu_settings_omit_account_and_qos(self):
        with patch.dict(os.environ, {'REUSE_ACCOUNT': 'gpu-account', 'REUSE_QOS': 'gpu-debug',
                                     'REUSE_CPU_ACCOUNT': '', 'REUSE_CPU_QOS': ''}):
            record, errors = submit.build_submission(self.args('fit'))
        self.assertFalse(errors)
        self.assertFalse(any(item.startswith(('--account=', '--qos='))
                             for item in record['command']))

    def test_unused_flags_omitted_and_no_shell(self):
        record, _ = submit.build_submission(self.args())
        self.assertFalse(any(item.startswith(('--account=', '--qos=')) for item in record['command']))
        self.assertIsInstance(record['command'], list)

    def test_gpu_setting_cannot_request_multiple_devices(self):
        with patch.dict(os.environ, {'REUSE_GPU_REQUEST': 'gpu:2'}):
            with self.assertRaisesRegex(ValueError, 'Exactly one GPU'):
                submit.build_submission(self.args())

    def test_array_shards_and_retry_indices_fixed(self):
        record, errors = submit.build_submission(self.args('capture', num_shards=4, concurrency=2,
                                                          indices='1,3', resume=True))
        self.assertFalse(errors)
        self.assertIn('--array=1,3%2', record['command'])
        self.assertIn('--slurm-array', record['command'])
        self.assertIn('--resume', record['command'])
        self.assertEqual(record['selected_indices'], [1, 3])
        self.assertEqual(record['estimates']['max_gpu_hours'], 1)
        self.assertEqual(record['estimates']['forwards'], 100)

    def test_smoke_and_pilot_cannot_be_arrays(self):
        with self.assertRaisesRegex(ValueError, 'Only baseline'):
            submit.build_submission(self.args(num_shards=2))
        with self.assertRaisesRegex(ValueError, 'single allocations'):
            submit.build_submission(self.args('baseline', pilot=True, num_shards=2))

    def test_pilot_requires_parity_but_not_existing_pilot(self):
        value = json.loads(self.gates.read_text())
        value.pop('pilot')
        self.gates.write_text(json.dumps(value))
        record, errors = submit.build_submission(self.args('baseline', pilot=True))
        self.assertFalse(errors)
        self.assertIn('--pilot', record['command'])
        self.assertTrue(submit.build_submission(self.args('capture'))[1])

    def test_cpu_merge_passes_original_count_without_scheduler_array(self):
        record, errors = submit.build_submission(self.args('merge', num_shards=4))
        self.assertFalse(errors)
        self.assertFalse(record['is_array'])
        self.assertFalse(any(item.startswith('--array=') for item in record['command']))
        self.assertEqual(record['tasks'], {'0': {'state': 'RESERVED'}})
        self.assertEqual(record['command'][-2:], ['--num-shards', '4'])

    def test_discovery_and_confirmation_cpu_stages_have_separate_retry_keys(self):
        first, _ = submit.build_submission(self.args('merge', num_shards=4))
        second = dict(first, stage_key='merge:confirmation', config_hash='a-different-stage-config')
        self.assertFalse(submit.has_duplicate({'jobs': [first]}, second))
        same_stage_changed = dict(first, config_hash='changed-within-stage')
        with self.assertRaisesRegex(ValueError, 'Stage retries must preserve'):
            submit.has_duplicate({'jobs': [first]}, same_stage_changed)

    def test_cpu_report_receives_run_dir_not_config(self):
        record, errors = submit.build_submission(self.args('report'))
        self.assertFalse(errors)
        command = record['command']
        self.assertNotIn('--config', command)
        self.assertEqual(command[-2:], ['--run-dir', str((self.results / 'test-run').resolve())])

    def test_stale_fingerprint_blocks_confirmation(self):
        value = json.loads(self.gates.read_text())
        value['discovery']['study_fingerprint'] = 'old-study'
        self.gates.write_text(json.dumps(value))
        self.assertIn('Gate discovery missing/failed/stale', submit.build_submission(self.args('confirm'))[1])

    def test_backend_signature_exported_and_missing_signature_blocks(self):
        record, errors = submit.build_submission(self.args('capture'))
        self.assertFalse(errors)
        exported = next(item for item in record['command'] if item.startswith('--export='))
        self.assertIn('REUSE_EXPECTED_BACKEND=' + 'b' * 64, exported)
        value = json.loads(self.gates.read_text())
        del value['parity']['backend_signature']
        self.gates.write_text(json.dumps(value))
        self.assertTrue(submit.build_submission(self.args('capture'))[1])

    def test_cap_includes_outstanding_retries_and_all_array_tasks(self):
        with patch.dict(os.environ, {'REUSE_GPU_HOUR_CAP': '1'}):
            with self.assertRaisesRegex(ValueError, 'cap exceeded'):
                self.run_submit(self.args('capture', num_shards=3))

    def test_concurrent_duplicate_refused(self):
        self.run_submit(self.args())
        with self.assertRaisesRegex(ValueError, 'Concurrent duplicate'):
            self.run_submit(self.args(resume=True))

    def test_uncertain_submission_retains_worst_case_reservation(self):
        self.fake.write_text('#!/usr/bin/env python3\nimport sys\nsys.exit(1)\n')
        with self.assertRaisesRegex(ValueError, 'reservation retained'):
            self.run_submit(self.args())
        ledger = json.loads((self.results / 'scheduler_ledger.json').read_text())
        self.assertEqual(ledger['jobs'][0]['submission_state'], 'UNKNOWN_RECONCILE_REQUIRED')
        self.assertEqual(submit.budget(ledger)['outstanding_max_gpu_hours'], 0.5)

    def test_changed_manifest_rejected_against_submission_hash(self):
        original = submit.digest(self.project / 'snapshot_manifest.json')
        manifest = json.loads((self.project / 'snapshot_manifest.json').read_text())
        manifest['source_commit'] = 'changed'
        (self.project / 'snapshot_manifest.json').write_text(json.dumps(manifest))
        result = subprocess.run([sys.executable, str(self.project / 'scripts/slurm/freeze.py'),
                                 '--verify', str(self.project), '--expected-sha256', original],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('manifest changed since submission', result.stderr)

    def test_sacct_no_step_or_array_parent_double_count(self):
        raw = ('12345|COMPLETED|900|cpu=2,gres/gpu=1|\n'
               '12345.batch|COMPLETED|900|cpu=2,gres/gpu=1|\n'
               '12345.extern|COMPLETED|900|gres/gpu=1|\n'
               '12345.0|COMPLETED|900|gres/gpu=1|\n'
               '12346_[0-3]|PENDING|0||\n'
               '12346_0|COMPLETED|600|gres/gpu=1,gres/gpu:a100=1|\n')
        rows = submit.parse_sacct(raw)
        self.assertEqual(set(rows), {'12345', '12346_0'})
        self.assertEqual(rows['12346_0']['allocated_gpus'], 1)
        self.run_submit(self.args())
        accounting = self.root / 'sacct.txt'
        accounting.write_text(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            submit.account(self.args('account', sacct_file=str(accounting)))
        ledger = json.loads((self.results / 'scheduler_ledger.json').read_text())
        self.assertEqual(submit.budget(ledger)['consumed_gpu_hours'], 0.25)
        self.assertEqual(submit.budget(ledger)['outstanding_max_gpu_hours'], 0)

    def test_cpu_wrapper_has_no_gpu_preflight_or_request(self):
        script = (ROOT / 'scripts/slurm/run_cpu.sbatch').read_text()
        self.assertNotIn('cluster_preflight', script)
        record, errors = submit.build_submission(self.args('fit'))
        self.assertFalse(errors)
        self.assertFalse(any(item.startswith(('--gres=', '--gpus=')) for item in record['command']))
        self.assertEqual(record['estimates']['max_gpu_hours'], 0)

    def test_unscheduled_preflight_fails_before_importing_torch(self):
        env = dict(os.environ)
        env.pop('SLURM_JOB_ID', None)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/cluster_preflight.py')],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        value = json.loads(result.stdout)
        self.assertEqual(value['status'], 'failed')
        self.assertIn('Slurm allocation', value['error'])
        self.assertNotIn('torch_version', value)

    def test_shell_syntax(self):
        for name in ['environment.sh', 'run_cpu.sbatch', 'run_gpu.sbatch']:
            result = subprocess.run(['bash', '-n', str(ROOT / 'scripts/slurm' / name)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
