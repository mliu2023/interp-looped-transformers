"""CPU preparation contract: no network, scheduler, installation, or weights."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cluster_setup', ROOT / 'scripts/cluster_setup.py')
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class ClusterSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source'
        self.base = self.root / 'work'
        (self.source / 'scripts/slurm').mkdir(parents=True)
        (self.source / 'configs').mkdir()
        shutil.copy2(ROOT / 'scripts/slurm/freeze.py', self.source / 'scripts/slurm/freeze.py')
        shutil.copy2(ROOT / 'scripts/slurm/submit.py', self.source / 'scripts/slurm/submit.py')
        (self.source / 'src/reuseinterp').mkdir(parents=True)
        shutil.copy2(ROOT / 'src/reuseinterp/io.py', self.source / 'src/reuseinterp/io.py')
        config = json.loads((ROOT / 'configs/smoke.yaml').read_text())
        for name in setup.STAGES:
            (self.source / 'configs' / (name + '.yaml')).write_text(json.dumps(config))
        self.args = argparse.Namespace(base_dir=str(self.base), source_dir=str(self.source),
            python=sys.executable, site_env=None, env_file=None)

    def configure(self, **kwargs):
        values = vars(self.args).copy()
        values.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return setup.configure(argparse.Namespace(**values))

    def install_args(self, **kwargs):
        values = {'use_installed_torch': False, 'torch_index_url': 'https://example.edu/whl/cu130',
                  'torch_version': '2.12.1', 'torchvision_version': '0.27.1'}
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_configuration_keeps_source_unchanged_and_ready_false(self):
        before = (self.source / 'configs/smoke.yaml').read_bytes()
        state = self.configure()
        self.assertEqual(state, setup.load_state(str(self.base)))
        self.assertEqual(state['environment']['REUSE_CACHE_READY'], '0')
        self.assertEqual(state['environment']['REUSE_GPU_REQUEST'], 'gpu:nvidia_rtx_a6000:1')
        self.assertEqual(state['environment']['REUSE_CPU_QOS'], 'normal')
        self.assertEqual((self.source / 'configs/smoke.yaml').read_bytes(), before)
        self.assertIn('export REUSE_PROJECT_DIR=', Path(state['env_file']).read_text())
        self.assertEqual(Path(state['env_file']).stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.configure()

    def test_configuration_rejects_nested_runtime_source(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.configure(base_dir=str(self.source / 'work'))
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.configure(base_dir=str(self.source))

    def test_configuration_rejects_tracked_env_file(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.configure(env_file=str(self.source / 'cluster.env'))

    def test_configuration_copies_reviewed_site_script(self):
        site = self.root / 'reviewed-site.sh'
        site.write_text('module load site-python\n')
        state = self.configure(site_env=str(site))
        self.assertEqual(Path(state['site_env']).read_text(), site.read_text())
        self.assertEqual(state['environment']['REUSE_SITE_ENV'], str(self.base / 'snapshot/configs/site-environment.sh'))

    def test_load_rejects_state_path_escape(self):
        state = self.configure()
        state['environment']['REUSE_MODEL_DIR'] = str(self.root / 'outside')
        setup.save_json(self.base / 'setup-state.json', state)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            setup.load_state(str(self.base))

    def test_runner_passes_commands_as_argv_with_module_source(self):
        state = self.configure()
        state['site_env'] = '/some path/reviewed.sh'
        command = ['python', 'argument $(not executed)', "it's literal"]
        result = subprocess.CompletedProcess([], 0, stdout='okay')
        with patch.object(setup.subprocess, 'run', return_value=result) as mocked:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.run(state, command, capture=True, offline=True), 'okay')
        args, kwargs = mocked.call_args
        self.assertEqual(args[0][4:], [state['site_env']] + command)
        self.assertIn('set -e;', args[0][2])
        self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
        self.assertTrue(kwargs['check'])

    def test_platform_check_rejects_old_python_and_macos(self):
        state = self.configure()
        for info in ({'system': 'Linux', 'version': [3, 11, 10]},
                     {'system': 'Darwin', 'version': [3, 12, 1]}):
            with patch.object(setup, 'run', return_value=json.dumps(info)):
                with self.assertRaisesRegex(ValueError, 'Linux Python'):
                    setup.check_platform(state)

    def test_install_requires_explicit_credential_free_index(self):
        state = self.configure()
        with patch.object(setup, 'check_platform', return_value={'system': 'Linux', 'version': [3, 12, 1]}):
            for url in (None, 'http://example.edu/whl', 'https://user:secret@example.edu/whl'):
                with self.assertRaises(ValueError):
                    setup.install(self.install_args(torch_index_url=url), state)
        self.assertFalse(setup.load_state(self.base)['installed'])

    def test_install_pins_binary_cuda_pair_and_records_final_operator_probe(self):
        state = self.configure()
        commands = []
        pair = {'torch': '2.12.1+cu130', 'torchvision': '0.27.1+cu130', 'cuda_runtime': '13.0'}
        def runner(actual_state, command, **kwargs):
            commands.append(list(map(str, command)))
            return json.dumps(pair) if '-c' in command else ''
        with patch.object(setup, 'check_platform', return_value={'system': 'Linux', 'version': [3, 12, 1]}):
            with patch.object(setup, 'official_source', return_value=self.base / 'LoopVL'):
                with patch.object(setup, 'run', side_effect=runner):
                    with contextlib.redirect_stdout(io.StringIO()):
                        setup.install(self.install_args(), state)
        package_command = next(command for command in commands if '--only-binary=:all:' in command)
        self.assertIn('torch==2.12.1', package_command)
        self.assertIn('torchvision==0.27.1', package_command)
        self.assertEqual(commands[-1][1], '-c')
        self.assertEqual(state['torch_pair'], pair)
        self.assertTrue(setup.load_state(self.base)['installed'])
        self.assertEqual(state['environment']['REUSE_CACHE_READY'], '0')

    def test_all_stage_pins_must_match(self):
        setup.pinned_configuration(self.source)
        path = self.source / 'configs/confirmation.yaml'
        config = json.loads(path.read_text())
        config['dataset_hash'] = 'different'
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'same model/source/data'):
            setup.pinned_configuration(self.source)

    def test_checkpoint_checks_bytes(self):
        directory = self.root / 'model'
        directory.mkdir()
        model = directory / 'model.safetensors'
        model.write_bytes(b'fixture, never model weights')
        config = {'checkpoint_hashes': {'model.safetensors': setup.sha(model)}}
        self.assertEqual(setup.verify_checkpoint(directory, config), config['checkpoint_hashes'])
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            setup.verify_checkpoint(directory, config)

    def make_archive(self, members):
        archive = self.root / 'scenes-v1.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, data, kind in members:
                info = tarfile.TarInfo(name)
                info.type = kind
                info.size = len(data) if kind == tarfile.REGTYPE else 0
                if kind == tarfile.SYMTYPE:
                    info.linkname = '/tmp/outside'
                bundle.addfile(info, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        checksum = self.root / 'scenes-v1.tar.gz.sha256'
        checksum.write_text(setup.sha(archive) + '  ' + archive.name + '\n')
        return archive, checksum

    def test_archive_extract_preserves_bytes_and_resumes_existing_data(self):
        archive, checksum = self.make_archive([('scenes-v1/manifest.json', b'{}\n', tarfile.REGTYPE),
                                               ('scenes-v1/images/a.png', b'png', tarfile.REGTYPE)])
        destination = self.root / 'datasets/scenes-v1'
        setup.extract_dataset(archive, checksum, destination)
        self.assertEqual((destination / 'images/a.png').read_bytes(), b'png')
        setup.extract_dataset(archive, checksum, destination)
        checksum.write_text('0' * 64 + '  ' + archive.name + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            setup.extract_dataset(archive, checksum, destination)

    def test_archive_rejects_traversal_links_and_duplicate_paths(self):
        for members in ([('../escape', b'x', tarfile.REGTYPE)],
                        [('scenes-v1/link', b'', tarfile.SYMTYPE)],
                        [('scenes-v1/manifest.json', b'{}', tarfile.REGTYPE)] * 2):
            archive, checksum = self.make_archive(members)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                setup.extract_dataset(archive, checksum, self.root / 'datasets/scenes-v1')
        self.assertFalse((self.root / 'escape').exists())

    def test_failed_stage_resets_readiness_before_any_download(self):
        state = self.configure()
        state.update(installed=True, staged=True)
        state['environment']['REUSE_CACHE_READY'] = '1'
        with patch.object(setup, 'extract_dataset', side_effect=ValueError('fixture failure')):
            with self.assertRaisesRegex(ValueError, 'fixture failure'):
                setup.stage(argparse.Namespace(), state)
        reloaded = setup.load_state(self.base)
        self.assertFalse(reloaded['staged'])
        self.assertEqual(reloaded['environment']['REUSE_CACHE_READY'], '0')

    def test_successful_stage_only_sets_ready_after_offline_checks(self):
        state = self.configure()
        state['installed'] = True
        evidence = {'offline_components': {'weights_loaded': False}}
        with patch.object(setup, 'extract_dataset'), patch.object(setup, 'run') as runner:
            def verified(actual_state):
                self.assertEqual(actual_state['environment']['REUSE_CACHE_READY'], '0')
                return evidence
            with patch.object(setup, 'verify_staged', side_effect=verified):
                with contextlib.redirect_stdout(io.StringIO()):
                    setup.stage(argparse.Namespace(), state)
        command = runner.call_args.args[1]
        self.assertIn(setup.CHECKPOINT_REVISION, command)
        self.assertEqual(json.loads((self.base / 'staging-evidence.json').read_text()), evidence)
        self.assertEqual(setup.load_state(self.base)['environment']['REUSE_CACHE_READY'], '1')

    def test_freeze_transforms_copy_and_keeps_original_commit(self):
        state = self.configure()
        state.update(installed=True, staged=True)
        (self.base / 'staging-evidence.json').write_text('{}')
        before = (self.source / 'configs/smoke.yaml').read_bytes()
        commit = 'a' * 40
        def runner(actual_state, command, **kwargs):
            argv = list(map(str, command))
            if 'freeze.py' in argv[1]:
                return subprocess.run([sys.executable] + argv[1:], check=True, capture_output=True, text=True).stdout
            if 'freeze' in argv:
                return 'pip==25.0\nreuseinterp==0.1.0\n'
            if 'rev-parse' in argv:
                return commit + '\n'
            return ''
        with patch.object(setup, 'verify_staged', return_value={}), patch.object(setup, 'run', side_effect=runner):
            with contextlib.redirect_stdout(io.StringIO()):
                setup.freeze(argparse.Namespace(), state)
        snapshot = self.base / 'snapshot'
        manifest = json.loads((snapshot / 'snapshot_manifest.json').read_text())
        self.assertEqual(manifest['source_commit'], commit)
        self.assertEqual((self.source / 'configs/smoke.yaml').read_bytes(), before)
        config = json.loads((snapshot / 'configs/smoke.yaml').read_text())
        self.assertEqual(config['dataset_dir'], state['environment']['REUSE_DATASET_DIR'])
        self.assertEqual(setup.freeze_module(self.source).verify(snapshot), state['snapshot_sha256'])
        self.assertTrue((snapshot / 'source_provenance.json').is_file())
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            setup.freeze(argparse.Namespace(), state)

    def recovery_fixture(self):
        state = self.configure()
        lock = self.root / 'original-dependencies.lock'
        lock.write_text('fixture-package==1.0\n')
        subprocess.run([sys.executable, str(ROOT / 'scripts/slurm/freeze.py'),
                        '--source', str(self.source), '--destination', state['environment']['REUSE_PROJECT_DIR'],
                        '--dependency-lock', str(lock)], check=True, capture_output=True, text=True)
        state.update(installed=True, staged=True, frozen=True,
                     snapshot_sha256=setup.freeze_module(self.source).verify(state['environment']['REUSE_PROJECT_DIR']))
        state['environment']['REUSE_CACHE_READY'] = '1'
        setup.persist(state)
        (self.base / 'staging-evidence.json').write_text('{"original":true}\n')
        run_root = Path(state['environment']['REUSE_RUN_ROOT'])
        (run_root / 'scheduler-accounting').mkdir(parents=True)
        accounting = run_root / 'scheduler-accounting/original.sacct.txt'
        accounting.write_text('7489067|FAILED|8|gres/gpu=1\n')
        ledger = {'schema_version': 1, 'gpu_hour_cap': 30,
                  'jobs': [{'job_id': '7489067', 'kind': 'gpu', 'stage': 'validate',
                            'num_shards': 1, 'wall_seconds': 3600, 'tasks': {
                                '0': {'state': 'FAILED', 'elapsed_seconds': 8, 'allocated_gpus': 1}}}],
                  'accounting_history': [{'source': str(accounting), 'sha256': setup.sha(accounting)}]}
        ledger_path = run_root / 'scheduler_ledger.json'
        setup.save_json(ledger_path, ledger)
        (run_root / 'preflight').mkdir()
        setup.save_json(run_root / 'preflight/7489067_single.json', {
            'slurm_job_id': '7489067', 'status': 'failed', 'cuda_available': False,
            'torch_version': '2.12.1+cu130', 'cuda_runtime': '13.0'})
        return state, ledger, ledger_path

    def recovery_args(self, **kwargs):
        values = {'name': 'cu126', 'failed_job_id': '7489067',
                  'torch_index_url': 'https://download.pytorch.org/whl/cu126',
                  'torch_version': '2.12.1+cu126', 'torchvision_version': '0.27.1+cu126'}
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_recovery_requires_reconciled_terminal_gpu_jobs(self):
        state, ledger, path = self.recovery_fixture()
        evidence = setup.recovery_accounting(state, '7489067')
        self.assertAlmostEqual(evidence['budget_at_recovery']['consumed_gpu_hours'], 8 / 3600)
        ledger['jobs'][0]['tasks']['0']['state'] = 'RUNNING'
        setup.save_json(path, ledger)
        with self.assertRaisesRegex(ValueError, 'Outstanding or unreconciled'):
            setup.recover(self.recovery_args(), state)
        self.assertFalse((self.base / 'setup-state-cu126.json').exists())

    def test_recovery_rejects_missing_or_changed_accounting_evidence(self):
        state, ledger, path = self.recovery_fixture()
        history = ledger.pop('accounting_history')
        setup.save_json(path, ledger)
        with self.assertRaisesRegex(ValueError, 'Refresh accounting first'):
            setup.recovery_accounting(state, '7489067')
        ledger['accounting_history'] = history
        setup.save_json(path, ledger)
        Path(history[0]['source']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            setup.recovery_accounting(state, '7489067')

    def test_recovery_rejects_exhausted_budget_without_resetting_ledger(self):
        state, ledger, path = self.recovery_fixture()
        ledger['jobs'][0]['tasks']['0']['elapsed_seconds'] = 108000
        accounting = Path(ledger['accounting_history'][0]['source'])
        accounting.write_text('7489067|FAILED|108000|gres/gpu=1\n')
        ledger['accounting_history'][0]['sha256'] = setup.sha(accounting)
        setup.save_json(path, ledger)
        original_bytes = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'no room for recovery'):
            setup.recover(self.recovery_args(), state)
        self.assertEqual(path.read_bytes(), original_bytes)
        self.assertFalse((self.base / 'setup-state-cu126.json').exists())

    def test_recovery_requires_preserved_cuda13_failure_and_exact_cuda126_pins(self):
        state, ledger, path = self.recovery_fixture()
        with self.assertRaisesRegex(ValueError, r'exact \+cu126'):
            setup.recover(self.recovery_args(torch_version='2.12.1'), state)
        preflight = Path(state['environment']['REUSE_RUN_ROOT']) / 'preflight/7489067_single.json'
        data = json.loads(preflight.read_text())
        data['status'] = 'passed'
        setup.save_json(preflight, data)
        with self.assertRaisesRegex(ValueError, 'expected CUDA 13'):
            setup.recover(self.recovery_args(), state)
        self.assertFalse((self.base / 'setup-state-cu126.json').exists())

    def test_recovery_build_preserves_prior_snapshot_assets_and_budget(self):
        state, ledger, ledger_path = self.recovery_fixture()
        original_state = (self.base / 'setup-state.json').read_bytes()
        original_env = (self.base / 'cluster.env').read_bytes()
        original_evidence = (self.base / 'staging-evidence.json').read_bytes()
        original_ledger = ledger_path.read_bytes()
        original_snapshot = state['snapshot_sha256']
        old_venv = Path(state['environment']['REUSE_ENV_DIR'])
        old_venv.mkdir()
        (old_venv / 'preserved.txt').write_text('original CUDA13 environment')
        def installed(args, recovered):
            self.assertEqual(args.torch_version, '2.12.1+cu126')
            self.assertEqual(args.torchvision_version, '0.27.1+cu126')
            self.assertFalse(Path(recovered['environment']['REUSE_ENV_DIR']).exists())
            self.assertEqual(recovered['environment']['REUSE_ENV_DIR'], str(self.base / 'venv-cu126'))
            recovered['installed'] = True
            recovered['torch_pair'] = {'torch': args.torch_version, 'torchvision': args.torchvision_version, 'cuda_runtime': '12.6'}
            setup.persist(recovered)
        def runner(actual_state, command, **kwargs):
            argv = list(map(str, command))
            if 'freeze.py' in argv[1]:
                return subprocess.run([sys.executable] + argv[1:], check=True, capture_output=True, text=True).stdout
            if 'freeze' in argv:
                return 'torch==2.12.1+cu126\ntorchvision==0.27.1+cu126\n'
            if 'rev-parse' in argv:
                return 'a' * 40 + '\n'
            return ''
        with patch.object(setup, 'install', side_effect=installed), patch.object(setup, 'verify_staged', return_value={}):
            with patch.object(setup, 'extract_dataset') as extractor, patch.object(setup, 'run', side_effect=runner):
                with contextlib.redirect_stdout(io.StringIO()):
                    recovered = setup.recover(self.recovery_args(), state)
        extractor.assert_not_called()
        self.assertEqual((self.base / 'setup-state.json').read_bytes(), original_state)
        self.assertEqual((self.base / 'cluster.env').read_bytes(), original_env)
        self.assertEqual((self.base / 'staging-evidence.json').read_bytes(), original_evidence)
        self.assertEqual(ledger_path.read_bytes(), original_ledger)
        self.assertEqual((old_venv / 'preserved.txt').read_text(), 'original CUDA13 environment')
        self.assertEqual(setup.freeze_module(self.source).verify(state['environment']['REUSE_PROJECT_DIR']), original_snapshot)
        self.assertEqual(setup.load_state(self.base, 'cu126'), recovered)
        self.assertEqual(recovered['environment']['REUSE_RUN_ROOT'], state['environment']['REUSE_RUN_ROOT'])
        for name in ('REUSE_MODEL_DIR', 'REUSE_DATASET_DIR', 'REUSE_LOOPVL_DIR', 'REUSE_CACHE_ROOT'):
            self.assertEqual(recovered['environment'][name], state['environment'][name])
        old_config = json.loads((self.source / 'configs/smoke.yaml').read_text())
        for name in setup.STAGES:
            config = json.loads((self.base / 'snapshot-cu126/configs' / (name + '.yaml')).read_text())
            self.assertEqual(config['run_id'], old_config['run_id'] + '-cu126')
            self.assertNotEqual(config['study_fingerprint'], old_config['study_fingerprint'])
            self.assertEqual(config['checkpoint_hashes'], old_config['checkpoint_hashes'])
            self.assertEqual(config['dataset_hash'], old_config['dataset_hash'])
            self.assertEqual(config['runtime_variant']['cuda_runtime'], '12.6')
        new_run = Path(recovered['environment']['REUSE_RUN_ROOT']) / (old_config['run_id'] + '-cu126')
        self.assertFalse((new_run / 'gates.json').exists())
        with self.assertRaisesRegex(ValueError, 'profile already exists'):
            setup.recover(self.recovery_args(), state)

    def test_recovery_install_rejects_final_runtime_or_build_mismatch(self):
        state = self.configure()
        args = self.recovery_args()
        state['runtime_variant'] = {'profile': args.name, 'torch': args.torch_version,
            'torchvision': args.torchvision_version, 'cuda_runtime': '12.6', 'torch_index_url': args.torch_index_url}
        state['profile'] = 'cu126'
        state['env_file'] = str(self.base / 'cluster-cu126.env')
        state['environment']['REUSE_ENV_DIR'] = str(self.base / 'venv-cu126')
        install_args = self.install_args(torch_version=args.torch_version,
            torchvision_version=args.torchvision_version, torch_index_url=args.torch_index_url)
        wrong_pair = {'torch': '2.12.1+cu130', 'torchvision': '0.27.1+cu130', 'cuda_runtime': '13.0'}
        def runner(actual_state, command, **kwargs):
            return json.dumps(wrong_pair) if '-c' in command else ''
        with patch.object(setup, 'check_platform', return_value={'system': 'Linux', 'version': [3, 12, 1]}):
            with patch.object(setup, 'official_source', return_value=self.base / 'LoopVL'):
                with patch.object(setup, 'run', side_effect=runner):
                    with self.assertRaisesRegex(ValueError, 'differs from declared CUDA build'):
                        setup.install(install_args, state)
        self.assertFalse(setup.load_state(self.base, 'cu126')['installed'])
        self.assertEqual(state['environment']['REUSE_CACHE_READY'], '0')

    def test_freeze_refuses_credentials_in_dependency_lock(self):
        state = self.configure()
        state.update(installed=True, staged=True)
        with patch.object(setup, 'verify_staged', return_value={}):
            with patch.object(setup, 'run', return_value='lib @ https://user:token@example.edu/lib.whl\n'):
                with self.assertRaisesRegex(ValueError, 'credential-bearing'):
                    setup.freeze(argparse.Namespace(), state)
        self.assertFalse((self.base / 'snapshot').exists())


if __name__ == '__main__':
    unittest.main()
