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

    def test_install_constrains_followup_resolution_to_exact_cuda_build(self):
        state = self.configure()
        pair = {'torch': '2.8.0+cu128', 'torchvision': '0.23.0+cu128', 'cuda_runtime': '12.8'}
        commands = []
        def runner(actual_state, command, **kwargs):
            commands.append(list(map(str, command)))
            return json.dumps(pair) if '-c' in command else ''
        with patch.object(setup, 'check_platform', return_value={'system': 'Linux', 'version': [3, 12, 1]}), \
                patch.object(setup, 'official_source', return_value=self.base / 'LoopVL'), \
                patch.object(setup, 'run', side_effect=runner), \
                contextlib.redirect_stdout(io.StringIO()):
            setup.install(self.install_args(torch_version='2.8.0+cu128', torchvision_version='0.23.0+cu128'), state)
        constraint = self.base / 'torch-pair.constraints.txt'
        self.assertEqual(constraint.read_text(), 'torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\n')
        followups = [command for command in commands if 'install' in command and '--only-binary=:all:' not in command]
        self.assertEqual(len(followups), 2)
        for command in followups:
            self.assertEqual(command[command.index('--constraint') + 1], str(constraint))

    def test_install_rejects_silent_cuda_build_replacement(self):
        state = self.configure()
        wrong = {'torch': '2.8.0+cu130', 'torchvision': '0.23.0+cu130', 'cuda_runtime': '13.0'}
        def runner(actual_state, command, **kwargs):
            return json.dumps(wrong) if '-c' in command else ''
        with patch.object(setup, 'check_platform', return_value={'system': 'Linux', 'version': [3, 12, 1]}), \
                patch.object(setup, 'official_source', return_value=self.base / 'LoopVL'), \
                patch.object(setup, 'run', side_effect=runner), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'requested CUDA pair'):
                setup.install(self.install_args(torch_version='2.8.0+cu128', torchvision_version='0.23.0+cu128'), state)
        self.assertFalse(setup.load_state(self.base)['installed'])

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
