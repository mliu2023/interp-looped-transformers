#!/usr/bin/env python3
"""Build a fresh CUDA 12.8 environment using already staged study inputs.

Run through recover_cuda12.sbatch. Never changes the original setup, snapshot,
weights, dataset, or GPU-hour ledger, and never submits a GPU job itself.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import cluster_setup as setup

TORCH = '2.8.0+cu128'
TORCHVISION = '0.23.0+cu128'
INDEX = 'https://download.pytorch.org/whl/cu128'


def recovery_paths(base, name):
    if not re.fullmatch(r'recovery-[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('--name must start with recovery- and contain only simple path characters')
    parent = setup.load_state(base)
    if not parent.get('frozen') or parent['environment']['REUSE_CACHE_READY'] != '1':
        raise ValueError('Recovery requires the existing fully staged and frozen setup')
    destination = Path(parent['base_dir']) / name
    if destination.exists():
        raise ValueError('Recovery destination already exists; inspect it and choose a fresh --name')
    return parent, destination


def recover(base, name):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Environment installation and tests require a scheduled CPU allocation')
    parent, destination = recovery_paths(base, name)
    source = Path(__file__).resolve().parents[1]
    # Verify the original frozen evidence before creating anything.
    original_hash = setup.freeze_module(source).verify(parent['environment']['REUSE_PROJECT_DIR'])
    if original_hash != parent['snapshot_sha256']:
        raise ValueError('Original snapshot differs from the saved preparation state')
    destination.mkdir()
    environment = dict(parent['environment'])
    environment.update(REUSE_SOURCE_DIR=str(source), REUSE_ENV_DIR=str(destination / 'venv'),
                       REUSE_PROJECT_DIR=str(destination / 'snapshot'), REUSE_CACHE_READY='0')
    state = dict(parent, source_dir=str(source), environment=environment,
                 env_file=str(destination / 'cluster.env'))
    record = {'schema_version': 1, 'status': 'running', 'slurm_job_id': os.environ['SLURM_JOB_ID'],
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'parent_snapshot_sha256': original_hash, 'parent_base': parent['base_dir'],
              'torch': TORCH, 'torchvision': TORCHVISION, 'index': INDEX,
              'gpu_experiments': 'pending fresh numerical parity; none launched by recovery'}
    status_file = destination / 'recovery_state.json'
    setup.save_json(status_file, record)
    try:
        python = str(destination / 'venv/bin/python')
        constraints = destination / 'torch-pair.constraints.txt'
        constraints.write_text('torch==' + TORCH + '\ntorchvision==' + TORCHVISION + '\n')
        setup.run(state, [parent['python'], '-m', 'venv', destination / 'venv'])
        setup.run(state, [python, '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', INDEX,
                          'torch==' + TORCH, 'torchvision==' + TORCHVISION])
        setup.run(state, [python, '-m', 'pip', 'install', '--constraint', constraints,
                          '-r', Path(environment['REUSE_LOOPVL_DIR']) / 'runtime/requirements.txt'])
        setup.run(state, [python, '-m', 'pip', 'install', '--constraint', constraints, str(source) + '[plots]'])
        setup.run(state, [python, '-m', 'pip', 'check'])
        pair = json.loads(setup.run(state, [python, '-c',
            'import json,torch,torchvision; from torchvision.ops import nms; '
            'nms(torch.empty((0,4)),torch.empty((0,)),0.5); '
            'print(json.dumps({"torch":torch.__version__,"torchvision":torchvision.__version__,"cuda_runtime":torch.version.cuda}))'], capture=True))
        if pair != {'torch': TORCH, 'torchvision': TORCHVISION, 'cuda_runtime': '12.8'}:
            raise ValueError('Installed CUDA pair differs from the recovery pins: ' + str(pair))
        record['torch_pair'] = pair
        # The CPU allocation runs the full suite. No model or CUDA initialization.
        test_environment = dict(os.environ, **environment, PYTHONPATH=str(source / 'src'))
        test_environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                CUDA_VISIBLE_DEVICES='', TMPDIR=os.environ.get('TMPDIR', '/tmp'))
        with (destination / 'tests.log').open('w') as stream:
            subprocess.run([python, '-m', 'unittest', 'discover', '-s', str(source / 'tests'), '-v'],
                           env=test_environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record['tests'] = {'passed': True, 'log': str(destination / 'tests.log')}
        evidence = setup.verify_staged(state)
        setup.save_json(destination / 'staging-evidence.json', evidence)
        lock = destination / 'cluster-dependencies.lock'
        lock_text = setup.run(state, [python, '-m', 'pip', 'freeze', '--all'], capture=True)
        if re.search(r'https?://[^\s/]*@', lock_text):
            raise ValueError('Refusing credential-bearing dependency lock')
        lock.write_text(lock_text)
        preparation = destination / 'preparation'
        preparation.mkdir()
        freezer = setup.freeze_module(source)
        for path in freezer.files(source):
            if 'local' in path.name and path.suffix in {'.env', '.sh'}:
                continue
            target = preparation / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for path in (preparation / 'configs').glob('*.yaml'):
            config = json.loads(path.read_text())
            if 'dataset_dir' in config:
                config['dataset_dir'] = environment['REUSE_DATASET_DIR']
                setup.save_json(path, config)
        if state.get('site_env'):
            shutil.copy2(state['site_env'], preparation / 'configs/site-environment.sh')
            environment['REUSE_SITE_ENV'] = str(destination / 'snapshot/configs/site-environment.sh')
        record['source_commit'] = setup.run(state, ['git', '-C', source, 'rev-parse', 'HEAD'], capture=True).strip()
        record['source_diff_sha256'] = __import__('hashlib').sha256(
            setup.run(state, ['git', '-C', source, 'diff', '--binary', 'HEAD'], capture=True).encode()).hexdigest()
        setup.save_json(preparation / 'recovery_provenance.json', record)
        setup.run(state, [python, source / 'scripts/slurm/freeze.py', '--source', preparation,
                          '--destination', destination / 'snapshot', '--dependency-lock', lock])
        record.update(status='ready_for_numerical_smoke',
                      snapshot_sha256=freezer.verify(destination / 'snapshot'))
        environment['REUSE_CACHE_READY'] = '1'
        setup.write_environment(state)
        print(json.dumps({'environment': state['env_file'], 'status': record['status'],
                          'next': 'Submit singleton validate using this environment; do not reuse old parity'}, indent=2))
    except Exception as exc:
        record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        setup.save_json(status_file, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--name', default='recovery-cu128-20261016')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        parent, destination = recovery_paths(args.base_dir, args.name)
        print(json.dumps({'destination': str(destination), 'torch': TORCH, 'torchvision': TORCHVISION,
                          'index': INDEX, 'inputs': {key: parent['environment'][key] for key in
                          ('REUSE_LOOPVL_DIR', 'REUSE_MODEL_DIR', 'REUSE_DATASET_DIR')},
                          'scheduled_cpu_required': True, 'gpu_hours_reserved': 0}, indent=2))
    else:
        recover(args.base_dir, args.name)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('recover_cuda12: ' + str(exc), file=sys.stderr)
        sys.exit(2)
