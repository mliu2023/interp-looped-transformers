#!/usr/bin/env python3
"""Submit immutable single-GPU shards with gates and conservative GPU-hour reservations."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

GPU_STAGES = {'validate', 'baseline', 'capture', 'discover-causal', 'confirm', 'poles-pilot', 'image-pilot'}
CPU_STAGES = {'audit', 'data', 'fit', 'lock', 'merge', 'report', 'verify-results', 'fixtures'}
ARRAY_STAGES = {'baseline', 'capture', 'confirm'}
ENV_NAMES = ('REUSE_PROJECT_DIR', 'REUSE_ENV_DIR', 'REUSE_RUN_ROOT', 'REUSE_CACHE_ROOT',
             'REUSE_LOOPVL_DIR', 'REUSE_MODEL_DIR',
             'REUSE_SITE_ENV', 'REUSE_CACHE_READY', 'REUSE_DTYPE', 'REUSE_MIN_VRAM_GIB')
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'PREEMPTED',
            'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read JSON configuration/artifact {path}: {exc}') from exc


def wall_seconds(value):
    # sbatch accepts [days-]hours:minutes:seconds; require this unambiguous form.
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', value)
    if not match:
        raise ValueError('REUSE_TIME must be [days-]HH:MM:SS')
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError('Invalid walltime minutes/seconds')
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError('Walltime must be positive')
    return total


def indices(value, count):
    if value is None:
        return list(range(count))
    selected = set()
    for component in value.split(','):
        if re.fullmatch(r'\d+', component):
            selected.add(int(component))
        elif re.fullmatch(r'\d+-\d+', component):
            start, stop = map(int, component.split('-'))
            if start > stop:
                raise ValueError('Reversed array range')
            selected.update(range(start, stop + 1))
        else:
            raise ValueError('Array indices must be comma-separated integers or ranges')
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError('Array indices must be within the original --num-shards')
    return sorted(selected)


@contextlib.contextmanager
def ledger_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def empty_ledger(cap):
    return {'schema_version': 1, 'gpu_hour_cap': cap, 'jobs': [], 'accounting_history': []}


def budget(ledger):
    consumed = 0.0
    outstanding = 0.0
    for job in ledger['jobs']:
        if job['kind'] != 'gpu':
            continue
        for task in job['tasks'].values():
            if task.get('state') in TERMINAL:
                consumed += task['elapsed_seconds'] * task.get('allocated_gpus', 1) / 3600
            else:
                outstanding += job['wall_seconds'] / 3600
    return {'consumed_gpu_hours': consumed, 'outstanding_max_gpu_hours': outstanding,
            'remaining_gpu_hours': ledger['gpu_hour_cap'] - consumed - outstanding}


def gate_errors(stage, config, path, pilot=False):
    if stage == 'validate' or stage not in GPU_STAGES:
        return []
    if stage == 'image-pilot':
        # This separate image protocol checks native-grid capture parity inside
        # its own allocation. Authenticate the parent backend evidence without
        # copying its population gates or reusing its synthetic study identity.
        specification = config.get('image_pilot', {})
        try:
            if path.resolve() != Path(specification['parent_gates']).resolve():
                return ['Image pilot must use its declared parent parity gates']
            gates = read_json(path)
            parity = gates.get('parity', {})
            parent = specification['parent_study_fingerprint']
            if not parent or gates.get('study_fingerprint') != parent or parity.get('study_fingerprint') != parent:
                return ['Parent parity study identity missing or mismatched']
            if not config.get('study_fingerprint') or config['study_fingerprint'] == parent:
                return ['Real-image protocol requires its own study identity']
            report = Path(specification['parent_parity_report'])
            if parity.get('passed') is not True or read_json(report).get('passed') is not True or digest(report) != parity.get('report_sha256'):
                return ['Parent parity report missing, failed, or changed']
            return []
        except (KeyError, ValueError, OSError) as exc:
            return [f'Invalid image-pilot parent parity evidence: {exc}']
    if not path.is_file():
        return [f'No completed prerequisite gates: {path}']
    gates = read_json(path)
    fingerprint = config.get('study_fingerprint')
    if not fingerprint or gates.get('study_fingerprint') != fingerprint:
        return ['study_fingerprint missing or mismatched between configuration and gates']
    required = ['parity']
    # The bounded signed-activity diagnostic is a separate exploratory protocol,
    # not population capture or a task/causal claim. It performs new-site parity
    # inside its allocation and never writes the original study's gates.
    if not pilot and stage != 'poles-pilot':
        required.append('pilot')
    if stage in {'capture', 'discover-causal', 'confirm'}:
        required.append('capability')
    if stage == 'confirm':
        required += ['discovery', 'protocol_lock']
    errors = []
    for name in required:
        item = gates.get(name, {})
        if item.get('passed') is not True or item.get('study_fingerprint') != fingerprint:
            errors.append(f'Gate {name} missing/failed/stale')
        if name == 'pilot' and item.get('measured_examples', 0) < 32:
            errors.append('Pilot must measure at least 32 examples')
        if name == 'protocol_lock' and not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
            errors.append('Protocol lock must provide its SHA256')
    return errors


def verified_snapshot(project):
    spec = importlib.util.spec_from_file_location('reuse_snapshot_freeze', project / 'scripts/slurm/freeze.py')
    if spec is None or spec.loader is None:
        raise ValueError('Missing immutable-snapshot verifier')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify(project)


def runtime_environment(kind='gpu'):
    values = {}
    for name in ENV_NAMES:
        value = os.getenv(name, '')
        if ',' in value or '\n' in value or '\r' in value:
            raise ValueError(f'{name} contains a delimiter unsupported by Slurm --export')
        if (name in ENV_NAMES[:4] or (kind == 'gpu' and name in ENV_NAMES[4:6])) and (not value.startswith('/') or 'REPLACE' in value):
            raise ValueError(f'{name} must be a configured absolute path')
        values[name] = value
    return values


def build_submission(args):
    kind = 'gpu' if args.stage in GPU_STAGES else 'cpu'
    environment = runtime_environment(kind)
    project = Path(environment['REUSE_PROJECT_DIR']).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project / config_path
    config_path = config_path.resolve()
    if project not in config_path.parents:
        raise ValueError('Configuration must live inside the immutable project snapshot')
    snapshot_hash = verified_snapshot(project)
    environment['REUSE_EXPECTED_SNAPSHOT_SHA256'] = snapshot_hash
    if environment['REUSE_SITE_ENV']:
        site_script = Path(environment['REUSE_SITE_ENV']).resolve()
        if project not in site_script.parents or not site_script.is_file():
            raise ValueError('Site environment script must live inside the immutable snapshot')
    config = read_json(config_path)
    configured_cap = float(config.get('gpu_hour_cap', 30))
    if not math.isfinite(configured_cap) or not 0 < configured_cap <= 30:
        raise ValueError('Configuration GPU-hour cap must be positive and at most 30')
    run_id = config.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('config run_id must be a simple directory name')
    if args.pilot and args.stage != 'baseline':
        raise ValueError('--pilot is supported only for single baseline timing/capability runs')
    if args.num_shards < 1:
        raise ValueError('--num-shards must be positive')
    if args.num_shards > 1 and args.stage not in ARRAY_STAGES | {'merge'}:
        raise ValueError('Only baseline/capture/confirm support GPU arrays; merge accepts their original shard count')
    if (args.stage == 'validate' or args.pilot) and args.num_shards != 1:
        raise ValueError('Smoke and initial pilot must be single allocations')
    is_array = args.num_shards > 1 and args.stage in ARRAY_STAGES
    if args.indices is not None and not is_array:
        raise ValueError('--indices requires an array stage and the original --num-shards > 1')
    selected = indices(args.indices, args.num_shards) if is_array else [0]
    if args.concurrency < 1:
        raise ValueError('--concurrency must be positive')
    if args.gates:
        gates_path = Path(args.gates)
    elif args.stage == 'image-pilot':
        gates_path = Path(config.get('image_pilot', {}).get('parent_gates', 'missing-parent-gates.json'))
    else:
        gates_path = Path(environment['REUSE_RUN_ROOT']) / run_id / 'gates.json'
    errors = gate_errors(args.stage, config, gates_path, pilot=args.pilot)
    # A GPU/backend change must trigger a fresh numerical validate run.
    environment['REUSE_EXPECTED_BACKEND'] = ''
    if kind == 'gpu' and args.stage != 'validate' and gates_path.is_file():
        parity = read_json(gates_path).get('parity', {})
        signature = parity.get('backend_signature', '')
        if not re.fullmatch(r'[0-9a-f]{64}', signature):
            errors.append('Parity gate must record the scheduled preflight backend_signature')
        else:
            environment['REUSE_EXPECTED_BACKEND'] = signature
    if kind == 'gpu' and environment['REUSE_CACHE_READY'] != '1':
        errors.append('REUSE_CACHE_READY must be 1 after complete offline-cache verification')
    partition = os.getenv('REUSE_PARTITION' if kind == 'gpu' else 'REUSE_CPU_PARTITION', '')
    if not partition or 'REPLACE' in partition:
        raise ValueError('Configure the verified GPU/CPU partition')
    wall = wall_seconds(os.getenv('REUSE_TIME', '01:00:00'))
    cpus = os.getenv('REUSE_CPUS', '8')
    if not cpus.isdigit() or int(cpus) < 1:
        raise ValueError('REUSE_CPUS must be positive')
    mem = os.getenv('REUSE_MEM', '32G')
    if not re.fullmatch(r'\d+(?:[KMGT])?', mem, flags=re.I):
        raise ValueError('Invalid host-memory request')
    logs = Path(environment['REUSE_RUN_ROOT']) / 'logs'
    # Create log directories before sbatch even in dry-run, but never touch the ledger.
    logs.mkdir(parents=True, exist_ok=True)
    suffix = '%A_%a' if is_array else '%j'
    command = [args.sbatch, '--parsable', '--no-requeue', f'--partition={partition}', f'--cpus-per-task={cpus}',
               f'--mem={mem}', f'--time={os.getenv("REUSE_TIME", "01:00:00")}',
               f'--output={logs / (suffix + ".out")}', f'--error={logs / (suffix + ".err")}',
               '--export=' + ','.join(f'{key}={value}' for key, value in environment.items())]
    if getattr(args, 'wait', False):
        command.append('--wait')
    for name, cpu_name, flag in [('REUSE_ACCOUNT', 'REUSE_CPU_ACCOUNT', '--account'),
                                 ('REUSE_QOS', 'REUSE_CPU_QOS', '--qos')]:
        # An explicitly empty CPU override asks Slurm to use its own default.
        selected_name = cpu_name if kind == 'cpu' and cpu_name in os.environ else name
        if os.getenv(selected_name):
            command.append(f'{flag}={os.environ[selected_name]}')
    if kind == 'gpu':
        flag = os.getenv('REUSE_GPU_FLAG', '--gres')
        request = os.getenv('REUSE_GPU_REQUEST', 'gpu:1')
        if flag not in {'--gres', '--gpus', '--gpus-per-node', '--gpus-per-task'}:
            raise ValueError('Unsupported GPU request flag')
        if (flag == '--gres' and not re.fullmatch(r'gpu(?::[A-Za-z0-9_-]+)?:1', request)) or (flag != '--gres' and request != '1'):
            raise ValueError('Exactly one GPU per independent shard is required')
        command.append(f'{flag}={request}')
    if is_array:
        command.append('--array=' + ','.join(map(str, selected)) + f'%{min(args.concurrency, len(selected))}')
    if args.dependency:
        if not re.fullmatch(r'\d+(?::\d+)*', args.dependency):
            raise ValueError('--dependency must contain actual numeric job IDs separated by colons')
        command.append(f'--dependency=afterok:{args.dependency}')
    command += [str(project / f'scripts/slurm/run_{kind}.sbatch'), args.stage]
    if args.stage in {'report', 'verify-results'}:
        run_dir = getattr(args, 'run_dir', None) or str(Path(environment['REUSE_RUN_ROOT']) / run_id)
        command += ['--run-dir', str(Path(run_dir).resolve())]
    else:
        command += ['--config', str(config_path)]
    if is_array:
        command += ['--slurm-array', '--num-shards', str(args.num_shards)]
    elif args.stage == 'merge':
        command += ['--num-shards', str(args.num_shards)]
    if args.pilot:
        command.append('--pilot')
    if args.resume:
        command.append('--resume')
    estimates = config.get('resource_estimates', {})
    reservation = wall / 3600 * len(selected) if kind == 'gpu' else 0
    stage_key = args.stage
    if args.stage == 'merge':
        stage_key += ':' + config.get('merge_stage', 'capture')
    elif args.stage == 'fit':
        stage_key += ':' + config.get('analysis_stage', 'discovery')
    elif args.stage == 'baseline' and args.pilot:
        stage_key = 'timing-pilot'
    record = {'reservation_id': uuid.uuid4().hex, 'created_utc': utcnow(), 'kind': kind,
              'stage': args.stage, 'stage_key': stage_key, 'run_id': run_id, 'config_hash': digest(config_path),
              'configured_gpu_hour_cap': configured_cap,
              'study_fingerprint': config.get('study_fingerprint'), 'snapshot_sha256': snapshot_hash,
              'num_shards': args.num_shards, 'is_array': is_array, 'selected_indices': selected, 'wall_seconds': wall,
              'tasks': {str(index): {'state': 'RESERVED'} for index in selected}, 'command': command,
              'estimates': {'max_gpu_hours': reservation, 'forwards': estimates.get(args.stage, {}).get('forwards'),
                            'storage_bytes': estimates.get(args.stage, {}).get('storage_bytes'),
                            'measured_estimates_available': bool(estimates.get(args.stage))},
              'gate_errors': errors}
    return record, errors


def has_duplicate(ledger, record):
    for job in ledger['jobs']:
        if (job['run_id'], job.get('stage_key', job['stage'])) != (record['run_id'], record.get('stage_key', record['stage'])):
            continue
        if job['config_hash'] != record['config_hash'] or job['snapshot_sha256'] != record['snapshot_sha256'] or job['num_shards'] != record['num_shards']:
            # A failed engineering validation may be retried after repairing the
            # environment. Successful/unknown attempts and research shards keep
            # their original immutable snapshot/configuration requirements.
            failed_validation = (
                job['stage'] == record['stage'] == 'validate'
                and job['num_shards'] == record['num_shards']
                and job.get('job_id')
                and job.get('study_fingerprint')
                and job['study_fingerprint'] == record.get('study_fingerprint')
                and job['tasks']
                and all(task.get('state') in TERMINAL and task.get('state') != 'COMPLETED'
                        for task in job['tasks'].values()))
            if failed_validation:
                continue
            raise ValueError('Stage retries must preserve snapshot/configuration hashes and original shard count')
        overlap = set(job['tasks']) & set(record['tasks'])
        if any(job['tasks'][index].get('state') not in TERMINAL for index in overlap):
            return True
    return False


def parse_sacct(text):
    rows = {}
    for line in text.splitlines():
        cells = line.strip().split('|')
        if len(cells) < 4 or not re.fullmatch(r'\d+(?:_\d+)?', cells[0]):
            continue # Ignore headers, array aggregate ranges, and ALL job-step rows.
        job_id, state, elapsed, tres = cells[:4]
        if not elapsed.isdigit():
            raise ValueError(f'Invalid sacct elapsed seconds for {job_id}')
        match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres)
        if not match:
            typed = re.findall(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?=,|$)', tres)
            gpus = sum(map(int, typed)) if typed else 0
        else:
            gpus = int(match.group(1))
        rows[job_id] = {'state': state.split()[0].rstrip('+'), 'elapsed_seconds': int(elapsed),
                        'allocated_gpus': gpus}
    return rows


def account(args):
    root = Path(os.environ['REUSE_RUN_ROOT'])
    path = Path(args.ledger) if args.ledger else root / 'scheduler_ledger.json'
    with ledger_lock(path):
        ledger = read_json(path) if path.exists() else empty_ledger(min(30.0, float(os.getenv('REUSE_GPU_HOUR_CAP', 30))))
        if args.sacct_file or args.refresh:
            if args.sacct_file:
                raw = Path(args.sacct_file).read_text()
            else:
                job_ids = [job['job_id'] for job in ledger['jobs'] if job.get('job_id')]
                if not job_ids:
                    print(json.dumps(budget(ledger), indent=2))
                    return
                result = subprocess.run([args.sacct, '-j', ','.join(job_ids), '--noheader', '--parsable2',
                                         '--format=JobID,State,ElapsedRaw,AllocTRES'], capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(f'sacct failed: {result.stderr.strip()}')
                raw = result.stdout
            rows = parse_sacct(raw)
            for job in ledger['jobs']:
                job_id = job.get('job_id')
                for index in job['tasks']:
                    key = f'{job_id}_{index}' if job.get('is_array', job['num_shards'] > 1) else job_id
                    if key in rows:
                        row = rows[key]
                        if job['kind'] == 'gpu' and row['elapsed_seconds'] > 0 and row['allocated_gpus'] < 1:
                            raise ValueError(f'sacct omitted GPU allocation for {key}; keep reservation until corrected')
                        job['tasks'][index] = row
            evidence = root / 'scheduler-accounting' / f'{uuid.uuid4().hex}.sacct.txt'
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text(raw)
            ledger['accounting_history'].append({'utc': utcnow(), 'source': str(evidence), 'sha256': digest(evidence)})
            atomic_json(path, ledger)
        print(json.dumps(budget(ledger), indent=2))


def submit(args):
    record, errors = build_submission(args)
    cap = min(30.0, record['configured_gpu_hour_cap'], float(os.getenv('REUSE_GPU_HOUR_CAP', '30')))
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('GPU-hour cap must be finite and positive (at most 30)')
    path = Path(args.ledger) if args.ledger else Path(os.environ['REUSE_RUN_ROOT']) / 'scheduler_ledger.json'
    with ledger_lock(path):
        ledger = read_json(path) if path.exists() else empty_ledger(cap)
        # Lowering a cap is allowed; increasing it past the original cap is not.
        ledger['gpu_hour_cap'] = min(cap, ledger['gpu_hour_cap'])
        existing = budget(ledger)
        if existing['remaining_gpu_hours'] + 1e-12 < record['estimates']['max_gpu_hours']:
            errors.append('Project GPU-hour cap exceeded by consumed plus outstanding worst-case allocations')
        if has_duplicate(ledger, record):
            errors.append('Concurrent duplicate stage/shard submission refused; reconcile accounting before retry')
        print(json.dumps({'command': shlex.join(record['command']), 'estimates': record['estimates'],
                          'budget_before': existing, 'blocked_reasons': errors, 'dry_run': args.dry_run}, indent=2))
        if args.dry_run:
            return 0
        if errors:
            raise ValueError('; '.join(errors))
        # Persist a conservative reservation before invoking sbatch. An uncertain response
        # keeps its reservation; never silently free budget for a possibly submitted job.
        ledger['jobs'].append(record)
        atomic_json(path, ledger)
        # Unrelated login-shell SBATCH_* defaults must not add GPUs, arrays, or
        # time requests that are absent from this budgeted argument list.
        scheduler_environment = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
        result = subprocess.run(record['command'], capture_output=True, text=True, env=scheduler_environment)
        record['sbatch_returncode'] = result.returncode
        record['sbatch_stdout'] = result.stdout.strip()
        record['sbatch_stderr'] = result.stderr.strip()
        match = re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip())
        if (result.returncode and not getattr(args, 'wait', False)) or not match:
            record['submission_state'] = 'UNKNOWN_RECONCILE_REQUIRED'
            atomic_json(path, ledger)
            raise ValueError('sbatch did not return an unambiguous job ID; reservation retained. Inspect scheduler before retry.')
        record['job_id'] = match.group(1)
        record['submission_state'] = 'SUBMITTED'
        if getattr(args, 'wait', False):
            # --wait returns the batch exit status, not submission success.
            # Keep the full reservation until normal accounting reconciliation.
            record['wait_exit_code'] = result.returncode
        for task in record['tasks'].values():
            task['state'] = 'PENDING'
        atomic_json(path, ledger)
        print(json.dumps({'job_id': record['job_id'], 'ledger': str(path), 'budget_after': budget(ledger)}))
        if getattr(args, 'wait', False) and result.returncode:
            raise ValueError(f'Job {record["job_id"]} completed with exit status {result.returncode}; inspect its logs')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=sorted(GPU_STAGES | CPU_STAGES | {'account'}))
    parser.add_argument('--config')
    parser.add_argument('--run-dir', help='CPU report/verify destination; defaults to configured run')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--indices', help='Retry indices, preserving original --num-shards')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot', action='store_true', help='Single 32-example baseline timing/capability pilot')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--wait', action='store_true', help='Use Slurm completion notification; does not poll the queue')
    parser.add_argument('--gates')
    parser.add_argument('--dependency', help='Actual afterok job IDs separated by colons')
    parser.add_argument('--ledger')
    parser.add_argument('--sbatch', default='sbatch')
    parser.add_argument('--sacct', default='sacct')
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--sacct-file')
    args = parser.parse_args()
    if args.stage == 'account':
        account(args)
        return 0
    if not args.config:
        parser.error('submission needs --config')
    return submit(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'submit: {exc}', file=sys.stderr)
        sys.exit(2)
