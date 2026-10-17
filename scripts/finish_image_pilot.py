#!/usr/bin/env python3
"""Complete the prespecified image experiment after its successful timing job.

Run after successful timing in a scheduled CPU allocation. It submits one GPU
job and a dependent CPU postprocessing job, without polling or automatic retries.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from reuseinterp.image_pilot import verify
from reuseinterp.io import atomic_json, config_load, digest, file_hash, read_json


def timing_bound(summary, records):
    """Conservative wall bound for 64 images, eight controls, and validation."""
    primary = [record for record in records if record['condition'] == 'primary']
    if len(primary) != 2 or len(records) != 2:
        raise ValueError('Timing evidence must contain exactly two primary image captures')
    elapsed = [float(record['elapsed_seconds']) for record in primary]
    tokens = [int(record['visual_tokens']) for record in primary]
    total = float(summary['elapsed_seconds'])
    if any(not math.isfinite(value) or value <= 0 for value in elapsed+[total]) or not all(256 <= n <= 1024 for n in tokens):
        raise ValueError('Timing evidence contains invalid elapsed times or native token counts')
    overhead = max(0., total-sum(elapsed))
    multiplier = (1024/min(tokens))**2
    seconds = overhead + 72*max(elapsed)*multiplier*2 + 180
    rounded = int(math.ceil(seconds/300)*300)
    if rounded > 3600:
        raise ValueError(f'Measured conservative allocation requires {rounded} seconds, above the one-hour bound')
    return {'load_and_preparation_overhead_seconds': overhead,
            'maximum_timing_record_seconds': max(elapsed), 'minimum_timing_visual_tokens': min(tokens),
            'maximum_full_visual_tokens': 1024, 'quadratic_token_multiplier': multiplier,
            'capture_tasks': 72, 'safety_multiplier': 2, 'additional_slack_seconds': 180,
            'unrounded_wall_seconds': seconds, 'wall_seconds': rounded,
            'reserved_gpu_hours': rounded/3600,
            'timing_peak_allocated_vram_bytes': summary['peak_allocated_vram_bytes']}


def run(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run completion orchestration in a scheduled CPU allocation')
    os.umask(0o077)
    experiment = Path(args.experiment_dir).resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    state_path = experiment / 'pipeline_state.json'
    if args.after_capture:
        state = read_json(state_path)
        if (state.get('snapshot_sha256') != args.snapshot_sha256 or
                state.get('expected_timing_config_hash') != args.expected_timing_config_hash or
                state.get('full_job_id') != args.gpu_job_id or
                state.get('status') not in ('waiting_for_capture', 'postprocess_submitted')):
            raise ValueError('Postprocessing identity/state differs from the submitted experiment')
    else:
        state = {'status': 'initializing', 'controller_job_id': os.environ['SLURM_JOB_ID'],
                 'snapshot_sha256': args.snapshot_sha256, 'experiment_dir': str(experiment),
                 'started_utc': datetime.now(timezone.utc).isoformat()}
        # An existing state may represent a submitted GPU job. Never start twice.
        with state_path.open('x') as stream:
            os.chmod(state_path, 0o600)
            json.dump(state, stream, indent=2)

    def save(status, **values):
        state.update(status=status, **values)
        atomic_json(state_path, state)

    def command(argv, name, environment):
        log = experiment / (name + '.stdout')
        error = experiment / (name + '.stderr')
        with log.open('x') as stdout, error.open('x') as stderr:
            os.chmod(log, 0o600)
            os.chmod(error, 0o600)
            result = subprocess.run(argv, cwd=PROJECT, env=environment, stdout=stdout, stderr=stderr)
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {error}')
        return log.read_text()

    def job_id(output):
        match = re.fullmatch(r'(\d+)(?:;[A-Za-z0-9_.-]+)?', output.strip())
        if not match:
            raise RuntimeError('Ambiguous sbatch response; inspect before any resubmission')
        return match.group(1)

    def private_log(path):
        with path.open('x'):
            pass
        os.chmod(path, 0o600)
        return str(path)

    try:
        base = dict(os.environ)
        base['REUSE_PROJECT_DIR'] = str(PROJECT)
        base['REUSE_EXPECTED_SNAPSHOT_SHA256'] = args.snapshot_sha256
        # UI packages must never change the frozen GPU environment's pip lock.
        base['PYTHONPATH'] = str(PROJECT/'src')
        config_path = Path(args.config).resolve()
        if PROJECT not in config_path.parents:
            raise ValueError('The full prespecified configuration must be inside this frozen snapshot')
        config = config_load(config_path)
        spec = config['image_pilot']
        if spec.get('image_limit') != 64 or spec.get('mode') != 'pilot' or len(spec.get('paraphrase_image_ids', [])) != 8:
            raise ValueError('Full configuration must prespecify 64 original images and eight prompt controls')
        full_run = Path(base['REUSE_RUN_ROOT']).resolve() / config['run_id']
        if not args.after_capture and (full_run/'protocol.json').exists():
            raise FileExistsError('Full experiment already has a protocol; refusing an automatic retry')
        command([sys.executable, str(PROJECT/'scripts/slurm/freeze.py'), '--verify', str(PROJECT),
                 '--verify-environment', '--expected-sha256', args.snapshot_sha256],
                'snapshot-postprocess' if args.after_capture else 'snapshot-verification', base)
        submitter = str(PROJECT/'scripts/slurm/submit.py')
        ledger_path = Path(base['REUSE_RUN_ROOT'])/'scheduler_ledger.json'
        if not args.after_capture:
            timing_run = Path(args.timing_run).resolve()
            if spec.get('timing_run_id') != timing_run.name or spec.get('timing_config_hash') != args.expected_timing_config_hash:
                raise ValueError('Full prespecification does not authorize this timing run')
            protocol = read_json(timing_run/'protocol.json')
            if protocol['config_hash'] != args.expected_timing_config_hash or protocol['specification'].get('mode') != 'timing':
                raise ValueError('Timing protocol does not match the authorized frozen configuration')
            for key in ('manifest_sha256', 'manifest_path', 'channels', 'layer_order', 'scope', 'prompt', 'paraphrase', 'sample_seed'):
                if spec[key] != protocol['specification'][key]:
                    raise ValueError('Timing/full protocols differ in a frozen field: ' + key)
            if protocol.get('checkpoint_revision') != config['checkpoint_revision'] or protocol.get('code_revision') != config['code_revision']:
                raise ValueError('Timing/full model identities differ')
            timing_summary = read_json(timing_run/'summary.json')
            if not timing_summary['capture_validation']['passed']:
                raise ValueError('Native timing capture parity failed')
            timing_job = read_json(timing_run/'execution_state.json')['attempts'][-1]['slurm_job_id']
            save('validating_timing', timing_job_id=timing_job, full_run=str(full_run),
                 full_config_sha256=file_hash(config_path), full_config_hash=digest(config),
                 expected_timing_config_hash=args.expected_timing_config_hash)
            timing_verification = verify(timing_run)
            bound = timing_bound(timing_summary, read_json(timing_run/'capture_index.json')['records'])
            bound.update(timing_summary_sha256=file_hash(timing_run/'summary.json'),
                         timing_protocol_sha256=file_hash(timing_run/'protocol.json'),
                         timing_verification=timing_verification)
            before = json.loads(command([sys.executable, submitter, 'account', '--refresh'], 'account-before', base))
            timing_jobs = [job for job in read_json(ledger_path)['jobs'] if job.get('job_id') == timing_job]
            if len(timing_jobs) != 1 or any(task.get('state') != 'COMPLETED' for task in timing_jobs[0]['tasks'].values()):
                raise RuntimeError('Timing allocation lacks fresh COMPLETED accounting; refusing full submission')
            for key in ('consumed_gpu_hours', 'outstanding_max_gpu_hours', 'remaining_gpu_hours'):
                if not isinstance(before.get(key), (int, float)) or not math.isfinite(before[key]):
                    raise ValueError('Accounting did not return finite GPU-hour totals')
            if before['remaining_gpu_hours'] + 1e-12 < bound['reserved_gpu_hours']:
                raise ValueError('Measured full allocation exceeds remaining shared GPU budget')
            if any(job['run_id'] == config['run_id'] and job['stage'] == 'image-pilot' for job in read_json(ledger_path)['jobs']):
                raise ValueError('A full-experiment submission already exists; automatic retries are forbidden')
            save('submitting_experiment', resource_bound=bound, budget_before=before)
            gpu = dict(base)
            seconds = bound['wall_seconds']
            gpu['REUSE_TIME'] = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
            try:
                command([sys.executable, submitter, 'image-pilot', '--config', str(config_path)], 'gpu-submission', gpu)
            finally:
                matching = [job for job in read_json(ledger_path)['jobs'] if job['run_id'] == config['run_id'] and job['stage'] == 'image-pilot']
                if len(matching) == 1 and matching[0].get('job_id'):
                    save(state['status'], full_job_id=matching[0]['job_id'])
            if not state.get('full_job_id'):
                raise RuntimeError('GPU submission did not persist an unambiguous job ID')
            save('waiting_for_capture')
            followup = {key: value for key, value in base.items() if not key.startswith('SBATCH_')}
            output = command(['sbatch', '--parsable', '--no-requeue', '--export=ALL',
                              '--dependency=afterany:' + state['full_job_id'], '--kill-on-invalid-dep=yes',
                              '--output=' + private_log(experiment/'postprocess.stdout'),
                              '--error=' + private_log(experiment/'postprocess.stderr'),
                              str(PROJECT/'scripts/slurm/finish_image_pilot.sbatch'),
                              '--after-capture', '--gpu-job-id', state['full_job_id'], '--config', str(config_path),
                              '--timing-run', str(timing_run), '--experiment-dir', str(experiment),
                              '--expected-timing-config-hash', args.expected_timing_config_hash,
                              '--snapshot-sha256', args.snapshot_sha256], 'postprocess-submission', followup)
            child = job_id(output)
            save('postprocess_submitted', postprocess_job_id=child, verification_job_id=child)
            print(json.dumps(state, indent=2))
            return 0
        if state.get('full_config_hash') != digest(config) or state.get('full_run') != str(full_run):
            raise ValueError('Postprocessing configuration differs from the submitted experiment')
        save('validating_capture', verification_job_id=os.environ['SLURM_JOB_ID'])
        after = json.loads(command([sys.executable, submitter, 'account', '--refresh'], 'account-after', base))
        ledger = read_json(ledger_path)
        jobs = [job for job in ledger['jobs'] if job.get('job_id') == args.gpu_job_id]
        if len(jobs) != 1 or any(task.get('state') != 'COMPLETED' for task in jobs[0]['tasks'].values()):
            raise RuntimeError('Full GPU allocation lacks fresh COMPLETED accounting; inspect the retained reservation')
        if read_json(full_run/'protocol.json')['config_hash'] != state['full_config_hash']:
            raise ValueError('Completed capture configuration does not match the prespecified experiment')
        save('experiment_complete', budget_after=after)
        analysis = experiment/'analysis-v1'
        analysis.mkdir()  # Derived evidence is never overwritten by a retry.
        ui = dict(base)
        ui['PYTHONPATH'] = str(experiment/'ui-deps') + os.pathsep + str(PROJECT/'src')
        ui['PLAYWRIGHT_BROWSERS_PATH'] = str(experiment/'browsers')
        bundle = analysis/'bundle'
        save('validating_viewer', bundle_dir=str(bundle))
        command([sys.executable, str(PROJECT/'scripts/validate_image_viewer.py'), '--run-dir', str(full_run),
                 '--dataset-dir', config['dataset_dir'], '--bundle-dir', str(bundle),
                 '--ready-file', str(analysis/'validation-connection.json'),
                 '--screenshot', str(analysis/'viewer_preview.png'), '--report', str(analysis/'viewer_validation.json')],
                'viewer-validation', ui)
        validation = read_json(analysis/'viewer_validation.json')
        if validation.get('passed') is not True:
            raise RuntimeError('Viewer export/browser validation did not pass')
        sessions = Path(base['REUSE_RUN_ROOT'])/'viewer-sessions'
        sessions.mkdir(mode=0o700, exist_ok=True)
        os.chmod(sessions, 0o700)
        viewer = {key: value for key, value in base.items() if not key.startswith('SBATCH_')}
        viewer['LOOPVL_VIEWER_BUNDLE'] = str(bundle)
        save('submitting_viewer', viewer_validation_sha256=file_hash(analysis/'viewer_validation.json'))
        output = command(['sbatch', '--parsable', '--no-requeue', '--export=ALL',
                          '--output=' + private_log(sessions/(os.environ['SLURM_JOB_ID']+'-viewer.out')),
                          '--error=' + private_log(sessions/(os.environ['SLURM_JOB_ID']+'-viewer.err')),
                          str(PROJECT/'scripts/slurm/serve_viewer.sbatch')], 'viewer-submission', viewer).strip()
        viewer_job = job_id(output)
        save('completed', viewer_job_id=viewer_job,
             viewer_connection_file=str(sessions/(viewer_job+'.json')),
             finished_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps(state, indent=2))
        return 0
    except Exception:
        save('failed', failure_stage=state['status'], traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--timing-run', required=True)
    parser.add_argument('--experiment-dir', required=True)
    parser.add_argument('--expected-timing-config-hash', required=True)
    parser.add_argument('--snapshot-sha256', required=True)
    parser.add_argument('--after-capture', action='store_true')
    parser.add_argument('--gpu-job-id')
    return run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(main())
