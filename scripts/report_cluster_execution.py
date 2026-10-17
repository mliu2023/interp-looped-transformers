#!/usr/bin/env python3
"""Rebuild the cluster execution status package from small saved raw records."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def build_report(root):
    root = Path(root)
    manifest = json.loads((root / 'evidence_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Execution evidence changed: ' + name)
    read = lambda name: json.loads((root / name).read_text())
    ledger = read('raw/scheduler_ledger.json')
    preflight = read('raw/preflight_7489067.json')
    submission = read('recovery_submission.json')
    tests = read('tests_summary.json')
    spec = importlib.util.spec_from_file_location('execution_submit', Path(__file__).parent / 'slurm/submit.py')
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    accounting = submit.budget(ledger)
    if preflight['status'] != 'failed' or preflight['cuda_available'] is not False:
        raise ValueError('This status package requires the saved failed pre-model CUDA preflight')
    metrics = {'schema_version': 1, 'status': 'blocked_before_model_loading',
               'model_examples_measured': 0, 'model_endpoints': {'P1': None, 'P2': None, 'P3': None},
               'prior_gpu_job_ids': [job['job_id'] for job in ledger['jobs'] if job['kind'] == 'gpu'],
               'gpu_hour_cap': ledger['gpu_hour_cap'], 'saved_accounting': accounting,
               'accounting_scope': 'saved scheduler ledger, not a live scheduler reconciliation',
               'recovery_submission_state': submission['state'],
               'recovery_job_id': submission.get('job_id'),
               'targeted_cpu_checks': tests}
    limitations = {'status': metrics['status'], 'model_findings': [],
                   'pending': ['working scheduler access and resolution of uncertain CPU submission',
                               'scheduled CUDA 12.8 repair and full CPU suite',
                               'fresh GPU preflight and loaded architecture audit',
                               'official inference parity and intervention validity',
                               'timing/memory pilot and capability/image controls',
                               'representative capture, discovery and remaining control orchestration',
                               'causal collection executor after loaded audit',
                               'immutable protocol, confirmation and scientific figures'],
                   'distribution': 'deterministic 2D scenes; not CLEVR',
                   'gpu_compatibility': 'proposed CUDA 12.8 environment remains untested on GPU',
                   'uncertain_submission': 'No job ID returned; resolve with scheduler before retrying',
                   'scientific_uncertainty': 'No scientific effect estimated; no inferential intervals applicable'}
    for name, value in (('metrics.json', metrics), ('limitations.json', limitations)):
        (root / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    fields = ['claim_id', 'claim', 'classification', 'evidence', 'denominator_or_estimand', 'uncertainty']
    rows = [
        ['CLUSTER-1', 'Prior GPU smoke failed CUDA preflight before model loading', 'observed_status',
         'raw/preflight_7489067.json; raw/7489067.err; raw/7489067.out', '0 model examples', 'not applicable'],
        ['CLUSTER-2', 'Saved GPU-hour usage includes the failed smoke', 'resource_accounting',
         'raw/scheduler_ledger.json; raw/7489067.sacct.txt', str(accounting['consumed_gpu_hours']),
         'saved accounting only; live reconciliation required'],
        ['CLUSTER-3', 'Targeted CPU engineering tests passed', 'engineering_check',
         'tests.log; tests_summary.json', str(tests['tests_run']) + ' tests', 'no model-validity inference'],
        ['CLUSTER-4', 'CPU recovery submission returned no job ID', 'observed_status',
         'recovery_submission.json', '1 submission attempt; ' + submission['state'], 'acceptance by Slurm unknown']]
    with (root / 'claim_ledger.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    (root / 'report.md').write_text(
        '# Cluster execution status\n\n'
        'The study remains blocked before model loading. No scientific endpoints have been measured.\n\n'
        'Existing job 7489067 failed CUDA preflight: torch 2.12.1+cu130 could not initialize on the assigned driver. '
        'A fresh CUDA 12.8 environment repair is prepared; it has not run.\n\n'
        'The saved ledger records %.12f GPU-hours consumed, %.12f outstanding maximum GPU-hours, '
        'and %.12f remaining of the 30 GPU-hour ceiling. These are saved records, not live scheduler state.\n\n'
        '%d targeted CPU tests passed. The full CPU suite in the repaired environment remains pending.\n\n'
        'The CPU recovery submission timed out without a job ID; its state is unknown. Resolve it before retrying. '
        'See [the continuation runbook](../../docs/CLUSTER_EXECUTION.md) for exact commands and pending gates.\n\n'
        'Recompute this package without loading a model:\n\n'
        '```bash\npython3 scripts/report_cluster_execution.py --run-dir results/cluster-execution-20261016\n```\n'
        % (accounting['consumed_gpu_hours'], accounting['outstanding_max_gpu_hours'],
           accounting['remaining_gpu_hours'], tests['tests_run']))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_dir), indent=2, allow_nan=False))
