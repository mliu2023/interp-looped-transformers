#!/usr/bin/env python3
"""Print saved viewer connection details without querying Slurm or starting jobs."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
from urllib.parse import urlencode


SSH_LOGIN = 'mliu171@ssh.ccv.brown.edu'
DEFAULT_LOCAL_PORT = 8766


def _port(value):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError('Viewer ports must be integers between 1 and 65535')
    return value


def connection_details(ready, local_port=DEFAULT_LOCAL_PORT):
    """Reconstruct commands from validated primitives, never a saved URL/command."""
    if not isinstance(ready, dict):
        raise ValueError('Ready file must contain a JSON object')
    local_port, remote_port = _port(local_port), _port(ready.get('port'))
    address_text = ready.get('compute_ip')
    if not isinstance(address_text, str) or not re.fullmatch(r'[0-9A-Fa-f:.]+', address_text):
        raise ValueError('Ready file must record an unscoped compute-node IP address')
    try:
        address = ipaddress.ip_address(address_text)
    except (ValueError, TypeError):
        raise ValueError('Ready file must record a valid compute-node IP address') from None
    if address.is_unspecified or address.is_loopback or address.is_multicast:
        raise ValueError('Ready file needs a compute-node IP reachable from the Oscar login node')
    token = ready.get('token')
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{16,256}', token):
        raise ValueError('Ready file has an invalid session token')
    target = f'[{address}]' if address.version == 6 else str(address)
    command = f'ssh -N -L {local_port}:{target}:{remote_port} {SSH_LOGIN}'
    url = f'http://127.0.0.1:{local_port}/?'+urlencode({'token': token})
    return command, url


def describe(experiment_dir=None, ready_file=None, local_port=DEFAULT_LOCAL_PORT, environ=None):
    """Read one snapshot of state. Missing files are an expected queued status."""
    _port(local_port)
    lines = []
    if experiment_dir is not None:
        state_file = Path(experiment_dir).expanduser().resolve()/'pipeline_state.json'
        if not state_file.is_file():
            receipt = state_file.with_name('queued_state.json')
            if receipt.is_file():
                state_file = receipt
        if not state_file.is_file():
            return ['Pipeline has not written its state yet.', f'State file: {state_file}',
                    'Run this command again after the queued pipeline starts.']
        state = json.loads(state_file.read_text())
        if not isinstance(state, dict):
            raise ValueError('Pipeline state must contain a JSON object')
        lines.append('Pipeline status: '+str(state.get('status', 'unknown')))
        if state.get('status') == 'failed':
            lines.append('Failure stage: '+str(state.get('failure_stage', 'not recorded')))
        for key, label in (('timing_job_id', 'Timing GPU job'), ('controller_job_id', 'Pipeline CPU job'), ('full_job_id', 'Image GPU job'),
                           ('verification_job_id', 'Verification CPU job'), ('viewer_job_id', 'Viewer CPU job')):
            if state.get(key) is not None:
                lines.append(f'{label}: {state[key]}')
        lines.append(f'State file: {state_file}')
        ready_file = state.get('viewer_connection_file')
        if not ready_file and state.get('viewer_job_id') is not None:
            environment = os.environ if environ is None else environ
            run_root = state.get('run_root') or environment.get('REUSE_RUN_ROOT')
            job_id = str(state['viewer_job_id'])
            if run_root and re.fullmatch(r'[0-9]+', job_id):
                ready_file = str(Path(run_root)/'viewer-sessions'/f'{job_id}.json')
        if not ready_file:
            lines.extend(['Viewer connection details are not recorded yet.',
                          'Run this command again after the pipeline advances.'])
            return lines
    if ready_file is None:
        raise ValueError('Select an experiment directory or a ready file')
    path = Path(ready_file).expanduser().resolve()
    if not path.is_file():
        lines.extend(['Viewer connection file has not appeared yet.', f'Connection file: {path}',
                      'Run this command again after the viewer job starts.'])
        return lines
    ready = json.loads(path.read_text())
    command, url = connection_details(ready, local_port)
    lines.extend([f'Connection file: {path}', '', 'Run on your computer and keep the terminal open:',
                  command, '', 'Open in your local browser:', url])
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--experiment-dir', help='Directory containing pipeline_state.json')
    source.add_argument('--ready-file', help='Explicit private viewer-sessions/JOB_ID.json shortcut')
    parser.add_argument('--local-port', type=int, default=DEFAULT_LOCAL_PORT,
                        help='Port on your computer (default: 8766)')
    args = parser.parse_args(argv)
    try:
        lines = describe(args.experiment_dir, args.ready_file, args.local_port)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f'Cannot read viewer connection details: {error}\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
