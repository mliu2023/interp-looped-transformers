#!/usr/bin/env python3
"""Verify a real-image run, export its viewer, and exercise a browser on CPU.

Run in a scheduled CPU allocation with Playwright available for validation.
The exported viewer needs only NumPy and Pillow during ordinary browsing.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reuseinterp.io import atomic_json, file_hash
from reuseinterp.viewer import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dataset-dir')
    parser.add_argument('--kind', choices=('training-source', 'historical-shapes'), default='training-source')
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--ready-file', required=True)
    parser.add_argument('--screenshot', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a scheduled CPU allocation for capture verification and browser testing')
    screenshot = Path(args.screenshot).resolve()
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == 'historical-shapes':
        from reuseinterp.poles import verify_recomputation as verify
    else:
        from reuseinterp.image_pilot import verify
    verification = verify(args.run_dir)
    exported = export_bundle(args.run_dir, args.bundle_dir, args.dataset_dir, kind=args.kind)
    server_command = [sys.executable, str(Path(__file__).with_name('serve_viewer.py')),
                      'serve', '--bundle-dir', args.bundle_dir, '--port', '0',
                      '--ready-file', args.ready_file]
    log = screenshot.parent / 'viewer-validation.log'
    with log.open('x') as server_log:
        os.chmod(log, 0o600)
        server = subprocess.Popen(server_command, stdout=subprocess.PIPE,
                                  stderr=server_log, text=True)
        try:
            # The first line is emitted only after every bundle hash is checked
            # and the listening socket and private ready file exist. No polling.
            ready = json.loads(server.stdout.readline())
            if ready.get('status') != 'ready':
                raise RuntimeError('Viewer did not report readiness')
            browser = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('smoke_viewer_browser.py')),
                 '--ready-file', args.ready_file, '--screenshot', str(screenshot)],
                capture_output=True, text=True, timeout=180)
            (screenshot.parent / 'browser-validation.stderr').write_text(browser.stderr)
            if browser.returncode:
                raise RuntimeError('Browser validation failed: ' + browser.stderr[-4000:])
            browser_result = json.loads(browser.stdout)
            if browser_result.get('passed') is not True:
                raise RuntimeError('Browser validation did not pass')
            result = {'passed': True, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                      'verification': verification, 'export': exported,
                      'browser': browser_result, 'screenshot_sha256': file_hash(screenshot),
                      'validation_script_sha256': file_hash(__file__)}
            atomic_json(args.report, result)
            print(json.dumps(result, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
