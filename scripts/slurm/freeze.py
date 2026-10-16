#!/usr/bin/env python3
"""Freeze reviewable sources/configs, and refuse queued jobs if any byte changed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

EXCLUDE = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'results',
           'model-cache', '.aws', '.ssh', '.codex', '.agents', 'build', 'dist'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root):
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and not (Path(base) == root and d in {'data', 'datasets', 'vendor'}))
        for directory in dirs:
            if (Path(base) / directory).is_symlink():
                raise ValueError(f'Snapshot must not contain symlink directories: {Path(base) / directory}')
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError(f'Snapshot must not contain symlinks: {path}')
            if name in EXCLUDE or name == 'snapshot_manifest.json' or name.endswith(('.pyc', '.env')) or name == '.DS_Store':
                continue
            yield path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'snapshot_manifest.json').read_text())
    actual = {str(p.relative_to(root)): sha(p) for p in files(root)}
    if actual != manifest['files']:
        changed = sorted(k for k in set(actual) | set(manifest['files']) if actual.get(k) != manifest['files'].get(k))
        raise ValueError(f'Frozen snapshot changed: {changed[:12]}')
    return sha(root / 'snapshot_manifest.json')


def verify_environment(root):
    expected = (Path(root) / 'cluster-dependencies.lock').read_text().splitlines()
    actual = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True)
    if actual.returncode:
        raise ValueError('Cannot inspect installed dependency versions')
    normalize = lambda lines: sorted(line.strip() for line in lines if line.strip() and not line.startswith('#'))
    if normalize(actual.stdout.splitlines()) != normalize(expected):
        raise ValueError('Installed environment differs from the frozen dependency lock; rebuild or freeze a new stage snapshot')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--verify-environment', action='store_true')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--dependency-lock', type=Path)
    args = parser.parse_args()
    if args.verify:
        actual_sha = verify(args.verify)
        if args.expected_sha256 and actual_sha != args.expected_sha256:
            raise ValueError('Snapshot manifest changed since submission')
        if args.verify_environment:
            verify_environment(args.verify)
        print(json.dumps({'snapshot_sha256': actual_sha}))
        return
    if not all((args.source, args.destination, args.dependency_lock)):
        parser.error('freeze needs --source, --destination, and --dependency-lock')
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source in destination.parents:
        parser.error('destination must be outside source, to prevent recursive copies')
    if destination.exists():
        parser.error('destination already exists; never overwrite a snapshot')
    lock = args.dependency_lock.resolve()
    if not lock.is_file():
        parser.error('dependency lock does not exist')
    destination.mkdir(parents=True)
    for path in files(source):
        relative = path.relative_to(source)
        # Site configuration and credentials are not source artifacts.
        if 'local' in path.name and path.suffix in {'.env', '.sh'}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(lock, destination / 'cluster-dependencies.lock')
    commit = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    manifest = {'schema_version': 1, 'source_commit': commit.stdout.strip() if commit.returncode == 0 else None,
                'files': {str(p.relative_to(destination)): sha(p) for p in files(destination)},
                'dependency_lock_sha256': sha(lock)}
    (destination / 'snapshot_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'snapshot': str(destination), 'snapshot_sha256': verify(destination)}))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f'freeze: {exc}', file=sys.stderr)
        sys.exit(2)
