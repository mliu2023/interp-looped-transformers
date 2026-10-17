"""Read-only, dependency-light explorer for saved LoopVL activity captures.

``export`` creates a portable, hash-checked bundle. ``serve`` reads only files
named by that bundle; it never loads a model or serves a filesystem directory.
"""
import argparse
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import sys
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image, ImageOps


STREAMS = ('product', 'gate_pre', 'gate_effective', 'value')
SCHEMA_VERSION = 1


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, relative):
    """Resolve a manifest path without allowing traversal, symlinks, or drives."""
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('Invalid bundle path')
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in ('..', '.') for p in parts.parts):
        raise ValueError('Bundle paths must be relative without traversal')
    root = Path(root).resolve()
    candidate = root.joinpath(*parts.parts)
    # Do not permit symlinks even when their current target happens to be inside.
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent != root and root in parent.parents):
        raise ValueError('Bundle symlinks are not permitted')
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('Bundle path escaped its root')
    return candidate


def _source_file(root, location):
    root = Path(root).resolve()
    path = Path(location)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Source artifact must be a file within its selected source root')
    return path


def _ordered_axes(protocol, summary, validation):
    spec = protocol.get('specification', protocol.get('poles_pilot', protocol))
    channels = spec.get('channels')
    if not isinstance(channels, dict) or not channels:
        raise ValueError('Protocol must record channel identities by physical layer')
    order = validation.get('layers') or summary.get('capture_validation', {}).get('layers')
    if order is None:
        order = spec.get('layer_order') or protocol.get('layer_order')
    # Explicit axes avoid the historical JSON sorting 0, 15, 7 pitfall. Raw
    # arrays have layer-specific keys, so a numeric fallback is also unambiguous.
    if order is None:
        order = sorted(map(int, channels))
    if len(order) != len(set(map(int, order))) or set(map(str, order)) != set(channels):
        raise ValueError('Recorded layer order disagrees with channel identities')
    axes = [{'layer': int(layer), 'channels': [int(c) for c in channels[str(layer)]]}
            for layer in order]
    if any(not a['channels'] or len(a['channels']) != len(set(a['channels'])) for a in axes):
        raise ValueError('Channels must be nonempty and unique within each layer')
    return spec, axes


def export_bundle(run_dir, output_dir, dataset_dir=None, label=None, kind=None):
    """Copy original images and verified raw captures, and derive all stream means.

    The raw NPZ contract is one file per image, layer_<n>__<stream> arrays shaped
    [six invocations, visual tokens, selected channels within physical layer].
    Original source hashes remain in the manifest. Display images are EXIF
    oriented JPEG previews; the exact original bytes are separately retained.
    """
    run, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Choose a new bundle directory; never overwrite an export')
    protocol, rows = _read(run/'protocol.json'), _read(run/'image_rows.json')
    summary = _read(run/'summary.json') if (run/'summary.json').exists() else {}
    validation = _read(run/'capture_validation.json') if (run/'capture_validation.json').exists() else {}
    spec, axes = _ordered_axes(protocol, summary, validation)
    index = _read(run/'capture_index.json')
    if not index.get('complete'):
        raise ValueError('Only a complete capture index can be exported')
    primary = {}
    for record in index['records']:
        if record.get('condition') == 'primary':
            identity = str(record.get('scene_id', record.get('image_id', '')))
            if not identity or identity in primary:
                raise ValueError('Duplicate or absent primary capture identity')
            primary[identity] = record
    if not rows or len(rows) != len(primary):
        raise ValueError('Image rows and primary captures must match one to one')
    if kind is None:
        kind = 'historical-shapes' if any('objects' in row for row in rows) else 'training-source'
    if kind not in ('training-source', 'historical-shapes'):
        raise ValueError('Unknown dataset kind')
    if kind == 'training-source' and any('objects' in row for row in rows):
        raise ValueError('Synthetic shape records cannot be labeled training-source images')
    dataset = Path(dataset_dir).resolve() if dataset_dir else run
    output.mkdir(parents=True)
    (output/'files').mkdir()
    units = [(a['layer'], channel) for a in axes for channel in a['channels']]
    means = np.empty((len(STREAMS), len(rows), 6, len(units)), dtype=np.float32)
    scale = np.zeros((len(STREAMS), len(units)), dtype=np.float32)
    images, files, seen = [], {}, set()

    def register(relative, source=None, expected_hash=None):
        target = _inside(output, relative)
        if source is not None:
            actual = _hash(source)
            if expected_hash and actual != expected_hash:
                raise ValueError('A source file no longer matches its recorded SHA256')
            shutil.copyfile(source, target)
        files[relative] = {'sha256': _hash(target), 'bytes': target.stat().st_size,
                           'mime': mimetypes.guess_type(relative)[0] or 'application/octet-stream'}
        return target

    for i, row in enumerate(rows):
        identity = str(row.get('scene_id', row.get('image_id', '')))
        if not identity or identity in seen or identity not in primary:
            raise ValueError('Image identities must uniquely match primary captures')
        seen.add(identity)
        record = primary[identity]
        grid = record.get('grid', row.get('spatial_layout', {}).get('effective_grid'))
        if not isinstance(grid, list) or len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
            raise ValueError('Each capture must record its positive effective [height,width] grid')
        layout = record.get('spatial_layout', row.get('spatial_layout', {}))
        if layout and (layout.get('no_padding') is False or layout.get('no_crop') is False or layout.get('merge_size', 1) != 1):
            raise ValueError('Overlay mapping needs the validated no-crop/no-pad merge-one layout')
        raw = _source_file(run, record['artifact'])
        raw_name = f'files/activation_{i:04d}.npz'
        register(raw_name, raw, record.get('sha256'))
        with np.load(raw, allow_pickle=False) as capture:
            for si, stream in enumerate(STREAMS):
                offset = 0
                for axis in axes:
                    key = f'layer_{axis["layer"]}__{stream}'
                    values = np.asarray(capture[key])
                    n = len(axis['channels'])
                    if values.shape != (6, grid[0]*grid[1], n) or not np.isfinite(values).all():
                        raise ValueError('Nonfinite or misaligned raw capture arrays')
                    means[si, i, :, offset:offset+n] = values.mean(axis=1)
                    scale[si, offset:offset+n] = np.maximum(scale[si, offset:offset+n], np.abs(values).max(axis=(0, 1)))
                    offset += n
        image_location = row.get('image_path', row.get('relative_path'))
        original = _source_file(dataset, image_location)
        with Image.open(original) as source:
            original_format = source.format
            preview = ImageOps.exif_transpose(source).convert('RGB')
            oriented_size = list(preview.size)
            preview.thumbnail((1200, 1200))
            preview_name = f'files/preview_{i:04d}.jpg'
            preview.save(output/preview_name, quality=92)
        extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(original_format, '.image')
        original_name = f'files/original_{i:04d}{extension}'
        register(original_name, original, row.get('image_sha256'))
        register(preview_name)
        images.append({'id': identity, 'family_id': row.get('family_id', identity),
                       'image_file': preview_name, 'original_file': original_name,
                       'activation_file': raw_name, 'grid': grid,
                       'oriented_image_size': oriented_size,
                       'source_sha256': files[original_name]['sha256'],
                       'source_hash_verified': bool(row.get('image_sha256')),
                       'spatial_layout': layout or {'kind': 'recorded_row_major_grid',
                           'limitation': 'Historical grid mapping; native natural-image validation was not part of this run'},
                       'provenance': row.get('provenance', {}), 'source_record': row})
    np.savez_compressed(output/'files/measurements.npz', means=means, absolute_scale=scale)
    register('files/measurements.npz')
    for name in ('protocol.json', 'capture_validation.json', 'capture_index.json',
                 'summary.json', 'image_rows.json', 'source_manifest.json',
                 'source_manifest_original.json'):
        if (run/name).is_file():
            register('files/'+name, run/name)
    limitations = [
        'Exploratory signed responses and image rankings. Positive/negative values do not establish semantic poles, memorization, circuits, or superposition.',
        'Channels and image families are a bounded sample. Galleries are sorted by spatial mean; both tails can have the same sign.',
        'Overlays show an interpolated visual-token grid, not pixel attribution. One symmetric scale spans every image and all six invocations for this channel and stream.',
        'Interpret comparisons conditional on the recorded image prompt, checkpoint, preprocessing, and sampled physical channels.',
    ]
    if kind == 'historical-shapes':
        limitations.insert(0, 'Historical synthetic-shape engineering pilot. These are not LoopVL training photographs.')
    else:
        memberships = sorted({str(image['provenance'].get('training_membership', 'unknown')) for image in images})
        limitations.insert(0, 'Training membership evidence: '+', '.join(memberships)+'. Reported-corpus membership alone does not prove this exact checkpoint consumed the image.')
    manifest = {'schema_version': SCHEMA_VERSION, 'label': label or run.name, 'kind': kind,
                'run_id': run.name, 'source_protocol_sha256': _hash(run/'protocol.json'),
                'checkpoint_revision': summary.get('checkpoint_revision', protocol.get('checkpoint_revision', spec.get('checkpoint_revision'))),
                'code_revision': protocol.get('code_revision', spec.get('code_revision')),
                'prompt': summary.get('prompt', spec.get('prompt', 'Not recorded')),
                'channel_axes': axes, 'streams': list(STREAMS), 'invocations': list(range(1, 7)),
                'images': images, 'files': files, 'measurements_file': 'files/measurements.npz',
                'limitations': limitations, 'capture_validation': validation or summary.get('capture_validation', {}),
                'summary': summary}
    _write(output/'manifest.json', manifest)
    return {'bundle_dir': str(output), 'images': len(images), 'physical_channels': len(units),
            'bytes': sum(item['bytes'] for item in files.values()), 'manifest_sha256': _hash(output/'manifest.json')}


class Bundle:
    """Validated manifest-only data access; no user-controlled filesystem roots."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = _read(self.directory/'manifest.json')
        if self.manifest.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported viewer bundle schema')
        self.files = {}
        for name, record in self.manifest['files'].items():
            path = _inside(self.directory, name)
            if not path.is_file() or path.stat().st_size != record['bytes'] or _hash(path) != record['sha256']:
                raise ValueError('Missing or changed bundle artifact: '+name)
            self.files[name] = path
        self.images = self.manifest['images']
        if not self.images or self.manifest['streams'] != list(STREAMS) or self.manifest['invocations'] != list(range(1, 7)):
            raise ValueError('Invalid image, stream, or invocation axes')
        for image in self.images:
            if any(image[key] not in self.files for key in ('image_file', 'original_file', 'activation_file')):
                raise ValueError('Image references an unlisted artifact')
        self.units = [(a['layer'], c) for a in self.manifest['channel_axes'] for c in a['channels']]
        if len(self.units) != len(set(self.units)) or not self.units:
            raise ValueError('Physical channel identities must be unique')
        with np.load(self.file(self.manifest['measurements_file']), allow_pickle=False) as data:
            self.means = data['means'].copy()
            self.scale = data['absolute_scale'].copy()
        expected = (len(STREAMS), len(self.images), 6, len(self.units))
        if self.means.shape != expected or self.scale.shape != (len(STREAMS), len(self.units)) or not np.isfinite(self.means).all() or not np.isfinite(self.scale).all() or (self.scale < 0).any():
            raise ValueError('Invalid saved measurement axes or values')
        self._cached = None
        self._lock = threading.Lock()

    def file(self, relative):
        if relative not in self.files:
            raise KeyError('Artifact is not listed by the bundle manifest')
        # Recheck resolved containment on each access, including a late symlink.
        path = _inside(self.directory, relative)
        if path != self.files[relative]:
            raise ValueError('Artifact path changed')
        return path

    def _selection(self, layer, channel, stream):
        try:
            return self.units.index((int(layer), int(channel))), STREAMS.index(stream)
        except (ValueError, TypeError):
            raise ValueError('Unknown physical layer, channel, or stream') from None

    def unit(self, layer, channel, stream):
        unit, si = self._selection(layer, channel, stream)
        return {'means': self.means[si, :, :, unit].tolist(),
                'absolute_scale': float(self.scale[si, unit]),
                'layer': int(layer), 'channel': int(channel), 'stream': stream}

    def image(self, image_index, layer, channel, stream):
        unit, si = self._selection(layer, channel, stream)
        image_index = int(image_index)
        if not 0 <= image_index < len(self.images):
            raise ValueError('Image index outside this bundle')
        image = self.images[image_index]
        axis = next(a for a in self.manifest['channel_axes'] if a['layer'] == int(layer))
        local = axis['channels'].index(int(channel))
        key = (image_index, int(layer), int(channel), stream)
        with self._lock:
            if self._cached is not None and self._cached[0] == key:
                maps = self._cached[1]
            else:
                with np.load(self.file(image['activation_file']), allow_pickle=False) as data:
                    raw = data[f'layer_{int(layer)}__{stream}']
                    expected = (6, image['grid'][0]*image['grid'][1], len(axis['channels']))
                    if raw.shape != expected or not np.isfinite(raw).all():
                        raise ValueError('Raw arrays do not match the image layout')
                    maps = raw[:, :, local].reshape((6, *image['grid'])).copy()
                self._cached = (key, maps)
        return {'maps': maps.tolist(), 'grid': image['grid'],
                'means': self.means[si, image_index, :, unit].tolist(),
                'absolute_scale': float(self.scale[si, unit]), 'image_index': image_index}


def handler_for(bundle, token):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'LoopVLViewer/1'

        def log_message(self, format, *args):
            # Never put URL credentials in a server log.
            sys.stderr.write(json.dumps({'request': self.command, 'path': urlsplit(self.path).path,
                                         'client': self.client_address[0]})+'\n')

        def _send(self, status, content, mime='application/json', cookie=False):
            if isinstance(content, str):
                content = content.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if cookie:
                self.send_header('Set-Cookie', 'loopvl_viewer='+token+'; Path=/; HttpOnly; SameSite=Strict')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(content)

        def do_GET(self):
            parsed = urlsplit(self.path)
            args = parse_qs(parsed.query, keep_blank_values=True)
            supplied = args.get('token', [''])[0]
            bearer = self.headers.get('Authorization', '')
            if bearer.startswith('Bearer '):
                supplied = bearer[7:]
            if not supplied:
                try:
                    supplied = SimpleCookie(self.headers.get('Cookie', '')).get('loopvl_viewer')
                    supplied = supplied.value if supplied else ''
                except Exception:
                    supplied = ''
            if not hmac.compare_digest(supplied.encode('utf-8'), token.encode('utf-8')):
                self._send(403, json.dumps({'error': 'Open the session URL printed by the server; authentication is required.'}))
                return
            if parsed.path == '/' and 'token' in args:
                self.send_response(303)
                self.send_header('Location', '/')
                self.send_header('Set-Cookie', 'loopvl_viewer='+token+'; Path=/; HttpOnly; SameSite=Strict')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            try:
                if parsed.path == '/':
                    self._send(200, HTML, 'text/html; charset=utf-8')
                elif parsed.path == '/app.js':
                    self._send(200, JS, 'text/javascript; charset=utf-8')
                elif parsed.path == '/style.css':
                    self._send(200, CSS, 'text/css; charset=utf-8')
                elif parsed.path == '/api/manifest':
                    self._send(200, json.dumps(bundle.manifest, allow_nan=False))
                elif parsed.path in ('/api/unit', '/api/image'):
                    selection = [args[name][0] for name in ('layer', 'channel', 'stream')]
                    value = bundle.unit(*selection) if parsed.path == '/api/unit' else bundle.image(args['image'][0], *selection)
                    self._send(200, json.dumps(value, allow_nan=False))
                elif parsed.path.startswith('/file/'):
                    # Exact manifest membership; no directory fallback, decoding,
                    # globbing, or general filesystem serving.
                    relative = parsed.path[len('/file/'):]
                    path = bundle.file(relative)
                    self._send(200, path.read_bytes(), bundle.manifest['files'][relative]['mime'])
                else:
                    self._send(404, json.dumps({'error': 'Unknown endpoint'}))
            except (KeyError, ValueError, IndexError, TypeError):
                self._send(400, json.dumps({'error': 'Unknown or invalid bundle selection'}))

    return Handler


def serve_bundle(directory, host='127.0.0.1', port=8765, ready_file=None):
    bundle = Bundle(directory)
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer((host, port), handler_for(bundle, token))
    server.daemon_threads = True
    node = socket.gethostname()
    try:
        node_ip = socket.gethostbyname(node)
    except socket.gaierror:
        node_ip = None
    ready = {'status': 'ready', 'host': host, 'node': node, 'compute_ip': node_ip,
             'port': server.server_address[1], 'token': token,
             'local_url': f'http://127.0.0.1:{server.server_address[1]}/?token={token}',
             'bundle_dir': str(bundle.directory), 'images': len(bundle.images),
             'physical_channels': len(bundle.units)}
    if ready_file:
        path = Path(ready_file)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(ready, stream, indent=2)
            stream.write('\n')
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


HTML = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoopVL activity explorer</title><link rel="stylesheet" href="/style.css"><script src="/app.js" defer></script></head>
<body><header><div><p class="eyebrow">FROZEN MODEL · SAVED CAPTURES</p><h1>LoopVL activity explorer</h1>
<p id="run-label">Loading the verified bundle…</p></div><div><span id="kind" class="badge"></span> <span id="status" class="badge" data-state="loading" role="status">Loading</span></div></header>
<main><div id="error" role="alert" hidden></div><section class="context"><p id="metadata"></p>
<details><summary>Provenance, prompt and limits</summary><div id="limitations"></div><pre id="provenance"></pre></details></section>
<section class="controls" aria-label="Activity selection">
<label>Physical L layer<select id="layer"></select></label><label>Channel<select id="channel"></select></label>
<label>Captured stream<select id="stream"><option value="product">Gated product · down_proj input</option><option value="gate_pre">Gate before activation</option><option value="gate_effective">Gate after activation</option><option value="value">Value branch</option></select></label>
<label>Rank images at<select id="invocation"></select></label><label>Gallery size<select id="tail"><option>4</option><option selected>8</option><option>16</option></select></label>
</section><p class="notice" id="notice"></p>
<section><div class="section-heading"><h2>Signed image-mean responses</h2><p>Click any image to inspect all six invocations.</p></div>
<div class="tails"><div><h3>Highest responses <span id="top-signs"></span></h3><div id="top" class="gallery"></div></div>
<div><h3>Lowest responses <span id="bottom-signs"></span></h3><div id="bottom" class="gallery"></div></div></div></section>
<section><div class="section-heading"><h2>Image across the loop</h2><label>Image<select id="image-select"></select></label>
<label>Overlay opacity <input id="opacity" type="range" min="0" max="100" value="62"></label></div>
<p id="selected-caption"></p><p class="legend"><span class="negative">Blue: negative</span><span>White: zero</span><span class="positive">Red: positive</span><strong id="scale-label"></strong></p>
<div id="overlays" class="overlays"></div><details><summary>Selected image record and source evidence</summary><pre id="image-provenance"></pre><p><a id="original-link">Download exact source image</a> · <a id="raw-link">Download this image's raw captures</a></p></details></section>
<section class="charts"><div><h2>Response by invocation</h2><p>Selected image, spatial mean. The same physical weights are reused.</p><canvas id="trajectory" width="680" height="330" aria-label="Signed mean response across six invocations"></canvas></div>
<div><div class="section-heading"><h2>Image rank transfer</h2><label>Compare with L1<select id="compare"></select></label></div>
<p id="scatter-caption"></p><canvas id="scatter" width="680" height="330" aria-label="Image response rank scatter"></canvas><p class="subtle">Each point is one image; edited or duplicate relatives can share a family. Click a point to select its image.</p></div></section>
<footer>Read-only exploration of saved model responses. No generated semantic descriptions or causal claims.</footer></main></body></html>'''

CSS = '''*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#172332;font:15px/1.5 system-ui,-apple-system,sans-serif}header{padding:27px max(24px,calc((100vw - 1600px)/2));background:#122b3c;color:white;display:flex;align-items:center;justify-content:space-between;gap:20px}h1{font-size:29px;margin:2px 0}h2{font-size:20px;margin:0}h3{font-size:16px;margin:6px 0 13px}p{margin:5px 0 12px}.eyebrow{font-size:11px;letter-spacing:2px;color:#a8c9d9;margin:0}.badge{padding:7px 11px;border:1px solid #5a8398;border-radius:5px;font-size:12px}main{max-width:1650px;margin:auto;padding:20px 24px}section{background:white;border:1px solid #dce2e8;border-radius:8px;margin-bottom:20px;padding:20px}.context{padding:13px 20px}.context p{margin-bottom:7px}.controls{display:flex;gap:18px;flex-wrap:wrap;position:sticky;top:0;z-index:2;box-shadow:0 4px 14px #1723320a}label{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:600;color:#425367}select{font:14px system-ui;padding:7px 10px;max-width:100%;background:#fff;border:1px solid #bdc9d4;border-radius:4px;color:#172332}.section-heading{display:flex;align-items:center;gap:18px;flex-wrap:wrap;justify-content:space-between;margin-bottom:14px}.section-heading p{font-size:13px;color:#586879;margin:0}.notice{font-size:13px;padding:10px 16px;background:#fff7df;border-left:3px solid #d5a727;margin:-5px 0 20px}.tails{display:grid;grid-template-columns:1fr 1fr;gap:26px}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.card{cursor:pointer;border:2px solid transparent;border-radius:5px;background:#f5f7f9;overflow:hidden;padding:0;text-align:left;color:#253649;min-width:0}.card:hover{border-color:#8cb8cf}.card.selected{border-color:#15749e}.card img{width:100%;aspect-ratio:1.2;object-fit:contain;display:block;background:#e8edf1}.card span{display:block;font-size:12px;padding:3px 5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.card .response{font-weight:650;font-variant-numeric:tabular-nums;padding-top:7px}.card .id{font-size:10px;color:#617183}.overlays{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:14px 0 20px}.map{margin:0;min-width:0}.map canvas{width:100%;height:auto;display:block;background:#f3f5f7;border-radius:4px}.map figcaption{font-size:12px;padding-top:5px}.map figcaption strong{font-size:14px;margin-right:6px}.legend{font-size:12px;display:flex;flex-wrap:wrap;gap:15px;align-items:center}.negative{color:#2569b1}.positive{color:#ba3f45}#scale-label{font-weight:500;margin-left:auto}#selected-caption{font-size:13px;overflow-wrap:anywhere}.charts{display:grid;grid-template-columns:1fr 1fr;gap:25px}.charts>div{min-width:0}.charts p{font-size:13px;color:#586879}.charts canvas{width:100%;height:auto;display:block;cursor:crosshair}details{font-size:13px}summary{cursor:pointer;color:#286684}pre{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:#f4f6f8;padding:12px;border-radius:4px}.context li{margin:6px 0}.subtle{font-size:12px!important;color:#6e7a87}footer{text-align:center;color:#6c7987;font-size:12px;padding:8px 0 22px}#error{border:1px solid #ce5656;color:#872424;background:#fff0f0;padding:14px;margin-bottom:15px}a{color:#086e9c}input[type=range]{min-width:130px}h3 span{font-size:11px;font-weight:400;color:#63717f}@media(max-width:1100px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.overlays{grid-template-columns:repeat(3,minmax(0,1fr))}.charts{grid-template-columns:1fr}}@media(max-width:650px){header{padding:20px;align-items:flex-start;flex-direction:column}main{padding:12px}section{padding:14px}.tails{grid-template-columns:1fr;gap:22px}.overlays{grid-template-columns:repeat(2,minmax(0,1fr))}.controls{position:static;gap:10px}h1{font-size:25px}}'''

JS = r'''"use strict";
const $ = id => document.getElementById(id);
let manifest, unit, selected=0, imageData, loadedImage, unitVersion=0, imageVersion=0, scatterPoints=[];
const fmt=x=>Number.isFinite(x)?Number(x).toPrecision(4):"undefined";
const signed=x=>(x>=0?"+":"")+fmt(x);
const asset=name=>"/file/"+name;
function error(e){$("error").hidden=false;$("error").textContent=String(e.message||e);$("status").dataset.state="error";$("status").textContent="Error";}
async function api(url){const response=await fetch(url);if(!response.ok)throw Error("Viewer request failed ("+response.status+"). Check the session URL and bundle.");return response.json();}
function options(select,values,label=x=>x){select.replaceChildren();for(const value of values){const option=document.createElement("option");option.value=value;option.textContent=label(value);select.append(option);}}
function query(){return new URLSearchParams({layer:$("layer").value,channel:$("channel").value,stream:$("stream").value});}
function ranks(values){const order=values.map((v,i)=>({v,i})).sort((a,b)=>a.v-b.v||a.i-b.i), out=Array(values.length);let j=0;while(j<order.length){let k=j+1;while(k<order.length&&order[k].v===order[j].v)k++;for(let n=j;n<k;n++)out[order[n].i]=(j+1+k)/2;j=k;}return out;}
function correlation(a,b){const ma=a.reduce((s,x)=>s+x,0)/a.length,mb=b.reduce((s,x)=>s+x,0)/b.length;let aa=0,bb=0,ab=0;for(let i=0;i<a.length;i++){aa+=(a[i]-ma)**2;bb+=(b[i]-mb)**2;ab+=(a[i]-ma)*(b[i]-mb);}return aa*bb>0?ab/Math.sqrt(aa*bb):NaN;}
async function chooseUnit(){const version=++unitVersion;$("status").dataset.state="loading";$("status").textContent="Loading";try{const next=await api("/api/unit?"+query());if(version!==unitVersion)return;unit=next;renderGalleries();await chooseImage(selected);if(version!==unitVersion)return;renderScatter();$("status").dataset.state="ready";$("status").textContent="Ready";}catch(e){error(e);}}
function selectLayer(){const axis=manifest.channel_axes.find(a=>a.layer===Number($("layer").value));options($("channel"),axis.channels);chooseUnit();}
function renderGalleries(){if(!unit)return;const pass=Number($("invocation").value)-1,n=Number($("tail").value);const order=unit.means.map((v,i)=>({i,value:v[pass]})).sort((a,b)=>a.value-b.value||a.i-b.i);
 const high=order.slice(-n).reverse(),low=order.slice(0,n);for(const [where,items] of [["top",high],["bottom",low]]){const parent=$(where);parent.replaceChildren();for(const item of items){const button=document.createElement("button");button.className="card"+(item.i===selected?" selected":"");button.type="button";button.title=manifest.images[item.i].id+"; family "+manifest.images[item.i].family_id;const img=document.createElement("img");img.src=asset(manifest.images[item.i].image_file);img.alt=manifest.images[item.i].id;img.loading="lazy";const value=document.createElement("span");value.className="response";value.textContent=signed(item.value);const id=document.createElement("span");id.className="id";id.textContent=manifest.images[item.i].id;button.append(img,value,id);button.onclick=()=>chooseImage(item.i).catch(error);parent.append(button);}
 $(where+"-signs").textContent=items.length&&items.every(x=>x.value>0)?"all positive":items.length&&items.every(x=>x.value<0)?"all negative":"mixed signs / zeros";}
}
async function chooseImage(index){selected=Number(index);$("image-select").value=String(selected);renderGalleries();const version=++imageVersion;const q=query();q.set("image",selected);const next=await api("/api/image?"+q);if(version!==imageVersion)return;imageData=next;const metadata=manifest.images[selected];const img=new Image();img.src=asset(metadata.image_file);await img.decode();if(version!==imageVersion)return;loadedImage=img;
 $("selected-caption").textContent=metadata.id+" · family "+metadata.family_id+" · "+metadata.oriented_image_size.join(" × ")+" original pixels · "+metadata.grid.join(" × ")+" visual-token grid";
 $("image-provenance").textContent=JSON.stringify({source_sha256:metadata.source_sha256,source_hash_verified:metadata.source_hash_verified,spatial_layout:metadata.spatial_layout,provenance:metadata.provenance,source_record:metadata.source_record},null,2);
 $("original-link").href=asset(metadata.original_file);$("original-link").download=metadata.original_file.split("/").pop();$("raw-link").href=asset(metadata.activation_file);$("raw-link").download=metadata.activation_file.split("/").pop();
 $("scale-label").textContent="Scale across this channel/stream: ±"+fmt(unit.absolute_scale);renderOverlays();renderTrajectory();renderScatter();}
function renderOverlays(){if(!imageData||!loadedImage)return;const parent=$("overlays");parent.replaceChildren();const lim=imageData.absolute_scale||1,opacity=Number($("opacity").value)/100;
 imageData.maps.forEach((map,q)=>{const fig=document.createElement("figure");fig.className="map";const canvas=document.createElement("canvas");const maxSide=420,ratio=loadedImage.naturalWidth/loadedImage.naturalHeight;canvas.width=Math.round(ratio>=1?maxSide:maxSide*ratio);canvas.height=Math.round(ratio>=1?maxSide/ratio:maxSide);const ctx=canvas.getContext("2d");ctx.drawImage(loadedImage,0,0,canvas.width,canvas.height);
 const heat=document.createElement("canvas");heat.width=imageData.grid[1];heat.height=imageData.grid[0];const hctx=heat.getContext("2d"),pixels=hctx.createImageData(heat.width,heat.height);for(let y=0;y<heat.height;y++)for(let x=0;x<heat.width;x++){const v=Math.max(-1,Math.min(1,map[y][x]/lim)),s=Math.abs(v),i=(y*heat.width+x)*4;const color=v>=0?[190,43,49]:[36,102,180];for(let c=0;c<3;c++)pixels.data[i+c]=Math.round(255+(color[c]-255)*s);pixels.data[i+3]=Math.round(255*opacity*s);}hctx.putImageData(pixels,0,0);ctx.imageSmoothingEnabled=true;ctx.drawImage(heat,0,0,canvas.width,canvas.height);const caption=document.createElement("figcaption"),strong=document.createElement("strong");strong.textContent="L"+(q+1);caption.append(strong,document.createTextNode("mean "+signed(imageData.means[q])));fig.append(canvas,caption);parent.append(fig);});
}
function axes(canvas,xlabel,ylabel){const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height,p={left:76,right:25,top:28,bottom:54};ctx.clearRect(0,0,w,h);ctx.font="12px system-ui";ctx.strokeStyle="#cbd5df";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p.left,p.top);ctx.lineTo(p.left,h-p.bottom);ctx.lineTo(w-p.right,h-p.bottom);ctx.stroke();ctx.fillStyle="#526272";ctx.textAlign="center";ctx.fillText(xlabel,(w+p.left-p.right)/2,h-11);ctx.save();ctx.translate(17,(h+p.top-p.bottom)/2);ctx.rotate(-Math.PI/2);ctx.fillText(ylabel,0,0);ctx.restore();return{ctx,w,h,p};}
function renderTrajectory(){if(!imageData)return;const {ctx,w,h,p}=axes($("trajectory"),"L invocation","Mean signed response");const values=imageData.means;let lo=Math.min(0,...values),hi=Math.max(0,...values),pad=(hi-lo)*.15||1;lo-=pad;hi+=pad;const x=q=>p.left+q*(w-p.left-p.right)/5,y=v=>h-p.bottom-(v-lo)/(hi-lo)*(h-p.top-p.bottom);ctx.textAlign="right";for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4;ctx.fillText(fmt(v),p.left-9,y(v)+4);ctx.strokeStyle="#edf1f5";ctx.beginPath();ctx.moveTo(p.left,y(v));ctx.lineTo(w-p.right,y(v));ctx.stroke();}ctx.strokeStyle="#a5b3c1";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(p.left,y(0));ctx.lineTo(w-p.right,y(0));ctx.stroke();ctx.setLineDash([]);ctx.strokeStyle="#167da6";ctx.lineWidth=2.4;ctx.beginPath();values.forEach((v,q)=>q?ctx.lineTo(x(q),y(v)):ctx.moveTo(x(q),y(v)));ctx.stroke();values.forEach((v,q)=>{ctx.fillStyle=q===Number($("invocation").value)-1?"#d77534":"#167da6";ctx.beginPath();ctx.arc(x(q),y(v),5,0,Math.PI*2);ctx.fill();ctx.textAlign="center";ctx.fillStyle="#526272";ctx.fillText("L"+(q+1),x(q),h-p.bottom+21);});}
function renderScatter(){if(!unit)return;const compare=Number($("compare").value)-1,a=ranks(unit.means.map(v=>v[0])),b=ranks(unit.means.map(v=>v[compare])),rho=correlation(a,b);$("scatter-caption").textContent="L1 → L"+(compare+1)+" Spearman ρ = "+fmt(rho)+" · "+unit.means.length+" images (average ranks for ties)";const {ctx,w,h,p}=axes($("scatter"),"Image response rank at L1","Image response rank at L"+(compare+1));const n=Math.max(2,a.length),x=v=>p.left+(v-1)/(n-1)*(w-p.left-p.right),y=v=>h-p.bottom-(v-1)/(n-1)*(h-p.top-p.bottom);ctx.strokeStyle="#dce4ea";ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x(1),y(1));ctx.lineTo(x(n),y(n));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#526272";for(const v of [...new Set([1,Math.round(n/2),n])]){ctx.textAlign="center";ctx.fillText(String(v),x(v),h-p.bottom+21);ctx.textAlign="right";ctx.fillText(String(v),p.left-8,y(v)+4);}scatterPoints=a.map((v,i)=>({x:x(v),y:y(b[i]),i}));for(const point of scatterPoints){ctx.fillStyle=point.i===selected?"#d77534":"#197ba699";ctx.beginPath();ctx.arc(point.x,point.y,point.i===selected?6:3.7,0,2*Math.PI);ctx.fill();}const point=scatterPoints[selected];if(point){ctx.strokeStyle="#ad5021";ctx.beginPath();ctx.arc(point.x,point.y,9,0,2*Math.PI);ctx.stroke();}}
async function start(){try{manifest=await api("/api/manifest");$("run-label").textContent=manifest.label;$("kind").textContent=manifest.kind==="historical-shapes"?"HISTORICAL SHAPES":"TRAINING-SOURCE IMAGES";const count=manifest.channel_axes.reduce((n,a)=>n+a.channels.length,0);$("metadata").textContent=manifest.images.length+" images · "+count+" sampled channels · 6 L invocations · checkpoint "+(manifest.checkpoint_revision||"not recorded");const list=document.createElement("ul");for(const text of manifest.limitations){const item=document.createElement("li");item.textContent=text;list.append(item);}$("limitations").append(list);$("provenance").textContent=JSON.stringify({run_id:manifest.run_id,source_protocol_sha256:manifest.source_protocol_sha256,checkpoint_revision:manifest.checkpoint_revision,code_revision:manifest.code_revision,prompt:manifest.prompt,capture_validation:manifest.capture_validation},null,2);$("notice").textContent=manifest.limitations[0]+" Tail ranking and negative values alone do not establish semantic poles.";options($("layer"),manifest.channel_axes.map(a=>a.layer));options($("invocation"),[1,2,3,4,5,6],n=>"L"+n);$("invocation").value="4";options($("compare"),[2,3,4,5,6],n=>"L"+n);$("compare").value="4";options($("image-select"),manifest.images.map((_,i)=>i),i=>manifest.images[i].id);$("layer").onchange=selectLayer;$("channel").onchange=chooseUnit;$("stream").onchange=chooseUnit;$("invocation").onchange=()=>{renderGalleries();renderTrajectory();};$("tail").onchange=renderGalleries;$("compare").onchange=renderScatter;$("image-select").onchange=()=>chooseImage($("image-select").value).catch(error);$("opacity").oninput=renderOverlays;$("scatter").onclick=event=>{const c=$("scatter"),rect=c.getBoundingClientRect(),x=(event.clientX-rect.left)*c.width/rect.width,y=(event.clientY-rect.top)*c.height/rect.height;let nearest=null,best=18;for(const p of scatterPoints){const d=Math.hypot(p.x-x,p.y-y);if(d<best){best=d;nearest=p;}}if(nearest)chooseImage(nearest.i).catch(error);};selectLayer();}catch(e){error(e);}}
start();
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='Create a new portable bundle from complete saved captures')
    export.add_argument('--run-dir', required=True)
    export.add_argument('--dataset-dir', help='Root of original image_path values; defaults to run-dir')
    export.add_argument('--output-dir', required=True)
    export.add_argument('--label')
    export.add_argument('--kind', choices=('training-source', 'historical-shapes'))
    serve = commands.add_parser('serve', help='Verify and serve one exported bundle; never loads a model')
    serve.add_argument('--bundle-dir', required=True)
    serve.add_argument('--host', default='127.0.0.1', help='Default loopback; cluster interface is allowed with mandatory session authentication')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--ready-file', help='Write private connection JSON once (0600); must not already exist')
    args = parser.parse_args(argv)
    if args.command == 'export':
        print(json.dumps(export_bundle(args.run_dir, args.output_dir, args.dataset_dir, args.label, args.kind), indent=2))
    else:
        serve_bundle(args.bundle_dir, args.host, args.port, args.ready_file)


if __name__ == '__main__':
    main()
