"""Small tests of raw-backed plotting data, export identity and safe HTTP access."""
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import build_opener, HTTPCookieProcessor, Request, urlopen

import numpy as np
from PIL import Image

from reuseinterp.viewer import Bundle, STREAMS, _hash, export_bundle, handler_for


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(root):
    """Variable grids and unequal channel counts catch accidental fixed axes."""
    run, dataset = root/'run', root/'data'
    run.mkdir()
    dataset.mkdir()
    channels = {'0': [9], '7': [5, 2], '15': [11]}
    # Sort JSON keys deliberately; the recorded list remains 0,7,15.
    write(run/'protocol.json', {'specification': {'channels': channels, 'prompt': 'Describe the image.'}})
    write(run/'capture_validation.json', {'passed': True, 'layers': [0, 7, 15]})
    write(run/'summary.json', {'checkpoint_revision': 'test-checkpoint'})
    records, rows, values = [], [], []
    for i, grid in enumerate(([2, 3], [3, 1])):
        original = dataset/f'photo{i}.jpg'
        exif = Image.Exif()
        exif[274] = 6  # Display width/height swap after native EXIF transpose.
        Image.new('RGB', (8, 4), (10+i, 100, 80)).save(original, exif=exif)
        raw = {}
        for layer, selected in channels.items():
            for si, stream in enumerate(STREAMS):
                array = np.arange(6*np.prod(grid)*len(selected), dtype=np.float32).reshape(6, np.prod(grid), len(selected))
                raw[f'layer_{layer}__{stream}'] = array/10 + int(layer) - 6 + i*3 + si*4
        target = run/f'raw{i}.npz'
        np.savez_compressed(target, **raw)
        values.append(raw)
        rows.append({'image_id': f'photo{i}', 'family_id': f'family{i}',
                     'image_path': original.name, 'image_sha256': _hash(original),
                     'provenance': {'dataset_id': 'fixture-images', 'training_membership': 'reported_corpus_member'}})
        records.append({'image_id': f'photo{i}', 'condition': 'primary', 'artifact': target.name,
                        'sha256': _hash(target), 'grid': grid,
                        'spatial_layout': {'kind': 'native_resize_row_major', 'merge_size': 1,
                                           'no_padding': True, 'no_crop': True, 'exif_transposed': True}})
    write(run/'image_rows.json', rows)
    write(run/'capture_index.json', {'complete': True, 'records': records})
    return run, dataset, values


class ExportTests(unittest.TestCase):
    def test_native_variable_grids_and_all_streams_follow_physical_axes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, dataset, raw = fixture(root)
            result = export_bundle(run, root/'bundle', dataset)
            self.assertEqual(result['images'], 2)
            bundle = Bundle(root/'bundle')
            self.assertEqual(bundle.units, [(0, 9), (7, 5), (7, 2), (15, 11)])
            for stream in STREAMS:
                got = bundle.unit(7, 2, stream)
                expected = np.stack([r[f'layer_7__{stream}'][:, :, 1].mean(1) for r in raw])
                np.testing.assert_allclose(got['means'], expected, rtol=0, atol=0)
                self.assertAlmostEqual(got['absolute_scale'], max(float(np.abs(r[f'layer_7__{stream}'][:, :, 1]).max()) for r in raw))
                image = bundle.image(1, 7, 2, stream)
                np.testing.assert_array_equal(np.asarray(image['maps']), raw[1][f'layer_7__{stream}'][:, :, 1].reshape(6, 3, 1))
            self.assertEqual(bundle.images[0]['oriented_image_size'], [4, 8])
            self.assertEqual(_hash(bundle.file(bundle.images[0]['original_file'])), _hash(dataset/'photo0.jpg'))
            self.assertTrue(bundle.images[0]['source_hash_verified'])
            self.assertIn('does not prove', bundle.manifest['limitations'][0])
            with self.assertRaises(FileExistsError):
                export_bundle(run, root/'bundle', dataset)

    def test_export_rejects_changed_source_image_and_misaligned_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, dataset, _ = fixture(root)
            Image.new('RGB', (8, 4), 'red').save(dataset/'photo0.jpg')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                export_bundle(run, root/'bad-image', dataset)
            index = json.loads((run/'capture_index.json').read_text())
            index['records'][0]['grid'] = [3, 3]
            write(run/'capture_index.json', index)
            with self.assertRaisesRegex(ValueError, 'misaligned'):
                export_bundle(run, root/'bad-grid', dataset)

    def test_only_manifest_files_and_unchanged_artifacts_are_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, dataset, _ = fixture(root)
            export_bundle(run, root/'bundle', dataset)
            bundle = Bundle(root/'bundle')
            for name in ('../protocol.json', '/etc/passwd', 'manifest.json', 'files/not-listed'):
                with self.assertRaises(KeyError):
                    bundle.file(name)
            selected = bundle.images[0]['activation_file']
            bundle.file(selected).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed bundle'):
                Bundle(root/'bundle')

    def test_manifest_symlinks_and_source_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, dataset, _ = fixture(root)
            export_bundle(run, root/'bundle', dataset)
            target = root/'bundle/files/preview_0000.jpg'
            target.unlink()
            target.symlink_to(dataset/'photo0.jpg')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                Bundle(root/'bundle')
            rows = json.loads((run/'image_rows.json').read_text())
            rows[0]['image_path'] = '../run/protocol.json'
            write(run/'image_rows.json', rows)
            with self.assertRaisesRegex(ValueError, 'source root'):
                export_bundle(run, root/'bad-source', dataset)


class HTTPTests(unittest.TestCase):
    def test_token_cookie_data_and_path_restrictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, dataset, _ = fixture(root)
            export_bundle(run, root/'bundle', dataset)
            bundle = Bundle(root/'bundle')
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(bundle, 'test-session-secret'))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            opener = build_opener(HTTPCookieProcessor(CookieJar()))
            try:
                with self.assertRaises(HTTPError) as denied:
                    urlopen(base+'/api/manifest')
                self.assertEqual(denied.exception.code, 403)
                with opener.open(base+'/?token=test-session-secret') as response:
                    self.assertEqual(response.geturl(), base+'/')
                    self.assertIn(b'LoopVL activity explorer', response.read())
                    self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
                with opener.open(base+'/api/unit?layer=7&channel=2&stream=value') as response:
                    self.assertEqual(len(json.load(response)['means']), 2)
                with opener.open(base+'/api/image?image=1&layer=7&channel=2&stream=product') as response:
                    self.assertEqual(json.load(response)['grid'], [3, 1])
                for path in ('/file/../manifest.json', '/file/%2e%2e/manifest.json', '/file//etc/passwd',
                             '/api/image?image=-1&layer=7&channel=2&stream=product',
                             '/api/unit?layer=7&channel=9999&stream=value'):
                    with self.assertRaises(HTTPError) as invalid:
                        opener.open(base+path)
                    self.assertEqual(invalid.exception.code, 400)
                request = Request(base+'/api/manifest', headers={'Authorization': 'Bearer test-session-secret'})
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['schema_version'], 1)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
