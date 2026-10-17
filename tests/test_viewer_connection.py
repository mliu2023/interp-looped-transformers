"""Queue-friendly connection reporting and validation of copyable SSH commands."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('viewer_connection', Path(__file__).resolve().parents[1]/'scripts/viewer_connection.py')
connection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connection)


class ConnectionTests(unittest.TestCase):
    def test_builds_tunnel_and_local_url_from_primitives(self):
        ready = {'compute_ip': '10.2.3.4', 'port': 8765, 'token': 'test_session_token_12345',
                 'local_url': 'https://untrusted.example/ignored'}
        command, url = connection.connection_details(ready, local_port=8766)
        self.assertEqual(command, 'ssh -N -L 8766:10.2.3.4:8765 mliu171@ssh.ccv.brown.edu')
        self.assertEqual(url, 'http://127.0.0.1:8766/?token=test_session_token_12345')
        ready['compute_ip'] = 'fd00::1234'
        self.assertIn('8765:[fd00::1234]:8765', connection.connection_details(ready)[0])

    def test_malformed_ip_port_or_token_cannot_form_shell_command(self):
        valid = {'compute_ip': '10.2.3.4', 'port': 8765, 'token': 'test_session_token_12345'}
        for field, value in (('compute_ip', '10.2.3.4; echo unsafe'), ('compute_ip', 'fd00::1%iface;unsafe'),
                             ('compute_ip', '127.0.0.1'),
                             ('compute_ip', '0.0.0.0'), ('port', '8765'), ('port', True),
                             ('port', 65536), ('token', 'bad-token\nsecond line')):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                connection.connection_details({**valid, field: value})

    def test_reads_pipeline_once_and_reports_queued_or_ready_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIn('not written', '\n'.join(connection.describe(experiment_dir=root)))
            (root/'queued_state.json').write_text(json.dumps({
                'status': 'waiting_for_timing', 'timing_job_id': '123', 'controller_job_id': '122'}))
            queued = '\n'.join(connection.describe(experiment_dir=root))
            self.assertIn('Pipeline CPU job: 122', queued)
            self.assertIn('waiting_for_timing', queued)
            ready = root/'session.json'
            state = {'status': 'submitting_viewer', 'timing_job_id': '123',
                     'full_job_id': '124', 'verification_job_id': '125',
                     'viewer_job_id': '126', 'viewer_connection_file': str(ready)}
            (root/'pipeline_state.json').write_text(json.dumps(state))
            pending = '\n'.join(connection.describe(experiment_dir=root))
            self.assertIn('Viewer CPU job: 126', pending)
            self.assertIn('has not appeared', pending)
            ready.write_text(json.dumps({'compute_ip': '10.2.3.4', 'port': 8765, 'token': 'test_session_token_12345'}))
            complete = '\n'.join(connection.describe(experiment_dir=root))
            self.assertIn('ssh -N -L 8765:10.2.3.4:8765', complete)
            self.assertIn('http://127.0.0.1:8765/?token=', complete)


if __name__ == '__main__':
    unittest.main()
