"""CPU failures for corrupted, incomplete, and leaking execution state."""
import tempfile
from pathlib import Path
import unittest
from reuseinterp.io import atomic_json, read_json
from reuseinterp.shards import FamilyShard, merge_shards, shard_for, StopRequested
from reuseinterp.protocol import create_lock, verify_lock
from reuseinterp.schemas import BehavioralMeasurement, validate_rows


class IntegrityTests(unittest.TestCase):
    def test_interrupted_resume_equals_uninterrupted(self):
        with tempfile.TemporaryDirectory() as d:
            families = ["a", "b", "c", "d", "e"]
            root = Path(d)
            for index in range(2):
                with FamilyShard(root, {"config_hash": "frozen"}, families, index, 2) as shard:
                    pending = list(shard.pending())
                    if pending:
                        shard.commit(pending[0], [{"example_id": pending[0], "condition": "clean"}])
                with self.assertRaises(FileNotFoundError):
                    merge_shards(root, families, 2)
                with FamilyShard(root, {"config_hash": "frozen"}, families, index, 2, resume=True) as shard:
                    for family in shard.pending():
                        shard.commit(family, [{"example_id": family, "condition": "clean"}])
                    shard.finish()
            rows = merge_shards(root, families, 2, ["clean"])
            self.assertEqual(sorted(r["example_id"] for r in rows), families)

    def test_concurrent_writer_refused_and_lock_released(self):
        with tempfile.TemporaryDirectory() as d:
            with FamilyShard(d, {}, ["a"], 0, 1):
                with self.assertRaises(FileExistsError):
                    with FamilyShard(d, {}, ["a"], 0, 1):
                        pass
            self.assertFalse((Path(d) / "shard-0000/writer.lock").exists())

    def test_config_and_topology_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with FamilyShard(d, {"config_hash": "first"}, ["a"], 0, 1):
                pass
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                with FamilyShard(d, {"config_hash": "changed"}, ["a"], 0, 1, resume=True):
                    pass

    def test_missing_condition_prevents_merge(self):
        with tempfile.TemporaryDirectory() as d:
            with FamilyShard(d, {}, ["a"], 0, 1) as shard:
                shard.commit("a", [{"example_id": "a", "condition": "plus"}])
                shard.finish()
            with self.assertRaisesRegex(ValueError, "condition"):
                merge_shards(d, ["a"], 1, ["plus", "minus"])

    def test_corrupted_committed_result_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            with FamilyShard(d, {}, ["a"], 0, 1) as shard:
                shard.commit("a", [{"value": 1}])
            record = next(p for p in (Path(d) / "shard-0000").glob("*.json") if p.name != "manifest.json")
            value = read_json(record)
            value["rows"][0]["value"] = 2
            atomic_json(record, value)
            with FamilyShard(d, {}, ["a"], 0, 1, resume=True) as shard:
                with self.assertRaisesRegex(ValueError, "checksum"):
                    list(shard.pending())

    def test_signal_handler_only_requests_stop(self):
        import signal
        with StopRequested() as stop:
            signal.raise_signal(signal.SIGUSR1)
            self.assertTrue(stop.requested)

    def test_draft_cannot_be_locked(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "gate"):
                create_lock(Path(d) / "protocol.json", {"status": "DRAFT"}, {})

    def test_invalid_lock_tampering_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "protocol.json"
            atomic_json(p, {"status": "locked", "protocol": {}, "protocol_hash": "wrong"})
            with self.assertRaisesRegex(ValueError, "tampered"):
                verify_lock(p)

    def test_schema_rejects_duplicate_or_nonfinite_scores(self):
        row = BehavioralMeasurement("run", "config", "checkpoint", "pilot", "family", "example", "counting", "clean", "1", {"1": 0.0}, "1", True).validate()
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_rows([row, row])
        row["candidate_scores"]["1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "scores"):
            validate_rows([row])

    def test_study_identity_ignores_paths_but_rejects_changed_science(self):
        from reuseinterp.io import study_fingerprint, config_load
        import json
        config = json.loads((Path(__file__).parents[1] / "configs/smoke.yaml").read_text())
        original = study_fingerprint(config)
        config["dataset_dir"] = "/cluster/data"
        self.assertEqual(study_fingerprint(config), original)
        config["state_layers"] = [0, 5, 10]
        self.assertNotEqual(study_fingerprint(config), original)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "changed.json"
            atomic_json(p, config)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                config_load(p)

    def test_incomplete_controls_cannot_lock(self):
        from reuseinterp.protocol import REQUIRED_FIELDS
        protocol = {name: {} for name in REQUIRED_FIELDS}
        protocol["confirmation_opened"] = False
        gates = {gate: {"passed": True} for gate in ("parity", "pilot", "discovery")}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Mandatory controls"):
                create_lock(Path(d) / "lock.json", protocol, gates)


if __name__ == "__main__":
    unittest.main()
