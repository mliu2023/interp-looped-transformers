"""Family-atomic resumable shards. A process owns exactly one directory."""
import os
import signal
from pathlib import Path
from .io import atomic_json, digest, read_json, file_hash


def shard_for(family_id, num_shards):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    return int(digest(str(family_id))[:16], 16) % num_shards


class StopRequested:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def handler(signum, frame):
            self.requested = True
        for name in ("SIGUSR1", "SIGTERM"):
            if hasattr(signal, name):
                sig = getattr(signal, name)
                self.previous[sig] = signal.signal(sig, handler)
        return self

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


class FamilyShard:
    def __init__(self, root, identity, families, index, num_shards, resume=False):
        if not 0 <= index < num_shards:
            raise ValueError("shard index outside original num_shards")
        self.path = Path(root) / ("shard-%04d" % index)
        self.path.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity, num_shards=num_shards, shard_index=index)
        self.families = sorted(f for f in families if shard_for(f, num_shards) == index)
        self.resume = resume
        self.fd = None

    def __enter__(self):
        lock = self.path / "writer.lock"
        self.fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(self.fd, str(os.getpid()).encode())
        try:
            manifest = self.path / "manifest.json"
            wanted = {"identity": self.identity, "families": self.families}
            if manifest.exists():
                if read_json(manifest) != wanted:
                    raise ValueError("Incompatible shard manifest")
                if not self.resume:
                    raise ValueError("Existing shard requires --resume")
            else:
                atomic_json(manifest, wanted)
            return self
        except BaseException:
            self.__exit__()
            raise

    def pending(self):
        for family in self.families:
            record = self.path / (digest(family) + ".json")
            if record.exists():
                value = read_json(record)
                if value["family_id"] != family or value["identity"] != self.identity:
                    raise ValueError("Invalid committed family identity")
                if digest(value["rows"]) != value["rows_hash"]:
                    raise ValueError("Committed family checksum mismatch")
            else:
                yield family

    def commit(self, family, rows, metadata=None):
        if family not in self.families or not rows:
            raise ValueError("Empty or out-of-shard family")
        record = {
            "family_id": family, "identity": self.identity,
            "rows": rows, "rows_hash": digest(rows)}
        if metadata is not None:
            record["metadata"] = metadata
        atomic_json(self.path / (digest(family) + ".json"), record)

    def finish(self):
        if list(self.pending()):
            raise ValueError("Cannot complete a partial shard")
        atomic_json(self.path / "COMPLETE.json", {
            "identity": self.identity, "family_count": len(self.families),
            "files": {p.name: file_hash(p) for p in sorted(self.path.glob("*.json"))
                      if p.name != "COMPLETE.json"}})

    def __exit__(self, *args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            (self.path / "writer.lock").unlink()


def merge_shards(root, families, num_shards, required_conditions=None):
    rows, seen, shared_identity = [], set(), None
    for index in range(num_shards):
        directory = Path(root) / ("shard-%04d" % index)
        complete = read_json(directory / "COMPLETE.json")
        identity = complete["identity"]
        if identity["shard_index"] != index or identity["num_shards"] != num_shards:
            raise ValueError("Wrong shard topology")
        common = {k: v for k, v in identity.items() if k != "shard_index"}
        if shared_identity is not None and shared_identity != common:
            raise ValueError("Incompatible manifests")
        shared_identity = common
        for name, checksum in complete["files"].items():
            if file_hash(directory / name) != checksum:
                raise ValueError("Shard checksum mismatch")
        expected = sorted(f for f in families if shard_for(f, num_shards) == index)
        if read_json(directory / "manifest.json")["families"] != expected:
            raise ValueError("Family coverage mismatch")
        for family in expected:
            record = read_json(directory / (digest(family) + ".json"))
            if record["identity"] != identity or record["family_id"] != family:
                raise ValueError("Family identity mismatch")
            if digest(record["rows"]) != record["rows_hash"] or family in seen:
                raise ValueError("Duplicate or corrupt family")
            if required_conditions:
                examples = {r["example_id"] for r in record["rows"]}
                for example in examples:
                    conditions = [r["condition"] for r in record["rows"] if r["example_id"] == example]
                    if len(conditions) != len(set(conditions)) or set(conditions) != set(required_conditions):
                        raise ValueError("Missing/duplicated condition rows")
            seen.add(family)
            rows.extend(record["rows"])
    if seen != set(families):
        raise ValueError("Missing families")
    return rows
