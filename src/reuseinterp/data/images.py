"""Bounded, provenance-preserving access to LoopVL's reported image sources.

Public corpus membership is not proof that a particular record survived the
authors' filtering or was consumed by this exact checkpoint. The manifest keeps
that distinction explicit. Image selection never uses model activations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random
import re
import urllib.request

from PIL import Image


SCHEMA_VERSION = "reuseinterp.training_images.v1"
DATASET_ID = "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data"
DATASET_REVISION = "0efb7ad47f8ba36e1d262545cadec4cd8960dc5f"
CHECKPOINT_REVISION = "92aa28bc678f523f8641fdb2f9e5159f614d1c02"
PAPER_URL = "https://arxiv.org/html/2609.38426#S4.SS1"

# Selected before inference: natural-photograph components of the paper's SFT
# source, rather than diagrams, CLEVR renders, documents, or arbitrary web photos.
# Metadata comes from the immutable Hugging Face tree API at DATASET_REVISION.
SOURCES = (
    {"config": "coco", "shard": "coco/train-00000-of-00032.parquet",
     "size": 493699808, "lfs_sha256": "ca210ea40b4c9eadc5e0ee994e65e838ade52c6998346ab633ef82fb5e3d3059"},
    {"config": "gqa", "shard": "gqa/train-00000-of-00020.parquet",
     "size": 505627740, "lfs_sha256": "0b373375474f246660fa0a5f1199f2fb5bc78064538e86da9c2a81ef76234b17"},
    {"config": "vg", "shard": "vg/train-00000-of-00059.parquet",
     "size": 516099324, "lfs_sha256": "4f8a439cf9c5fe26986d8ca1166a22bd2e0978b67906de80814e31806a64c35b"},
)

INSPECTION_SOURCES = SOURCES + (
    {"config": "aokvqa", "shard": "aokvqa/train-00000-of-00019.parquet",
     "size": 504381640, "lfs_sha256": "4ee7a6a86916148b18a84e81c3d06e4f7056efa007efe7491add2118ea309673"},
    {"config": "VizWiz", "shard": "VizWiz/train-00000-of-00075.parquet",
     "size": 506108675, "lfs_sha256": "f1304332c4318bcf79ba7d6c2bbffd68e095405d87d7239a57a19b70f72ddb5b"},
    {"config": "allava", "shard": "allava/train-00000-of-00082.parquet",
     "size": 486230014, "lfs_sha256": "fb3b3431513fb83701f17607f15c86304b51902f929bd66e0dc706c625f80383"},
)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass
class DownloadBudget:
    max_bytes: int
    bytes_received: int = 0
    requests: int = 0

    def require(self, count: int) -> None:
        if count < 0 or self.bytes_received + count > self.max_bytes:
            raise ValueError("Training-image network byte cap would be exceeded")


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP reader that refuses full-file fallbacks and oversized reads.

    PyArrow can read the footer and selected row groups through this reader. No
    Hugging Face cache or background prefetch can silently fetch whole shards.
    """

    def __init__(self, url: str, size: int, budget: DownloadBudget, *, opener=None):
        super().__init__()
        self.url, self.size, self.budget = url, int(size), budget
        self._position = 0
        self._opener = opener or urllib.request.urlopen
        self._cache_start, self._cache = 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (offset if whence == io.SEEK_SET else
                    self._position + offset if whence == io.SEEK_CUR else
                    self.size + offset if whence == io.SEEK_END else None)
        if position is None or position < 0:
            raise ValueError("Invalid seek")
        self._position = int(position)
        return self._position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed range reader")
        count = max(0, self.size - self._position)
        if size is not None and size >= 0:
            count = min(count, int(size))
        if not count:
            return b""
        start, end = self._position, self._position + count - 1
        if self._cache_start <= start and end < self._cache_start + len(self._cache):
            offset = start - self._cache_start
            self._position += count
            return self._cache[offset:offset + count]
        self.budget.require(count)
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}",
                               "Accept-Encoding": "identity",
                               "User-Agent": "reuseinterp-training-image-pilot/1"})
        self.budget.requests += 1
        with self._opener(request, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server ignored HTTP Range; refusing whole-shard download")
            expected = f"bytes {start}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("HTTP Content-Range does not match pinned shard size/offset")
            payload = response.read(count)
            self.budget.bytes_received += len(payload)
            if len(payload) != count:
                raise ValueError("Incomplete HTTP Range response")
        self._position += count
        # A bounded single-block cache covers repeated footer/metadata reads;
        # large image-column reads are never retained in an unbounded cache.
        if len(payload) <= 2 * 1024 * 1024:
            self._cache_start, self._cache = start, payload
        return payload

    def readinto(self, buffer):
        payload = self.read(len(buffer))
        buffer[:len(payload)] = payload
        return len(payload)


def image_fingerprints(payload: bytes) -> dict:
    """Hash original bytes and decoded RGB; use dHash for conservative deduping."""
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(payload)) as image:
        width, height, image_format = image.width, image.height, image.format
        if width < 32 or height < 32 or width * height > 40_000_000:
            raise ValueError("Image dimensions outside bounded pilot limits")
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(
            f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
        thumb = list(rgb.convert("L").resize((9, 8)).getdata())
        dhash = 0
        for row in range(8):
            for column in range(8):
                dhash = (dhash << 1) | int(thumb[row * 9 + column] > thumb[row * 9 + column + 1])
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
    if not extension:
        raise ValueError(f"Unsupported original image format: {image_format}")
    return {"image_sha256": hashlib.sha256(payload).hexdigest(),
            "pixel_sha256": pixel_hash, "dhash": f"{dhash:016x}",
            "width": width, "height": height, "extension": extension,
            "image_format": image_format}


def first_instruction(row: dict) -> str | None:
    for turn in row.get("conversations") or []:
        if (turn.get("role") or turn.get("from")) in ("user", "human"):
            value = turn.get("content") or turn.get("value")
            if isinstance(value, str):
                return value.replace("<image>", "").strip()
    return None


def _source_url(source: dict) -> str:
    return (f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
            f"{DATASET_REVISION}/{source['shard']}")


def inspect_training_sources() -> dict:
    """Read only small Parquet footers to estimate bounded image-column costs."""
    import pyarrow.parquet as pq
    budget = DownloadBudget(16 * 1024 * 1024)
    rows = []
    for source in INSPECTION_SOURCES:
        with HTTPRangeReader(_source_url(source), source["size"], budget) as reader:
            parquet = pq.ParquetFile(reader, pre_buffer=False)
            groups = []
            for index in range(min(2, parquet.num_row_groups)):
                group = parquet.metadata.row_group(index)
                groups.append({"row_group": index, "rows": group.num_rows,
                               "compressed_bytes": sum(group.column(i).total_compressed_size
                                                       for i in range(group.num_columns))})
            rows.append({**source, "row_groups": parquet.num_row_groups,
                         "rows": parquet.metadata.num_rows, "first_groups": groups})
    return {"sources": rows, "metadata_bytes_received": budget.bytes_received,
            "metadata_requests": budget.requests}


def stage_training_images(output: str | Path, *, count: int = 64,
                          max_download_bytes: int = 500 * 1024 * 1024,
                          max_row_groups_per_source: int = 2, seed: int = 20261017,
                          components: tuple[str, ...] | None = None) -> dict:
    """Stage an equal-stratum, deterministic convenience sample on a CPU node.

    Requires optional pyarrow. A seeded shuffle selects distinct families within
    bounded first-shard row groups; this is not a representative mixture estimate.
    An incomplete attempt deliberately has no completed manifest.json.
    """
    import pyarrow.parquet as pq

    sources = SOURCES
    if components:
        source_by_name = {source["config"]: source for source in INSPECTION_SOURCES}
        if len(set(components)) != len(components) or any(name not in source_by_name for name in components):
            raise ValueError("Components must be distinct audited source names")
        sources = tuple(source_by_name[name] for name in components)
    if count < len(sources) or max_row_groups_per_source < 1 or max_download_bytes < 1:
        raise ValueError("count must cover source strata and bounds must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Staging output must be empty; use a new immutable dataset directory")
    (output / "images").mkdir(parents=True)
    (output / "metadata").mkdir()
    budget = DownloadBudget(max_download_bytes)
    images, seen_pixels, seen_dhash, audit = [], set(), [], []
    try:
        for source_index, source in enumerate(sources):
            quota = count // len(sources) + int(source_index < count % len(sources))
            selected, skipped, row_index, groups_read = 0, [], 0, []
            source_start_bytes = budget.bytes_received
            reader = HTTPRangeReader(_source_url(source), source["size"], budget)
            with reader:
                parquet = pq.ParquetFile(reader, pre_buffer=False)
                schema = parquet.schema_arrow.names
                if "image" not in schema:
                    raise ValueError("Pinned source lacks embedded image column")
                columns = [name for name in ("id", "image", "conversations", "data_source") if name in schema]
                for group_index in range(min(parquet.num_row_groups, max_row_groups_per_source)):
                    group_metadata = parquet.metadata.row_group(group_index)
                    groups_read.append({"index": group_index, "rows": group_metadata.num_rows,
                                        "compressed_bytes": sum(group_metadata.column(i).total_compressed_size
                                                                for i in range(group_metadata.num_columns))})
                    table = parquet.read_row_group(group_index, columns=columns, use_threads=False)
                    group_rows = table.to_pylist()
                    indices = list(range(len(group_rows)))
                    random.Random(f"{seed}:{source['config']}:{group_index}").shuffle(indices)
                    group_start = sum(parquet.metadata.row_group(i).num_rows for i in range(group_index))
                    for index in indices:
                        row = group_rows[index]
                        current_index, row_index = group_start + index, row_index + 1
                        embedded = row.get("image")
                        payload = embedded.get("bytes") if isinstance(embedded, dict) else None
                        if not isinstance(payload, bytes):
                            skipped.append({"record_index": current_index, "reason": "no_embedded_image_bytes"})
                            continue
                        try:
                            identity = image_fingerprints(payload)
                        except (ValueError, OSError) as error:
                            skipped.append({"record_index": current_index, "reason": str(error)})
                            continue
                        dhash = int(identity["dhash"], 16)
                        if identity["pixel_sha256"] in seen_pixels or any(
                                bin(dhash ^ previous).count("1") <= 3 for previous in seen_dhash):
                            skipped.append({"record_index": current_index, "reason": "duplicate_or_near_duplicate"})
                            continue
                        image_id = f"{source['config'].lower()}_{current_index:07d}_{identity['image_sha256'][:12]}"
                        relative_path = f"images/{image_id}{identity['extension']}"
                        (output / relative_path).write_bytes(payload)
                        metadata = {key: value for key, value in row.items() if key != "image"}
                        metadata.update({"original_image_path": embedded.get("path"),
                                         "image_sha256": identity["image_sha256"]})
                        metadata_payload = canonical_json(metadata)
                        (output / "metadata" / f"{image_id}.json").write_bytes(metadata_payload + b"\n")
                        provenance = {
                            "dataset_id": DATASET_ID, "revision": DATASET_REVISION,
                            "config": source["config"], "split": "train", "shard": source["shard"],
                            "record_index": current_index, "source_record_id": str(row.get("id", "")),
                            "source_data_source": row.get("data_source"),
                            "shard_size": source["size"], "shard_lfs_sha256": source["lfs_sha256"],
                            "shard_full_hash_verified": False,
                            "source_record_metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
                            "training_membership": "reported_corpus_member",
                            "source_stage": "supervised_fine_tuning",
                            "checkpoint_revision": CHECKPOINT_REVISION,
                            "checkpoint_retained_record_membership_verified": False,
                            "evidence_urls": [PAPER_URL,
                                f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}/{source['config']}"]}
                        image_row = {"image_id": image_id, "scene_id": image_id,
                                     "family_id": f"original_{identity['pixel_sha256'][:24]}",
                                     "image_path": relative_path, "provenance": provenance,
                                     **{k: v for k, v in identity.items() if k != "extension"}}
                        instruction = first_instruction(row)
                        if instruction:
                            image_row["original_instruction"] = instruction
                        images.append(image_row)
                        seen_pixels.add(identity["pixel_sha256"])
                        seen_dhash.append(dhash)
                        selected += 1
                        if selected == quota:
                            break
                    if selected == quota:
                        break
            audit.append({**source, "selected": selected, "rows_examined": row_index,
                          "bytes_received": budget.bytes_received - source_start_bytes,
                          "groups_read": groups_read, "skipped": skipped})
            if selected != quota:
                raise ValueError(f"Insufficient distinct families in bounded {source['config']} row groups")
    except Exception as error:
        (output / "incomplete_audit.json").write_bytes(canonical_json({
            "complete": False, "error": str(error), "selected": len(images),
            "bytes_received": budget.bytes_received, "requests": budget.requests,
            "sources": audit}) + b"\n")
        raise
    manifest = {"schema_version": SCHEMA_VERSION,
                "dataset_name": f"loopvl_reported_training_photos_{count}",
                "complete": True, "images": images,
                "sampling": {"selection": "seeded_shuffle_of_bounded_pinned_first_shard_row_groups",
                             "seed": seed,
                             "source_strata": [s["config"] for s in sources],
                             "count_per_stratum": {s["config"]: count // len(sources) + int(i < count % len(sources))
                                                   for i, s in enumerate(sources)},
                             "activation_independent": True,
                             "representative_training_mixture": False,
                             "duplicate_rule": "same_decoded_RGB_or_64bit_dhash_hamming_at_most_3"},
                "membership_limitation": "The paper reports this SFT corpus; no authors' retained-record manifest or exact checkpoint stage lineage is available. These are original public training-source records, not verified optimizer-consumed records.",
                "bytes_received": budget.bytes_received, "network_requests": budget.requests,
                "max_download_bytes": max_download_bytes, "source_audit": audit}
    manifest["dataset_fingerprint"] = hashlib.sha256(canonical_json(manifest)).hexdigest()
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def load_image_manifest(path: str | Path) -> dict:
    """Validate native originals before capture and resolve manifest-relative paths."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get("complete"):
        raise ValueError("Not a complete supported training-image manifest")
    stored_fingerprint = manifest.get("dataset_fingerprint")
    fingerprint_payload = {key: value for key, value in manifest.items() if key != "dataset_fingerprint"}
    if stored_fingerprint != hashlib.sha256(canonical_json(fingerprint_payload)).hexdigest():
        raise ValueError("Training-image manifest fingerprint mismatch")
    if not manifest.get("images"):
        raise ValueError("Training-image manifest is empty")
    seen_ids, seen_families, seen_pixels = set(), set(), set()
    for row in manifest["images"]:
        image_path = Path(row["image_path"])
        image_path = image_path if image_path.is_absolute() else path.parent / image_path
        image_path = image_path.resolve()
        if path.parent not in image_path.parents:
            raise ValueError("Training image path leaves manifest directory")
        identity = image_fingerprints(image_path.read_bytes())
        for key in ("image_sha256", "pixel_sha256", "width", "height"):
            if row[key] != identity[key]:
                raise ValueError(f"Training image integrity mismatch: {key}")
        if row["image_id"] in seen_ids or row["family_id"] in seen_families or row["pixel_sha256"] in seen_pixels:
            raise ValueError("Training-image manifest repeats an image family")
        provenance = row.get("provenance", {})
        if provenance.get("training_membership") != "reported_corpus_member":
            raise ValueError("Unsupported or overstated training membership")
        if not re.fullmatch(r"[a-f0-9]{40}", provenance.get("revision", "")):
            raise ValueError("Training source must have an immutable revision")
        if (provenance.get("dataset_id") != DATASET_ID or
                provenance.get("revision") != DATASET_REVISION or
                provenance.get("checkpoint_revision") != CHECKPOINT_REVISION):
            raise ValueError("Training source or checkpoint differs from audited pins")
        seen_ids.add(row["image_id"])
        seen_families.add(row["family_id"])
        seen_pixels.add(row["pixel_sha256"])
        row["image_path"] = str(image_path)
    return manifest
