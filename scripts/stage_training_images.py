#!/usr/bin/env python3
"""Run only in a scheduled CPU job; stage a strictly bounded original-image pool."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reuseinterp.data.images import inspect_training_sources, stage_training_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-download-mib", type=int, default=500)
    parser.add_argument("--max-row-groups-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        print(json.dumps(inspect_training_sources(), indent=2))
        return
    if not args.output:
        parser.error("--output is required for staging")
    manifest = stage_training_images(
        args.output, count=args.count, max_download_bytes=args.max_download_mib * 1024 * 1024,
        max_row_groups_per_source=args.max_row_groups_per_source, seed=args.seed,
        components=tuple(args.components) if args.components else None)
    print(json.dumps({"manifest": str(Path(args.output).resolve() / "manifest.json"),
                      "images": len(manifest["images"]),
                      "bytes_received": manifest["bytes_received"],
                      "dataset_fingerprint": manifest["dataset_fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
