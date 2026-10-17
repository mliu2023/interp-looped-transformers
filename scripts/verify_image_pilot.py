#!/usr/bin/env python3
"""Recompute a real-image pilot from its original image bytes and raw captures."""
import argparse
import json

from reuseinterp.image_pilot import verify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2))
