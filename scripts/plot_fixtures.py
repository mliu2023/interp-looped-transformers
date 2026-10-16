#!/usr/bin/env python3
"""Optional SYNTHETIC engineering figure, recomputed from saved raw arrays."""
import argparse
from pathlib import Path
import os
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from reuseinterp.plotting import plot_fixtures
    plot_fixtures(args.fixture_dir, args.output)


if __name__ == "__main__":
    main()
