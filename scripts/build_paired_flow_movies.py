#!/usr/bin/env python3
"""Publish the missing animated peculiar-flow views at the two principal epochs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.flow_movie import make_paired_flow_movies

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("atlas", type=Path)
args = parser.parse_args()
for snapshot, redshift in ((21, 14.4), (85, 0.)):
    print(make_paired_flow_movies(args.atlas, snapshot, redshift), flush=True)
