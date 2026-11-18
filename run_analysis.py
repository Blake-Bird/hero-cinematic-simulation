#!/usr/bin/env python3
"""Analyze saved snapshots and export explicitly selected scientific figures.

CLI entrypoint for post-simulation scientific processing, halo finding,
HMF calculations, matter power spectra, and visual exports.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

if __name__ == "__main__":
    from commands import run_analysis_main
    sys.exit(run_analysis_main() or 0)

