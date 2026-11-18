#!/usr/bin/env python3
"""Re-seal selected native analysis suites against the corrected implementation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.analysis import analyze_selected_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', type=Path)
    parser.add_argument('model', choices=('EDE', 'LCDM'))
    args = parser.parse_args()
    workspace = args.atlas.resolve() / 'Source/Native_workspaces' / args.model
    prior = sorted(workspace.glob('results/analysis_products_*/analysis_products_summary.json'))[-1]
    summary = json.loads(prior.read_text())
    analyze_selected_products(workspace, summary['products'], include_pdf_report=False,
                              snapshots=summary['selected_snapshot_ids'])


if __name__ == '__main__':
    main()
