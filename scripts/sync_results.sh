#!/usr/bin/env bash
# Synchronize lightweight scientific results and deliverables (reports, figures, videos)
# to remote destination (SharePoint / workstation) while keeping heavy multi-terabyte raw snapshots on cluster.
set -euo pipefail

DESTINATION="${1:-}"
if [[ -z "$DESTINATION" ]]; then
    echo "Usage: bash scripts/sync_results.sh <user@remote-host:/path/to/sync_dir>"
    echo "Example: bash scripts/sync_results.sh bbird@my-macbook:~/ede-campaign-exports"
    exit 1
fi

STATE_ROOT="${RESEARCH2_STATE_ROOT:-$HOME/ede-runtime}"
CAMPAIGN="${CAMPAIGN:-matched40}"
SOURCE_DIR="$STATE_ROOT/campaigns/$CAMPAIGN"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: Campaign directory $SOURCE_DIR not found!" >&2
    exit 1
fi

echo "=== Packaging and syncing scientific deliverables for campaign: $CAMPAIGN ==="
echo "Excluding raw particle snapshots and temporary meshes..."

rsync -avz --progress \
    --exclude="data/snapshots/*" \
    --exclude="*.raw" \
    --exclude="*.tmp" \
    --exclude="restart.*" \
    --include="*/" \
    --include="*.json" \
    --include="*.csv" \
    --include="*.html" \
    --include="*.png" \
    --include="*.pdf" \
    --include="*.svg" \
    --include="*.mp4" \
    --include="*.blend" \
    --include="*.npz" \
    --exclude="*" \
    "$SOURCE_DIR/" "$DESTINATION/"

echo "=== Sync complete! Deliverables safely mirrored to $DESTINATION ==="

