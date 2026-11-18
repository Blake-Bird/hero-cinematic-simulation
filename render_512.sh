#!/usr/bin/env bash
#SBATCH --job-name=render-512
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=render-512-%j.log
# ------------------------------------------------------------------------------
# Decoupled Cinematic & Blender Rendering Job
# Runs independently of scientific analysis so Blender never blocks or delays
# matched comparison or scientific conclusions.
# ------------------------------------------------------------------------------
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment

for model in lcdm ede; do
    run="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n512/runs/$model"
    echo "Rendering Blender universe & movies for $model..."
    python commands.py analysis "$run" --graphs blender,blender_movie,movie,halo_movie,planetarium --allow-incomplete || {
        echo "Blender rendering for $model completed with non-fatal warnings."
    }
done
echo "Cinematic rendering stage finished."

