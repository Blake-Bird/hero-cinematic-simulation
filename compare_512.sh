#!/usr/bin/env bash
#SBATCH --job-name=compare-512
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=compare-512-%j.log
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment
python scripts/compare_pair.py --n 512
