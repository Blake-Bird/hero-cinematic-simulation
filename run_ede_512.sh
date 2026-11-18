#!/usr/bin/env bash
#SBATCH --job-name=ede-512
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=ede-512-%j.log
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment
run_model ede 512
