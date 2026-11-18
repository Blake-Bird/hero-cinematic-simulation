#!/usr/bin/env bash
#SBATCH --job-name=bench-512
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=bench-512-%j.log
# ------------------------------------------------------------------------------
# 512^3 Resource & Performance Benchmark
# Runs 20 initial timesteps (z=99 -> z=90) to measure peak RAM, MPI communication
# scaling, and snapshot I/O rate before committing to the 48-hour production run.
# ------------------------------------------------------------------------------
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
export CAMPAIGN="${CAMPAIGN:-bench512}"
export BENCHMARK=1
export STAGE=sim
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment

echo "======================================================================"
echo " Starting 512^3 Scaling & Performance Benchmark"
echo " Cores: 64 physical ranks | RAM Requested: 400G | Mode: N-body sim only"
echo "======================================================================"

run_model lcdm 512

echo "======================================================================"
echo " Benchmark completed successfully!"
echo " Inspect timing & memory report: $RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n512/runs/lcdm/timing_report.json"
echo "======================================================================"

