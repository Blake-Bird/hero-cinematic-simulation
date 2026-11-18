#!/usr/bin/env bash
#SBATCH --job-name=pair-32
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=pair-32-%j.log
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment

export SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE:-32768}"

total_tasks="${SLURM_NTASKS:-24}"
if [[ -z "${SLURM_JOB_ID:-}" && "$(uname -s)" == "Darwin" ]]; then
    total_tasks="${TASKS_LOCAL:-16}"
fi
tasks_per_model=$(( total_tasks / 2 ))
tasks_per_model="${TASKS_PER_MODEL:-$tasks_per_model}"

echo "=== Starting matched 32^3 pair (LCDM & EDE in parallel, ${tasks_per_model} cores each) ==="

(
    export SLURM_NTASKS="$tasks_per_model"
    export SLURM_CPUS_PER_TASK=1
    run_model lcdm 32
) > "lcdm-32-${SLURM_JOB_ID:-local}.log" 2>&1 &
pid_lcdm=$!

(
    export SLURM_NTASKS="$tasks_per_model"
    export SLURM_CPUS_PER_TASK=1
    run_model ede 32
) > "ede-32-${SLURM_JOB_ID:-local}.log" 2>&1 &
pid_ede=$!

echo "Simulations launched: LCDM (PID $pid_lcdm), EDE (PID $pid_ede)"
wait "$pid_lcdm" || { echo "ERROR: LCDM 32 run failed! Check lcdm-32-${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
wait "$pid_ede" || { echo "ERROR: EDE 32 run failed! Check ede-32-${SLURM_JOB_ID:-local}.log" >&2; exit 1; }

echo "=== Both simulations finished successfully. Running matched pair comparison ==="
python scripts/compare_pair.py --n 32
echo "=== Matched pair comparison complete ==="
