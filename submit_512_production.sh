#!/usr/bin/env bash
# Submit LCDM (node 1, 64 cores) and EDE (node 2, 64 cores) simultaneously,
# followed by the matched comparison job upon completion.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo " Submitting 512^3 Matched EDE/LCDM Cosmological Simulation Campaign"
echo " Configuration: 2 nodes, 64 cores each on partition 'cs'"
echo "======================================================================"

# Submit LCDM on node 1 (64 cores)
echo "Submitting LCDM-512 job to partition cs..."
JOB_LCDM=$(sbatch --parsable run_lcdm_512.sh)
echo "  -> LCDM-512 Job ID: $JOB_LCDM"

# Submit EDE on node 2 (64 cores)
echo "Submitting EDE-512 job to partition cs..."
JOB_EDE=$(sbatch --parsable run_ede_512.sh)
echo "  -> EDE-512 Job ID:  $JOB_EDE"

# Submit Comparison job with dependency (triggers immediately when both simulations finish)
echo "Submitting Comparison job (triggers automatically after both finish)..."
JOB_COMP=$(sbatch --parsable --dependency=afterok:${JOB_LCDM}:${JOB_EDE} compare_512.sh)
echo "  -> Compare Job ID:  $JOB_COMP"

# Optionally submit independent Blender rendering job (non-blocking)
if [[ "${QUEUE_RENDER:-0}" == 1 ]]; then
    echo "Submitting decoupled Blender rendering job..."
    JOB_RENDER=$(sbatch --parsable --dependency=afterok:${JOB_LCDM}:${JOB_EDE} render_512.sh)
    echo "  -> Render Job ID:   $JOB_RENDER"
fi

echo "======================================================================"
echo " All jobs queued successfully!"
echo "   1. LCDM-512 (64 cores):   Job $JOB_LCDM"
echo "   2. EDE-512  (64 cores):   Job $JOB_EDE"
echo "   3. Compare  (16 cores):   Job $JOB_COMP (depends on $JOB_LCDM & $JOB_EDE)"
if [[ "${QUEUE_RENDER:-0}" == 1 ]]; then
    echo "   4. Render   (16 cores):   Job $JOB_RENDER (non-blocking visualization)"
fi
echo ""
echo " Check queue status:   squeue -u \$USER"
echo " Check LCDM log:       tail -f lcdm-512-${JOB_LCDM}.log"
echo " Check EDE log:        tail -f ede-512-${JOB_EDE}.log"
echo " Check Compare log:    tail -f compare-512-${JOB_COMP}.log"
echo "======================================================================"

