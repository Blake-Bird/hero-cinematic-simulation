#!/usr/bin/env bash
set -euo pipefail

# Ensure standard bin paths are in PATH (for macOS Homebrew and Linux user bins)
for p in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin"; do
    if [[ -d "$p" && ":$PATH:" != *":$p:"* ]]; then
        export PATH="$p:$PATH"
    fi
done

# Auto-detect HDF5 headers and libraries on cluster nodes
for cand in "$HOME/.local" /opt/anaconda3 /usr/local /usr; do
    if [[ -f "$cand/include/hdf5.h" ]]; then
        export HDF5_ROOT="$cand"
        export CPATH="${cand}/include${CPATH:+:$CPATH}"
        export C_INCLUDE_PATH="${cand}/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
        export CPLUS_INCLUDE_PATH="${cand}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
        if [[ -d "$cand/lib" ]]; then
            export LIBRARY_PATH="${cand}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
            export LD_LIBRARY_PATH="${cand}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        if [[ -d "$cand/lib64" ]]; then
            export LIBRARY_PATH="${cand}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
            export LD_LIBRARY_PATH="${cand}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        break
    fi
done

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$PROJECT_DIR"
export RESEARCH2_STATE_ROOT="${RESEARCH2_STATE_ROOT:-$HOME/ede-runtime}"
export CAMPAIGN="${CAMPAIGN:-matched40}"
export BOX_MPC_H="${BOX_MPC_H:-40}"
export SEED="${SEED:-42}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
load_cluster_modules() {
    if [[ -n "${EDE_MODULES:-}" ]]; then
        if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
        type module >/dev/null 2>&1 || { echo 'Environment modules are unavailable; load compiler/MPI libraries in your shell.' >&2; exit 2; }
        read -r -a selected_modules <<< "$EDE_MODULES"
        module load "${selected_modules[@]}"
    fi
}
require_allocation() {
    if [[ "${ALLOW_LOCAL:-0}" == 1 || "$(uname -s)" == "Darwin" ]]; then
        return 0
    fi
    [[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch. Simulation and setup launchers require a compute allocation.' >&2; exit 2; }
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || { echo 'This profile supports one node; multi-node budgeting needs a separate configuration.' >&2; exit 2; }
}
ensure_blender() {
    if command -v blender >/dev/null 2>&1; then
        return 0
    fi
    if [[ -n "${BLENDER:-}" && -x "$BLENDER" ]]; then
        export PATH="$(dirname -- "$BLENDER"):$PATH"
        return 0
    fi
    local state_root="${RESEARCH2_STATE_ROOT:-$HOME/ede-runtime}"
    if [[ -x "$state_root/tools/blender/blender" ]]; then
        export PATH="$state_root/tools/blender:$PATH"
        return 0
    fi
    if [[ -x "$HOME/.local/bin/blender" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi
    if [[ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]]; then
        mkdir -p "$HOME/.local/bin"
        cat << 'EOF' > "$HOME/.local/bin/blender"
#!/usr/bin/env bash
exec /Applications/Blender.app/Contents/MacOS/Blender "$@"
EOF
        chmod +x "$HOME/.local/bin/blender"
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi
    if [[ "$(uname -s)" == "Linux" ]]; then
        echo "Blender not found on PATH. Attempting to install portable Blender LTS..."
        mkdir -p "$state_root/tools"
        local b_url="https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz"
        local b_archive="/tmp/blender-4.2.3-linux-x64.tar.xz"
        local downloaded=0
        if command -v curl >/dev/null 2>&1; then
            if curl -fsSL --connect-timeout 8 "$b_url" -o "$b_archive" 2>/dev/null; then
                downloaded=1
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q --timeout=8 "$b_url" -O "$b_archive" 2>/dev/null; then
                downloaded=1
            fi
        fi
        if [[ "$downloaded" -eq 1 && -f "$b_archive" ]]; then
            tar -xJf "$b_archive" -C "$state_root/tools"
            mv "$state_root/tools"/blender-4.2.3-linux-x64 "$state_root/tools/blender"
            rm -f "$b_archive"
            export PATH="$state_root/tools/blender:$PATH"
            echo "Installed portable Blender to $state_root/tools/blender"
            return 0
        else
            echo "NOTE: Could not download Blender (node may have no direct internet access). 3D Blender export will be optional." >&2
            return 1
        fi
    fi
    return 1
}
activate_environment() {
    load_cluster_modules
    source "$RESEARCH2_STATE_ROOT/environments/default/bin/activate"
    if ! ensure_blender; then
        echo 'NOTE: Blender not found; 3D visualization and .blend export will be skipped during simulation run.' >&2
    fi
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
}
run_model() {
    local model="$1" n="$2"
    local stage="${STAGE:-all}"
    local inputs="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n$n/inputs/$model"
    local run="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n$n/runs/$model"
    python scripts/prepare_cluster_inputs.py "$model" "$n"
    
    local t0=$(date +%s)
    echo "=== Running model: $model (n=$n, stage=$stage, campaign=$CAMPAIGN) ==="
    
    if [[ "$stage" == "sim" || "$stage" == "all" ]]; then
        local t_sim_start=$(date +%s)
        if [[ "${RESUME:-0}" == 1 ]]; then
            python scripts/audit_cluster_run.py "$run" --simulation-only --quiet || python commands.py run --run-dir "$run" --resume
        else
            if [[ "$stage" == "sim" ]]; then
                python commands.py run --linear-config "$inputs/linear.json" --answers "$inputs/run.json" --run-dir "$run" --sim-only
            else
                python commands.py run --linear-config "$inputs/linear.json" --answers "$inputs/run.json" --run-dir "$run"
            fi
        fi
        python scripts/audit_cluster_run.py "$run" --simulation-only
        local t_sim_end=$(date +%s)
        echo "Simulation stage completed in $((t_sim_end - t_sim_start)) seconds."
    fi

    if [[ "$stage" == "catalogs" ]]; then
        local t_cat_start=$(date +%s)
        echo "Running halo finder catalogs stage..."
        python commands.py analysis "$run" --find-halos --graphs finder,subhalos
        local t_cat_end=$(date +%s)
        echo "Catalogs stage completed in $((t_cat_end - t_cat_start)) seconds."
    fi

    if [[ "$stage" == "analysis" || "$stage" == "all" ]]; then
        local t_ana_start=$(date +%s)
        if [[ "$stage" == "analysis" ]]; then
            echo "Running post-simulation analysis..."
            python commands.py analysis "$run" --find-halos --graphs available --allow-incomplete
        fi
        local export_dir="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n$n/exports/$model"
        if [[ -e "$export_dir" ]]; then export_dir="${export_dir}_job${SLURM_JOB_ID:-$(date +%s)}"; fi
        python commands.py hmf-learning export-gadget --run "$run" --output "$export_dir"
        python commands.py results "$run" --verify
        python scripts/audit_cluster_run.py "$run"
        local t_ana_end=$(date +%s)
        echo "Analysis and exports stage completed in $((t_ana_end - t_ana_start)) seconds."
    fi

    local t1=$(date +%s)
    python -c "
import json, os, time
from pathlib import Path
path = Path('$run') / 'timing_report.json'
report = {'model': '$model', 'n': $n, 'stage': '$stage', 'total_duration_seconds': $((t1 - t0)), 'timestamp': time.time()}
path.write_text(json.dumps(report, indent=2))
" 2>/dev/null || true
    echo "=== Completed model $model in $((t1 - t0)) seconds ==="
}
