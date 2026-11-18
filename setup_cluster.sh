#!/usr/bin/env bash
#SBATCH --job-name=ede-setup
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ede-setup-%j.log
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
if [[ "${1:-}" == --inspect ]]; then
    command -v sinfo >/dev/null && sinfo -N -o '%N %P %c %m %t'
    command -v scontrol >/dev/null && scontrol show partition
    if type module >/dev/null 2>&1; then module list; fi
    exit 0
fi
# Auto-detect and activate Python >= 3.10 if default python3 is older
ensure_modern_python() {
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            return 0
        fi
    fi
    for cand in python3.12 python3.11 python3.10 /usr/bin/python3.12 /usr/bin/python3.11 /usr/bin/python3.10 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10; do
        if [[ -x "$cand" || $(command -v "$cand" 2>/dev/null) ]]; then
            if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                mkdir -p "$HOME/.local/bin"
                ln -sf "$(command -v "$cand")" "$HOME/.local/bin/python3"
                ln -sf "$(command -v "$cand")" "$HOME/.local/bin/python"
                export PATH="$HOME/.local/bin:$PATH"
                echo "Activated $cand as python3 ($($cand --version))"
                return 0
            fi
        fi
    done
    if ! type module >/dev/null 2>&1; then
        for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
            if [[ -r "$f" ]]; then source "$f" 2>/dev/null || true; break; fi
        done
    fi
    if type module >/dev/null 2>&1; then
        for mod in $(module -t avail 2>&1 | grep -iE 'python|anaconda|miniconda' | grep -vE '2\.|3\.[0-9]\b' | sort -V -r); do
            mod_clean=$(echo "$mod" | tr -d ' :' | sed 's/(default)//')
            if [[ -n "$mod_clean" ]]; then
                if module load "$mod_clean" 2>/dev/null; then
                    if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                        echo "Loaded environment module: $mod_clean ($(python3 --version 2>&1))"
                        export EDE_MODULES="${EDE_MODULES:-}${EDE_MODULES:+ }$mod_clean"
                        return 0
                    else
                        module unload "$mod_clean" 2>/dev/null || true
                    fi
                fi
            fi
        done
    fi
    return 1
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Running setup directly on login node ($HOSTNAME)..."
else
    echo "Running setup inside Slurm allocation (Job ID: $SLURM_JOB_ID)..."
    require_allocation
fi
ensure_modern_python || true
load_cluster_modules
export SIM_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"
ensure_blender || true
bash scripts/setup.sh --no-system-packages --engines all
# Future allocations repeat the same module loads through EDE_MODULES.
