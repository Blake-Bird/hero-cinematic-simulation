#!/usr/bin/env bash
# Ubuntu 24.04 / WSL2 setup. Cluster users may supply libraries through modules.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
install_system=1
profile=base
engines=class,axiclass,rockstar,consistent-trees
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-system-packages) install_system=0; shift ;;
        --profile)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--profile requires a dependency profile (base)." >&2; exit 2
            fi
            profile="$2"; shift 2 ;;
        --engines)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--engines requires comma-separated engine names, all, or none." >&2; exit 2
            fi
            engines="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash scripts/setup.sh [--no-system-packages] [--profile PROFILE] [--engines NAMES|all|none]"
            echo "Default profile: base. Default engines: class,axiclass,rockstar,consistent-trees. Optional profiles and engines are explicit."
            exit 0 ;;
        *) echo "Unknown option: $1; use --help." >&2; exit 2 ;;
    esac
done
build_flags=()
if [[ "$engines" == all ]]; then
    build_flags+=(--build-all)
elif [[ "$engines" != none ]]; then
    IFS=',' read -r -a selected_engines <<< "$engines"
    for engine in "${selected_engines[@]}"; do
        case "$engine" in
            class|axiclass|monofonic|music2|rockstar|consistent-trees|velociraptor|disperse)
                build_flags+=("--build-$engine") ;;
            *) echo "Unknown engine: $engine" >&2; exit 2 ;;
        esac
    done
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "Run inside Ubuntu Linux or Ubuntu 24.04 under WSL2." >&2
    exit 1
fi
if [[ "$install_system" == 1 ]]; then
    if ! command -v apt-get >/dev/null; then
        echo "Load cluster development libraries and use --no-system-packages." >&2
        exit 1
    fi
    if [[ "$EUID" == 0 ]]; then
        echo "Run as your normal Ubuntu user; sudo is used only for system packages." >&2
        exit 1
    fi
    sudo apt-get update
    sudo apt-get install -y git build-essential gfortran cmake pkg-config perl python3-dev python3-venv libopenmpi-dev openmpi-bin libgsl-dev libfftw3-dev libfftw3-mpi-dev libhdf5-dev libcfitsio-dev libtirpc-dev libgl1 libegl1 libxrender1 libxkbcommon0 blender
fi
PYTHON_CMD="${PYTHON:-}"
if [[ -z "$PYTHON_CMD" ]]; then
    for cand in python3.12 python3.11 python3.10 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
                PYTHON_CMD="$cand"
                break
            fi
        fi
    done
fi
if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: Python 3.10 or newer is required, but $(python3 --version 2>&1) was found." >&2
    echo "Please load a modern Python module (e.g. 'module load python/3.11' or 'module avail python') and rerun." >&2
    exit 1
fi
echo "Using Python interpreter: $PYTHON_CMD ($($PYTHON_CMD --version))"
"$PYTHON_CMD" bootstrap.py --install --profile "$profile" --check "${build_flags[@]}" --jobs "${SIM_BUILD_JOBS:-2}"
echo "Build complete. Environment is under external RESEARCH2_STATE_ROOT (default: ../ede-runtime)."
echo "GADGET is compiled by the run driver using the requested physics configuration."
if ! command -v blender >/dev/null 2>&1; then
    source "$(dirname -- "${BASH_SOURCE[0]}")/cluster_common.sh" 2>/dev/null || true
    if type ensure_blender >/dev/null 2>&1; then
        ensure_blender
    fi
fi
if ! command -v blender >/dev/null 2>&1; then
    echo "ERROR: Blender is not available. On Ubuntu rerun setup without --no-system-packages; on HPC load a Blender module or set BLENDER to its executable." >&2
    exit 1
fi
blender --version | head -n 1
source "${RESEARCH2_STATE_ROOT:-$PWD/../ede-runtime}/environments/default/bin/activate"
export EDE_SETUP_ENGINES="$engines"
python - <<'PY_CHECK_NATIVE'
from pathlib import Path
import os
from core.native import native_engine_directory
requested = os.environ['EDE_SETUP_ENGINES']
selected = ('class', 'axiclass', 'monofonic', 'music2', 'rockstar', 'consistent-trees', 'velociraptor', 'disperse') if requested == 'all' else (() if requested == 'none' else tuple(requested.split(',')))
for engine in selected:
    directory = native_engine_directory(engine)
    print(f'Native engine {engine}: {directory}')
    if not directory.is_dir():
        raise SystemExit(f'Native engine missing: {engine}')
for engine in ('class', 'axiclass'):
    if engine not in selected:
        continue
    binary = native_engine_directory(engine)/'class'
    if not binary.is_file():
        raise SystemExit(f'Native solver executable missing: {binary}')
print('All requested native engines located (not a simulation acceptance test).')
PY_CHECK_NATIVE
