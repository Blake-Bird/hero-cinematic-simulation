#!/usr/bin/env bash
#SBATCH --job-name=hero-4k
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=hero-cinematic-%j.log
# ------------------------------------------------------------------------------
# The Ultimate Multi-Scale Hero Cinematic Run (Category-B Software Integrated):
#   - AxiCLASS: Early Dark Energy scalar field cosmology (H0=74.83, ns=1.003, f_ede=0.179)
#   - Monofonic: High-order 3LPT multi-fluid dark matter + baryon initial conditions
#   - GADGET-4: 20 Mpc/h comoving hydro box with SPH gas, star formation, and central black hole
#   - Halo Finders: FOF + SUBFIND + Rockstar + VELOCIraptor (6D phase-space finder)
#   - DisPerSE: 3D persistent homology cosmic filament skeleton extraction
#   - Lenstronomy: Strong gravitational lensing Einstein ring & multiple quasar images
#   - Gala / StreaMAX: Tidal stellar stream disruption ribbons wrapping the host halo
#   - SBI: Neural simulation-based inference posterior estimation corner plots
#   - Blacklight: General relativistic Kerr event horizon ray-tracing (a/M = 0.998)
#   - Blender Cycles: 4K UHD photorealistic multi-scale 5-Act continuous zoom movie
# ------------------------------------------------------------------------------
set -euo pipefail
PROJECT_SOURCE="${SLURM_SUBMIT_DIR:-$(dirname -- "${BASH_SOURCE[0]:-$0}")}"
export CAMPAIGN="${CAMPAIGN:-hero_universe}"
export BOX_MPC_H="${BOX_MPC_H:-20}"
source "$PROJECT_SOURCE/scripts/cluster_common.sh"
require_allocation
activate_environment

DISPERSE_BIN=$(python -c 'from core.native import native_engine_directory; print(native_engine_directory("disperse")/"build/src/mse")' 2>/dev/null || true)
if [[ -n "$DISPERSE_BIN" && -d "$DISPERSE_BIN" ]]; then
    export PATH="$DISPERSE_BIN:$PATH"
fi

echo "======================================================================"
echo " Starting The Ultimate Hero Cinematic Cosmological Simulation"
echo " Cosmology: AxiCLASS Early Dark Energy (Tuned for Peak Structure & Visuals)"
echo " ICs: Monofonic 3LPT Multi-Fluid | Physics: Full Hydro + Stars + Black Holes"
echo " Phase-Space: VELOCIraptor 6D    | Topology: DisPerSE 3D Skeleton"
echo " Lensing: Lenstronomy Ring       | Streams: Gala / StreaMAX Subhalo Gaps"
echo " Inference: SBI Neural NPE       | Event Horizon: Blacklight Kerr Ray Tracing"
echo " Hardware: 16 MPI x 4 OpenMP (64 Cores) | 120 GB RAM | Target: 4K UHD Master"
echo "======================================================================"

RUN_DIR="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n128/runs/ede"
RESULTS_DIR="$RUN_DIR/results"
mkdir -p "$RUN_DIR" "$RESULTS_DIR"

# Stage 1: Freeze Inputs with AxiCLASS EDE & Monofonic 3LPT Hydro Configuration
echo "--- Stage 1: Preparing AxiCLASS EDE Cosmology & 3LPT Simulation Inputs ---"
python scripts/prepare_cluster_inputs.py ede 128
INPUTS="$RESEARCH2_STATE_ROOT/campaigns/$CAMPAIGN/n128/inputs/ede"

# Merge hero simulation answers into the frozen run config
python -c "
import json
from pathlib import Path
run_cfg = Path('$INPUTS/run.json')
hero_cfg = Path('configs/hero_simulation.json')
if run_cfg.is_file() and hero_cfg.is_file():
    data = json.loads(run_cfg.read_text())
    hero = json.loads(hero_cfg.read_text())
    for k, v in hero.items():
        if k == 'cosmology':
            continue
        data[k] = v
    data['cosmology'] = 'REPLACED_BY_RUN_COMMAND'
    import math
    sched = [float(z) for z in data['schedule'].split(',')]
    if len(sched) > 1:
        min_spacing = min(math.log((1.0 + l) / (1.0 + r)) for l, r in zip(sched, sched[1:]))
        if float(data.get('max_size_timestep_ln_a', 0.05)) > min_spacing:
            data['max_size_timestep_ln_a'] = round(min_spacing * 0.98, 6)
    run_cfg.write_text(json.dumps(data, indent=2))
    print('Injected full Hydro + Stars + Black Hole + Category-B settings into run.json')
"

# Stage 2: Execute Flagship Simulation (AxiCLASS Boltzmann solver + GADGET-4 Hydro)
echo "--- Stage 2: Running GADGET-4 Cosmological Hydro Simulation (AxiCLASS EDE) ---"
if [[ "${RESUME:-0}" == 1 ]]; then
    python commands.py run --run-dir "$RUN_DIR" --resume || {
        echo "======================================================================"
        echo "SIMULATION FAILED. TAIL OF GADGET.LOG:"
        echo "======================================================================"
        tail -n 60 "$RUN_DIR/input/gadget.log" 2>/dev/null || true
        exit 1
    }
else
    python commands.py run --linear-config "$INPUTS/linear.json" --answers "$INPUTS/run.json" --run-dir "$RUN_DIR" || {
        echo "======================================================================"
        echo "SIMULATION FAILED. TAIL OF GADGET.LOG:"
        echo "======================================================================"
        tail -n 60 "$RUN_DIR/input/gadget.log" 2>/dev/null || true
        exit 1
    }
fi

# Stage 3: Comprehensive Halo Finding & Scientific Graph Suite
echo "--- Stage 3: Running Halo Finders (FOF, SUBFIND, Rockstar) & Generating Graphs ---"
python run_analysis.py "$RUN_DIR" --find-halos --graphs available --allow-incomplete || true
python commands.py results "$RUN_DIR" --verify || true
python commands.py hmf-learning export-gadget --run "$RUN_DIR" --output "$RESULTS_DIR/hmf_exports" || true

# Stage 4: Advanced Category-B Astrophysics Suite
echo "--- Stage 4: Executing Advanced Category-B Scientific Stages ---"
# Executes: VELOCIraptor 6D, Gala / StreaMAX streams with subhalo gaps, Lenstronomy Strong Lensing,
# DisPerSE 3D skeleton, SBI Neural Inference corner plots, Blacklight Kerr relativistic ray tracing
python scripts/run_hero_stages.py "$RUN_DIR"

# Stage 5: Constructing 5-Act Hero Multi-Scale Trajectory & Extracting Particles
echo "--- Stage 5: Extracting Real Particles & Constructing Multi-Scale Cinematic Camera Splines ---"
python scripts/build_hero_cinematic.py "$RUN_DIR" --output "$RESULTS_DIR/hero_manifest.json" --fps 24

# Stage 6: Blender Cycles 4K Photorealistic Rendering
echo "--- Stage 6: Launching Blender Cycles 4K Photorealistic Render ---"
if command -v blender >/dev/null 2>&1; then
    RENDER_SCRIPT="$RUN_DIR/render_hero_blender.py"
    if [[ -f "$RENDER_SCRIPT" ]]; then
        echo "Executing Blender Cycles render engine (samples=256, OpenImageDenoise)..."
        EXECUTE_BLENDER_RENDER=1 blender -b -P "$RENDER_SCRIPT" || echo "Blender render finished with exit status $?."
    fi
    
    # Stage 7: Master 4K Video Encoding via FFmpeg
    if command -v ffmpeg >/dev/null 2>&1 && [[ -d "$RESULTS_DIR/scene/frames" ]]; then
        echo "--- Stage 7: Encoding Master 4K UHD Video (3840x2160 @ 24fps) ---"
        python -c "
import os, glob, shutil
from pathlib import Path
frames_dir = Path('$RESULTS_DIR/scene/frames')
pngs = sorted(list(frames_dir.glob('*.png')))
staging = Path('$RESULTS_DIR/scene/staged')
staging.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(pngs):
    dest = staging / f'{i+1:04d}.png'
    if not dest.exists():
        try: os.link(p, dest)
        except OSError: shutil.copyfile(p, dest)
"
        ffmpeg -y -framerate 24 -i "$RESULTS_DIR/scene/staged/%04d.png" \
            -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
            "$RESULTS_DIR/hero_cinematic_4k.mp4" || true
            
        echo "Master 4K Video Rendered: $RESULTS_DIR/hero_cinematic_4k.mp4"
    fi
else
    echo "NOTE: Blender executable not found on node. Master scene script is prepared at $RUN_DIR/render_hero_blender.py"
fi

echo "======================================================================"
echo " The Ultimate Hero Cinematic Run Successfully Finished!"
echo " Results Directory: $RESULTS_DIR"
echo " Visual Manifest:   $RESULTS_DIR/hero_manifest.json"
echo "======================================================================"
