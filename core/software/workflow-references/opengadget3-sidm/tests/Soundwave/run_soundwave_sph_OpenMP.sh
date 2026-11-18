#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source Build/LMUPhysik_build.sh GPU

rm -rf snap_???
make clean CONFIG=tests/Soundwave/SOUNDWAVE-SPH.OpenMP.Config.sh
make CONFIG=tests/Soundwave/SOUNDWAVE-SPH.OpenMP.Config.sh -j

mpirun -np 1 OpenGadget3 tests/Soundwave/SOUNDWAVE-SPH.par

source tests/create_venv.sh

python3 tests/Soundwave/verify_soundwave.py SPH-OpenMP
