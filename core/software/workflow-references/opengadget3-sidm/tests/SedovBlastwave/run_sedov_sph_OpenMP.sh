#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source Build/LMUPhysik_build.sh GPU

rm -rf snap_???
make clean CONFIG=tests/SedovBlastwave/SEDOV-SPH.OpenMP.Config.sh
make CONFIG=tests/SedovBlastwave/SEDOV-SPH.OpenMP.Config.sh -j

mpirun -np 1 OpenGadget3 tests/SedovBlastwave/SEDOV-SPH.parameters.tex

source tests/create_venv.sh

python3 tests/SedovBlastwave/verify_sedov.py SPH-OpenMP
