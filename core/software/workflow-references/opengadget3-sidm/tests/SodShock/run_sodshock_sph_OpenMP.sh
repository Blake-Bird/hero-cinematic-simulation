#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source Build/LMUPhysik_build.sh GPU

rm -rf snap_???
make clean CONFIG=tests/SodShock/ADSOD-SPH.OpenMP.Config.sh
make CONFIG=tests/SodShock/ADSOD-SPH.OpenMP.Config.sh -j

mpirun -np 1 OpenGadget3 tests/SodShock/ADSOD-SPH.par

source tests/create_venv.sh

python3 tests/SodShock/verify_sodshock.py SPH-OpenMP
