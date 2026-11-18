#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source Build/LMUPhysik_build.sh GPU

rm -rf snap_???
make clean
make CONFIG=tests/FreefallCollapse/FREEFALL-SPH-OpenACC.Config.sh -j
mpirun OpenGadget3 tests/FreefallCollapse/FREEFALL-SPH.parameters.tex

source tests/create_venv.sh

python tests/FreefallCollapse/verify_freefall.py SPH-OpenACC
