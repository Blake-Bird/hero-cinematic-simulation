#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/FreefallCollapse/FREEFALL-MFM.Config.sh
make CONFIG=tests/FreefallCollapse/FREEFALL-MFM.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/FreefallCollapse/FREEFALL-MFM.parameters.tex
python3 tests/FreefallCollapse/verify_freefall.py MFM
