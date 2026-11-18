#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/SodShock/ADSOD-SPH.Config.sh
make CONFIG=tests/SodShock/ADSOD-SPH.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/SodShock/ADSOD-SPH.par
python3 tests/SodShock/verify_sodshock.py SPH
