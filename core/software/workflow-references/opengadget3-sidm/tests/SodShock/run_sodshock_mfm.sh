#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/SodShock/ADSOD-MFM.Config.sh
make CONFIG=tests/SodShock/ADSOD-MFM.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/SodShock/ADSOD-MFM.par
python3 tests/SodShock/verify_sodshock.py MFM
