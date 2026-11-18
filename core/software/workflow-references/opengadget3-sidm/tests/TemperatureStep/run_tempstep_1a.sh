#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/TemperatureStep/conduction_1a_wendlandc6.Config.sh
make CONFIG=tests/TemperatureStep/conduction_1a_wendlandc6.Config.sh -j
$OPENGADGET3_EXE_COMMAND tests/TemperatureStep/conduction_1a_wendlandc6.par
python3 tests/TemperatureStep/verify_tempstep_1a.py
