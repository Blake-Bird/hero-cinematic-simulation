#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/SedovBlastwave/SEDOV-SPH.Config.sh
make CONFIG=tests/SedovBlastwave/SEDOV-SPH.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/SedovBlastwave/SEDOV-SPH.parameters.tex
python3 tests/SedovBlastwave/verify_sedov.py SPH
