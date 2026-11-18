#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/SedovBlastwave/SEDOV-MFM.Config.sh
make CONFIG=tests/SedovBlastwave/SEDOV-MFM.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/SedovBlastwave/SEDOV-MFM.parameters.tex
python3 tests/SedovBlastwave/verify_sedov.py MFM
