#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/Soundwave/SOUNDWAVE-PES.Config.sh
make CONFIG=tests/Soundwave/SOUNDWAVE-PES.Config.sh -j
$OPENGADGET3_MID_COMMAND tests/Soundwave/SOUNDWAVE-PES.par
python3 tests/Soundwave/verify_soundwave.py PES
