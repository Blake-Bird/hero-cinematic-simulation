#!/bin/bash 
set -e
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh

# make code
make clean CONFIG=tests/MHDTest/Config.sh
make CONFIG=tests/MHDTest/Config.sh -j

# run code
$OPENGADGET3_LARGE_COMMAND tests/MHDTest/paramfile_wc6.par
