#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???
make clean CONFIG=tests/KeplerDisk/KEPLER-SPH.Config.sh
make CONFIG=tests/KeplerDisk/KEPLER-SPH.Config.sh -j
wget https://homepages.usm.uni-muenchen.de/OpenGadget/InitialConditionsForTests/KeplerDisk/kepler.ic # get initial conditions
$OPENGADGET3_MID_COMMAND tests/KeplerDisk/KEPLER-SPH.parameters.tex
rm kepler.ic # remove initial conditions file after completion
python3 tests/KeplerDisk/verify_kepler.py SPH
