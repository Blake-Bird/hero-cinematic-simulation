#!/bin/bash
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh

# get results
wget -P tests/EvrardCollapse/ https://www.usm.uni-muenchen.de/~dolag/Hydro/Hydro/ppm.tar
tar -xvf tests/EvrardCollapse/ppm.tar -C tests/EvrardCollapse/  
rm tests/EvrardCollapse/ppm.tar

rm -rf snap_???
make clean CONFIG=tests/EvrardCollapse/Config.sh
make CONFIG=tests/EvrardCollapse/Config.sh -j4
$OPENGADGET3_MID_COMMAND tests/EvrardCollapse/box.param
python3 tests/EvrardCollapse/verify_Evrard.py
rm *.png
