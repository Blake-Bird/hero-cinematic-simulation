#!/bin/bash 
set -e
${OPENGADGET3_TEST_PREAMBLE_COMMAND}
source tests/configure_systype.sh
rm -rf snap_???

# Download files
./tests/asin1413816_1/download_supplementary_files.sh

# make code
make clean CONFIG=tests/asin1413816_1/asin1413816_1.Config.sh
make CONFIG=tests/asin1413816_1/asin1413816_1.Config.sh -j

# run code
$OPENGADGET3_LARGE_COMMAND tests/asin1413816_1/dianoga_gal_g1.par
# restart
$OPENGADGET3_LARGE_COMMAND tests/asin1413816_1/dianoga_gal_g1.par 1

# remove files
rm filters.dat* metals.dat* Chabrier_pow_new.IMF* SFs* Massive_Kobayashi.yields* Ia_thielemann.yields* AGB_Karakas.yields* d1413816_1d_6r_center_1920.ic_gas*
rm -rf CB07_new CoolingTablesWSS08

# make clean
make clean CONFIG=tests/asin1413816_1/asin1413816_1.Config.sh
