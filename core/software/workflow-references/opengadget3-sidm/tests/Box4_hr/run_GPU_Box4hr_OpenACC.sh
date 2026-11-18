#!/bin/bash 
set -e
${OPENGADGET3_TEST_PREAMBLE_COMMAND}

export OMP_NUM_THREADS=8

source tests/configure_systype.sh
rm -rf snap_???

# get required files
./tests/Box4_hr/download_supplementary_files.sh

# for NVTX timers add path to nsight systems
# export PATH="/e/ocean2/users/karademir/nsight-systems-2025.3.1/bin:$PATH"

# make code
source Build/LMUPhysik_build.sh GPU
make clean CONFIG=tests/Box4_hr/OpenACC_Box4_Config.sh
make -j8 CONFIG=tests/Box4_hr/OpenACC_Box4_Config.sh

# in case of nsys profiling run code this way
# nsys profile -t nvtx,openacc mpiexec -np 1 OpenGadget3 tests/Box4_hr/box4_hr.par

# run code on one GPU
mpirun -np 2 OpenGadget3 tests/Box4_hr/box4_hr.par

# remove files
rm filters.dat* metals.dat* Chabrier_pow_new.IMF* SFs* Massive_Kobayashi.yields* Ia_thielemann.yields* AGB_Karakas.yields* 
rm -rf CB07_new CoolingTablesWSS08 ICs

# make clean
make clean CONFIG=tests/ConfigTests/GPU_Box4_Config.sh
