#!/bin/bash
echo "Loading C2PAP LCG partition modules with Intel compiler"

export LC_CTYPE=en_US.utf8
export SYSTYPE=C2PAP_LCG
export OMP_PROC_BIND=close

module purge || true

ml stack/22.2.1
ml gcc/11.2.0
ml intel-oneapi-mpi/2021.4.0-gcc
ml hdf5/1.10.7-gcc11-impi
ml gsl/2.7-gcc11
ml fftw/3.3.10-gcc11-impi-openmp
