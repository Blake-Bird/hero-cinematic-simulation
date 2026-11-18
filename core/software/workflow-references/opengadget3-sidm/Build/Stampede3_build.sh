#!/bin/bash
echo "Loading modules for Stampede3 using the intel compilers"

export LC_CTYPE=en_US.utf8
export SYSTYPE=stampede3

module purge || true
module load intel/24.0
module load impi
module load fftw3/3.3.10
module load gsl/2.8
module load hdf5/1.10.11
