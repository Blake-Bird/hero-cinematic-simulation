#!/bin/bash
echo "Loading modules for Anvil using the gcc/openmpi compilers"

export LC_CTYPE=en_US.utf8
export SYSTYPE=Anvil

module purge || true
module load modtree/cpu
module load hdf5
module load fftw
module load gsl
