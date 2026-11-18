#!/bin/bash
echo "Loading SilkRiver modules"

export SYSTYPE=SilkRiver

module purge || true
module load compiler/intel/2021.3.0
module load mathlib/gsl/intel/2.6
module load mpi/intelmpi/2021.3.0
module load mathlib/fftw/intelmpi/3.3.9_double
module load mathlib/hdf5/intel/1.12.0
