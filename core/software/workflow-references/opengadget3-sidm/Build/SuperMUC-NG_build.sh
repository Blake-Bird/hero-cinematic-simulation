#!/bin/bash
echo "Loading modules for SuperMUC-NG"

export LC_CTYPE=en_US.utf8

module purge || true

echo "Loading modules for SuperMUC-NG using the intel compilers"
export SYSTYPE=SuperMUC-NG
ml stack/24.6.0
ml intel-toolkit/2025.1.0
ml gsl/2.8-intel25
ml fftw/3.3.10-gcc14-impi-openmp
ml hdf5/1.10.11-intel25-impi
ml gmake/4.4.1

export KMP_AFFINITY=granularity=fine,compact,1,0
