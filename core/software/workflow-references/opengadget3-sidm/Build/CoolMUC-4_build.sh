#!/bin/bash
export LC_CTYPE=en_US.utf8

# check if user is providing an argument. e.g: "gcc" or "intel"
if [ -n "$1" ]; then
    Compiler="$1"
fi

# If Compiler is still not set, default to intel
if [ -z "$Compiler" ]; then
    echo "Compiler is not set. Using intel as default."
    Compiler="intel"
fi

echo "Compiler is set to: $Compiler"

# Unload all modules
module purge || true

# Load modules based on the partition
if [ "$Compiler" == "gcc" ]; then
    echo "Loading CoolMUC-4 modules with GCC compiler."
    export SYSTYPE=CoolMUC-4_gcc
    export OMP_PROC_BIND=close

    ml stack/23.1.0
    ml gcc/12.2.0
    ml intel-mpi/2021.9.0
    ml gsl/2.7.1-gcc12
    ml fftw/3.3.10-gcc12-impi-openmp
    ml hdf5/1.8.23-gcc12
elif [ "$Compiler" == "intel" ]; then
    echo "Loading CoolMUC-4 modules with Intel compiler."
    export SYSTYPE=CoolMUC-4
    export KMP_AFFINITY=granularity=fine,compact,1,0

    ml stack/24.4.0
    ml intel/2024.1.0   
    ml intel-mpi/2021.12.0 
    ml fftw/3.3.10-intel24-impi-openmp 
    ml gsl/2.8-gcc13 
    ml hdf5/1.14.5-intel24-impi     
else
    echo "Invalid Compiler option: $Compiler. Please set it to either 'gcc' or 'intel'."
fi
