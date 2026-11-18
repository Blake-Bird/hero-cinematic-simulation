#!/bin/bash
echo "Loading modules for Discoverer"

# check if user is providing an argument. e.g: "CPU" or "GPU"
if [ -n "$1" ]; then
    Partition="$1"
fi

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (CPU or GPU):"
    read Partition
fi

echo "Partition is set to: $Partition"
ml purge || true

if [ "$Partition" == "CPU" ]; then
    export SYSTYPE=Discoverer-CPU
    ml openmpi/4/gcc/4.1.6 
    ml gsl/2/2.8-gcc
    ml fftw/3/3.3.10-gcc-openmpi
    ml hdf5/1/1.14/1.14.0-gcc-openmpi
elif [ "$Partition" == "GPU" ]; then
    export SYSTYPE=Discoverer-GPU
    echo "GPU modules not specified and missing SYSTYPE!!!"
else
    echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

export OMP_PROC_BIND=close
