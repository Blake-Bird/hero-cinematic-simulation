#!/bin/bash
echo "Loading Deucalion modules"

# check if user is providing an argument. e.g: "CPU" or "GPU"
if [ -n "$1" ]; then
    Partition="$1"
fi

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (x86, arm or GPU):"
    read Partition
fi

#module purge || true
if [ "$Partition" == "x86" ]; then
    echo "Loading x86 modules"
    export SYSTYPE=Deucalion-x86

    ml GSL/2.8-GCC-13.3.0
    ml HDF5/1.14.6-gompi-2025b
    ml FFTW.MPI/3.3.10-gompi-2025b

elif [ "$Partition" == "arm" ]; then
    echo "Modules for arm need to be selected manually."

elif [ "$Partition" == "GPU" ]; then
    echo "Loading GPU modules"
    export SYSTYPE=Deucalion-GPU
    echo "Warning: With Roxy 8 it is not possible to run the code. Waiting for update of system to Roxy 9."

    #ml NVHPC/25.1-CUDA-12.6.0

    echo "Note: GSL, FFTW and HDF5 have to be installed manually."
else
    echo "Invalid partition: $Partition. Please set it to either 'x86', 'arm' or 'GPU'."
fi

export OMP_PROC_BIND=close
