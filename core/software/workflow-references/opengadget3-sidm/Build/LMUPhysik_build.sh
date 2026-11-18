#!/bin/bash
echo "Loading modules for the Physik clusters at LMU"

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
    export SYSTYPE=LMUPhysik-CPU

    ml spack/2024.07
    ml gcc/13.2.0
    ml gsl/2.7.1-gcc-13.2.0
    ml openmpi/5.0.0-gcc-13.2.0
    ml hdf5/1.14.3-gcc-13.2.0-openmpi
    ml fftw/3.3.10-gcc-13.2.0-openmpi

elif [ "$Partition" == "GPU" ]; then

    export SYSTYPE=LMUPhysik-GPU

    ml spack/2024.07
    ml nvhpc/24.3
    ml openmpi/5.0.0-nvhpc-cuda12.4
    ml fftw/3.3.10-nvhpc-cuda12.4-openmpi
    ml gsl/2.7.1-nvhpc
    ml hdf5/1.14.3-nvhpc-cuda12.4-openmpi

    export HDF5_DISABLE_VERSION_CHECK=1
else
    echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

#export OMP_PROC_BIND=close
