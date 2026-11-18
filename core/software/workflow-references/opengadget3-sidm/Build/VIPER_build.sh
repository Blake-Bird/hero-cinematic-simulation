#!/bin/bash
echo "Loading VIPER modules and setting environment"

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

# Load modules based on the partition
if [ "$Partition" == "CPU" ]; then
    echo "Loading modules for VIPER-CPU"
    export SYSTYPE=VIPER-CPU
    ml intel/2024.0
    ml impi/2021.11
    ml gsl/2.7
    ml fftw-mpi/3.3.10
    ml hdf5-mpi/2.0.0
elif [ "$Partition" == "GPU" ]; then
    echo "Loading modules for VIPER-GPU"
    export SYSTYPE=VIPER-GPU
    ml amd-llvm/23.2
    ml openmpi/5.0
    ml gcc/14    
    ml fftw-mpi/3.3.10  
    ml hdf5-mpi/1.14.1  
    ml gsl/2.7
    ml rocm/7.2
    export OMPI_CXX=amdclang++
else
    echo "Invalid partition: $Partition, Please set it to either 'CPU' or 'GPU'"
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
