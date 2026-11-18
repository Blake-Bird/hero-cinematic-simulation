#!/bin/bash

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
    # Get the hostname
    hostname=$(hostname)
    
    module --force purge
    if [[ "$hostname" == *.barbora.it4i.cz ]]; then
        echo "Loading modules for Barbora CPU partition"
        ml GSL/2.7-GCCcore-12.2.0
    elif [[ "$hostname" == *.karolina.it4i.cz ]]; then # karolina
        echo "Loading modules for Karolina CPU partition"
        ml GSL/2.7-GCC-12.3.0
    else
        echo "Unknown hostname: $hostname"
        echo "Load modules manually"
    fi
    # Works for both Barbora and Karolina
    export SYSTYPE=KarolinaBarbora-CPU

    ml GCCcore/13.2.0
    ml OpenMPI/4.1.6-GCC-13.2.0
    ml HDF5/1.14.3-gompi-2023b
    ml FFTW.MPI/3.3.10-gompi-2023b

elif [ "$Partition" == "GPU" ]; then
    echo "Loading modules for Karolina GPU partition"
    export SYSTYPE=Karolina-GPU
    ml CUDA/12.4.0
    ml HDF5/1.14.3-NVHPC-24.1-CUDA-12.4.0
    ml NVHPC/24.1-CUDA-12.4.0
    ml GSL/2.7-GCC-12.3.0
    ml OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0
else
    echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

export OMP_PROC_BIND=close
