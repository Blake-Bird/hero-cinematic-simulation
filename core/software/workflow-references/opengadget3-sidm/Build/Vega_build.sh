#!/bin/bash
echo "Loading VEGA modules."

# check if user is providing an argument.
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

# Load modules based on the partition
if [ "$Partition" == "CPU" ]; then
   echo "Loading CPU modules"
   export SYSTYPE=Vega-CPU
   ml OpenMPI/4.1.1-GCC-11.2.0
   ml GSL/2.7-GCC-11.2.0
   ml HDF5/1.12.1-openmpi412
   ml FFTW/3.3.10-gompi-2021b
elif [ "$Partition" == "GPU" ]; then
   echo "Loading GPU modules"
   export SYSTYPE=Vega-GPU
   ml NVHPC-HPCX-CUDA12/24.9
   ml CUDA/12.8.0

   echo "Note: GSL, FFTW and HDF5 need to be installed manually."
else
   echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
