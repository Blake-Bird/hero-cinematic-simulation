#!/bin/bash
echo "Loading modules for NgarrguTindebeek clusters at Ozstar"

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
    export SYSTYPE=NgarrguTindebeek-CPU
    ml gcc/12.3.0
    ml openmpi/4.1.5
    ml fftw.mpi/3.3.10
    ml hdf5/1.14.0
    ml gsl/2.7
elif [ "$Partition" == "GPU" ]; then
    export SYSTYPE=NgarrguTindebeek-GPU
    ml nvidia-compilers/25.3-cuda-12.8.0
    ml gcc/12.3.0
    ml openmpi/4.1.5
    ml fftw.mpi/3.3.10
    ml hdf5/1.14.0
    ml gsl/2.7
else
    echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

export OMP_PROC_BIND=close
