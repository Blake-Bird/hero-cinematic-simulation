#!/bin/bash -l
#SBATCH -N 1
#SBATCH -A p200526
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -q test
#SBATCH --time 00:05:00

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (CPU or GPU):"
    read Partition
fi

export LC_CTYPE=en_US.utf8

echo "Partition is set to: $Partition"

if [ "$Partition" == "CPU" ]; then
    echo "Loading modules for CPU partition"
    export SYSTYPE=MeluXina-CPU
    module use /apps/USE/easybuild/staging/2022.1/modules/all/
    ml OpenMPI/4.1.4-GCC-11.3.0
    ml FFTW.MPI/3.3.10-gompi-2022a
    ml HDF5/1.12.2-gompi-2022a
    ml GSL/2.7-GCC-11.3.0
    source tests/ConfigTests/compile_config1.sh
elif [ "$Partition" == "GPU" ]; then
    echo "Loading modules for GPU partition"
    export SYSTYPE=MeluXina-GPU
    module use /apps/USE/easybuild/staging/2023.1/modules/all/
    ml CUDA/12.2.0
    ml NVHPC/23.7-CUDA-12.2.0
    ml GSL/2.7-GCC-12.3.0
    ml HDF5/1.14.0-gompi-2023a
    ml OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0
    source tests/ConfigTests/compile_config1.sh
else
    echo "Invalid partition: $Partition. Please set it to either 'CPU' or 'GPU'."
fi

export OMP_PROC_BIND=close
