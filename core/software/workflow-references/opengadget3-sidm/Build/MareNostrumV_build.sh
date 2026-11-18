

#!/bin/bash
module purge || true

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (GPP or ACC):"
    read Partition
fi

echo "Partition is set to: $Partition"

# Load modules based on the partition
if [ "$Partition" == "GPP" ]; then
    echo "Loading modules for MareNostrumV-GPP"
    export SYSTYPE=MareNostrumV-GPP
    ml ucx/1.16.0-gcc
    ml openmpi/4.1.5-gcc
    ml hdf5/1.14.1-2-gcc-openmpi
    ml fftw/3.3.10-gcc-ompi   
    ml gsl/2.7.1-gcc
elif [ "$Partition" == "ACC" ]; then
    export SYSTYPE=MareNostrumV-ACC
    module purge
    module load nvidia-hpc-sdk/25.7
    module load gsl/2.8-gcc
    module load fftw/3.3.10-gcc-nvhpcx
else
    echo "Invalid partition: $Partition. Please set it to either 'GPP' or 'ACC'."
fi

export OMP_PROC_BIND=close
