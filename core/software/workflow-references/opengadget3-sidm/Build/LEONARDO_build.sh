#!/bin/bash

module purge || true

# check if user is providing an argument. e.g: "DCGP" or "Booster"
if [ -n "$1" ]; then
    Partition="$1"
fi

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (DCGP or Booster):"
    read Partition
fi

echo "Partition is set to: $Partition"

# Load modules based on the partition
if [ "$Partition" == "DCGP" ]; then
    echo "Loading modules for Leonardo-DCGP"
    export SYSTYPE=Leonardo-DCGP
    ml openmpi/4.1.4--gcc--11.3.0-cuda-11.8 
    ml fftw/3.3.10--openmpi--4.1.4--gcc--11.3.0 
    ml hdf5/1.12.2--openmpi--4.1.4--gcc--11.3.0 
    ml gsl/2.7.1--gcc--11.3.0
elif [ "$Partition" == "Booster" ]; then
    echo "Loading modules for Leonardo-Booster"
    export SYSTYPE=Leonardo-Booster
    ml cuda/12.3
    ml nvhpc/23.11
    ml openmpi/4.1.6--nvhpc--23.11
    ml fftw/3.3.10--openmpi--4.1.6--nvhpc--23.11
    ml hdf5/1.14.3--openmpi--4.1.6--nvhpc--23.11
    ml gsl/2.7.1--oneapi--2023.2.0
else
    echo "Invalid partition: $Partition. Please set it to either 'DCGP' or 'Booster'."
fi

export OMP_PROC_BIND=close
