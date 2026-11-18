#!/bin/bash
echo "Loading LUMI modules and setting environment without multithreading."
export LC_CTYPE=en_US.utf8

# check if user is providing an argument. e.g: "LUMI-C" or "LUMI-G"
if [ -n "$1" ]; then
    Partition="$1"
fi

# Check if Partition is set, if not ask for input
if [ -z "$Partition" ]; then
    echo "Partition is not set. Please enter a partition (LUMI-C or LUMI-G):"
    read Partition
fi

echo "Partition is set to: $Partition"

# Load modules based on the partition
if [ "$Partition" == "LUMI-C" ]; then
    echo "Loading modules for LUMI-C"
    export SYSTYPE=LUMI-C

    ml LUMI/24.03 partition/C
    ml cray-fftw/3.3.10.7
    ml GSL/2.7.1-cpeCray-24.03-OpenMP
    ml cray-hdf5-parallel/1.12.2.11
elif [ "$Partition" == "LUMI-G" ]; then
    echo "Loading modules for LUMI-G"
    export SYSTYPE=LUMI-G
    export MPICH_OFI_NIC_POLICY=NUMA

    ml LUMI/25.03 partition/G
    ml PrgEnv-amd
    ml rocm/6.3.4
    ml GSL/2.8-cpeAMD-25.03        
    ml cray-fftw/3.3.10.10
    ml cray-hdf5-parallel/1.14.3.5
    export LIBRARY_PATH=/opt/rocm-6.3.4/lib/llvm/lib/libdevice:/opt/rocm-6.3.4/lib/llvm/lib:$LIBRARY_PATH

    printf '%s\n' \ 'NOTE: When running full node jobs use correct GCD/CCD affinity to improve performance'\
    'see: https://docs.lumi-supercomputer.eu/'
else
    echo "Invalid partition: $Partition. Please set it to either 'LUMI-C' or 'LUMI-G'."
fi

export OMP_PROC_BIND=close
export OMP_PLACES=cores
