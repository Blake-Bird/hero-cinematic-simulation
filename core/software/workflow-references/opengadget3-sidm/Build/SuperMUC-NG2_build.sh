#!/bin/bash
echo "Loading SuperMUC-NG2 GPU modules"
module unload stack
ml stack/24.5.0
ml intel-toolkit/2025.1.0
ml gsl/2.8-intel25
ml hdf5/1.10.11-intel25-impi

export SYSTYPE=SuperMUC-NG2

if [ -z "$MKLFFTW" ]; then
  echo "MKLFFTW not defined, loading fftw module"
  ml fftw/3.3.10-gcc14-impi-openmp
else
  echo "MKLFFTW defined, using MKL fftw $MKLFFTW mode"
  FOLDER=${SYSTYPE}_mklfftw3  # Compile mkl advanced API
  if [ ! -d "$FOLDER" ]; then # no need, if alread compiled
    echo "Building mkl's fftw3x_cdft locally (una tantum)"
    HERE=$PWD
    mkdir -p folder
    cd $MKLROOT/share/mkl/interfaces/fftw3x_cdft
    make INSTALL_DIR=$HERE/$FOLDER libintel64 interface=ilp64
    cd $HERE
  fi
fi

export I_MPI_OFFLOAD=1
export I_MPI_OFI_PROVIDER=mlx
export I_MPI_OFFLOAD_CELL=tile
export I_MPI_CBWR=1
export I_MPI_OFFLOAD_RDMA=0
export I_MPI_OFFLOAD_IPC=0
