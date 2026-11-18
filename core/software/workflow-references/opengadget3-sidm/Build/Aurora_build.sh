#!/bin/bash
echo "Loading modules for Aurora using the intel compilers"

export LC_CTYPE=en_US.utf8
export SYSTYPE=Aurora
export OMP_PROC_BIND=close

# all other modules are default
module load hdf5/1.14.5

# gsl is installed in my home directory which is publicly readable
# /home/lboess/libs/gsl-2.8
# /home/lboess/libs/fftw3

# Current (28.8.2025) status:
# Currently Loaded Modules:
#   1) gcc-runtime/13.3.0-ghotoln (H)   7) libiconv/1.17-jjpb4sl         (H)  13) cray-pals/1.4.0                  19) autoconf/2.72
#   2) gmp/6.3.0-mtokfaw          (H)   8) libxml2/2.13.5                     14) cray-libpals/1.4.0               20) gmake/4.4.1
#   3) mpfr/4.2.1-gkcdl5w         (H)   9) hwloc/2.11.3-mpich-level-zero      15) bzip2/1.0.8                      21) cmake/3.30.5
#   4) mpc/1.3.1-rdrlvsl          (H)  10) yaksa/0.3-7ks5f26             (H)  16) gdbm/1.23                        22) hdf5/1.14.5
#   5) gcc/13.3.0                      11) mpich/opt/develop-git.6037a7a      17) berkeley-db/18.1.40-64t6wec (H)
#   6) oneapi/release/2025.0.5         12) libfabric/1.22.0                   18) perl/5.40.0
# 
#   Where:
#    H:  Hidden Module