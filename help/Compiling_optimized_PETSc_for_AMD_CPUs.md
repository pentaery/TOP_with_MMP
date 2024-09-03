### Introduction
In this experiment, we want to compile an optimized PETSc library for AMD 7452 CPUs. We follow the AMD document [HPC tuning guide for 7002 series processors] to build optimized libraries, and we expected it may finally accelerate the computing time of our numerical experiments.


### Set compiling flags
#!/bin/sh

unset PETSC_DIR  
unset PETSC_ARCH  
module purge

export CC=clang
export FC=flang
export CXX=clang++
export F90=flang
export F77=flang
export AR=llvm-ar
export RANLIB=llvm-ranlib
export NM=llvm-nm

export AOCC_ROOT=${HOME}/cqye/aocc-compiler-4.0.0
export OPENMPI_ROOT=${HOME}/cqye/openmpi-4.1.4/linux-amd-opt
export AOCL_ROOT=/public1/home/scb8261/cqye/aocl/4.0
export PATH=${OPENMPI_ROOT}/bin:${AOCC_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${OPENMPI_ROOT}/lib:${AOCL_ROOT}/lib:${AOCC_ROOT}/lib:${LD_LIBRARY_PATH}

export CFLAGS="-O3 -ffast-math -march=znver2 -fopenmp"
export CXXFLAGS="-O3 -ffast-math -march=znver2 -fopenmp"
export FFLAGS="-O3 -ffast-math -march=znver2 -fopenmp"
export LDFLAGS="-lalm -lm"

export PETSC_DIR=${HOME}/cqye/petsc
export PETSC_ARCH=linux-amd-opt
echo "cqye will use PETSc="${PETSC_DIR}/${PETSC_ARCH}
echo "Don't panic!"
export OMPI_MCA_btl=^openib


### Build OpenMPI
Note the cluster is managed by Slurm, and configure OpenMPI with "--with-slurm".

./configure --prefix=${OPENMPI_ROOT} --with-slurm \
CC=${CC} CXX=${CXX} FC=${FC} CFLAGS="${CFLAGS}" \
CXXFLAGS="${CXXFLAGS}" FCFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}" 

make -j

make install

Give up OpenMPI...

### Build mpich
./configure --prefix=${MPICH_ROOT} --with-pm=no \
--with-pmi=slurm --with-slurm-include=/usr/include/slurm \
--with-slurm-lib=/usr/lib64 \
--with-ucx=embedded \
CC=${CC} CXX=${CXX} FC=${FC} CFLAGS="${CFLAGS}" \
CXXFLAGS="${CXXFLAGS}" FCFLAGS="${FFLAGS}" LDFLAGS="${LDFLAGS}" |& tee c.txt

make 2>&1 | tee m.txt

make install |& tee mi.txt


### Build PETSc with OpenMPI, failed
Note that libflame.so is a pre-built library from AOCL that cannot work in our system. Delete all *.so files and only use *.a files.

It seems SuiteSparse cannot be built in our system via PETSc, and I do not why...

./configure PETSC_ARCH=linux-amd-opt --with-debugging=0 \
--with-mpi-dir=${OPENMPI_ROOT} AR=${AR} RANLIB=${RANLIB} \
--CFLAGS="${CFLAGS}" --FFLAGS="${FFLAGS}" --CXXFLAGS="${CXXFLAGS}" --LDFLAGS="${LDFLAGS}" \
--with-blaslapack-dir=${AOCL_ROOT} \
--with-scalapack-dir=${AOCL_ROOT} \
--download-cmake=externalpackages/cmake-3.24.2.tar.gz \
--download-slepc=externalpackages/slepc-3811.tar.gz \
--download-scalapack=externalpackages/scalapack-2.2.0.tar.gz \
--download-mumps=externalpackages/MUMPS_5.5.1.tar.gz \
--download-hdf5=externalpackages/hdf5-1.12.1.tar.gz \
--download-sowing=externalpackages/petsc-pkg-sowing.tar.gz \
--download-metis=externalpackages/petsc-pkg-metis.tar.gz \
--download-parmetis=externalpackages/petsc-pkg-parmetis.tar.gz \
--download-superlu_dist=externalpackages/superlu_dist-8.1.2.tar.gz


### Build PETSc with aocc, aocl and mpich
./configure PETSC_ARCH=linux-aocc-aocl-mpich --with-debugging=0 \
--with-mpi-dir=${MPICH_ROOT} AR=${AR} RANLIB=${RANLIB} \
--CFLAGS="${CFLAGS}" --FFLAGS="${FFLAGS}" --CXXFLAGS="${CXXFLAGS}" --LDFLAGS="${LDFLAGS}" \
--with-blaslapack-dir=${AOCL_ROOT} \
--with-scalapack-dir=${AOCL_ROOT} \
--download-cmake=externalpackages/cmake-3.24.2.tar.gz \
--download-slepc=externalpackages/slepc-3811.tar.gz \
--download-scalapack=externalpackages/scalapack-2.2.0.tar.gz \
--download-mumps=externalpackages/MUMPS_5.5.1.tar.gz \
--download-hdf5=externalpackages/hdf5-1.12.1.tar.gz \
--download-sowing=externalpackages/petsc-pkg-sowing.tar.gz \
--download-metis=externalpackages/petsc-pkg-metis.tar.gz \
--download-parmetis=externalpackages/petsc-pkg-parmetis.tar.gz \
--download-superlu_dist=externalpackages/superlu_dist-8.1.2.tar.gz

It finally works...















