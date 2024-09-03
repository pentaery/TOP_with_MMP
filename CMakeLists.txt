#
#  This is a sample CMakeLists.txt suitable for direct use with a new PETSc application or
#  you can add fragments of material below to an existing application CMakeLists.txt
#
#  Usage:  To build ex1 from ex1.c
#    rm -fr build
#    mkdir build
#    cd build
#    cmake ..
#    make
#    ./ex1
#
#  By default it gets both the compiler information and the library information from PETSc
#
cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_TYPE Debug)

# Use verbose output.
set(CMAKE_VERBOSE_MAKEFILE ON)

# set the root location to find PETSc's pkg-config
# set(ENV{PETSC_DIR} "/home/cqye/petsc")
# set(ENV{PETSC_ARCH} "linux-oneapi-int64")
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

# Remove the lines below if you do not wish to have PETSc determine the compilers
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
if (CXX_COMPILER)
  SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
endif (CXX_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=fcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE FORTRAN_COMPILER)
if (FORTRAN_COMPILER)
  SET(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER})
  enable_language(Fortran)
endif (FORTRAN_COMPILER)

# Get flags from PETSc.pc
execute_process ( COMMAND pkg-config PETSc --variable=cflags_extra COMMAND tr -d '\n' OUTPUT_VARIABLE C_FLAGS)
SET(CMAKE_C_FLAGS ${C_FLAGS})
execute_process ( COMMAND pkg-config PETSc --variable=cxxflags_extra COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_FLAGS)
if (CXX_FLAGS)
  SET(CMAKE_CXX_FLAGS ${CXX_FLAGS})
endif (CXX_FLAGS)

#  tells CMake to build the application ex1 from the source file ex1.c
#  this must appear AFTER the compilers are set
project(PreMixFEM)
set(SOURCES 
 src/PreMixFEM_3D.c
 src/main.c
 src/topopt3D/optimization.c
 src/topopt3D/system.c
 src/topopt3D/oCriteria.c
 src/topopt3D/MMA.c
)
SET(OUTPUT_NAME test)
add_executable(${OUTPUT_NAME} ${SOURCES})
target_include_directories(${OUTPUT_NAME}
  PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/topopt3D
)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET slepc)

#  Learned from https://stackoverflow.com/questions/8774593/cmake-link-to-external-library
#  Thanks!
# add_library(slepc SHARED IMPORTED)
# set_target_properties(slepc PROPERTIES IMPORTED_LOCATION ${PETSC}/lib/libslepc.so)
target_link_libraries(${OUTPUT_NAME} PkgConfig::PETSC PkgConfig::SLEPC m)
