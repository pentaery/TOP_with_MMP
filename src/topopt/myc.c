
#include "mpi.h"
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "This is a demo.";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscInt sum, i;
  for (i = 0; i < 20; ++i) {
    PetscCallMPI(
        MPI_Allreduce(&i, &sum, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sum: %d\n", sum));
  }

  PetscCall(PetscFinalize());
}