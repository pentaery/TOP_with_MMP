#include "MMA.h"
#include "PreMixFEM_3D.h"
#include "mpi.h"
#include "oCriteria.h"
#include "optimization.h"
#include "system.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petsctime.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>
#include <stdio.h>

int main(int argc, char **argv) {
  PetscCall(
      SlepcInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));

  DM dm1, dm2;
  PetscInt grid;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, grid, grid, grid,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
                         NULL, NULL, &dm1));
  PetscCall(DMSetUp(dm1));
  PetscCall(DMCoarsen(dm1, PETSC_COMM_WORLD, &dm2));
  PetscInt mx, my, mz;
  DMDAGetInfo(dm2, NULL, &mx, &my, &mz, NULL, NULL, NULL, NULL, NULL, NULL,
              NULL, NULL, NULL);
  PetscPrintf(PETSC_COMM_WORLD, "mx = %d, my = %d, mz = %d\n", mx, my, mz);
  PetscCall(SlepcFinalize());
}
