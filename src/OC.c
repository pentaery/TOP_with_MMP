#include "PreMixFEM_3D.h"
#include "oCriteria.h"
#include "system.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  PetscInt grid = 20;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0;
  PetscScalar change = 1;
  PetscScalar cost0 = 0;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(x, 0.5));
  PetscCall(formBoundary(&test));

  while (change > 0.01) {
    loop += 1;

    PetscViewer viewer;
    sprintf(str, "../data/output/change%04d.vtr", loop);
    PetscCall(
        PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(formkappa(&test, x, 3));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, 3));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", iter));
    if (loop == 1) {
      PetscCall(computeCost1(&test, t, &cost0));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost0: %f\n", cost0));
    }
    PetscCall(computeCost1(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    // PetscCall(computeCost(&test, t, rhs, &cost));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));

    // PetscCall(computeGradient(&test, x, t, dc));
    PetscCall(filter(&test, dc, x));
    PetscCall(optimalCriteria(&test, x, dc, &change));
    // PetscCall(
    //     genOptimalCriteria(&test, x, dc, &g, &glast, &lmid, &change, cost0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}