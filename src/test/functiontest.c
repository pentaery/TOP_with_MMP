#include "PreMixFEM_3D.h"
#include "optimization.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  PetscInt grid = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscInt iter;
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  Vec a, b;
  // PetscScalar cost = 0, change = 0, error = 0;
  // Mat A;
  // Vec rhs, t, x, dc;
  // KSP ksp;
  PetscScalar c = 1.0/3+1;

  PetscPrintf(PETSC_COMM_WORLD, "%f\n", c);
  // PetscCall(PC_init(&test, dom, mesh));
  // PetscCall(PC_print_info(&test));

  // PetscCall(DMCreateMatrix(test.dm, &A));
  // PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  // PetscCall(DMCreateGlobalVector(test.dm, &x));
  // PetscCall(VecSet(x, 0.5));
  // PetscCall(VecSet(rhs, 3.0));
  // PetscCall(VecPointwiseMult(x, x, rhs));
  // PetscCall(VecView(rhs, PETSC_VIEWER_STDOUT_WORLD));
  // PetscCall(DMCreateGlobalVector(test.dm, &t));
  // PetscCall(DMCreateGlobalVector(test.dm, &dc));

  // PetscCall(formBoundary(&test));
  // PetscCall(formkappa(&test, x));
  // PetscCall(formMatrix(&test, A));
  // PetscCall(formRHS(&test, rhs, x));

  // PetscCall(MatDestroy(&A));
  // PetscCall(VecDestroy(&rhs));
  // PetscCall(VecDestroy(&t));
  // PetscCall(VecDestroy(&dc));
  // PetscCall(VecDestroy(&x));
  // PetscCall(KSPDestroy(&ksp));
  // PetscCall(PetscFinalize());
  // PetscCall(DMCreateGlobalVector(test.dm, &a));
  // PetscCall(DMCreateGlobalVector(test.dm, &b));
  // PetscCall(VecSet(a, 1));
  // PetscCall(VecSet(b, 2));
  // PetscCall(VecPointwiseMax(a, a, b));
  // PetscCall(VecView(a, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscFinalize());
}