#include "func.h"
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "A test for TOOP.";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  Mat A;
  Vec x, xold;
  Vec dc;
  Vec b, u;
  KSP ksp;
  PetscInt M = 3, N = 3;
  PetscReal cost = 0;

  PetscReal change = 1;
  PetscInt loop = 0;
  PetscReal volfrac = (M - 1) * (N - 1) * 0.5;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 2,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateMatrix(dm, &A));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &xold));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &dc));
  PetscCall(formx(dm, x, M, N));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  PetscCall(formRHS(dm, b, N));

  while (change > 0.01) {
    loop += 1;
    PetscCall(VecCopy(x, xold));
    PetscCall(formMatrix(dm, A, x, M, N));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, b, u));
    PetscCall(computeCost(dm, &cost, u, dc, x, M, N));
    // PetscCall(filter(dm, dc, x, M, N));
    PetscCall(optimalCriteria(dm, x, dc, volfrac, M, N));
    // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecAXPY(xold, -1, x));

    PetscCall(VecNorm(xold, NORM_INFINITY, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "norm: %f\n", change));
    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
    // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    // if (loop == 1) {
    //   break;
    // }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d", loop));
  // PetscCall(VecView(dc, PETSC_VIEWER_STDOUT_WORLD));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "total cost: %f\n", cost));

  // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}
