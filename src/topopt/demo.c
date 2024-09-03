#include "func.h"
#include <petsc.h>
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
static char help[] = "This is a demo.";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  PetscInt M = 3, N = 3;
  Mat A;
  Vec x, b, u, dc;
  KSP ksp;
  PetscScalar cost;
  PetscScalar volfrac = (M - 1) * (N - 1) * 0.3;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 2,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &dc));
  PetscCall(formx(dm, x, M, N));
  PetscCall(formMatrix(dm, A, x, M, N));
  PetscCall(formRHS(dm, b, N));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSolve(ksp, b, u));

  PetscCall(computeCost(dm, &cost, u, dc, x, M, N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));

  PetscCall(optimalCriteria(dm, x, dc, volfrac, M, N));


  PetscCall(PetscFinalize());
}