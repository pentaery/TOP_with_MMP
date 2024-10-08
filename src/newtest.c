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
#include <petscpctypes.h>
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
  PCCtx test;
  MMAx mmax;
  PetscInt grid = 8;
  PetscInt iter_number = 60;
  PetscBool petsc_default = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};

  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt penal = 3;

  PetscCall(PC_init(&test, dom, mesh));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPGCR));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  // PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(VecSet(x, volfrac));

  PetscCall(formBoundary(&test));

  PetscCall(formkappa(&test, x, penal));
  PetscCall(formMatrix(&test, A));
  PetscCall(formRHS(&test, rhs, x, penal));
  PetscCall(KSPSetOperators(ksp, A, A));

  // PetscCall(MatView(test.Rcc, PETSC_VIEWER_STDOUT_WORLD));
  PetscLogDouble time;
  PetscTime(&time);
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  if (!petsc_default) {
    PetscCall(PC_setup(&test));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using GMsFEM\n"));
    KSP kspCoarse, kspSmoother1, kspSmoother2;
    PC pcCoarse, pcSmoother1, pcSmoother2;
    // 设置三层multigrid
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, 3, NULL));
    // 设为V-cycle
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    // 设置coarse solver
    PetscCall(PCMGGetCoarseSolve(pc, &kspCoarse));
    PetscCall(KSPSetType(kspCoarse, KSPPREONLY));
    PetscCall(KSPGetPC(kspCoarse, &pcCoarse));
    PetscCall(PCSetType(pcCoarse, PCLU));
    PetscCall(PCFactorSetMatSolverType(pcCoarse, MATSOLVERSUPERLU_DIST));
    PetscCall(KSPSetErrorIfNotConverged(kspCoarse, PETSC_TRUE));
    // 设置一阶smoother
    PetscCall(PCMGGetSmoother(pc, 2, &kspSmoother1));
    PetscCall(KSPGetPC(kspSmoother1, &pcSmoother1));
    PetscCall(PCSetType(pcSmoother1, PCBJACOBI));
    PetscCall(
        KSPSetTolerances(kspSmoother1, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 5));
    // 设置二阶smoother
    PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother2));
    PetscCall(KSPGetPC(kspSmoother2, &pcSmoother2));
    PetscCall(PCSetType(pcSmoother2, PCBJACOBI));
    PetscCall(
        KSPSetTolerances(kspSmoother2, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 5));
    // 设置Prolongation
    PetscCall(PCMGSetInterpolation(pc, 2, test.Rc));
    PetscCall(PCMGSetInterpolation(pc, 1, test.Rcc));
    PetscCall(
        PCShellSetName(pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using GAMG\n"));
    PetscCall(PCSetType(pc, PCGAMG));
  }

  PetscCall(KSPSolve(ksp, rhs, t));
  PetscTimeSubtract(&time);
  PetscPrintf(PETSC_COMM_WORLD, "Time: %f\n", -time);

  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  // PetscCall(mmaFinal(&mmax));
  PetscCall(PC_final(&test));

  PetscCall(SlepcFinalize());
}
