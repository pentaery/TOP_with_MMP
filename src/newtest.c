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

  PetscCall(PC_setup(&test));
  PetscCall(MatView(test.Rcc, PETSC_VIEWER_STDOUT_WORLD));

  Vec rhsc, rhscc;
  Vec xfine, xc, xcc;
  Vec r, rc, rcc;
  PetscCall(DMCreateGlobalVector(test.dm, &r));
  PetscCall(DMCreateGlobalVector(test.dm, &xfine));
  PetscCall(MatCreateVecs(test.Rc, &xc, NULL));
  PetscCall(MatCreateVecs(test.Rc, &rc, NULL));
  PetscCall(MatCreateVecs(test.Rc, &rhsc, NULL));
  PetscCall(MatCreateVecs(test.Rcc, &xcc, NULL));
  PetscCall(MatCreateVecs(test.Rcc, &rcc, NULL));
  PetscCall(MatCreateVecs(test.Rcc, &rhscc, NULL));
  PC pc;
  PetscCall(KSPGetPC(ksp, &pc));
  if (!petsc_default) {
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
    PetscCall(KSPSetErrorIfNotConverged(kspSmoother1, PETSC_TRUE));
    // 设置二阶smoother
    PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother2));
    PetscCall(KSPGetPC(kspSmoother2, &pcSmoother2));
    PetscCall(PCSetType(pcSmoother2, PCBJACOBI));
    PetscCall(KSPSetErrorIfNotConverged(kspSmoother2, PETSC_TRUE));
    // 设置Prolongation
    PetscCall(PCMGSetInterpolation(pc, 2, test.Rc));
    PetscCall(PCMGSetInterpolation(pc, 1, test.Rcc));
    // 设置工作变量
    // PetscCall(PCMGSetX(pc, 0, xcc));
    // PetscCall(PCMGSetRhs(pc, 0, rhscc));
    // PetscCall(PCMGSetX(pc, 1, xc));
    // PetscCall(PCMGSetRhs(pc, 1, rhsc));
    // PetscCall(PCMGSetR(pc, 1, rc));
    // PetscCall(PCMGSetX(pc, 2, xfine));
    // PetscCall(PCMGSetRhs(pc, 2, rhs));
    PetscCall(
        PCShellSetName(pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
  } else {
    PetscCall(PCSetType(pc, PCGAMG));
  }

  PetscCall(KSPSolve(ksp, rhs, t));

  PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  // PetscCall(mmaFinal(&mmax));

  PetscCall(SlepcFinalize());
}
