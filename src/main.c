#include "MMA.h"
#include "PreMixFEM_3D.h"
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
  PCCtx test;
  MMAx mmax;
  PetscInt grid = 20;
  PetscInt iter_number = 60;
  PetscInt output_frequency = 10;
  PetscLogEvent LS1, optimize, setUp;
  PetscBool xvalue_default = PETSC_TRUE, petsc_default = PETSC_FALSE;
  PetscCall(PetscLogEventRegister("LS1", 0, &LS1));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));
  PetscCall(PetscLogEventRegister("setUp", 2, &setUp));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number, NULL));
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-xvalue_default", &xvalue_default));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-frequency", &output_frequency, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  PetscLogDouble mmatime[iter_number];
  PetscInt it[iter_number];
  PetscScalar tem[iter_number];
  Mat A;
  Vec rhs, t, x, dc, y;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));
  PetscCall(DMCreateGlobalVector(test.dm, &y));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)mmax.xlast, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  // PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));

  if (!xvalue_default) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using loaded vector\n"));
    PetscViewer h5_viewer;
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,
                                  "../data/save/xvalue0020.h5", FILE_MODE_READ,
                                  &h5_viewer));
    PetscCall(VecLoad(x, h5_viewer));
    PetscCall(PetscViewerDestroy(&h5_viewer));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using default vector\n"));
    PetscCall(VecSet(x, volfrac));
  }
  PetscCall(VecCopy(x, mmax.xlast));
  // PetscCall(VecSet(mmax.xlast, volfrac));

  PetscCall(formBoundary(&test));
  while (PETSC_TRUE) {
    PetscCall(PetscTime(&mmatime[loop]));
    if (loop >= iter_number) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test.M * test.N * test.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/save/xvalue%04d.h5", loop);
      PetscCall(
          PetscViewerHDF5Open(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(x, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    // if (loop % output_frequency == 0) {
    //   PetscViewer viewer;
    //   sprintf(str, "../data/output2/tvalue%04d.vtr", loop);
    //   PetscCall(
    //       PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE,
    //       &viewer));
    //   PetscCall(VecView(t, viewer));
    //   PetscCall(PetscViewerDestroy(&viewer));
    // }

    PetscCall(formkappa(&test, x, penal));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));

    if (!petsc_default) {
      PetscCall(PetscLogEventBegin(setUp, 0, 0, 0, 0));
      PetscCall(PC_setup(&test));
      PetscCall(PetscLogEventEnd(setUp, 0, 0, 0, 0));
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
      PetscCall(KSPSetTolerances(kspSmoother1, PETSC_DEFAULT, PETSC_DEFAULT,
                                 PETSC_DEFAULT, 1));
      PetscCall(KSPGetPC(kspSmoother1, &pcSmoother1));
      PetscCall(PCSetType(pcSmoother1, PCBJACOBI));
      // 设置二阶smoother
      PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother2));
      PetscCall(KSPSetTolerances(kspSmoother2, PETSC_DEFAULT, PETSC_DEFAULT,
                                 PETSC_DEFAULT, 1));
      PetscCall(KSPGetPC(kspSmoother2, &pcSmoother2));
      PetscCall(PCSetType(pcSmoother2, PCBJACOBI));
      // 设置Prolongation
      PetscCall(PCMGSetInterpolation(pc, 2, test.Rc));
      PetscCall(PCMGSetInterpolation(pc, 1, test.Rcc));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Using GAMG\n"));
      PetscCall(PCSetType(pc, PCGAMG));
    }

    PetscCall(PetscLogEventBegin(LS1, 0, 0, 0, 0));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(PetscLogEventEnd(LS1, 0, 0, 0, 0));

    PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    it[loop - 1] = iter;
    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL;
    tau /= f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));
    PetscCall(computeCostMMA(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    tem[loop - 1] = cost;

    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test, &mmax, loop));
    PetscCall(mmaSub(&test, &mmax, dc));
    PetscCall(subSolv(&test, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
    PetscCall(PetscTimeSubtract(&mmatime[loop - 1]));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MMA time: \n"));
  for (loop = 0; loop < iter_number; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f, ", -mmatime[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration: \n"));
  for (loop = 0; loop < iter_number; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d, ", it[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost function: \n"));
  for (loop = 0; loop < iter_number; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f, ", tem[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(PC_final(&test));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));

  PetscCall(SlepcFinalize());
}
