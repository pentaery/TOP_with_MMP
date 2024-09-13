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
#include <petscsys.h>
#include <petscsystypes.h>
#include <petsctime.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>

int main(int argc, char **argv) {
  PetscCall(
      SlepcInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  MMAx mmax;
  PetscInt grid = 20;
  PetscInt iter_number1 = 90, iter_number2 = 60, iter_number3 = 30,
           output_frequency = 30;
  PetscBool petsc_default = PETSC_FALSE;
  PetscLogEvent LS1, optimize, setUp;
  PetscLogStage stage1, stage2, stage3;
  PetscCall(PetscLogEventRegister("LS1", 0, &LS1));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));
  PetscCall(PetscLogEventRegister("setUp", 2, &setUp));
  PetscCall(PetscLogStageRegister("Stage1", &stage1));
  PetscCall(PetscLogStageRegister("Stage2", &stage2));
  PetscCall(PetscLogStageRegister("Stage3", &stage3));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number1, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter2", &iter_number2, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter3", &iter_number3, NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-frequency", &output_frequency, NULL));
  PetscLogDouble mmatime[iter_number1 + iter_number2 + iter_number3];
  PetscInt it[iter_number1 + iter_number2 + iter_number3];
  PetscScalar tem[iter_number1 + iter_number2 + iter_number3];
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;

  char str[80];

  PetscCall(PetscLogStagePush(stage1));

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(mmax.xlast, volfrac));
  PetscCall(VecSet(x, volfrac));
  PetscCall(formBoundary(&test));
  while (PETSC_TRUE) {
    PetscCall(PetscTime(&mmatime[loop]));
    if (loop == iter_number1) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test.M * test.N * test.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output1/xvalue%04d.h5", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
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
    PetscCall(PCSetType(pc, PCGAMG));

    PetscCall(KSPSolve(ksp, rhs, t));

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

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));

  PetscCall(KSPDestroy(&ksp));
  Vec xmid, xlastmid, xllastmid, xlllastmid;
  PetscCall(DMCreateGlobalVector(test.dm, &xmid));
  PetscCall(DMCreateGlobalVector(test.dm, &xlastmid));
  PetscCall(DMCreateGlobalVector(test.dm, &xllastmid));
  PetscCall(DMCreateGlobalVector(test.dm, &xlllastmid));
  PetscCall(VecCopy(x, xmid));
  PetscCall(VecCopy(mmax.xlast, xlastmid));
  PetscCall(VecCopy(mmax.xllast, xllastmid));
  PetscCall(VecCopy(mmax.xlllast, xlllastmid));
  PetscCall(VecDestroy(&x));
  PetscCall(mmaFinal(&mmax));

  PetscCall(PetscLogStagePop());

  // round 2

  PetscCall(PetscLogStagePush(stage2));
  PCCtx test2;
  mesh[0] = grid * 2;
  mesh[1] = grid * 2;
  mesh[2] = grid * 2;
  PetscCall(PC_init(&test2, dom, mesh));
  PetscCall(mmaInit(&test2, &mmax));

  PetscCall(DMCreateMatrix(test2.dm, &A));
  PetscCall(DMCreateGlobalVector(test2.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test2.dm, &x));
  PetscCall(DMCreateGlobalVector(test2.dm, &t));
  PetscCall(DMCreateGlobalVector(test2.dm, &dc));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(xScaling(test.dm, test2.dm, xmid, x));
  PetscCall(xScaling(test.dm, test2.dm, xlastmid, mmax.xlast));
  PetscCall(xScaling(test.dm, test2.dm, xllastmid, mmax.xllast));
  PetscCall(xScaling(test.dm, test2.dm, xlllastmid, mmax.xlllast));
  PetscCall(PC_final(&test));
  PetscCall(VecDestroy(&xmid));
  PetscCall(VecDestroy(&xlastmid));
  PetscCall(VecDestroy(&xllastmid));
  PetscCall(VecDestroy(&xlllastmid));
  PetscCall(formBoundary(&test2));
  while (PETSC_TRUE) {
    PetscCall(PetscTime(&mmatime[loop]));
    if (loop == iter_number1 + iter_number2) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test2.M * test2.N * test2.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output1/xvalue%04d.vtr", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(x, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    // if (loop % output_frequency == 0) {
    //    PetscViewer viewer;
    //    sprintf(str, "../data/output2/tvalue%04d.vtr", loop);
    //    PetscCall(
    //        PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE,
    //        &viewer));
    //    PetscCall(VecView(t, viewer));
    //    PetscCall(PetscViewerDestroy(&viewer));
    // }

    PetscCall(formkappa(&test2, x, penal));
    PetscCall(formMatrix(&test2, A));
    PetscCall(formRHS(&test2, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));

    PetscCall(KSPSolve(ksp, rhs, t));

    PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    it[loop - 1] = iter;
    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL;
    tau /= f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));

    PetscCall(computeCostMMA(&test2, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    tem[loop - 1] = cost;
    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test2, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test2, &mmax, loop));
    PetscCall(mmaSub(&test2, &mmax, dc));
    PetscCall(subSolv(&test2, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
    PetscCall(PetscTimeSubtract(&mmatime[loop - 1]));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(DMCreateGlobalVector(test2.dm, &xmid));
  PetscCall(DMCreateGlobalVector(test2.dm, &xlastmid));
  PetscCall(DMCreateGlobalVector(test2.dm, &xllastmid));
  PetscCall(DMCreateGlobalVector(test2.dm, &xlllastmid));
  PetscCall(VecCopy(x, xmid));
  PetscCall(VecCopy(mmax.xlast, xlastmid));
  PetscCall(VecCopy(mmax.xllast, xllastmid));
  PetscCall(VecCopy(mmax.xlllast, xlllastmid));
  PetscCall(VecDestroy(&x));
  PetscCall(mmaFinal(&mmax));
  PetscCall(PetscLogStagePop());

  // round3

  PetscCall(PetscLogStagePush(stage3));
  PCCtx test3;
  mesh[0] = grid * 4;
  mesh[1] = grid * 4;
  mesh[2] = grid * 4;
  PetscCall(PC_init(&test3, dom, mesh));
  PetscCall(mmaInit(&test3, &mmax));

  PetscCall(DMCreateMatrix(test3.dm, &A));
  PetscCall(DMCreateGlobalVector(test3.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test3.dm, &x));
  PetscCall(DMCreateGlobalVector(test3.dm, &t));
  PetscCall(DMCreateGlobalVector(test3.dm, &dc));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(xScaling(test2.dm, test3.dm, xmid, x));
  PetscCall(xScaling(test2.dm, test3.dm, xlastmid, mmax.xlast));
  PetscCall(xScaling(test2.dm, test3.dm, xllastmid, mmax.xllast));
  PetscCall(xScaling(test2.dm, test3.dm, xlllastmid, mmax.xlllast));
  PetscCall(PC_final(&test2));
  PetscCall(VecDestroy(&xmid));
  PetscCall(VecDestroy(&xlastmid));
  PetscCall(VecDestroy(&xllastmid));
  PetscCall(VecDestroy(&xlllastmid));
  PetscCall(formBoundary(&test3));
  while (PETSC_TRUE) {
    PetscCall(PetscTime(&mmatime[loop]));
    if (loop == iter_number1 + iter_number2 + iter_number3) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test3.M * test3.N * test3.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output1/xvalue%04d.h5", loop);
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
    PetscCall(formkappa(&test3, x, penal));
    PetscCall(formMatrix(&test3, A));
    PetscCall(formRHS(&test3, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (!petsc_default) {
      PetscCall(PetscLogEventBegin(setUp, 0, 0, 0, 0));
      PetscCall(PC_setup(&test3));
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
      PetscCall(KSPGetPC(kspSmoother1, &pcSmoother1));
      PetscCall(PCSetType(pcSmoother1, PCBJACOBI));
      PetscCall(KSPSetErrorIfNotConverged(kspSmoother1, PETSC_TRUE));
      // 设置二阶smoother
      PetscCall(PCMGGetSmoother(pc, 1, &kspSmoother2));
      PetscCall(KSPGetPC(kspSmoother2, &pcSmoother2));
      PetscCall(PCSetType(pcSmoother2, PCBJACOBI));
      PetscCall(KSPSetErrorIfNotConverged(kspSmoother2, PETSC_TRUE));
      // 设置Prolongation
      PetscCall(PCMGSetInterpolation(pc, 2, test3.Rc));
      PetscCall(PCMGSetInterpolation(pc, 1, test3.Rcc));
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

    PetscCall(computeCostMMA(&test3, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    tem[loop - 1] = cost;
    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test3, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test3, &mmax, loop));
    PetscCall(mmaSub(&test3, &mmax, dc));
    PetscCall(subSolv(&test3, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
    PetscCall(PetscTimeSubtract(&mmatime[loop - 1]));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MMA time: \n"));
  for (loop = 0; loop < iter_number1 + iter_number2 + iter_number3; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f, ", -mmatime[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration: \n"));
  for (loop = 0; loop < iter_number1 + iter_number2 + iter_number3; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%d, ", it[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost function: \n"));
  for (loop = 0; loop < iter_number1 + iter_number2 + iter_number3; ++loop) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f, ", tem[loop]));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));

  PetscCall(PC_final(&test3));

  PetscCall(PetscLogStagePop());

  PetscCall(SlepcFinalize());
}
