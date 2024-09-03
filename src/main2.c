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
  PetscInt iter_number1 = 10, iter_number2 = 10, output_frequency = 20;
  PetscLogEvent linearsolve, optimize;
  PetscBool petsc_default = PETSC_FALSE;
  PetscCall(PetscLogEventRegister("LinearSolve", 0, &linearsolve));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number1, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter2", &iter_number2, NULL));
  PetscInt transform = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-transform", &transform, NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-frequency", &output_frequency, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPGCR));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(mmax.xlast, volfrac));
  PetscCall(VecSet(x, volfrac));
  PetscCall(formBoundary(&test));
  while (PETSC_TRUE) {
    if (loop == iter_number1) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test.M * test.N * test.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output1/xvalue%04d.vtr", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(x, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output2/tvalue%04d.vtr", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(t, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(formkappa(&test, x, penal));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (!petsc_default) {
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &test));
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    } else {
      PetscCall(PCSetType(pc, PCGAMG));
    }
    PetscCall(KSPSolve(ksp, rhs, t));
    if(!petsc_default){
      PetscCall(PC_print_stat(&test));
    }
    PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, "The iteration number is: %d\n", iter));
    PetscCall(KSPGetIterationNumber(ksp, &iter));

    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL;
    tau /= f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));

    PetscCall(computeCostMMA(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test, &mmax, loop));
    PetscCall(mmaSub(&test, &mmax, dc));
    PetscCall(subSolv(&test, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
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


  // round 2

  PCCtx test2;
  mesh[0] = grid * transform;
  mesh[1] = grid * transform;
  mesh[2] = grid * transform;
  PetscCall(PC_init(&test2, dom, mesh));
  PetscCall(mmaInit(&test2, &mmax));
  PetscCall(PC_print_info(&test2));

  PetscCall(DMCreateMatrix(test2.dm, &A));
  PetscCall(DMCreateGlobalVector(test2.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test2.dm, &x));
  PetscCall(DMCreateGlobalVector(test2.dm, &t));
  PetscCall(DMCreateGlobalVector(test2.dm, &dc));
  PetscCall(PetscObjectSetName((PetscObject)x, "Design Variable"));
  PetscCall(PetscObjectSetName((PetscObject)t, "Temperature"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPGCR));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(xScaling(test.dm, test2.dm, xmid, x));
  PetscCall(xScaling(test.dm, test2.dm, xlastmid, mmax.xlast));
  PetscCall(xScaling(test.dm, test2.dm, xllastmid, mmax.xllast));
  PetscCall(xScaling(test.dm, test2.dm, xlllastmid, mmax.xlllast));
  PetscCall(formBoundary(&test2));
  while (PETSC_TRUE) {
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
    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output2/tvalue%04d.vtr", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(t, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(formkappa(&test2, x, penal));
    PetscCall(formMatrix(&test2, A));
    PetscCall(formRHS(&test2, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (!petsc_default) {
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &test2));
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    } else {
      PetscCall(PCSetType(pc, PCGAMG));
    }
    if(loop > 25){
      PetscCall(PetscLogEventBegin(linearsolve, 0, 0, 0, 0));
    }
    PetscCall(KSPSolve(ksp, rhs, t));
    if(loop > 25){
      PetscCall(PetscLogEventEnd(linearsolve, 0, 0, 0, 0));
    }
    if (!petsc_default) {
      PetscCall(PC_print_stat(&test));
    }
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(
        PetscPrintf(PETSC_COMM_WORLD, "The iteration number is: %d\n", iter));
    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL;
    tau /= f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));

    PetscCall(computeCostMMA(&test2, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test2, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test2, &mmax, loop));
    PetscCall(mmaSub(&test2, &mmax, dc));
    PetscCall(subSolv(&test2, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));
  if (!petsc_default) {
    PetscCall(PC_final(&test2));
  }
  PetscCall(PetscFinalize());
}
