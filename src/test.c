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
  PetscLogEvent linearsolve, optimize;
  PetscBool petsc_default = PETSC_FALSE;
  char filename[80] = "hello.txt";
  PetscCall(PetscLogEventRegister("LinearSolve", 0, &linearsolve));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-frequency", &output_frequency, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-itername", filename, 20, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  PetscInt it[iter_number];
  PetscScalar tem[iter_number];
  Mat A;
  Vec rhs, t, x, dc, y;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;
  PetscScalar residual;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));
  PetscCall(PC_print_info(&test));

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
  PetscCall(KSPSetType(ksp, KSPGCR));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  // PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetFromOptions(ksp));



  DM dm1;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 128,
                         128, 128, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                         NULL, NULL, NULL, &dm1));
  PetscCall(DMSetUp(dm1));
  Vec xcopy;
  PetscCall(DMCreateGlobalVector(dm1, &xcopy));
  PetscCall(PetscObjectSetName((PetscObject)xcopy, "Design Variable"));


  PetscViewer h5_viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,
                                "../../data/save/xvalue0025.h5", FILE_MODE_READ,
                                &h5_viewer));
  PetscCall(VecLoad(xcopy, h5_viewer));
  PetscCall(PetscViewerDestroy(&h5_viewer));
  PetscCall(xScaling(dm1, test.dm, xcopy, x));

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,
                                "../../data/save/xvalue0020.h5", FILE_MODE_READ,
                                &h5_viewer));
  PetscCall(VecLoad(xcopy, h5_viewer));
  PetscCall(PetscViewerDestroy(&h5_viewer));
  PetscCall(xScaling(dm1, test.dm, xcopy, mmax.xlast));

  PetscCall(DMDestroy(&dm1));
  PetscCall(VecDestroy(&xcopy));







  PetscCall(formBoundary(&test));
  while (PETSC_TRUE) {
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
    //       PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
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
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &test));
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    } else {
      PetscCall(PCSetType(pc, PCGAMG));
    }
    if(loop == 1){
    PetscCall(PetscLogEventBegin(linearsolve, 0, 0, 0, 0));
    }
    PetscCall(KSPSolve(ksp, rhs, t));
    if(loop == 1){
    PetscCall(PetscLogEventEnd(linearsolve, 0, 0, 0, 0));
    }
    if(!petsc_default){
      PetscCall(PC_print_stat(&test));
    }

    PetscCall(KSPConvergedReasonView(ksp, PETSC_VIEWER_STDOUT_WORLD));
    // PetscCall(MatMult(A, t, y));
    // PetscCall(VecAXPY(y, -1.0, rhs));
    // PetscCall(VecNorm(y, NORM_2, &residual));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The norm of residual is: %f\n", residual));
    // PetscCall(VecNorm(rhs, NORM_2, &residual));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "The norm of RHS is: %f\n", residual));
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


  }
  
  if (!petsc_default) {
    PetscCall(PC_final(&test));
  }



  FILE *fp = fopen(filename, "w");
  if(fp == NULL){
    printf("Error opening file\n");
    return 1;
  }
  for(int i = 0; i < iter_number; i++){
    fprintf(fp, "%d ", it[i]);
  }
  fprintf(fp, "\n");
  for(int i = 0; i < iter_number; i++){
    fprintf(fp, "%f ", tem[i]);
  }
  fclose(fp);



  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));

  PetscCall(SlepcFinalize());
}
