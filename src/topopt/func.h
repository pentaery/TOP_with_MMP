#pragma once
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

PetscErrorCode(formx(DM dm, Vec x, PetscInt M, PetscInt N));
PetscErrorCode formKE(PetscScalar KE[8][8], PetscScalar coef);
PetscErrorCode formMatrix(DM dm, Mat A, Vec x, PetscInt M, PetscInt N);
PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N);
PetscErrorCode computeCost(DM dm, PetscScalar *cost, Vec u, Vec dc, Vec x,
                           PetscInt M, PetscInt N);
PetscErrorCode filter(DM dm, Vec dc, Vec x, PetscInt M, PetscInt N);
PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscScalar volfrac,
                               PetscInt M, PetscInt N);