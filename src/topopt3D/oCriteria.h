#pragma once
#include <PreMixFEM_3D.h>
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
#define rmin 1.1
PetscErrorCode computeGradient(PCCtx *s_ctx, Vec x, Vec t, Vec dc,
                               PetscInt penal);
PetscErrorCode filter(PCCtx *s_ctx, Vec dc, Vec x);
PetscErrorCode computeCost(PCCtx *s_ctx, Vec t, Vec rhs, PetscScalar *cost);
PetscErrorCode optimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change);

PetscErrorCode genOptimalCriteria(PCCtx *s_ctx, Vec x, Vec dc, PetscScalar *g,
                                  PetscScalar *glast, PetscScalar *lmid,
                                  PetscScalar *change, PetscScalar cost0);
PetscErrorCode computeCost1(PCCtx *s_ctx, Vec t, PetscScalar *cost);