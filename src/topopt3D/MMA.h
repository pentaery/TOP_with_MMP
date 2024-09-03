#include "optimization.h"
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
#define xmin 0.0
#define xmax 1.0
#define epsimin 1e-7
PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode mmaLimit(PCCtx *s_ctx, MMAx *mmax, PetscInt penal);
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec dc);
PetscErrorCode gcmmaSub(PCCtx *s_ctx, MMAx *mmax, Vec dc, PetscScalar *raa,
                        PetscScalar *raa0);
PetscErrorCode subSolv(PCCtx *s_ctx, MMAx *mmax, Vec x);
PetscErrorCode omegaInitial(PCCtx *s_ctx, MMAx *mmax, Vec x);
PetscErrorCode computeResidual(PCCtx *s_ctx, MMAx *mmax, Vec x,
                               PetscScalar epsi, PetscScalar *residumax,
                               PetscScalar *residunorm);
PetscErrorCode computeDelta(PCCtx *s_ctx, MMAx *mmax, Vec x, PetscScalar epsi);
PetscErrorCode findStep(PCCtx *s_ctx, MMAx *mmax, Vec x, PetscScalar *step);
PetscErrorCode omegaUpdate(MMAx *mmax, Vec x, PetscScalar coef);
PetscErrorCode outputTest(MMAx *mmax, Vec dc);
PetscErrorCode raaInit(PCCtx *s_ctx, MMAx *mmax, PetscScalar *raa,
                       PetscScalar *raa0, Vec dc);
PetscErrorCode conCheck(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec xg, Vec dc,
                        PetscScalar *raa, PetscScalar *raa0,
                        PetscBool *conserve);