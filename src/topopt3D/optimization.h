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

#define mmas 0.85
#define mmas0 0.15
#define artificial 1
#define MMA_M 1

typedef struct mma_text {
  Vec mmaL, mmaU, mmaLlast, mmaUlast, alpha, beta, xlast, xllast, xlllast, lbd,
      ubd, xsign, dgT, zzz1, zzz2, zzz;
  Vec p0, q0, p[MMA_M], q[MMA_M], b[MMA_M], g[MMA_M];
  PetscScalar bval[MMA_M];
  Vec rex, dx, delx, diagx;
  PetscScalar y[MMA_M], rey[MMA_M], dy[MMA_M], dely[MMA_M];
  PetscScalar z, rez, dz, delz;
  PetscScalar lam[MMA_M], relam[MMA_M], dlam[MMA_M];
  Vec xsi, eta, rexsi, reeta, gvec[MMA_M], dxsi, deta;
  PetscScalar mu[MMA_M], remu[MMA_M], dmu[MMA_M];
  PetscScalar zet, rezet, dzet;
  PetscScalar s[MMA_M], res[MMA_M], ds[MMA_M];
  PetscScalar c[MMA_M], d[MMA_M], a[MMA_M], a0;
  Vec temp, temp1, temp2, temp3;
} MMAx;

PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode adjointGradient(PCCtx *s_ctx, MMAx *mma_text,KSP ksp, Mat A, Vec x,
                               Vec t, Vec dc, PetscInt penal);
PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                                PetscInt penal);

PetscErrorCode formLimit(PCCtx *s_ctx, MMAx *mma_text, PetscInt loop);
PetscErrorCode computeCostMMA(PCCtx *s_ctx, Vec t, PetscScalar *cost);
PetscErrorCode computeDerivative(PCCtx *s_ctx, PetscScalar y,
                                 PetscScalar *derivative, PetscScalar *dpartial,
                                 MMAx *mma_text, Vec dc, Vec x);
PetscErrorCode computeDerivative1(PCCtx *s_ctx, PetscScalar y,
                                  PetscScalar *derivative, MMAx *mma_text,
                                  Vec dc, Vec x);
PetscErrorCode mma(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                   PetscScalar *initial);
PetscErrorCode mma1(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                    PetscScalar *initial);
PetscErrorCode mma2(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                    PetscScalar *initial);

PetscErrorCode findX(PCCtx *s_ctx, PetscScalar y, MMAx *mma_text, Vec dc,
                     Vec x);
PetscErrorCode computeChange(MMAx *mma_text, Vec x, PetscScalar *change);

PetscErrorCode mmatest(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                       PetscScalar *initial);

PetscErrorCode computeWy(PCCtx *s_ctx, PetscScalar y, PetscScalar *wy,
                         MMAx *mma_text, Vec dc, Vec x);