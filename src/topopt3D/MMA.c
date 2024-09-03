#include "MMA.h"
#include "system.h"
#include <PreMixFEM_3D.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text) {
  PetscFunctionBeginUser;
  PetscInt i = 0;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaL));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaU));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xllast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xlllast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaLlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaUlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->alpha));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->beta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->lbd));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->ubd));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xsign));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->dgT));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->p0));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->q0));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->eta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->reeta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->deta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xsi));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->rexsi));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->dxsi));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->rex));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->dx));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->delx));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->diagx));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->rexsi));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->reeta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->zzz1));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->zzz2));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->zzz));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->temp));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->temp1));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->temp2));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->temp3));

  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->p[i]));
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->q[i]));
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->gvec[i]));
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->b[i]));
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->g[i]));
    mma_text->a[i] = 0;
    mma_text->d[i] = 1;
    mma_text->c[i] = 1000;
  }
  mma_text->a0 = 1;
  PetscCall(VecSet(mma_text->dgT, 1.0 / s_ctx->M / s_ctx->N / s_ctx->P));
  PetscCall(VecSet(mma_text->xlast, volfrac));
  PetscCall(VecSet(mma_text->lbd, xmin));
  PetscCall(VecSet(mma_text->ubd, xmax));
  PetscFunctionReturn(0);
}

PetscErrorCode mmaFinal(MMAx *mma_text) {
  PetscFunctionBeginUser;
  PetscInt i = 0;
  PetscCall(VecDestroy(&mma_text->mmaL));
  PetscCall(VecDestroy(&mma_text->mmaU));
  PetscCall(VecDestroy(&mma_text->xlast));
  PetscCall(VecDestroy(&mma_text->xllast));
  PetscCall(VecDestroy(&mma_text->xlllast));
  PetscCall(VecDestroy(&mma_text->mmaLlast));
  PetscCall(VecDestroy(&mma_text->mmaUlast));
  PetscCall(VecDestroy(&mma_text->alpha));
  PetscCall(VecDestroy(&mma_text->beta));
  PetscCall(VecDestroy(&mma_text->lbd));
  PetscCall(VecDestroy(&mma_text->ubd));
  PetscCall(VecDestroy(&mma_text->xsign));
  PetscCall(VecDestroy(&mma_text->dgT));
  PetscCall(VecDestroy(&mma_text->p0));
  PetscCall(VecDestroy(&mma_text->q0));
  PetscCall(VecDestroy(&mma_text->eta));
  PetscCall(VecDestroy(&mma_text->reeta));
  PetscCall(VecDestroy(&mma_text->deta));
  PetscCall(VecDestroy(&mma_text->xsi));
  PetscCall(VecDestroy(&mma_text->rexsi));
  PetscCall(VecDestroy(&mma_text->dxsi));
  PetscCall(VecDestroy(&mma_text->rex));
  PetscCall(VecDestroy(&mma_text->dx));
  PetscCall(VecDestroy(&mma_text->delx));
  PetscCall(VecDestroy(&mma_text->diagx));
  PetscCall(VecDestroy(&mma_text->rexsi));
  PetscCall(VecDestroy(&mma_text->reeta));
  PetscCall(VecDestroy(&mma_text->zzz1));
  PetscCall(VecDestroy(&mma_text->zzz2));
  PetscCall(VecDestroy(&mma_text->zzz));
  PetscCall(VecDestroy(&mma_text->temp));
  PetscCall(VecDestroy(&mma_text->temp1));
  PetscCall(VecDestroy(&mma_text->temp2));
  PetscCall(VecDestroy(&mma_text->temp3));

  for (i = 0; i < MMA_M; i++) {
    PetscCall(VecDestroy(&mma_text->p[i]));
    PetscCall(VecDestroy(&mma_text->q[i]));
    PetscCall(VecDestroy(&mma_text->gvec[i]));
    PetscCall(VecDestroy(&mma_text->b[i]));
    PetscCall(VecDestroy(&mma_text->g[i]));
  }
  PetscFunctionReturn(0);
}
PetscErrorCode mmaLimit(PCCtx *s_ctx, MMAx *mmax, PetscInt loop) {
  PetscFunctionBeginUser;
  PetscScalar asyinit = 0.5;
  PetscScalar asyincr = 1.0 / 0.85;
  PetscScalar asydecr = 0.85;
  PetscScalar sign = 0;
  PetscScalar albefa = 0.1;
  PetscScalar move = 0.5;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***xval, ***xold1, ***xold2, ***low, ***upp;
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlast, &xval));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xllast, &xold1));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlllast, &xold2));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  if (loop <= 2) {
    // PetscCall(VecWAXPY(mmax->mmaL, -asyinit, mmax->ubd, mmax->xlast));
    // PetscCall(VecAXPY(mmax->mmaL, asyinit, mmax->lbd));
    // PetscCall(VecWAXPY(mmax->mmaU, asyinit, mmax->ubd, mmax->xlast));
    // PetscCall(VecAXPY(mmax->mmaU, -asyinit, mmax->lbd));
    PetscCall(VecSet(mmax->mmaL, -0.15));
    PetscCall(VecSet(mmax->mmaU, 0.15));
  } else {
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->mmaL, &low));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->mmaU, &upp));
    for (ez = startz; ez < startz + nz; ++ez) {
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          sign = (xval[ez][ey][ex] - xold1[ez][ey][ex]) *
                 (xold1[ez][ey][ex] - xold2[ez][ey][ex]);
          if (sign > 0) {
            low[ez][ey][ex] = xval[ez][ey][ex] -
                              asyincr * (xold1[ez][ey][ex] - low[ez][ey][ex]);

            upp[ez][ey][ex] = xval[ez][ey][ex] +
                              asyincr * (upp[ez][ey][ex] - xold1[ez][ey][ex]);

          } else if (sign < 0) {
            low[ez][ey][ex] = xval[ez][ey][ex] -
                              asydecr * (xold1[ez][ey][ex] - low[ez][ey][ex]);
            upp[ez][ey][ex] = xval[ez][ey][ex] +
                              asydecr * (upp[ez][ey][ex] - xold1[ez][ey][ex]);
          } else {
            low[ez][ey][ex] =
                xval[ez][ey][ex] - (xold1[ez][ey][ex] - low[ez][ey][ex]);
            upp[ez][ey][ex] =
                xval[ez][ey][ex] + (upp[ez][ey][ex] - xold1[ez][ey][ex]);
          }
          low[ez][ey][ex] = PetscMax(low[ez][ey][ex],
                                     xval[ez][ey][ex] - 10.0 * (xmax - xmin));
          low[ez][ey][ex] = PetscMin(low[ez][ey][ex],
                                     xval[ez][ey][ex] - 0.01 * (xmax - xmin));
          upp[ez][ey][ex] = PetscMax(upp[ez][ey][ex],
                                     xval[ez][ey][ex] + 0.01 * (xmax - xmin));
          upp[ez][ey][ex] = PetscMin(upp[ez][ey][ex],
                                     xval[ez][ey][ex] + 10.0 * (xmax - xmin));
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->mmaL, &low));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->mmaU, &upp));
  }

  PetscCall(VecCopy(mmax->mmaL, mmax->zzz1));
  PetscCall(
      VecAXPBYPCZ(mmax->zzz1, albefa, -albefa, 1, mmax->xlast, mmax->mmaL));
  PetscCall(VecCopy(mmax->xlast, mmax->zzz2));
  PetscCall(VecShift(mmax->zzz2, move * (xmin - xmax)));
  PetscCall(VecPointwiseMax(mmax->zzz, mmax->zzz1, mmax->zzz2));
  PetscCall(VecPointwiseMax(mmax->alpha, mmax->zzz, mmax->lbd));

  PetscCall(VecCopy(mmax->mmaU, mmax->zzz1));
  PetscCall(
      VecAXPBYPCZ(mmax->zzz1, albefa, -albefa, 1, mmax->xlast, mmax->mmaU));
  PetscCall(VecCopy(mmax->xlast, mmax->zzz2));
  PetscCall(VecShift(mmax->zzz2, move * (xmax - xmin)));
  PetscCall(VecPointwiseMin(mmax->zzz, mmax->zzz1, mmax->zzz2));
  PetscCall(VecPointwiseMin(mmax->beta, mmax->zzz, mmax->ubd));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &xval));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xllast, &xold1));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlllast, &xold2));
  PetscFunctionReturn(0);
}
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec dc) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
  PetscScalar ***arraydc, ***arrayp0, ***arrayq0, ***arrayp[MMA_M], ***arrayq[MMA_M],
      ***arrayb[MMA_M], ***arrayL, ***arrayU, ***arrayx;
  PetscScalar raa = 1e-5;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayp0[ez][ey][ex] =
            PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) + raa / (xmax - xmin));
        arrayq0[ez][ey][ex] =
            PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, arraydc[ez][ey][ex]) + raa / (xmax - xmin));
        for (i = 0; i < MMA_M; ++i) {
          arrayp[i][ez][ey][ex] =
              PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
              (1.001 + raa / (xmax - xmin));
          arrayq[i][ez][ey][ex] =
              PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
              (0.001 + raa / (xmax - xmin));
          arrayb[i][ez][ey][ex] = (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                                      (1.001 + raa / (xmax - xmin)) +
                                  (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                                      (0.001 + raa / (xmax - xmin));
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscScalar fval = 0;
  PetscCall(VecSum(mmax->xlast, &fval));
  for (i = 0; i < MMA_M; ++i) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
    PetscCall(VecSum(mmax->b[i], &mmax->bval[i]));
    mmax->bval[i] -= fval;
    mmax->bval[i] += s_ctx->M * s_ctx->N * s_ctx->P * volfrac;
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }
  PetscFunctionReturn(0);
}
PetscErrorCode gcmmaSub(PCCtx *s_ctx, MMAx *mmax, Vec dc, PetscScalar *raa,
                        PetscScalar *raa0) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
  PetscScalar ***arraydc, ***arrayp0, ***arrayq0, ***arrayp[MMA_M], ***arrayq[MMA_M],
      ***arrayb[MMA_M], ***arrayL, ***arrayU, ***arrayx;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayp0[ez][ey][ex] =
            PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) +
             *raa0 / (xmax - xmin));
        arrayq0[ez][ey][ex] =
            PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, arraydc[ez][ey][ex]) +
             *raa0 / (xmax - xmin));
        for (i = 0; i < MMA_M; ++i) {
          arrayp[i][ez][ey][ex] =
              PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
              (1.001 + *raa / (xmax - xmin));
          arrayq[i][ez][ey][ex] =
              PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
              (0.001 + *raa / (xmax - xmin));
          arrayb[i][ez][ey][ex] = (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                                      (1.001 + *raa / (xmax - xmin)) +
                                  (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                                      (0.001 + *raa / (xmax - xmin));
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscScalar fval = 0;
  PetscCall(VecSum(mmax->xlast, &fval));
  for (i = 0; i < MMA_M; ++i) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
    PetscCall(VecSum(mmax->b[i], &mmax->bval[i]));
    mmax->bval[i] -= fval;
    mmax->bval[i] += s_ctx->M * s_ctx->N * s_ctx->P * volfrac;
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }
  PetscFunctionReturn(0);
}
PetscErrorCode subSolv(PCCtx *s_ctx, MMAx *mmax, Vec x) {
  PetscFunctionBeginUser;
  PetscCall(omegaInitial(s_ctx, mmax, x));
  PetscScalar epsi = 1, residumax, residunorm;
  // PetscInt itera = 0;
  while (epsi > epsimin) {
    PetscCall(computeResidual(s_ctx, mmax, x, epsi, &residumax, &residunorm));
    // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "residumax: %.12f\n",
    // residumax)); PetscCall(PetscPrintf(PETSC_COMM_WORLD, "residunorm:
    // %.12f\n", residunorm)); PetscCall(VecView(mmax->mmaL,
    // PETSC_VIEWER_STDOUT_WORLD));
    PetscInt ittt = 0;
    while (residumax > 0.9 * epsi && ittt < 200) {
      ittt += 1;
      PetscCall(computeDelta(s_ctx, mmax, x, epsi));
      PetscScalar step = 0, resinew;
      PetscCall(findStep(s_ctx, mmax, x, &step));
      PetscInt itto = 0;
      PetscCall(omegaUpdate(mmax, x, step));
      resinew = 2 * residunorm;
      while (resinew > residunorm && itto < 50) {
        itto += 1;
        if (itto > 1.5) {
          PetscCall(
              omegaUpdate(mmax, x, -PetscPowScalar(0.5, itto - 1) * step));
        }
        PetscCall(computeResidual(s_ctx, mmax, x, epsi, &residumax, &resinew));
        step /= 2;
      }
      residunorm = resinew;
      step *= 2;
      // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "itto: %d\n", itto));
    }
    epsi *= 0.1;
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ittt: %d\n", ittt));
  }
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "itera: %d\n", itera));
  PetscFunctionReturn(0);
}
PetscErrorCode omegaInitial(PCCtx *s_ctx, MMAx *mmax, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayeta, ***arrayxsi, ***arrayx, ***arrayalpha, ***arraybeta;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(VecAXPBYPCZ(x, 0.5, 0.5, 0, mmax->alpha, mmax->beta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->xsi, &arrayxsi));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayeta[ez][ey][ex] =
            PetscMax(1, 1 / (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]));
        arrayxsi[ez][ey][ex] =
            PetscMax(1, 1 / (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]));
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->xsi, &arrayxsi));
  mmax->z = 1;
  mmax->zet = 1;
  for (i = 0; i < MMA_M; ++i) {
    mmax->y[i] = 1;
    mmax->lam[i] = 1;
    mmax->s[i] = 1;
    mmax->mu[i] = PetscMax(1, mmax->c[i] / 2);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode computeResidual(PCCtx *s_ctx, MMAx *mmax, Vec x,
                               PetscScalar epsi, PetscScalar *residumax,
                               PetscScalar *residunorm) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayx, ***arrayrex, ***arrayxsi, ***arrayeta, ***arrayU,
      ***arrayL, ***arrayp0, ***arrayq0, ***arrayrexsi, ***arrayreeta,
      ***arrayalpha, ***arraybeta, ***arrayp[MMA_M], ***arrayq[MMA_M], ***arraygvec[MMA_M];
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rex, &arrayrex));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rexsi, &arrayrexsi));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->reeta, &arrayreeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->gvec[i], &arraygvec[i]));
  }

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayrex[ez][ey][ex] =
            arrayp0[ez][ey][ex] / ((arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                                   (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) -
            arrayq0[ez][ey][ex] / ((arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                                   (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]));
        arrayrexsi[ez][ey][ex] =
            arrayxsi[ez][ey][ex] *
                (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]) -
            epsi;
        arrayreeta[ez][ey][ex] = arrayeta[ez][ey][ex] * (arraybeta[ez][ey][ex] -
                                                         arrayx[ez][ey][ex]) -
                                 epsi;
        arrayrex[ez][ey][ex] += arrayeta[ez][ey][ex] - arrayxsi[ez][ey][ex];
        for (int i = 0; i < MMA_M; i++) {
          arrayrex[ez][ey][ex] +=
              arrayp[i][ez][ey][ex] * mmax->lam[i] /
                  ((arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                   (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) -
              arrayq[i][ez][ey][ex] * mmax->lam[0] /
                  ((arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                   (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]));
          arraygvec[i][ez][ey][ex] =
              arrayp[i][ez][ey][ex] /
                  (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) +
              arrayq[i][ez][ey][ex] / (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]);
        }
      }
    }
  }
  for (i = 0; i < MMA_M; i++) {
    mmax->rey[i] =
        mmax->c[i] + mmax->d[i] * mmax->y[i] - mmax->mu[i] - mmax->lam[i];
    mmax->remu[i] = mmax->mu[i] * mmax->y[i] - epsi;
    mmax->res[i] = mmax->lam[i] * mmax->s[i] - epsi;
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->gvec[i], &arraygvec[i]));
    PetscCall(VecSum(mmax->gvec[i], &mmax->relam[i]));
    mmax->relam[i] +=
        -mmax->a[i] * mmax->z - mmax->y[i] + mmax->s[i] - mmax->bval[i];
  }
  mmax->rez = mmax->a0 - mmax->zet;
  mmax->rezet = mmax->zet * mmax->z - epsi;
  for (i = 0; i < MMA_M; i++) {
    mmax->rez -= mmax->a[i] * mmax->lam[i];
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->rex, &arrayrex));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->rexsi, &arrayrexsi));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->reeta, &arrayreeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < MMA_M; ++i) {
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }

  PetscScalar maxrex, normrex, maxrexsi, normrexsi, maxreeta, normreeta;
  PetscCall(VecNorm(mmax->rex, NORM_INFINITY, &maxrex));
  PetscCall(VecNorm(mmax->rex, NORM_2, &normrex));
  PetscCall(VecNorm(mmax->rexsi, NORM_INFINITY, &maxrexsi));
  PetscCall(VecNorm(mmax->rexsi, NORM_2, &normrexsi));
  PetscCall(VecNorm(mmax->reeta, NORM_INFINITY, &maxreeta));
  PetscCall(VecNorm(mmax->reeta, NORM_2, &normreeta));
  *residunorm = normrex * normrex;
  *residunorm += normrexsi * normrexsi;
  *residunorm += normreeta * normreeta;
  *residumax = PetscMax(maxrex, maxrexsi);
  *residumax = PetscMax(*residumax, maxreeta);
  for (i = 0; i < MMA_M; ++i) {
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rey[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->remu[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->res[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->relam[i]));
    *residunorm += mmax->rey[i] * mmax->rey[i];
    *residunorm += mmax->remu[i] * mmax->remu[i];
    *residunorm += mmax->res[i] * mmax->res[i];
    *residunorm += mmax->relam[i] * mmax->relam[i];
  }
  *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rez));
  *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rezet));
  *residunorm += mmax->rez * mmax->rez;
  *residunorm += mmax->rezet * mmax->rezet;
  *residunorm = PetscSqrtScalar(*residunorm);

  PetscFunctionReturn(0);
}

PetscErrorCode computeDelta(PCCtx *s_ctx, MMAx *mmax, Vec x, PetscScalar epsi) {
  PetscFunctionBeginUser;
  PetscScalar mat11, mat12, mat21, mat22, rhs1, rhs2;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
  PetscScalar ***arraytemp, ***arraytemp1, ***arrayx, ***arrayalpha,
      ***arraybeta, ***arrayxsi, ***arrayeta, ***arrayp0, ***arrayq0,
      ***arrayp[MMA_M], ***arrayq[MMA_M], ***arrayL, ***arrayU, ***arraydelx,
      ***arraydiagx, ***arrayg[MMA_M];
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp, &arraytemp));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp1, &arraytemp1));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->delx, &arraydelx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->diagx, &arraydiagx));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  for (i = 0; i < MMA_M; ++i) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->g[i], &arrayg[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        for (i = 0; i < MMA_M; ++i) {
          arrayg[i][ez][ey][ex] =
              arrayp[0][ez][ey][ex] /
                  ((arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                   (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) -
              arrayq[0][ez][ey][ex] /
                  ((arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                   (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]));
        }
        arraydiagx[ez][ey][ex] =
            (2 * (arrayp0[ez][ey][ex] + mmax->lam[0] * arrayp[0][ez][ey][ex])) /
                PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 3) +
            (2 * (arrayq0[ez][ey][ex] + mmax->lam[0] * arrayq[0][ez][ey][ex])) /
                PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 3) +
            arrayxsi[ez][ey][ex] /
                (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]) +
            arrayeta[ez][ey][ex] / (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]);
        arraydelx[ez][ey][ex] =
            (arrayp0[ez][ey][ex] + mmax->lam[0] * arrayp[0][ez][ey][ex]) /
                PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) -
            (arrayq0[ez][ey][ex] + mmax->lam[0] * arrayq[0][ez][ey][ex]) /
                PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) -
            epsi / (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]) +
            epsi / (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]);
        arraytemp[ez][ey][ex] = arrayg[0][ez][ey][ex] * arrayg[0][ez][ey][ex] /
                                arraydiagx[ez][ey][ex];
        arraytemp1[ez][ey][ex] = -arrayg[0][ez][ey][ex] *
                                 arraydelx[ez][ey][ex] / arraydiagx[ez][ey][ex];
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp, &arraytemp));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp1, &arraytemp1));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q0, &arrayq0));

  for (i = 0; i < MMA_M; ++i) {
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }
  PetscCall(VecSum(mmax->temp, &mat11));
  PetscCall(VecSum(mmax->temp1, &rhs1));
  PetscScalar gx;
  PetscCall(VecSum(mmax->gvec[0], &gx));
  gx += -mmax->a[0] * mmax->z - mmax->y[0] - mmax->bval[0] +
        epsi / mmax->lam[0] +
        (mmax->c[0] + mmax->d[0] * mmax->y[0] - mmax->lam[0] -
         epsi / mmax->y[0]) /
            (mmax->d[0] + mmax->mu[0] / mmax->y[0]);
  rhs1 += gx;
  mat11 += mmax->s[0] / mmax->lam[0];
  mat11 += 1 / (mmax->d[0] + mmax->mu[0] / mmax->y[0]);
  mat12 = mmax->a[0];
  mat21 = mmax->a[0];
  mat22 = -mmax->zet / mmax->z;
  rhs2 = mmax->a0 - mmax->lam[0] * mmax->a[0] - epsi / mmax->z;
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mat11 %f\n", mat11));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mat12 %f\n", mat12));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mat21 %f\n", mat21));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mat22 %f\n", mat22));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rhs1 %f\n", rhs1));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rhs2 %f\n", rhs2));
  mmax->dlam[0] =
      (mat21 * rhs2 - mat22 * rhs1) / (mat12 * mat21 - mat11 * mat22);
  mmax->dz = (mat12 * rhs1 - mat11 * rhs2) / (mat12 * mat21 - mat11 * mat22);
  PetscScalar ***arraydx, ***arraydxsi, ***arraydeta;
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->dx, &arraydx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->dxsi, &arraydxsi));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->deta, &arraydeta));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraydx[ez][ey][ex] =
            -(arrayg[0][ez][ey][ex] * mmax->dlam[0] + arraydelx[ez][ey][ex]) /
            arraydiagx[ez][ey][ex];
        arraydxsi[ez][ey][ex] =
            (epsi - arrayxsi[ez][ey][ex] * arraydx[ez][ey][ex]) /
                (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]) -
            arrayxsi[ez][ey][ex];
        arraydeta[ez][ey][ex] =
            (epsi + arrayeta[ez][ey][ex] * arraydx[ez][ey][ex]) /
                (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]) -
            arrayeta[ez][ey][ex];
      }
    }
  }
  mmax->dy[0] = mmax->dlam[0] / (mmax->d[0] + mmax->mu[0] / mmax->y[0]) -
                (mmax->c[0] + mmax->d[0] * mmax->y[0] - mmax->lam[0] -
                 epsi / mmax->y[0]) /
                    (mmax->d[0] + mmax->mu[0] / mmax->y[0]);
  mmax->dmu[0] =
      -mmax->dy[0] * mmax->mu[0] / mmax->y[0] - mmax->mu[0] + epsi / mmax->y[0];
  mmax->dzet = -mmax->zet * mmax->dz / mmax->z - mmax->zet + epsi / mmax->z;
  mmax->ds[0] = -mmax->dlam[0] * mmax->s[0] / mmax->lam[0] - mmax->s[0] +
                epsi / mmax->lam[0];
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dy[0]: %f\n", mmax->dy[0]));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dmu[0]: %f\n", mmax->dmu[0]));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dzet: %f\n", mmax->dzet));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ds[0]: %f\n", mmax->ds[0]));
  for (i = 0; i < MMA_M; i++) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->g[i], &arrayg[i]));
  }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->dx, &arraydx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->dxsi, &arraydxsi));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->deta, &arraydeta));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->delx, &arraydelx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->diagx, &arraydiagx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscFunctionReturn(0);
}
PetscErrorCode findStep(PCCtx *s_ctx, MMAx *mmax, Vec x, PetscScalar *step) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz;
  PetscScalar ***arrayx, ***arrayalpha, ***arraybeta, ***arraytemp,
      ***arraytemp1, ***arraytemp2, ***arraytemp3, ***arraydx, ***arrayxsi,
      ***arraydxsi, ***arrayeta, ***arraydeta;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->dx, &arraydx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->dxsi, &arraydxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->deta, &arraydeta));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp, &arraytemp));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp1, &arraytemp1));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp2, &arraytemp2));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->temp3, &arraytemp3));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraytemp[ez][ey][ex] = -1.01 * arraydx[ez][ey][ex] /
                                (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]);
        arraytemp1[ez][ey][ex] = 1.01 * arraydx[ez][ey][ex] /
                                 (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]);
        arraytemp2[ez][ey][ex] =
            -1.01 * arraydxsi[ez][ey][ex] / arrayxsi[ez][ey][ex];
        arraytemp3[ez][ey][ex] =
            -1.01 * arraydeta[ez][ey][ex] / arrayeta[ez][ey][ex];
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->dx, &arraydx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->dxsi, &arraydxsi));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->deta, &arraydeta));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp, &arraytemp));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp1, &arraytemp1));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp2, &arraytemp2));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->temp3, &arraytemp3));
  PetscScalar step1, step2, step3, step4;
  PetscCall(VecMax(mmax->temp, NULL, &step1));
  PetscCall(VecMax(mmax->temp1, NULL, &step2));
  PetscCall(VecMax(mmax->temp2, NULL, &step3));
  PetscCall(VecMax(mmax->temp3, NULL, &step4));
  *step = PetscMax(step1, step2);
  *step = PetscMax(*step, step3);
  *step = PetscMax(*step, step4);
  *step = PetscMax(-1.01 * mmax->dy[0] / mmax->y[0], *step);
  *step = PetscMax(-1.01 * mmax->dmu[0] / mmax->mu[0], *step);
  *step = PetscMax(-1.01 * mmax->dzet / mmax->z, *step);
  *step = PetscMax(-1.01 * mmax->ds[0] / mmax->s[0], *step);
  *step = PetscMax(-1.01 * mmax->dlam[0] / mmax->lam[0], *step);
  *step = PetscMax(-1.01 * mmax->dzet / mmax->zet, *step);
  *step = PetscMax(1.0, *step);
  *step = 1.0 / *step;
  PetscFunctionReturn(0);
}
PetscErrorCode omegaUpdate(MMAx *mmax, Vec x, PetscScalar coef) {
  PetscFunctionBeginUser;
  PetscCall(VecAXPY(x, coef, mmax->dx));
  PetscCall(VecAXPY(mmax->xsi, coef, mmax->dxsi));
  PetscCall(VecAXPY(mmax->eta, coef, mmax->deta));
  PetscInt i = 0;
  for (i = 0; i < MMA_M; ++i) {
    mmax->y[i] += coef * mmax->dy[i];
    mmax->mu[i] += coef * mmax->dmu[i];
    mmax->s[i] += coef * mmax->ds[i];
    mmax->lam[i] += coef * mmax->dlam[i];
  }
  mmax->z += coef * mmax->dz;
  mmax->zet += coef * mmax->dzet;
  PetscFunctionReturn(0);
}
PetscErrorCode outputTest(MMAx *mmax, Vec dc) {
  PetscFunctionBeginUser;
  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,
                                  "../src/pymma/data/xval.bin", FILE_MODE_WRITE,
                                  &viewer));
  PetscCall(VecView(mmax->xlast, viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,
                                  "../src/pymma/data/xold1.bin",
                                  FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(mmax->xllast, viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,
                                  "../src/pymma/data/xold2.bin",
                                  FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(mmax->xlllast, viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../src/pymma/data/dc.bin",
                                  FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(dc, viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../src/pymma/data/upp.bin",
                                  FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(mmax->mmaU, viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../src/pymma/data/low.bin",
                                  FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(mmax->mmaL, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode raaInit(PCCtx *s_ctx, MMAx *mmax, PetscScalar *raa,
                       PetscScalar *raa0, Vec dc) {
  PetscFunctionBeginUser;
  PetscCall(VecCopy(dc, mmax->temp));
  PetscCall(VecAbs(mmax->temp));
  PetscCall(VecSum(mmax->temp, raa0));
  *raa0 *= 0.1 * (xmax - xmin) / s_ctx->M / s_ctx->N / s_ctx->P;
  *raa = 0.1 * (xmax - xmin);

  PetscFunctionReturn(0);
}
PetscErrorCode conCheck(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec xg, Vec dc,
                        PetscScalar *raa, PetscScalar *raa0,
                        PetscBool *conserve) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  Vec res0, res1, dx;
  PetscScalar ***arrayxg, ***arrayx, ***arrayres0, ***arrayres1, ***arrayU,
      ***arrayL, ***arrayp0, ***arrayp1, ***arrayq0, ***arrayq1, ***arraydx;
  PetscScalar re0, re1, d, del0, del1;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &res0));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &res1));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dx));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, xg, &arrayxg));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p[0], &arrayp1));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q[0], &arrayq1));
  PetscCall(DMDAVecGetArray(s_ctx->dm, res0, &arrayres0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, res1, &arrayres1));
  PetscCall(DMDAVecGetArray(s_ctx->dm, dx, &arraydx));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayres0[ez][ey][ex] =
            arrayp0[ez][ey][ex] / (arrayU[ez][ey][ex] - arrayxg[ez][ey][ex]) -
            arrayp0[ez][ey][ex] / (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) +
            arrayq0[ez][ey][ex] / (arrayxg[ez][ey][ex] - arrayL[ez][ey][ex]) -
            arrayq0[ez][ey][ex] / (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]);
        arrayres1[ez][ey][ex] =
            arrayp1[ez][ey][ex] / (arrayU[ez][ey][ex] - arrayxg[ez][ey][ex]) -
            arrayp1[ez][ey][ex] / (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) +
            arrayq1[ez][ey][ex] / (arrayxg[ez][ey][ex] - arrayL[ez][ey][ex]) -
            arrayq1[ez][ey][ex] / (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]);
        arraydx[ez][ey][ex] = (arrayU[ez][ey][ex] - arrayL[ez][ey][ex]) *
                              (arrayxg[ez][ey][ex] - arrayx[ez][ey][ex]) *
                              (arrayxg[ez][ey][ex] - arrayx[ez][ey][ex]) /
                              (arrayU[ez][ey][ex] - arrayxg[ez][ey][ex]) /
                              (arrayxg[ez][ey][ex] - arrayL[ez][ey][ex]) /
                              (xmax - xmin);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, xg, &arrayxg));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->p[0], &arrayp1));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->q[0], &arrayq1));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, res0, &arrayres0));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, res1, &arrayres1));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dx, &arraydx));
  PetscCall(VecSum(res0, &re0));
  PetscCall(VecSum(res1, &re1));
  PetscCall(VecSum(dx, &d));

  if (re0 >= 0 && re1 >= 0) {
    *conserve = PETSC_FALSE;
  } else {
    del0 = -re0 / d;
    del1 = -re1 / d;
    del0 = PetscMax(del0, 0);
    del1 = PetscMax(del1, 0);
    *raa0 = PetscMin(1.1 * (del0 + *raa0), 10 * *raa0);
    *raa = PetscMin(1.1 * (del1 + *raa), 10 * *raa);
  }
  PetscCall(VecDestroy(&res0));
  PetscCall(VecDestroy(&res1));
  PetscCall(VecDestroy(&dx));
  PetscFunctionReturn(0);
}