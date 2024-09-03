#include "PreMixFEM_3D.h"
#include "mpi.h"
#include "optimization.h"
#include "system.h"
#include <H5Spublic.h>
#include <oCriteria.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
PetscErrorCode computeCost(PCCtx *s_ctx, Vec t, Vec rhs, PetscScalar *cost) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  Vec c;
  PetscScalar cost1 = 0, cost2 = 0;
  PetscScalar ***arrayt, ***arraycost, ***arrayBoundary, ***arraykappa;
  PetscCall(VecDot(rhs, t, &cost1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost1: %f\n", cost1));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &c));
  PetscCall(VecSet(c, 0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t, &arrayt));
  PetscCall(DMDAVecGetArray(s_ctx->dm, c, &arraycost));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (arrayBoundary[ez][ey][ex] > 0.5) {
          arraycost[ez][ey][ex] = 2 * arraykappa[ez][ey][ex] * s_ctx->H_x *
                                  s_ctx->H_y / s_ctx->H_z * tD *
                                  (tD - 2 * arrayt[ez][ey][ex]);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t, &arrayt));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, c, &arraycost));
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));

  PetscCall(VecSum(c, &cost2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost2: %f\n", cost2));
  *cost = cost1 + cost2;

  PetscCall(VecDestroy(&c));
  PetscFunctionReturn(0);
}

PetscErrorCode computeGradient(PCCtx *s_ctx, Vec x, Vec t, Vec dc,
                               PetscInt penal) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayt, ***arraydc, ***arrayx, ***arrayBoundary,
      ***arr_kappa_3d[DIM];
  Vec t_loc;
  Vec kappa_loc[DIM];

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecGetArray(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex >= 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) / s_ctx->H_x *
                                 s_ctx->H_y * s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]));
        }
        if (ex < s_ctx->M - 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) / s_ctx->H_x *
                                 s_ctx->H_y * s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]));
        }
        if (ey >= 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) * s_ctx->H_x /
                                 s_ctx->H_y * s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]));
        }
        if (ey < s_ctx->N - 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) * s_ctx->H_x /
                                 s_ctx->H_y * s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]));
        }
        if (ez >= 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) * s_ctx->H_x *
                                 s_ctx->H_y / s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]));
        }
        if (ez < s_ctx->P - 1) {
          arraydc[ez][ey][ex] += 2 * penal * (kH - kL) * s_ctx->H_x *
                                 s_ctx->H_y / s_ctx->H_z *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]));
        }
        if (arrayBoundary[ez][ey][ex] > 0.5) {
          arraydc[ez][ey][ex] +=
              2 * penal * (kH - kL) * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (arrayt[ez][ey][ex] - tD) * (arrayt[ez][ey][ex] - tD);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dc, &arraydc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  PetscCall(VecDestroy(&t_loc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(VecDestroy(&kappa_loc[i]));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode filter(PCCtx *s_ctx, Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arraydc, ***arrayx, ***arraydcn, ***arraycoef;
  Vec localx, localdc, coef, dcn;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dcn));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &coef));
  PetscCall(VecSet(dcn, 0));
  PetscCall(VecSet(coef, 0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, coef, &arraycoef));

  PetscCall(DMCreateLocalVector(s_ctx->dm, &localx));
  PetscCall(DMCreateLocalVector(s_ctx->dm, &localdc));

  PetscCall(DMGlobalToLocal(s_ctx->dm, x, INSERT_VALUES, localx));
  PetscCall(DMGlobalToLocal(s_ctx->dm, dc, INSERT_VALUES, localdc));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, localdc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, dcn, &arraydcn));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, localx, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraydcn[ez][ey][ex] += rmin * arraydc[ez][ey][ex] * arrayx[ez][ey][ex];
        arraycoef[ez][ey][ex] += rmin * arrayx[ez][ey][ex];
        if (ex >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey][ex - 1] * arrayx[ez][ey][ex - 1];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ex < s_ctx->M - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey][ex + 1] * arrayx[ez][ey][ex + 1];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ey >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey - 1][ex] * arrayx[ez][ey - 1][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ey < s_ctx->N - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey + 1][ex] * arrayx[ez][ey + 1][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ez >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez - 1][ey][ex] * arrayx[ez - 1][ey][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ez < s_ctx->P - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez + 1][ey][ex] * arrayx[ez + 1][ey][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, localdc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, localx, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dcn, &arraydcn));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, coef, &arraycoef));

  // PetscCall(VecView(coef, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecPointwiseDivide(dc, dcn, coef));

  PetscCall(VecDestroy(&localx));
  PetscCall(VecDestroy(&localdc));
  PetscCall(VecDestroy(&dcn));
  PetscCall(VecDestroy(&coef));

  PetscFunctionReturn(0);
}

PetscErrorCode optimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change) {
  PetscFunctionBeginUser;
  PetscScalar l1 = 0, l2 = 100000, lmid;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar move = 0.2;
  PetscScalar ***arraydc, ***arrayx;
  PetscScalar sum;
  PetscScalar volume = volfrac * s_ctx->M * s_ctx->N * s_ctx->P;
  Vec xold;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &xold));
  PetscCall(VecCopy(x, xold));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));

  while (l2 - l1 > 1e-4) {
    PetscCall(VecCopy(xold, x));
    PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));
    lmid = (l1 + l2) / 2;
    for (ez = startz; ez < startz + nz; ++ez) {
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          if (arrayx[ez][ey][ex] * PetscSqrtScalar(arraydc[ez][ey][ex] / lmid) <
              PetscMax(0.001, arrayx[ez][ey][ex] - move)) {
            arrayx[ez][ey][ex] = PetscMax(0.001, arrayx[ez][ey][ex] - move);
          } else if (arrayx[ez][ey][ex] *
                         PetscSqrtScalar(arraydc[ez][ey][ex] / lmid) >
                     PetscMin(1, arrayx[ez][ey][ex] + move)) {
            arrayx[ez][ey][ex] = PetscMin(1, arrayx[ez][ey][ex] + move);
          } else {
            arrayx[ez][ey][ex] = arrayx[ez][ey][ex] *
                                 PetscSqrtScalar(arraydc[ez][ey][ex] / lmid);
          }
        }
      }
    }

    PetscCall(DMDAVecRestoreArray(s_ctx->dm, x, &arrayx));

    PetscCall(VecSum(x, &sum));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sum: %f\n", sum));
    if (sum > volume) {
      l1 = lmid;
    } else {
      l2 = lmid;
    }

    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "lambda: %f\n", lmid));
  }

  PetscCall(VecAXPY(xold, -1, x));
  PetscCall(VecMax(xold, NULL, change));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(VecDestroy(&xold));
  PetscFunctionReturn(0);
}

PetscErrorCode genOptimalCriteria(PCCtx *s_ctx, Vec x, Vec dc, PetscScalar *g,
                                  PetscScalar *glast, PetscScalar *lmid,
                                  PetscScalar *change, PetscScalar cost0) {
  PetscFunctionBeginUser;
  Vec xold;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar move = 0.2, eps = 0.05, p0 = 0;
  PetscScalar ***arraydc, ***arrayx;
  PetscScalar sum;
  PetscInt nelm = s_ctx->M * s_ctx->N * s_ctx->P;
  PetscCall(VecSum(x, &sum));
  *g = sum / (volfrac * nelm) - 1;
  PetscScalar dg = *g - *glast;
  *glast = *g;
  if ((g > 0 && dg > 0) || (g < 0 && dg < 0)) {
    p0 = 1.0;
  } else if ((g > 0 && dg > -eps) || (g < 0 && dg < eps)) {
    p0 = 0.5;
  } else {
    p0 = 0;
  }
  *lmid = *lmid * (1 + p0 * (*g + dg));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &xold));
  PetscCall(VecCopy(x, xold));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (arrayx[ez][ey][ex] *
                PetscSqrtScalar(arraydc[ez][ey][ex] / cost0 / (*lmid / nelm)) <
            PetscMax(0.001, arrayx[ez][ey][ex] - move)) {
          arrayx[ez][ey][ex] = PetscMax(0.001, arrayx[ez][ey][ex] - move);
        } else if (arrayx[ez][ey][ex] *
                       PetscSqrtScalar(arraydc[ez][ey][ex] / cost0 /
                                       (*lmid / nelm)) >
                   PetscMin(1, arrayx[ez][ey][ex] + move)) {
          arrayx[ez][ey][ex] = PetscMin(1, arrayx[ez][ey][ex] + move);
        } else {
          arrayx[ez][ey][ex] =
              arrayx[ez][ey][ex] *
              PetscSqrtScalar(arraydc[ez][ey][ex] / cost0 / (*lmid / nelm));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));

  PetscCall(VecAXPY(xold, -1, x));
  PetscCall(VecMax(xold, NULL, change));

  PetscCall(VecDestroy(&xold));
  PetscFunctionReturn(0);
}

PetscErrorCode computeCost1(PCCtx *s_ctx, Vec t, PetscScalar *cost) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayt, ***arrayc, ***arrayBoundary, ***arr_kappa_3d[DIM];
  Vec t_loc;
  Vec c;
  Vec kappa_loc[DIM];
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &c));
  PetscCall(VecSet(c, 0));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecGetArray(s_ctx->dm, c, &arrayc));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_y * s_ctx->H_z / s_ctx->H_x *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) /
                                (1 / arr_kappa_3d[0][ez][ey][ex] +
                                 1 / arr_kappa_3d[0][ez][ey][ex - 1]);
        }
        if (ey >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_z * s_ctx->H_x / s_ctx->H_y *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) /
                                (1 / arr_kappa_3d[1][ez][ey][ex] +
                                 1 / arr_kappa_3d[1][ez][ey - 1][ex]);
        }
        if (ez >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
                                (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
                                (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) /
                                (1 / arr_kappa_3d[2][ez][ey][ex] +
                                 1 / arr_kappa_3d[2][ez - 1][ey][ex]);
        }

        if (arrayBoundary[ez][ey][ex] == 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
                                (arrayt[ez][ey][ex] - tD) *
                                (arrayt[ez][ey][ex] - tD) *
                                arr_kappa_3d[2][ez][ey][ex];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, c, &arrayc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));

  PetscCall(VecSum(c, cost));

  PetscCall(VecDestroy(&t_loc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(VecDestroy(&kappa_loc[i]));
  }
  PetscCall(VecDestroy(&c));
  PetscFunctionReturn(0);
}