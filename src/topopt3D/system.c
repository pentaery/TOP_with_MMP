#include "system.h"
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

PetscErrorCode formBoundary(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***array;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->boundary));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->boundary, &array));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if(ex >= PetscFloorReal(0.45 * s_ctx->M) &&
            ex <= PetscCeilReal(0.55 * s_ctx->M) - 1 &&
            ey >= PetscFloorReal(0.45 * s_ctx->N) &&
            ey <= PetscCeilReal(0.55 * s_ctx->N) - 1 && ez == 0) {
          array[ez][ey][ex] = 1;
        }else{
          array[ez][ey][ex] = 0;
        }
        // array[ez][ey][ex] = 0;
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->boundary, &array));

  PetscFunctionReturn(0);
}

PetscErrorCode formkappa(PCCtx *s_ctx, Vec x, PetscInt penal) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayx, ***arraykappa[DIM];
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[i], &arraykappa[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        for (i = 0; i < DIM; ++i) {
          arraykappa[i][ez][ey][ex] =
              (kH - kL) * PetscPowScalar(arrayx[ez][ey][ex], penal) + kL;
          // arraykappa[i][ez][ey][ex] = 1.0;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->kappa[i], &arraykappa[i]));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A) {
  PetscFunctionBeginUser;

  Vec kappa_loc[DIM]; // Destroy later.
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arr_kappa_3d[DIM], ***arrayBoundary, val_A[2][2], avg_kappa_e;
  MatStencil row[2], col[2];

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(MatZeroEntries(A));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ++ey)
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex >= 1) {
          row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] +
                               1.0 / arr_kappa_3d[0][ez][ey][ex]);
          val_A[0][0] = s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * avg_kappa_e;
          val_A[0][1] = -s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * avg_kappa_e;
          val_A[1][0] = -s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * avg_kappa_e;
          val_A[1][1] = s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ey >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] +
                               1.0 / arr_kappa_3d[1][ez][ey][ex]);
          val_A[0][0] = s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * avg_kappa_e;
          val_A[0][1] = -s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * avg_kappa_e;
          val_A[1][0] = -s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * avg_kappa_e;
          val_A[1][1] = s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ez >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] +
                               1.0 / arr_kappa_3d[2][ez][ey][ex]);
          val_A[0][0] = s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * avg_kappa_e;
          val_A[0][1] = -s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * avg_kappa_e;
          val_A[1][0] = -s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * avg_kappa_e;
          val_A[1][1] = s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (arrayBoundary[ez][ey][ex] == 1) {
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = 2.0 * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
                        arr_kappa_3d[2][ez][ey][ex];
          PetscCall(MatSetValuesStencil(A, 1, &col[0], 1, &row[0], &val_A[0][0],
                                        ADD_VALUES));
        }
      }
  // A的赋值
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(VecDestroy(&kappa_loc[i]));
  }
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscFunctionReturn(0);
}

PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs, Vec x, PetscInt penal) {
  PetscFunctionBeginUser;
  PetscScalar ***array, ***arraykappa, ***arrayx, ***arrayBoundary;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  // Set RHS
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(VecSet(rhs, 0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, rhs, &array));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        array[ez][ey][ex] += s_ctx->H_x * s_ctx->H_y * s_ctx->H_z * f0 *
                             (1 - PetscPowScalar(arrayx[ez][ey][ex], penal));
        if (arrayBoundary[ez][ey][ex] > 0.5) {
          array[ez][ey][ex] += 2 * arraykappa[ez][ey][ex] * tD * s_ctx->H_x *
                               s_ctx->H_y / s_ctx->H_z;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, rhs, &array));

  PetscFunctionReturn(0);
}

PetscErrorCode xScaling(DM dm1, DM dm2, Vec x1, Vec x2) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, startx1, startx2, starty1, starty2, startz1, startz2,
      nx1, nx2, ny1, ny2, nz1, nz2;
  PetscScalar ***arrayx1, ***arrayx2;
  PetscCall(
      DMDAGetCorners(dm1, &startx1, &starty1, &startz1, &nx1, &ny1, &nz1));
  PetscCall(
      DMDAGetCorners(dm2, &startx2, &starty2, &startz2, &nx2, &ny2, &nz2));
  PetscCall(DMDAVecGetArray(dm1, x1, &arrayx1));
  PetscCall(DMDAVecGetArray(dm2, x2, &arrayx2));
  PetscInt q1 = nx2 / nx1;
  PetscInt q2 = ny2 / ny1;
  PetscInt q3 = nz2 / nz1;
  PetscInt r1 = nx2 % nx1;
  PetscInt r2 = ny2 % ny1;
  PetscInt r3 = nz2 % nz1;
  PetscInt eax[nx1 + 1], eay[ny1 + 1], eaz[nz1 + 1];
  eax[0] = startx2;
  eay[0] = starty2;
  eaz[0] = startz2;
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eax: %d\n", eax[0]));
  PetscInt i, j, k;
  for (i = 1; i <= nx1; ++i) {
    if (i <= r1) {
      eax[i] = eax[i - 1] + q1 + 1;
    } else {
      eax[i] = eax[i - 1] + q1;
    }
    // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eax: %d\n", eax[i]));
  }
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  for (i = 1; i <= ny1; ++i) {
    if (i <= r2) {
      eay[i] = eay[i - 1] + q2 + 1;
    } else {
      eay[i] = eay[i - 1] + q2;
    }
  }
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eay: %d\n", eay[0]));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eay: %d\n", eay[ny1]));
  for (i = 1; i <= nz1; ++i) {
    if (i <= r3) {
      eaz[i] = eaz[i - 1] + q3 + 1;
    } else {
      eaz[i] = eaz[i - 1] + q3;
    }
  }
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eaz: %d\n", eaz[0]));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "eaz: %d\n", eaz[nz1]));
  for (ez = startz2; ez < startz2 + nz2; ++ez) {
    for (ey = starty2; ey < starty2 + ny2; ++ey) {
      for (ex = startx2; ex < startx2 + nx2; ++ex) {
        for (i = 0; i < nx1; ++i) {
          if (ex >= eax[i] && ex < eax[i + 1]) {
            break;
          }
        }
        for (j = 0; j < ny1; ++j) {
          if (ey >= eay[j] && ey < eay[j + 1]) {
            break;
          }
        }
        for (k = 0; k < nz1; ++k) {
          if (ez >= eaz[k] && ez < eaz[k + 1]) {
            break;
          }
        }
        arrayx2[ez][ey][ex] = arrayx1[startz1 + k][starty1 + j][startx1 + i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(dm1, x1, &arrayx1));
  PetscCall(DMDAVecRestoreArray(dm2, x2, &arrayx2));

  PetscFunctionReturn(0);
}
