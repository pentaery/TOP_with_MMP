#include "PreMixFEM_3D.h"
#include "petscdmda.h"
#include "slepceps.h"
#include <petscdm.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>

int mkl_serv_intel_cpu_true() { return 1; }

typedef struct _internal_context {
  Vec *ms_bases_c_tmp;
  PetscScalar ***arr_kappa_3d[DIM];
  PetscScalar meas_elem, meas_face_xy, meas_face_yz, meas_face_zx;
  PetscInt max_eigen_num_lv1_upd, max_eigen_num_lv2_upd;
  PetscInt coarse_elem_num;
} _IntCtx;

// 获得当前coarse_element的xyz坐标
void _PC_get_coarse_elem_xyz(PCCtx *s_ctx, const PetscInt coarse_elem,
                             PetscInt *coarse_elem_x, PetscInt *coarse_elem_y,
                             PetscInt *coarse_elem_z) {
  *coarse_elem_x = coarse_elem % s_ctx->sub_domains;
  *coarse_elem_y = (coarse_elem / s_ctx->sub_domains) % s_ctx->sub_domains;
  *coarse_elem_z = coarse_elem / (s_ctx->sub_domains * s_ctx->sub_domains);
}

void _PC_get_coarse_elem_corners(PCCtx *s_ctx, const PetscInt coarse_elem,
                                 PetscInt *startx, PetscInt *nx,
                                 PetscInt *starty, PetscInt *ny,
                                 PetscInt *startz, PetscInt *nz) {
  PetscInt coarse_elem_x, coarse_elem_y, coarse_elem_z;
  _PC_get_coarse_elem_xyz(s_ctx, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                          &coarse_elem_z);
  *startx = s_ctx->coarse_startx[coarse_elem_x];
  *nx = s_ctx->coarse_lenx[coarse_elem_x];
  *starty = s_ctx->coarse_starty[coarse_elem_y];
  *ny = s_ctx->coarse_leny[coarse_elem_y];
  *startz = s_ctx->coarse_startz[coarse_elem_z];
  *nz = s_ctx->coarse_lenz[coarse_elem_z];
}

// 获取第coarse_elem个粗网格的自由度索引
// void _PC_get_dof_idx(PCCtx *s_ctx, PetscInt *dof_idx) {
//   PetscInt coarse_elem,
//       coarse_elem_num =
//           s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
//   dof_idx[0] = 0;
//   for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem)
//     dof_idx[coarse_elem + 1] =
//         dof_idx[coarse_elem] + s_ctx->eigen_num_lv1[coarse_elem];
// }

// 获得所有coarse element的起始点和长度
PetscErrorCode domainPartition(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, i;

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  s_ctx->coarse_startx[0] = proc_startx;
  s_ctx->coarse_starty[0] = proc_starty;
  s_ctx->coarse_startz[0] = proc_startz;
  s_ctx->coarse_lenx[0] =
      (proc_nx / s_ctx->sub_domains) + (proc_nx % s_ctx->sub_domains > 0);
  s_ctx->coarse_leny[0] =
      (proc_ny / s_ctx->sub_domains) + (proc_ny % s_ctx->sub_domains > 0);
  s_ctx->coarse_lenz[0] =
      (proc_nz / s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > 0);
  for (i = 1; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_startx[i] =
        s_ctx->coarse_startx[i - 1] + s_ctx->coarse_lenx[i - 1];
    s_ctx->coarse_lenx[i] =
        (proc_nx / s_ctx->sub_domains) + (proc_nx % s_ctx->sub_domains > i);
    s_ctx->coarse_starty[i] =
        s_ctx->coarse_starty[i - 1] + s_ctx->coarse_leny[i - 1];
    s_ctx->coarse_leny[i] =
        (proc_ny / s_ctx->sub_domains) + (proc_ny % s_ctx->sub_domains > i);
    s_ctx->coarse_startz[i] =
        s_ctx->coarse_startz[i - 1] + s_ctx->coarse_lenz[i - 1];
    s_ctx->coarse_lenz[i] =
        (proc_nz / s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > i);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode formRc(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  Mat A_i_inner, M_i;
  Vec diag_M_i, getIndex;
  PetscScalar ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv1], ***arr_M_i_3d,
      val_A[2][2], avg_kappa_e;
  PetscInt i, j, coarse_elem, startx, nx, ex, starty, ny, ey, startz, nz, ez,
      row[2], col[2];
  PetscInt colRc, rowRc, firstrow, firstcol, lastrow, lastcol;
  PetscInt proc_nx, proc_ny, proc_nz, proc_startx, proc_starty, proc_startz;
  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  PetscCall(DMGetGlobalVector(s_ctx->dm, &getIndex));
  PetscCall(VecGetOwnershipRange(getIndex, &firstcol, &lastcol));
  PetscCall(MatGetOwnershipRange(s_ctx->Rc, &firstrow, &lastrow));

  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &(int_ctx->ms_bases_c_tmp[i])));
    PetscCall(DMDAVecGetArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                              &arr_ms_bases_c_array[i]));
  }

  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    // 7 point stencil.
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nx * ny * nz, &diag_M_i));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz, 7,
                              NULL, &A_i_inner));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz, 1,
                              NULL, &M_i));
    PetscCall(VecGetArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          arr_M_i_3d[ez - startz][ey - starty][ex - startx] = 0.0;
          if (ex >= startx + 1) {
            row[0] =
                (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] =
                (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[0][1] = -int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][0] = -int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][1] = int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                   &val_A[0][0], ADD_VALUES));
            if (s_ctx->lv2_eigen_op == EIG_OP_MOD)
              arr_M_i_3d[ez - startz][ey - starty][ex - startx - 1] +=
                  2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
            if (s_ctx->lv2_eigen_op == EIG_OP_DIAG) {
              arr_M_i_3d[ez - startz][ey - starty][ex - startx - 1] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }
          } else if (ex >= 1 && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }
          if (ex + 1 != s_ctx->M && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex + 1]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }

          if (ey >= starty + 1) {
            row[0] =
                (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] =
                (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex - startx;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[0][1] = -int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][0] = -int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][1] = int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                   &val_A[0][0], ADD_VALUES));
            if (s_ctx->lv2_eigen_op == EIG_OP_MOD)
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                  2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
            if (s_ctx->lv2_eigen_op == EIG_OP_DIAG) {
              arr_M_i_3d[ez - startz][ey - starty - 1][ex - startx] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }
          } else if (ey >= 1 && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }
          if (ey + 1 != s_ctx->N && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey + 1][ex]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }

          if (ez >= startz + 1) {
            row[0] =
                (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
            row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] =
                (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex - startx;
            col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[0][1] = -int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][0] = -int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            val_A[1][1] = int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                   &val_A[0][0], ADD_VALUES));
            if (s_ctx->lv2_eigen_op == EIG_OP_MOD)
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                  2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
            if (s_ctx->lv2_eigen_op == EIG_OP_DIAG) {
              arr_M_i_3d[ez - startz - 1][ey - starty][ex - startx] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }
          } else if (ez >= 1 && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }
          if (ez + 1 != s_ctx->P && s_ctx->lv2_eigen_op == EIG_OP_MOD) {
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
                2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }

          if (s_ctx->lv2_eigen_op == EIG_OP_MEAN)
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] =
                int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                int_ctx->arr_kappa_3d[2][ez][ey][ex];
        }
      }

    PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(VecRestoreArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
    if (s_ctx->lv2_eigen_op == EIG_OP_MOD)
      PetscCall(VecScale(diag_M_i, int_ctx->meas_elem));
    PetscCall(VecScale(diag_M_i, 1e6));
    PetscCall(MatDiagonalSet(M_i, diag_M_i, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(M_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(M_i, MAT_FINAL_ASSEMBLY));
    PetscCall(VecDestroy(&diag_M_i));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, ***arr_eig_vec;
    Vec eig_vec;

    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, A_i_inner, M_i));
    PetscCall(EPSSetProblemType(eps, EPS_GHEP));
    PetscCall(EPSSetDimensions(eps, s_ctx->max_eigen_num_lv1, PETSC_DEFAULT,
                               PETSC_DEFAULT));
    PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));
    ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(
        nconv >= s_ctx->max_eigen_num_lv1, PETSC_COMM_WORLD, PETSC_ERR_USER,
        "SLEPc cannot find enough eigenvectors! (nconv=%d, eigen_num=%d)\n",
        nconv, s_ctx->max_eigen_num_lv1);
    for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem] = eig_val;

      if (j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
      }
      rowRc = firstrow + coarse_elem * s_ctx->max_eigen_num_lv1 + j;
      PetscCall(VecGetArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey) {
          PetscCall(PetscArraycpy(&arr_ms_bases_c_array[j][ez][ey][startx],
                                  &arr_eig_vec[ez - startz][ey - starty][0],
                                  nx));
          for (ex = startx; ex < startx + nx; ++ex) {
            colRc = firstcol + (ez - startz) * proc_ny * proc_nx +
                    (ey - starty) * proc_nx + ex - startx;
            PetscCall(
                MatSetValue(s_ctx->Rc, rowRc, colRc,
                            arr_eig_vec[ez - startz][ey - starty][ex - startx],
                            INSERT_VALUES));
          }
        }
      PetscCall(VecRestoreArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
    }

    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }

  PetscCall(MatAssemblyBegin(s_ctx->Rc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(s_ctx->Rc, MAT_FINAL_ASSEMBLY));
  // PetscScalar value;
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "firstrow=%d\n", firstrow));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "firstcol=%d\n", firstcol));
  // PetscCall(MatGetValue(s_ctx->Rc, firstrow, firstcol, &value));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "value: %f\n", value));

  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
  // We need to check the real maximum number of eigenvectors used, free what
  // those are not used. The first time clean ms_bases_c_tmp.

  Vec dummy_ms_bases_glo;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                              INSERT_VALUES, dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo, INSERT_VALUES,
                              int_ctx->ms_bases_c_tmp[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  PetscCall(VecDestroy(&getIndex));
  PetscFunctionReturn(0);
}

PetscErrorCode formRcc(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  Mat A_i_inner;
  Vec dummy_ms_bases_glo;
  PetscInt coarse_elem, coarse_elem_col, i, j, row, col;
  PetscInt coarse_elem_x, coarse_elem_y, coarse_elem_z, startx, nx, ex, starty,
      ny, ey, startz, nz, ez;
  PetscInt proc_startx, proc_nx, proc_starty, proc_ny, proc_startz, proc_nz;
  PetscScalar val_A, avg_kappa_e,
      ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv1];

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,
                            int_ctx->coarse_elem_num * s_ctx->max_eigen_num_lv1,
                            int_ctx->coarse_elem_num * s_ctx->max_eigen_num_lv1,
                            7 * s_ctx->max_eigen_num_lv1, NULL, &A_i_inner));
  PetscInt m, n;
  PetscCall(MatGetSize(A_i_inner, &m, &n));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "m=%d, n=%d\n", m, n));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "coarse_elem_num=%d\n",
                        int_ctx->coarse_elem_num));

  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_xyz(s_ctx, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                            &coarse_elem_z);
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
      row = coarse_elem * s_ctx->max_eigen_num_lv1 + i;
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "row=%d\n", row));
      for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
        col = coarse_elem * s_ctx->max_eigen_num_lv1 + j;
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "col=%d\n", col));
        val_A = 0.0;
        for (ez = startz; ez < startz + nz; ++ez)
          for (ey = starty; ey < starty + ny; ++ey)
            for (ex = startx; ex < startx + nx; ++ex) {
              if (ex >= startx + 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                           1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
                val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                         int_ctx->meas_elem * avg_kappa_e *
                         (arr_ms_bases_c_array[i][ez][ey][ex - 1] -
                          arr_ms_bases_c_array[i][ez][ey][ex]) *
                         (arr_ms_bases_c_array[j][ez][ey][ex - 1] -
                          arr_ms_bases_c_array[j][ez][ey][ex]);
              } else if (ex != proc_startx) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                           1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
                val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ex + 1 == startx + nx && ex + 1 != proc_startx + proc_nx) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex + 1]);
                val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }

              if (ey >= starty + 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                           1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
                val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                         int_ctx->meas_elem * avg_kappa_e *
                         (arr_ms_bases_c_array[i][ez][ey - 1][ex] -
                          arr_ms_bases_c_array[i][ez][ey][ex]) *
                         (arr_ms_bases_c_array[j][ez][ey - 1][ex] -
                          arr_ms_bases_c_array[j][ez][ey][ex]);
              } else if (ey != proc_starty) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                           1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
                val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ey + 1 == starty + ny && ey + 1 != proc_starty + proc_ny) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[1][ez][ey + 1][ex]);
                val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }

              if (ez >= startz + 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         (arr_ms_bases_c_array[i][ez - 1][ey][ex] -
                          arr_ms_bases_c_array[i][ez][ey][ex]) *
                         (arr_ms_bases_c_array[j][ez - 1][ey][ex] -
                          arr_ms_bases_c_array[j][ez][ey][ex]);
              } else if (ez != proc_startz) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ez + 1 == startz + nz && ez + 1 != proc_startz + proc_nz) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
            }
        PetscCall(
            MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "HELLO\n"));
      if (coarse_elem_x != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][startx - 1] +
                         1.0 / int_ctx->arr_kappa_3d[0][ez][ey][startx]);
              val_A -= int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][ez][ey][startx] *
                       arr_ms_bases_c_array[j][ez][ey][startx - 1];
            }
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_x != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x + 1;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey) {
              avg_kappa_e =
                  2.0 /
                  (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][startx + nx - 1] +
                   1.0 / int_ctx->arr_kappa_3d[0][ez][ey][startx + nx]);
              val_A -= int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][ez][ey][startx + nx - 1] *
                       arr_ms_bases_c_array[j][ez][ey][startx + nx];
            }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "rowA=%d\n", row));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "colA=%d\n", col));
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ex = startx; ex < startx + nx; ++ex) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][starty - 1][ex] +
                         1.0 / int_ctx->arr_kappa_3d[1][ez][starty][ex]);
              val_A -= int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][ez][starty][ex] *
                       arr_ms_bases_c_array[j][ez][starty - 1][ex];
            }
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y + 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ex = startx; ex < startx + nx; ++ex) {
              avg_kappa_e =
                  2.0 /
                  (1.0 / int_ctx->arr_kappa_3d[1][ez][starty + ny - 1][ex] +
                   1.0 / int_ctx->arr_kappa_3d[1][ez][starty + ny][ex]);
              val_A -= int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][ez][starty + ny - 1][ex] *
                       arr_ms_bases_c_array[j][ez][starty + ny][ex];
            }
          // PetscCall(PetscPrintf(PETSC_COMM_SELF, "rowA=%d\n", row));
          // PetscCall(PetscPrintf(PETSC_COMM_SELF, "colA=%d\n", col));
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != 0) {
        coarse_elem_col =
            (coarse_elem_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ey = starty; ey < starty + ny; ++ey)
            for (ex = startx; ex < startx + nx; ++ex) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[2][startz - 1][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[2][startz][ey][ex]);
              val_A -= int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][startz][ey][ex] *
                       arr_ms_bases_c_array[j][startz - 1][ey][ex];
            }

          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            (coarse_elem_z + 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < (coarse_elem_col * s_ctx->max_eigen_num_lv1); ++j) {
          col = coarse_elem_col * s_ctx->max_eigen_num_lv1 + j;
          val_A = 0.0;
          for (ey = starty; ey < starty + ny; ++ey)
            for (ex = startx; ex < startx + nx; ++ex) {
              avg_kappa_e =
                  2.0 /
                  (1.0 / int_ctx->arr_kappa_3d[2][startz + nz - 1][ey][ex] +
                   1.0 / int_ctx->arr_kappa_3d[2][startz + nz][ey][ex]);
              val_A -= int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_c_array[i][startz + nz - 1][ey][ex] *
                       arr_ms_bases_c_array[j][startz + nz][ey][ex];
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "val_A=%f\n", val_A));
              
            }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "rowA=%d\n", row));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "colA=%d\n", col));
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));

  for (i = 0; i < s_ctx->max_eigen_num_lv2; ++i)
    PetscCall(DMCreateLocalVector(s_ctx->dm, &s_ctx->ms_bases_cc[i]));

  EPS eps;
  PetscInt nconv;
  PetscScalar eig_val, *arr_eig_vec, ***arr_ms_bases_cc;
  Vec eig_vec;

  PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
  PetscCall(EPSSetOperators(eps, A_i_inner, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetDimensions(eps, s_ctx->max_eigen_num_lv2, PETSC_DEFAULT,
                             PETSC_DEFAULT));
  ST st;
  PetscCall(EPSGetST(eps, &st));
  PetscCall(STSetType(st, STSINVERT));
  PetscCall(EPSSetTarget(eps, -NIL));
  PetscCall(EPSSetOptionsPrefix(eps, "epsl2_"));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCheck(nconv >= s_ctx->max_eigen_num_lv2, PETSC_COMM_WORLD,
             PETSC_ERR_USER,
             "SLEPc cannot find enough eigenvectors for level-2! (nconv=%d, "
             "eigen_num=%d)\n",
             nconv, s_ctx->max_eigen_num_lv2);
  PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));
  for (j = 0; j < s_ctx->max_eigen_num_lv2; ++j) {
    PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
    if (j == 0)
      s_ctx->eigen_min_lv2 = eig_val;

    if (j == s_ctx->max_eigen_num_lv2 - 1) {
      s_ctx->eigen_max_lv2 = eig_val;
    }
    PetscCall(VecGetArray(eig_vec, &arr_eig_vec));

    // Construct coarse-coarse bases in the original P space.
    PetscCall(
        DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases_cc[j], &arr_ms_bases_cc));

    for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num;
         ++coarse_elem) {
      _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty,
                                  &ny, &startz, &nz);
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          for (ex = startx; ex < startx + nx; ++ex)
            for (i = coarse_elem * s_ctx->max_eigen_num_lv1;
                 i < (coarse_elem + 1) * s_ctx->max_eigen_num_lv1; ++i)
              arr_ms_bases_cc[ez][ey][ex] +=
                  arr_eig_vec[i] *
                  arr_ms_bases_c_array[i -
                                       coarse_elem * s_ctx->max_eigen_num_lv1]
                                      [ez][ey][ex];
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases_cc[j],
                                  &arr_ms_bases_cc));
    PetscCall(VecRestoreArray(eig_vec, &arr_eig_vec));
  }

  // We now use a new way to construct ms_bases_c.
  // Destroy ms_bases_c_tmp finally.
  PetscScalar ***arr_ms_bases_c;
  PetscInt dof;
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i) {
    for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num;
         ++coarse_elem) {
      dof = i + coarse_elem * s_ctx->max_eigen_num_lv1;
      if (dof < (coarse_elem + 1) * s_ctx->max_eigen_num_lv1) {
        _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty,
                                    &ny, &startz, &nz);
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx,
                               &s_ctx->ms_bases_c[dof]));
        PetscCall(VecGetArray3d(s_ctx->ms_bases_c[dof], nz, ny, nx, 0, 0, 0,
                                &arr_ms_bases_c));
        for (ez = startz; ez < startz + nz; ++ez)
          for (ey = starty; ey < starty + ny; ++ey)
            PetscArraycpy(&arr_ms_bases_c[ez - startz][ey - starty][0],
                          &arr_ms_bases_c_array[i][ez][ey][startx], nx);
        PetscCall(VecRestoreArray3d(s_ctx->ms_bases_c[dof], nz, ny, nx, 0, 0, 0,
                                    &arr_ms_bases_c));
      }
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &int_ctx->ms_bases_c_tmp[i]));
  }

  // Do some cleaning.
  PetscCall(VecDestroy(&eig_vec));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A_i_inner));

  // The first time clean some unused ms_bases_cc.
  PetscCallMPI(MPI_Allreduce(&s_ctx->eigen_num_lv2,
                             &int_ctx->max_eigen_num_lv2_upd, 1, MPI_INT,
                             MPI_MAX, PETSC_COMM_WORLD));
  for (i = int_ctx->max_eigen_num_lv2_upd; i < s_ctx->max_eigen_num_lv2; ++i)
    PetscCall(VecDestroy(&s_ctx->ms_bases_cc[i]));

  // Retrieve off-process information.
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < int_ctx->max_eigen_num_lv2_upd; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, s_ctx->ms_bases_cc[i], INSERT_VALUES,
                              dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo, INSERT_VALUES,
                              s_ctx->ms_bases_cc[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_setup(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  // Construct internal information passing.
  _IntCtx int_ctx;
  int i;
  Vec kappa_loc[DIM];
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(
        DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &int_ctx.arr_kappa_3d[i]));
  }
  // for (i = 0; i < DIM; ++i) {
  //   PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i],
  //                                     &int_ctx.arr_kappa_3d[i]));
  //   PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
  // }
  int_ctx.meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  int_ctx.meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  int_ctx.meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  int_ctx.meas_face_xy = s_ctx->H_x * s_ctx->H_y;
  int_ctx.coarse_elem_num =
      s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  int_ctx.ms_bases_c_tmp =
      (Vec *)malloc(s_ctx->max_eigen_num_lv1 * sizeof(Vec));

  PetscCall(domainPartition(s_ctx, &int_ctx));

  PetscInt proc_nx, proc_ny, proc_nz;
  PetscCall(DMDAGetCorners(s_ctx->dm, NULL, NULL, NULL, &proc_nx, &proc_ny,
                           &proc_nz));
  PetscInt coarse_elem_num =
      s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  PetscInt eigen_len_lv1, eigen_len_lv1_max;
  eigen_len_lv1 =
      s_ctx->coarse_lenx[0] * s_ctx->coarse_leny[0] * s_ctx->coarse_lenz[0];
  PetscCallMPI(MPI_Allreduce(&eigen_len_lv1, &eigen_len_lv1_max, 1, MPI_INT,
                             MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(
      MatCreateAIJ(PETSC_COMM_WORLD, coarse_elem_num * s_ctx->max_eigen_num_lv1,
                   proc_nx * proc_ny * proc_nz, PETSC_DEFAULT, PETSC_DEFAULT,
                   eigen_len_lv1_max, NULL, 0, NULL, &s_ctx->Rc));

  PetscCall(MatCreateAIJ(
      PETSC_COMM_WORLD, s_ctx->max_eigen_num_lv2,
      coarse_elem_num * s_ctx->max_eigen_num_lv1, PETSC_DEFAULT, PETSC_DEFAULT,
      coarse_elem_num * s_ctx->max_eigen_num_lv1, NULL, 0, NULL, &s_ctx->Rcc));

  PetscInt m, n;
  PetscCall(MatGetSize(s_ctx->Rc, &m, &n));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "m1=%d, n1=%d\n", m, n));
  PetscCall(MatGetSize(s_ctx->Rcc, &m, &n));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "m1=%d, n1=%d\n", m, n));

  PetscCall(formRc(s_ctx, &int_ctx));
  PetscCall(formRcc(s_ctx, &int_ctx));

  PetscFunctionReturn(0);
  // }
}

PetscErrorCode PC_init(PCCtx *s_ctx, PetscScalar *dom, PetscInt *mesh) {
  PetscFunctionBeginUser;

  PetscCheck((dom[0] > 0.0 && dom[1] > 0.0 && dom[2] > 0.0), PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Errors in dom(L, W, H)=[%.5f, %.5f, %.5f].\n", dom[0], dom[1],
             dom[2]);
  PetscCheck((mesh[0] > 0 && mesh[1] > 0 && mesh[2] > 0), PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG, "Errors in mesh(M, N, P)=[%d, %d, %d].\n",
             mesh[0], mesh[1], mesh[2]);
  s_ctx->L = dom[0];
  s_ctx->W = dom[1];
  s_ctx->H = dom[2];
  s_ctx->M = mesh[0];
  s_ctx->N = mesh[1];
  s_ctx->P = mesh[2];

  s_ctx->sub_domains = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sd", &s_ctx->sub_domains, NULL));
  PetscCheck(s_ctx->sub_domains >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Error in sub_domains=%d.\n", s_ctx->sub_domains);

  s_ctx->max_eigen_num_lv1 = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv1", &s_ctx->max_eigen_num_lv1,
                               NULL));
  PetscCheck(s_ctx->max_eigen_num_lv1 >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Error in max_eigen_num_lv1=%d for the level-1 problem.\n",
             s_ctx->max_eigen_num_lv1);

  s_ctx->max_eigen_num_lv2 = 1;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv2", &s_ctx->max_eigen_num_lv2,
                               NULL));
  PetscCheck(s_ctx->max_eigen_num_lv2 >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Error in max_eigen_num_lv2=%d for the level-2 problem.\n",
             s_ctx->max_eigen_num_lv2);

  s_ctx->lv2_eigen_op = EIG_OP_MOD;
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-eg_op", &s_ctx->lv2_eigen_op, NULL));
  PetscCheck(s_ctx->lv2_eigen_op == EIG_OP_MEAN ||
                 s_ctx->lv2_eigen_op == EIG_OP_DIAG ||
                 s_ctx->lv2_eigen_op == EIG_OP_MOD,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Error in lv2_eigen_op=%d.\n", s_ctx->lv2_eigen_op);

  s_ctx->H_x = dom[0] / (double)mesh[0];
  s_ctx->H_y = dom[1] / (double)mesh[1];
  s_ctx->H_z = dom[2] / (double)mesh[2];

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, s_ctx->M,
                         s_ctx->N, s_ctx->P, PETSC_DECIDE, PETSC_DECIDE,
                         PETSC_DECIDE, 1, 1, NULL, NULL, NULL, &(s_ctx->dm)));
  // If oversampling=1, DMDA has a ghost point width=1 now, and this will
  // change the construction of A_i in level-1.
  PetscCall(DMSetUp(s_ctx->dm));

  for (PetscInt i = 0; i < DIM; ++i)
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[i]));

  PetscInt m, n, p, coarse_elem_num;
  // Users should be responsible for constructing kappa.
  PetscCall(DMDAGetInfo(s_ctx->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL,
                        NULL, NULL, NULL, NULL, NULL));
  if (s_ctx->sub_domains == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "The subdomain may not be proper (too "
                          "long/wide/high), reset sub_domains from %d to 2.\n",
                          s_ctx->sub_domains));
    s_ctx->sub_domains = 2;
  }
  coarse_elem_num =
      s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;

  s_ctx->ms_bases_c =
      (Vec *)malloc(s_ctx->max_eigen_num_lv1 * coarse_elem_num * sizeof(Vec));
  s_ctx->ms_bases_cc = (Vec *)malloc(s_ctx->max_eigen_num_lv2 * sizeof(Vec));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_starty));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startz));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_leny));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenz));
  // PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_num_lv1));
  PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_max_lv1));
  PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_min_lv1));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscInt i, coarse_elem_num =
                  s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;

  for (i = 0; i < coarse_elem_num * s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(VecDestroy(&s_ctx->ms_bases_c[i]));

  PetscCall(PC_final_default(s_ctx));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final_default(PCCtx *s_ctx) {
  PetscFunctionBeginUser;

  PetscCall(DMDestroy(&s_ctx->dm));
  free(s_ctx->ms_bases_c);
  free(s_ctx->ms_bases_cc);
  PetscCall(PetscFree(s_ctx->coarse_startx));
  PetscCall(PetscFree(s_ctx->coarse_starty));
  PetscCall(PetscFree(s_ctx->coarse_startz));
  PetscCall(PetscFree(s_ctx->coarse_lenx));
  PetscCall(PetscFree(s_ctx->coarse_leny));
  PetscCall(PetscFree(s_ctx->coarse_lenz));
  // PetscCall(PetscFree(s_ctx->eigen_num_lv1));
  PetscCall(PetscFree(s_ctx->eigen_max_lv1));
  PetscCall(PetscFree(s_ctx->eigen_min_lv1));

  PetscFunctionReturn(0);
}
