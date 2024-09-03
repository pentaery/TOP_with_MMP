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

void _PC_get_dof_idx(PCCtx *s_ctx, PetscInt *dof_idx) {
  PetscInt coarse_elem,
      coarse_elem_num =
          s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  dof_idx[0] = 0;
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem)
    dof_idx[coarse_elem + 1] =
        dof_idx[coarse_elem] + s_ctx->eigen_num_lv1[coarse_elem];
}

PetscErrorCode _PC_setup_lv1_partition(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
  Get the boundaries of each coarse subdomain.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

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

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV1] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv1_J(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
    Get level-1 block Jacobi.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i;
  PetscScalar avg_kappa_e, val_A[2][2];
  PetscScalar ***arrayBoundary;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, coarse_elem, row[2],
      col[2];
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF, 1, nx * ny * nz, nx * ny * nz,
                                7, NULL, &A_i));
    PetscCall(MatSetOption(A_i, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex) {
          // We first handle homogeneous Neumann BCs.
          if (ex >= startx + 1)
          // Inner x-direction edges.
          {
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
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          } else if (ex >= 1)
          // Left boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }
          if (ex + 1 == startx + nx && ex + 1 != s_ctx->M)
          // Right boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex + 1]);
            val_A[0][0] = int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }

          if (ey >= starty + 1)
          // Inner y-direction edges.
          {
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
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          } else if (ey >= 1)
          // Down boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }
          if (ey + 1 == starty + ny && ey + 1 != s_ctx->N)
          // Up boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey + 1][ex]);
            val_A[0][0] = int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }

          if (ez >= startz + 1)
          // Inner z-direction edges.
          {
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
            PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          } else if (ez >= 1)
          // Back boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }
          if (ez + 1 == startz + nz && ez + 1 != s_ctx->P)
          // Front boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
            val_A[0][0] = int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }
          if (arrayBoundary[ez][ey][ex] == 1) {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = int_ctx->arr_kappa_3d[2][ez][ey][ex];
            val_A[0][0] = 2.0 * int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                          int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   ADD_VALUES));
          }
        }
    PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &(s_ctx->ksp_lv1[coarse_elem])));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv1[coarse_elem], A_i, A_i));
    PetscCall(KSPSetTolerances(s_ctx->ksp_lv1[coarse_elem], 1e-1, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetType(s_ctx->ksp_lv1[coarse_elem], KSPCG));
  
    PC pc_;
    PetscCall(KSPGetPC(s_ctx->ksp_lv1[coarse_elem], &pc_));
	 //if (s_ctx->use_full_Cholesky_lv1) {
    //  PetscCall(PCSetType(pc_, PCCHOLESKY));
    //  PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMKL_PARDISO));
    //} else {
    //  PetscCall(PCSetType(pc_, PCICC));
    //}
    PetscCall(PCSetType(pc_, PCSOR));
    PetscCall(PCSetOptionsPrefix(pc_, "kspl1_"));
    PetscCall(
        KSPSetErrorIfNotConverged(s_ctx->ksp_lv1[coarse_elem], PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc_));

    PetscCall(KSPSetUp(s_ctx->ksp_lv1[coarse_elem]));
    PetscCall(MatDestroy(&A_i));
  }
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV1] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv2_eigen(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
  Get eigenvectors from level-1 for level-2.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscScalar ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv1], ***arr_M_i_3d,
      val_A[2][2], avg_kappa_e;
  PetscInt max_eigen_num, i, j, coarse_elem, startx, nx, ex, starty, ny, ey,
      startz, nz, ez, row[2], col[2];

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
    // PetscCall(VecScale(diag_M_i, int_ctx->meas_elem * 0.25 * M_PI * M_PI /
    // (nx * nx * s_ctx->H_x * s_ctx->H_x + ny * ny * s_ctx->H_y * s_ctx->H_y +
    // nz * nz * s_ctx->H_z * s_ctx->H_z)));
    if (s_ctx->lv2_eigen_op == EIG_OP_MOD)
      PetscCall(VecScale(diag_M_i, int_ctx->meas_elem));
    PetscCall(VecScale(diag_M_i, 1e9));
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
    PetscBool find_max_eigen = PETSC_FALSE;
    for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem] = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv1) {
        s_ctx->eigen_num_lv1[coarse_elem] = j + 1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (!find_max_eigen && j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_num_lv1[coarse_elem] = s_ctx->max_eigen_num_lv1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
      }

      PetscCall(VecGetArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_c_array[j][ez][ey][startx],
                                  &arr_eig_vec[ez - startz][ey - starty][0],
                                  nx));
      PetscCall(VecRestoreArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
    }
    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }

  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
  // We need to check the real maximum number of eigenvectors used, free what
  // those are not used. The first time clean ms_bases_c_tmp.
  max_eigen_num = 0;
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem)
    max_eigen_num = PetscMax(max_eigen_num, s_ctx->eigen_num_lv1[coarse_elem]);
  PetscCallMPI(MPI_Allreduce(&max_eigen_num, &int_ctx->max_eigen_num_lv1_upd, 1,
                             MPI_INT, MPI_MAX, PETSC_COMM_WORLD));
  for (i = int_ctx->max_eigen_num_lv1_upd; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &int_ctx->ms_bases_c_tmp[i]));

  Vec dummy_ms_bases_glo;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                              INSERT_VALUES, dummy_ms_bases_glo));
    PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo, INSERT_VALUES,
                              int_ctx->ms_bases_c_tmp[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2_1] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv2_c(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;
  /*********************************************************************
    Get lever-2 A_c matrix for two-grids.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_c;
  PetscInt i, j, coarse_elem, coarse_elem_x, coarse_elem_y, coarse_elem_z,
      startx, nx, ex, starty, ny, ey, startz, nz, ez, row, col;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz;
  PetscInt dof_idx[NEIGH + 1][int_ctx->coarse_elem_num + 2], coarse_elem_col;
  const PetscMPIInt *ng_ranks;
  PetscScalar ***arr_ms_bases_c_array[int_ctx->max_eigen_num_lv1_upd], val_A,
      avg_kappa_e;

  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i)
    PetscCall(DMDAVecGetArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                              &arr_ms_bases_c_array[i]));
  _PC_get_dof_idx(s_ctx, &dof_idx[0][0]);

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, dof_idx[0][int_ctx->coarse_elem_num],
                         dof_idx[0][int_ctx->coarse_elem_num], PETSC_DETERMINE,
                         PETSC_DETERMINE, dof_idx[0][int_ctx->coarse_elem_num],
                         NULL, NEIGH * int_ctx->max_eigen_num_lv1_upd, NULL,
                         &A_c));
  PetscCall(MatGetOwnershipRange(A_c, &dof_idx[0][int_ctx->coarse_elem_num + 1],
                                 NULL));
  PetscCall(DMDAGetNeighbors(s_ctx->dm, &ng_ranks));

  // Send first!
  if (proc_startx != 0)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[12], ng_ranks[13],
                          PETSC_COMM_WORLD)); // Left.
  if (proc_startx + proc_nx != s_ctx->M)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[14], ng_ranks[13],
                          PETSC_COMM_WORLD)); // Right.
  if (proc_starty != 0)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[10], ng_ranks[13],
                          PETSC_COMM_WORLD)); // Down.
  if (proc_starty + proc_ny != s_ctx->N)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[16], ng_ranks[13], PETSC_COMM_WORLD)); // Up.
  if (proc_startz != 0)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[4], ng_ranks[13],
                          PETSC_COMM_WORLD)); // Back.
  if (proc_startz + proc_nz != s_ctx->P)
    PetscCallMPI(MPI_Send(&dof_idx[0][0], int_ctx->coarse_elem_num + 2, MPI_INT,
                          ng_ranks[22], ng_ranks[13],
                          PETSC_COMM_WORLD)); // Front.

  // Receive then.
  MPI_Status ista;
  if (proc_startx != 0)
    PetscCallMPI(MPI_Recv(&dof_idx[LEFT][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[12], ng_ranks[12], PETSC_COMM_WORLD,
                          &ista)); // Left.

  if (proc_startx + proc_nx != s_ctx->M)
    PetscCallMPI(MPI_Recv(&dof_idx[RIGHT][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[14], ng_ranks[14], PETSC_COMM_WORLD,
                          &ista)); // Right.

  if (proc_starty != 0)
    PetscCallMPI(MPI_Recv(&dof_idx[DOWN][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[10], ng_ranks[10], PETSC_COMM_WORLD,
                          &ista)); // Down.

  if (proc_starty + proc_ny != s_ctx->N)
    PetscCallMPI(MPI_Recv(&dof_idx[UP][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[16], ng_ranks[16], PETSC_COMM_WORLD,
                          &ista)); // Up.

  if (proc_startz != 0)
    PetscCallMPI(MPI_Recv(&dof_idx[BACK][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[4], ng_ranks[4], PETSC_COMM_WORLD,
                          &ista)); // Back.

  if (proc_startz + proc_nz != s_ctx->P)
    PetscCallMPI(MPI_Recv(&dof_idx[FRONT][0], int_ctx->coarse_elem_num + 2,
                          MPI_INT, ng_ranks[22], ng_ranks[22], PETSC_COMM_WORLD,
                          &ista)); // Front.

  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_xyz(s_ctx, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                            &coarse_elem_z);
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    for (i = 0; i < s_ctx->eigen_num_lv1[coarse_elem]; ++i) {
      row = i + dof_idx[0][coarse_elem] +
            dof_idx[0][int_ctx->coarse_elem_num + 1];
      for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem]; ++j) {
        col = j + dof_idx[0][coarse_elem] +
              dof_idx[0][int_ctx->coarse_elem_num + 1];
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
              } else if (ex >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                           1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
                val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ex + 1 == startx + nx && ex + 1 != s_ctx->M) {
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
              } else if (ey >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                           1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
                val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ey + 1 == starty + ny && ey + 1 != s_ctx->N) {
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
              } else if (ez >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ez + 1 == startz + nz && ez + 1 != s_ctx->P) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }

      if (coarse_elem_x != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_startx != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + s_ctx->sub_domains - 1;
        for (j = 0; j < dof_idx[LEFT][coarse_elem_col + 1] -
                            dof_idx[LEFT][coarse_elem_col];
             ++j) {
          col = j + dof_idx[LEFT][coarse_elem_col] +
                dof_idx[LEFT][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_x != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x + 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_startx + proc_nx != s_ctx->M) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + 0;
        for (j = 0; j < dof_idx[RIGHT][coarse_elem_col + 1] -
                            dof_idx[RIGHT][coarse_elem_col];
             ++j) {
          col = j + dof_idx[RIGHT][coarse_elem_col] +
                dof_idx[RIGHT][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_starty != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (s_ctx->sub_domains - 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < dof_idx[DOWN][coarse_elem_col + 1] -
                            dof_idx[DOWN][coarse_elem_col];
             ++j) {
          col = j + dof_idx[DOWN][coarse_elem_col] +
                dof_idx[DOWN][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y + 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_starty + proc_ny != s_ctx->N) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            0 * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < dof_idx[UP][coarse_elem_col + 1] -
                            dof_idx[UP][coarse_elem_col];
             ++j) {
          col = j + dof_idx[UP][coarse_elem_col] +
                dof_idx[UP][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != 0) {
        coarse_elem_col =
            (coarse_elem_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_startz != 0) {
        coarse_elem_col =
            (s_ctx->sub_domains - 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < dof_idx[BACK][coarse_elem_col + 1] -
                            dof_idx[BACK][coarse_elem_col];
             ++j) {
          col = j + dof_idx[BACK][coarse_elem_col] +
                dof_idx[BACK][int_ctx->coarse_elem_num + 1];
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
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            (coarse_elem_z + 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = j + dof_idx[0][coarse_elem_col] +
                dof_idx[0][int_ctx->coarse_elem_num + 1];
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
            }
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      } else if (proc_startz + proc_nz != s_ctx->P) {
        coarse_elem_col = 0 * s_ctx->sub_domains * s_ctx->sub_domains +
                          coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < dof_idx[FRONT][coarse_elem_col + 1] -
                            dof_idx[FRONT][coarse_elem_col];
             ++j) {
          col = j + dof_idx[FRONT][coarse_elem_col] +
                dof_idx[FRONT][int_ctx->coarse_elem_num + 1];
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
            }
          PetscCall(MatSetValues(A_c, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_c, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_c, MAT_FINAL_ASSEMBLY));
  if (!s_ctx->no_shift_A_cc) {
    PetscCall(MatShift(A_c, NIL));
  }

  // Destroy ms_bases_c_tmp finally.
  PetscScalar ***arr_ms_bases_c;
  PetscInt dof;
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i) {
    for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num;
         ++coarse_elem) {
      dof = i + dof_idx[0][coarse_elem];
      if (dof < dof_idx[0][coarse_elem + 1]) {
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

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &s_ctx->ksp_lv2));
  PetscCall(KSPSetOperators(s_ctx->ksp_lv2, A_c, A_c));
  PetscCall(KSPSetTolerances(s_ctx->ksp_lv2, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(s_ctx->ksp_lv2, KSPCG));
  {
    PC pc_;
    PetscCall(KSPGetPC(s_ctx->ksp_lv2, &pc_));
    //PetscCall(PCSetType(pc_, PCLU));
    // PetscCall(PCSetType(pc_, PCCHOLESKY));
    // PetscCall(PCFactorSetShiftType(pc_, MAT_SHIFT_NONZERO));
    //PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMKL_CPARDISO));
    PetscCall(PCSetType(pc_, PCBJACOBI));
    PetscCall(PCSetOptionsPrefix(pc_, "kspl2_"));
    PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv2, PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc_));
  }
  PetscCall(KSPSetUp(s_ctx->ksp_lv2));
  PetscCall(MatDestroy(&A_c));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2_2] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv2_J(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
    Get lever-2 block Jacobi.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i;
  PetscInt i, j, coarse_elem, startx, nx, ex, starty, ny, ey, startz, nz, ez,
      row, col;
  PetscInt dof_idx[int_ctx->coarse_elem_num + 1], coarse_elem_col,
      coarse_elem_x, coarse_elem_y, coarse_elem_z;
  PetscScalar ***arr_ms_bases_c_array[int_ctx->max_eigen_num_lv1_upd], val_A,
      avg_kappa_e;
  PetscScalar ***arrayBoundary;

  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i)
    PetscCall(DMDAVecGetArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                              &arr_ms_bases_c_array[i]));
  _PC_get_dof_idx(s_ctx, &dof_idx[0]);
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF, 1,
                              dof_idx[int_ctx->coarse_elem_num],
                              dof_idx[int_ctx->coarse_elem_num],
                              7 * int_ctx->max_eigen_num_lv1_upd, NULL, &A_i));
  PetscCall(MatSetOption(A_i, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_xyz(s_ctx, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                            &coarse_elem_z);
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    for (i = 0; i < s_ctx->eigen_num_lv1[coarse_elem]; ++i) {
      row = dof_idx[coarse_elem] + i;
      for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem]; ++j) {
        col = dof_idx[coarse_elem] + j;
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
              } else if (ex >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                           1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
                val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ex + 1 == startx + nx && ex + 1 != s_ctx->M) {
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
              } else if (ey >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                           1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
                val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ey + 1 == starty + ny && ey + 1 != s_ctx->N) {
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
              } else if (ez >= 1) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ez + 1 == startz + nz && ez + 1 != s_ctx->P) {
                avg_kappa_e =
                    2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                           1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
                val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (arrayBoundary[ez][ey][ex] == 1) {
                avg_kappa_e = int_ctx->arr_kappa_3d[2][ez][ey][ex];
                val_A += 2.0 * int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                         int_ctx->meas_elem * avg_kappa_e *
                         arr_ms_bases_c_array[i][ez][ey][ex] *
                         arr_ms_bases_c_array[j][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }

      if (coarse_elem_x != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_x != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x + 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y + 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != 0) {
        coarse_elem_col =
            (coarse_elem_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != s_ctx->sub_domains - 1) {
        coarse_elem_col =
            (coarse_elem_z + 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
            }
          PetscCall(MatSetValues(A_i, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
  PetscCall(KSPCreate(PETSC_COMM_SELF, &s_ctx->ksp_lv2));
  PetscCall(KSPSetOperators(s_ctx->ksp_lv2, A_i, A_i));
  PetscCall(KSPSetTolerances(s_ctx->ksp_lv2, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(s_ctx->ksp_lv2, KSPCG));
  {
    PC pc_;
    PetscCall(KSPGetPC(s_ctx->ksp_lv2, &pc_));
    //PetscCall(PCSetType(pc_, PCCHOLESKY));
    //PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMKL_PARDISO));
    PetscCall(PCSetType(pc_, PCBJACOBI));
    PetscCall(PCSetOptionsPrefix(pc_, "kspl2_"));
    PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv2, PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc_));
  }
  PetscCall(KSPSetUp(s_ctx->ksp_lv2));
  PetscCall(MatDestroy(&A_i));

  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2_2] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv3_eigen(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
    Get eigenvectors from lever-2 for level-3.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i_inner;
  Vec dummy_ms_bases_glo;
  PetscInt coarse_elem, coarse_elem_col, dof_idx[int_ctx->coarse_elem_num + 1],
      i, j, row, col;
  PetscInt coarse_elem_x, coarse_elem_y, coarse_elem_z, startx, nx, ex, starty,
      ny, ey, startz, nz, ez;
  PetscInt proc_startx, proc_nx, proc_starty, proc_ny, proc_startz, proc_nz;
  PetscScalar val_A, avg_kappa_e,
      ***arr_ms_bases_c_array[int_ctx->max_eigen_num_lv1_upd];

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  _PC_get_dof_idx(s_ctx, dof_idx);
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i)
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
                                  &arr_ms_bases_c_array[i]));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, dof_idx[int_ctx->coarse_elem_num],
                            dof_idx[int_ctx->coarse_elem_num],
                            7 * int_ctx->max_eigen_num_lv1_upd, NULL,
                            &A_i_inner));
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_xyz(s_ctx, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                            &coarse_elem_z);
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    for (i = 0; i < s_ctx->eigen_num_lv1[coarse_elem]; ++i) {
      row = dof_idx[coarse_elem] + i;
      for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem]; ++j) {
        col = dof_idx[coarse_elem] + j;
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

      if (coarse_elem_x != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_y != 0) {
        coarse_elem_col =
            coarse_elem_z * s_ctx->sub_domains * s_ctx->sub_domains +
            (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
          PetscCall(
              MatSetValues(A_i_inner, 1, &row, 1, &col, &val_A, INSERT_VALUES));
        }
      }

      if (coarse_elem_z != 0) {
        coarse_elem_col =
            (coarse_elem_z - 1) * s_ctx->sub_domains * s_ctx->sub_domains +
            coarse_elem_y * s_ctx->sub_domains + coarse_elem_x;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_col]; ++j) {
          col = dof_idx[coarse_elem_col] + j;
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
            }
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
  PetscBool find_max_eigen = PETSC_FALSE;
  for (j = 0; j < s_ctx->max_eigen_num_lv2; ++j) {
    PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
    if (!find_max_eigen && j == 0)
      s_ctx->eigen_min_lv2 = eig_val;

    if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv2) {
      s_ctx->eigen_num_lv2 = j + 1;
      s_ctx->eigen_max_lv2 = eig_val;
      find_max_eigen = PETSC_TRUE;
    }

    if (!find_max_eigen && j == s_ctx->max_eigen_num_lv2 - 1) {
      s_ctx->eigen_num_lv2 = s_ctx->max_eigen_num_lv2;
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
            for (i = dof_idx[coarse_elem]; i < dof_idx[coarse_elem + 1]; ++i)
              arr_ms_bases_cc[ez][ey][ex] +=
                  arr_eig_vec[i] *
                  arr_ms_bases_c_array[i - dof_idx[coarse_elem]][ez][ey][ex];
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases_cc[j],
                                  &arr_ms_bases_cc));
    PetscCall(VecRestoreArray(eig_vec, &arr_eig_vec));
  }

  // We now use a new way to construct ms_bases_c.
  // Destroy ms_bases_c_tmp finally.
  PetscScalar ***arr_ms_bases_c;
  PetscInt dof;
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i) {
    for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num;
         ++coarse_elem) {
      dof = i + dof_idx[coarse_elem];
      if (dof < dof_idx[coarse_elem + 1]) {
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

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV3_1] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv3_cc(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  /*********************************************************************
    Get level-3.
  *********************************************************************/
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_cc;
  PetscInt A_cc_range[NEIGH + 1][2];
  PetscInt i, ex, ey, ez, proc_startx, proc_nx, proc_starty, proc_ny,
      proc_startz, proc_nz, row, col, row_off, col_off;
  const PetscMPIInt *ng_ranks;
  PetscScalar ***arr_ms_bases_cc_array[int_ctx->max_eigen_num_lv2_upd], val_A,
      avg_kappa_e;
  PetscScalar ***arrayBoundary;

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  for (i = 0; i < int_ctx->max_eigen_num_lv2_upd; ++i)
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases_cc[i],
                              &arr_ms_bases_cc_array[i]));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));

  // PetscCall(MatCreateSBAIJ(PETSC_COMM_WORLD, 1, s_ctx->eigen_num_lv2,
  // s_ctx->eigen_num_lv2, PETSC_DETERMINE, PETSC_DETERMINE,
  // s_ctx->eigen_num_lv2, NULL, NEIGH * int_ctx->max_eigen_num_lv2_upd, NULL,
  // &A_cc));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, s_ctx->eigen_num_lv2,
                         s_ctx->eigen_num_lv2, PETSC_DETERMINE, PETSC_DETERMINE,
                         s_ctx->eigen_num_lv2, NULL,
                         NEIGH * int_ctx->max_eigen_num_lv2_upd, NULL, &A_cc));
  // PetscCall(MatSetOption(A_cc, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
  PetscCall(DMDAGetNeighbors(s_ctx->dm, &ng_ranks));

  // Send range first.
  if (proc_startx != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[12],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Left.
  if (proc_startx + proc_nx != s_ctx->M)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[14],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Right.
  if (proc_starty != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[10],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Down.
  if (proc_starty + proc_ny != s_ctx->N)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[16],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Up.
  if (proc_startz != 0)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[4],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Back.
  if (proc_startz + proc_nz != s_ctx->P)
    PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[22],
                          ng_ranks[13], PETSC_COMM_WORLD)); // Front.

  // Receive range then.
  MPI_Status ista;
  if (proc_startx != 0) {
    PetscCallMPI(MPI_Recv(&A_cc_range[LEFT][0], 2, MPI_INT, ng_ranks[12],
                          ng_ranks[12], PETSC_COMM_WORLD, &ista)); // Left.
  }
  if (proc_startx + proc_nx != s_ctx->M) {
    PetscCallMPI(MPI_Recv(&A_cc_range[RIGHT][0], 2, MPI_INT, ng_ranks[14],
                          ng_ranks[14], PETSC_COMM_WORLD, &ista)); // Right.
  }
  if (proc_starty != 0) {
    PetscCallMPI(MPI_Recv(&A_cc_range[DOWN][0], 2, MPI_INT, ng_ranks[10],
                          ng_ranks[10], PETSC_COMM_WORLD, &ista)); // Down.
  }
  if (proc_starty + proc_ny != s_ctx->N) {
    PetscCallMPI(MPI_Recv(&A_cc_range[UP][0], 2, MPI_INT, ng_ranks[16],
                          ng_ranks[16], PETSC_COMM_WORLD, &ista)); // Up.
  }
  if (proc_startz != 0) {
    PetscCallMPI(MPI_Recv(&A_cc_range[BACK][0], 2, MPI_INT, ng_ranks[4],
                          ng_ranks[4], PETSC_COMM_WORLD, &ista)); // Back.
  }
  if (proc_startz + proc_nz != s_ctx->P) {
    PetscCallMPI(MPI_Recv(&A_cc_range[FRONT][0], 2, MPI_INT, ng_ranks[22],
                          ng_ranks[22], PETSC_COMM_WORLD, &ista)); // Front.
  }

  for (row = A_cc_range[0][0]; row < A_cc_range[0][1]; ++row) {
    row_off = row - A_cc_range[0][0];
    for (col = A_cc_range[0][0]; col < A_cc_range[0][1]; ++col) {
      col_off = col - A_cc_range[0][0];
      val_A = 0.0;
      for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
        for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
          for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
            if (ex >= proc_startx + 1) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                         1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
              val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                       int_ctx->meas_elem * avg_kappa_e *
                       (arr_ms_bases_cc_array[row_off][ez][ey][ex - 1] -
                        arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                       (arr_ms_bases_cc_array[col_off][ez][ey][ex - 1] -
                        arr_ms_bases_cc_array[col_off][ez][ey][ex]);
            } else if (ex != 0) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                         1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
              val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }
            if (ex + 1 == proc_startx + proc_nx && ex + 1 != s_ctx->M) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex + 1]);
              val_A += int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }

            if (ey >= proc_starty + 1) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                         1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
              val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                       int_ctx->meas_elem * avg_kappa_e *
                       (arr_ms_bases_cc_array[row_off][ez][ey - 1][ex] -
                        arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                       (arr_ms_bases_cc_array[col_off][ez][ey - 1][ex] -
                        arr_ms_bases_cc_array[col_off][ez][ey][ex]);
            } else if (ey != 0) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                         1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
              val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }
            if (ey + 1 == proc_starty + proc_ny && ey + 1 != s_ctx->N) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[1][ez][ey + 1][ex]);
              val_A += int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }

            if (ez >= proc_startz + 1) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
              val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       (arr_ms_bases_cc_array[row_off][ez - 1][ey][ex] -
                        arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                       (arr_ms_bases_cc_array[col_off][ez - 1][ey][ex] -
                        arr_ms_bases_cc_array[col_off][ez][ey][ex]);
            } else if (ez != 0) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
              val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }
            if (ez + 1 == proc_startz + proc_nz && ez + 1 != s_ctx->P) {
              avg_kappa_e =
                  2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                         1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
              val_A += int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }
            if (arrayBoundary[ez][ey][ex] == 1) {
              avg_kappa_e = int_ctx->arr_kappa_3d[2][ez][ey][ex];
              val_A += 2.0 * int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                       int_ctx->meas_elem * avg_kappa_e *
                       arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                       arr_ms_bases_cc_array[col_off][ez][ey][ex];
            }
          }
      PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
    }

    if (proc_startx != 0)
      for (col = A_cc_range[LEFT][0]; col < A_cc_range[LEFT][1]; ++col) {
        col_off = col - A_cc_range[LEFT][0];
        val_A = 0.0;
        for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
            ex = proc_startx;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex]);
            val_A -= int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez][ey][ex - 1];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
    if (proc_startx + proc_nx != s_ctx->M)
      for (col = A_cc_range[RIGHT][0]; col < A_cc_range[RIGHT][1]; ++col) {
        col_off = col - A_cc_range[RIGHT][0];
        val_A = 0.0;
        for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
            ex = proc_startx + proc_nx - 1;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[0][ez][ey][ex + 1]);
            val_A -= int_ctx->meas_face_yz * int_ctx->meas_face_yz /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez][ey][ex + 1];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
    if (proc_starty != 0)
      for (col = A_cc_range[DOWN][0]; col < A_cc_range[DOWN][1]; ++col) {
        col_off = col - A_cc_range[DOWN][0];
        val_A = 0.0;
        for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
          for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
            ey = proc_starty;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]);
            val_A -= int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez][ey - 1][ex];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
    if (proc_starty + proc_ny != s_ctx->N)
      for (col = A_cc_range[UP][0]; col < A_cc_range[UP][1]; ++col) {
        col_off = col - A_cc_range[UP][0];
        val_A = 0.0;
        for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
          for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
            ey = proc_starty + proc_ny - 1;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[1][ez][ey + 1][ex]);
            val_A -= int_ctx->meas_face_zx * int_ctx->meas_face_zx /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez][ey + 1][ex];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
    if (proc_startz != 0)
      for (col = A_cc_range[BACK][0]; col < A_cc_range[BACK][1]; ++col) {
        col_off = col - A_cc_range[BACK][0];
        val_A = 0.0;
        for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
          for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
            ez = proc_startz;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]);
            val_A -= int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez - 1][ey][ex];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
    if (proc_startz + proc_nz != s_ctx->P)
      for (col = A_cc_range[FRONT][0]; col < A_cc_range[FRONT][1]; ++col) {
        col_off = col - A_cc_range[FRONT][0];
        val_A = 0.0;
        for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
          for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
            ez = proc_startz + proc_nz - 1;
            avg_kappa_e =
                2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex] +
                       1.0 / int_ctx->arr_kappa_3d[2][ez + 1][ey][ex]);
            val_A -= int_ctx->meas_face_xy * int_ctx->meas_face_xy /
                     int_ctx->meas_elem * avg_kappa_e *
                     arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                     arr_ms_bases_cc_array[col_off][ez + 1][ey][ex];
          }
        PetscCall(MatSetValues(A_cc, 1, &row, 1, &col, &val_A, INSERT_VALUES));
      }
  }
  for (i = 0; i < int_ctx->max_eigen_num_lv2_upd; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases_cc[i],
                                  &arr_ms_bases_cc_array[i]));
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(MatAssemblyBegin(A_cc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A_cc, MAT_FINAL_ASSEMBLY));
  if (!s_ctx->no_shift_A_cc) {
    PetscCall(MatShift(A_cc, NIL));
  }
  // MatView(A_cc, 0);
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &s_ctx->ksp_lv3));
  PetscCall(KSPSetOperators(s_ctx->ksp_lv3, A_cc, A_cc));
  PetscCall(KSPSetTolerances(s_ctx->ksp_lv3, 1e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetType(s_ctx->ksp_lv3, KSPCG));
  {
    PC pc_;
    PetscCall(KSPGetPC(s_ctx->ksp_lv3, &pc_));
    //PetscCall(PCSetType(pc_, PCLU));
    // PetscCall(PCSetType(pc_, PCCHOLESKY));
    // PetscCall(PCFactorSetShiftType(pc_, MAT_SHIFT_NONZERO));
    //PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMKL_CPARDISO));
    PetscCall(PCSetType(pc_, PCBJACOBI));
    PetscCall(PCSetOptionsPrefix(pc_, "kspl3_"));
    PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv3, PETSC_TRUE));
    PetscCall(PCSetFromOptions(pc_));
  }
  PetscCall(KSPSetUp(s_ctx->ksp_lv3));
  PetscCall(MatDestroy(&A_cc));

  // The second time clean some unused ms_bases_cc.
  for (i = s_ctx->max_eigen_num_lv2; i < int_ctx->max_eigen_num_lv2_upd; ++i)
    PetscCall(VecDestroy(&s_ctx->ms_bases_cc[i]));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV3_2] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup(PCCtx *s_ctx) {
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
  int_ctx.meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  int_ctx.meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  int_ctx.meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  int_ctx.meas_face_xy = s_ctx->H_x * s_ctx->H_y;
  int_ctx.coarse_elem_num =
      s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  int_ctx.ms_bases_c_tmp =
      (Vec *)malloc(s_ctx->max_eigen_num_lv1 * sizeof(Vec));

  PetscCall(_PC_setup_lv1_partition(s_ctx, &int_ctx));
  PetscCall(_PC_setup_lv1_J(s_ctx, &int_ctx));
  PetscCall(_PC_setup_lv2_eigen(s_ctx, &int_ctx));
  if (s_ctx->use_2level)
    PetscCall(_PC_setup_lv2_c(s_ctx, &int_ctx));
  else {
    PetscCall(_PC_setup_lv2_J(s_ctx, &int_ctx));
    PetscCall(_PC_setup_lv3_eigen(s_ctx, &int_ctx));
    PetscCall(_PC_setup_lv3_cc(s_ctx, &int_ctx));
  }

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i],
                                      &int_ctx.arr_kappa_3d[i]));
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv1(PCCtx *s_ctx, Vec x, Vec y) {
  // Input r, return y, additive.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;

  PetscCall(PetscTime(&time_tmp));
  PetscInt ey, ez, ex, startx, starty, startz, nx, ny, nz;
  PetscInt coarse_elem_p,
      coarse_elem_p_num =
          s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  Vec x_loc, y_loc, rhs, sol;
  PetscScalar ***arr_x, ***arr_y, ***arr_rhs, ***arr_sol;

  PetscCall(DMGetLocalVector(s_ctx->dm, &x_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, x, INSERT_VALUES, x_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x_loc, &arr_x));
  PetscCall(DMGetLocalVector(s_ctx->dm, &y_loc));
  PetscCall(VecZeroEntries(y_loc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, y_loc, &arr_y));

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem_p, &startx, &nx, &starty,
                                &ny, &startz, &nz);

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &rhs));
    PetscCall(VecDuplicate(rhs, &sol));
    PetscCall(VecGetArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_rhs[ez - startz][ey - starty][0],
                                &arr_x[ez][ey][startx], nx));
    PetscCall(VecRestoreArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
    PetscCall(KSPSolve(s_ctx->ksp_lv1[coarse_elem_p], rhs, sol));
    VecDestroy(&rhs);

    PetscCall(VecGetArray3d(sol, nz, ny, nx, 0, 0, 0, &arr_sol));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex)
          arr_y[ez][ey][ex] += arr_sol[ez - startz][ey - starty][ex - startx];

    PetscCall(VecRestoreArray3d(sol, nz, ny, nx, 0, 0, 0, &arr_sol));
    VecDestroy(&sol);
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(s_ctx->dm, x_loc, &arr_x));
  PetscCall(DMRestoreLocalVector(s_ctx->dm, &x_loc));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, y_loc, &arr_y));

  PetscCall(DMLocalToGlobal(s_ctx->dm, y_loc, ADD_VALUES, y));
  PetscCall(DMRestoreLocalVector(s_ctx->dm, &y_loc));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV_LV1] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv2_2level(PCCtx *s_ctx, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Vec rhs, sol, temp_loc;
  PetscScalar ***arr_x, ***arr_y, *arr_rhs, *arr_sol, ***arr_temp_loc;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, coarse_elem,
      coarse_elem_num =
          s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  PetscInt dof_idx[coarse_elem_num + 1];

  _PC_get_dof_idx(s_ctx, &dof_idx[0]);
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, dof_idx[coarse_elem_num],
                         PETSC_DETERMINE, &rhs));
  PetscCall(VecDuplicate(rhs, &sol));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arr_x));
  PetscCall(VecGetArray(rhs, &arr_rhs));
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(VecGetArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_temp_loc[ez - startz][ey - starty][0],
                                &arr_x[ez][ey][startx], nx));
    PetscCall(VecRestoreArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    PetscCall(VecMDot(temp_loc, s_ctx->eigen_num_lv1[coarse_elem],
                      &s_ctx->ms_bases_c[dof_idx[coarse_elem]],
                      &arr_rhs[dof_idx[coarse_elem]]));
    PetscCall(VecDestroy(&temp_loc));
  }
  PetscCall(VecRestoreArray(rhs, &arr_rhs));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arr_x));

  PetscCall(KSPSolve(s_ctx->ksp_lv2, rhs, sol));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecGetArray(sol, &arr_sol));
  PetscCall(DMDAVecGetArray(s_ctx->dm, y, &arr_y));

  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(VecMAXPY(temp_loc, s_ctx->eigen_num_lv1[coarse_elem],
                       &arr_sol[dof_idx[coarse_elem]],
                       &s_ctx->ms_bases_c[dof_idx[coarse_elem]]));
    PetscCall(VecGetArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex)
          arr_y[ez][ey][ex] +=
              arr_temp_loc[ez - startz][ey - starty][ex - startx];
    PetscCall(VecRestoreArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    PetscCall(VecDestroy(&temp_loc));
  }
  PetscCall(VecRestoreArray(sol, &arr_sol));
  PetscCall(VecDestroy(&sol));

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, y, &arr_y));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV_LV2] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv2(PCCtx *s_ctx, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Vec rhs, sol, temp_loc;
  PetscScalar ***arr_x, ***arr_y, *arr_rhs, *arr_sol, ***arr_temp_loc;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, coarse_elem,
      coarse_elem_num =
          s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  PetscInt dof_idx[coarse_elem_num + 1];

  _PC_get_dof_idx(s_ctx, &dof_idx[0]);
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, dof_idx[coarse_elem_num], &rhs));
  PetscCall(VecDuplicate(rhs, &sol));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arr_x));
  PetscCall(VecGetArray(rhs, &arr_rhs));
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(VecGetArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_temp_loc[ez - startz][ey - starty][0],
                                &arr_x[ez][ey][startx], nx));
    PetscCall(VecRestoreArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    PetscCall(VecMDot(temp_loc, s_ctx->eigen_num_lv1[coarse_elem],
                      &s_ctx->ms_bases_c[dof_idx[coarse_elem]],
                      &arr_rhs[dof_idx[coarse_elem]]));
    PetscCall(VecDestroy(&temp_loc));
  }
  PetscCall(VecRestoreArray(rhs, &arr_rhs));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arr_x));

  PetscCall(KSPSolve(s_ctx->ksp_lv2, rhs, sol));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecGetArray(sol, &arr_sol));
  PetscCall(DMDAVecGetArray(s_ctx->dm, y, &arr_y));

  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(VecMAXPY(temp_loc, s_ctx->eigen_num_lv1[coarse_elem],
                       &arr_sol[dof_idx[coarse_elem]],
                       &s_ctx->ms_bases_c[dof_idx[coarse_elem]]));
    PetscCall(VecGetArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex)
          arr_y[ez][ey][ex] +=
              arr_temp_loc[ez - startz][ey - starty][ex - startx];
    PetscCall(VecRestoreArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    PetscCall(VecDestroy(&temp_loc));
  }
  PetscCall(VecRestoreArray(sol, &arr_sol));
  PetscCall(VecDestroy(&sol));

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, y, &arr_y));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV_LV2] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv3(PCCtx *s_ctx, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;

  PetscCall(PetscTime(&time_tmp));
  Vec rhs, sol, temp_loc;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez;
  PetscScalar *arr_rhs, *arr_sol, ***arr_x, ***arr_y, ***arr_temp_loc;

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, s_ctx->eigen_num_lv2,
                         PETSC_DETERMINE, &rhs));
  PetscCall(VecDuplicate(rhs, &sol));
  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));

  PetscCall(DMGetLocalVector(s_ctx->dm, &temp_loc));
  PetscCall(VecZeroEntries(temp_loc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, temp_loc, &arr_temp_loc));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arr_x));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      PetscCall(PetscArraycpy(&arr_temp_loc[ez][ey][proc_startx],
                              &arr_x[ez][ey][proc_startx], proc_nx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arr_x));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, temp_loc, &arr_temp_loc));

  PetscCall(VecGetArray(rhs, &arr_rhs));
  PetscCall(VecMDot(temp_loc, s_ctx->eigen_num_lv2, &s_ctx->ms_bases_cc[0],
                    &arr_rhs[0]));
  PetscCall(VecRestoreArray(rhs, &arr_rhs));
  PetscCall(KSPSolve(s_ctx->ksp_lv3, rhs, sol));
  PetscCall(VecDestroy(&rhs));

  PetscCall(VecGetArray(sol, &arr_sol));
  PetscCall(VecZeroEntries(temp_loc));
  PetscCall(VecMAXPY(temp_loc, s_ctx->eigen_num_lv2, &arr_sol[0],
                     &s_ctx->ms_bases_cc[0]));
  PetscCall(VecRestoreArray(sol, &arr_sol));
  PetscCall(VecDestroy(&sol));

  PetscCall(DMDAVecGetArray(s_ctx->dm, temp_loc, &arr_temp_loc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, y, &arr_y));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex)
        arr_y[ez][ey][ex] += arr_temp_loc[ez][ey][ex];
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, temp_loc, &arr_temp_loc));
  PetscCall(DMRestoreLocalVector(s_ctx->dm, &temp_loc));

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, y, &arr_y));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV_LV3] -= time_tmp;

  PetscFunctionReturn(0);
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
  s_ctx->widthportion = 0.1;
  s_ctx->lengthportion = 0.1;

  s_ctx->smoothing_iters_lv1 = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-si_lv1",
                               &s_ctx->smoothing_iters_lv1, NULL));
  s_ctx->smoothing_iters_lv1 =
      s_ctx->smoothing_iters_lv1 >= 1 ? s_ctx->smoothing_iters_lv1 : 1;

  s_ctx->smoothing_iters_lv2 = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-si_lv2",
                               &s_ctx->smoothing_iters_lv2, NULL));
  s_ctx->smoothing_iters_lv2 =
      s_ctx->smoothing_iters_lv2 >= 1 ? s_ctx->smoothing_iters_lv2 : 1;

  s_ctx->sub_domains = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sd", &s_ctx->sub_domains, NULL));
  PetscCheck(s_ctx->sub_domains >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Error in sub_domains=%d.\n", s_ctx->sub_domains);

  s_ctx->max_eigen_num_lv1 = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv1", &s_ctx->max_eigen_num_lv1,
                               NULL));
  PetscCheck(s_ctx->max_eigen_num_lv1 >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Error in max_eigen_num_lv1=%d for the level-1 problem.\n",
             s_ctx->max_eigen_num_lv1);

  s_ctx->max_eigen_num_lv2 = 4;
  s_ctx->eigen_num_lv2 = 4;
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

  s_ctx->eigen_bd_lv1 = -1.0;
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-eb_lv1", &s_ctx->eigen_bd_lv1, NULL));
  s_ctx->eigen_bd_lv1 =
      s_ctx->eigen_bd_lv1 < 0.0 ? 1.0 / NIL : s_ctx->eigen_bd_lv1;

  s_ctx->eigen_bd_lv2 = -1.0;
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-eb_lv2", &s_ctx->eigen_bd_lv2, NULL));
  s_ctx->eigen_bd_lv2 =
      s_ctx->eigen_bd_lv2 < 0.0 ? 1.0 / NIL : s_ctx->eigen_bd_lv2;

  s_ctx->use_W_cycle = PETSC_FALSE;
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-use_W_cycle", &s_ctx->use_W_cycle));

  s_ctx->no_shift_A_cc = PETSC_FALSE;
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-no_shift_A_cc", &s_ctx->no_shift_A_cc));

  s_ctx->use_full_Cholesky_lv1 = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-use_full_Cholesky_lv1",
                                &s_ctx->use_full_Cholesky_lv1));

  s_ctx->use_2level = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(NULL, NULL, "-use_2level", &s_ctx->use_2level));

  s_ctx->H_x = dom[0] / (double)mesh[0];
  s_ctx->H_y = dom[1] / (double)mesh[1];
  s_ctx->H_z = dom[2] / (double)mesh[2];

  // PetscInt progress = 1;
  // PetscCall(PetscOptionsGetInt(NULL, NULL, "-progress", &progress, NULL));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, s_ctx->M,
                         s_ctx->N, s_ctx->P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1,
                         NULL, NULL, NULL, &(s_ctx->dm)));
  // If oversampling=1, DMDA has a ghost point width=1 now, and this will change
  // the construction of A_i in level-1.
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
  s_ctx->ksp_lv1 = (KSP *)malloc(coarse_elem_num * sizeof(KSP));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_starty));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_startz));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenx));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_leny));
  PetscCall(PetscMalloc1(s_ctx->sub_domains, &s_ctx->coarse_lenz));
  PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_num_lv1));
  PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_max_lv1));
  PetscCall(PetscMalloc1(coarse_elem_num, &s_ctx->eigen_min_lv1));

  // Log the time used in every statge.
  PetscCall(PetscMemzero(&s_ctx->t_stages[0], sizeof(s_ctx->t_stages)));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_print_info(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DIM=%d, M=%d, N=%d, P=%d.\n", 3,
                        s_ctx->M, s_ctx->N, s_ctx->P));
  PetscInt m, n, p;
  PetscCall(DMDAGetInfo(s_ctx->dm, NULL, NULL, NULL, NULL, &m, &n, &p, NULL,
                        NULL, NULL, NULL, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "MPI Processes in each direction: X=%d, Y=%d, Z=%d.\n",
                        m, n, p));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "smoothing_iters_lv1=%d, smoothing_iters_lv2=%d, "
                        "sub_domains=%d, lv2_eigen_op=%d\n",
                        s_ctx->smoothing_iters_lv1, s_ctx->smoothing_iters_lv2,
                        s_ctx->sub_domains, s_ctx->lv2_eigen_op));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "max_eigen_num_lv1=%d, eigen_bd_lv1=%.5E\n",
                        s_ctx->max_eigen_num_lv1, s_ctx->eigen_bd_lv1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "max_eigen_num_lv2=%d, eigen_bd_lv2=%.5E\n",
                        s_ctx->max_eigen_num_lv2, s_ctx->eigen_bd_lv2));
  if (s_ctx->use_2level)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Only use the 2-level version!\n"));
  PetscFunctionReturn(0);
}

PetscErrorCode PC_create_A(PCCtx *s_ctx, Mat *A) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(s_ctx->dm, A));
  Vec kappa_loc[DIM]; // Destroy later.
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez, i;
  // 赋值迭代变量
  PetscScalar ***arr_kappa_3d[DIM], val_A[2][2], meas_elem, meas_face_yz,
      meas_face_zx, meas_face_xy, avg_kappa_e;
  MatStencil row[2], col[2];

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  // 将每个进程中的kappa取出来

  meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  meas_face_xy = s_ctx->H_x * s_ctx->H_y;
  // 计算网格体积和面积

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if (ex >= 1) {
          row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] +
                               1.0 / arr_kappa_3d[0][ez][ey][ex]);
          val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0],
                                        &val_A[0][0], ADD_VALUES));
        }
        if (ey >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] +
                               1.0 / arr_kappa_3d[1][ez][ey][ex]);
          val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0],
                                        &val_A[0][0], ADD_VALUES));
        }
        if (ez >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] +
                               1.0 / arr_kappa_3d[2][ez][ey][ex]);
          val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(*A, 2, &row[0], 2, &col[0],
                                        &val_A[0][0], ADD_VALUES));
        }
      }
  // A的赋值
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PC_setup(PC pc) {
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  PetscCall(_PC_setup(s_ctx));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y) {
  // Input x, return y.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  PCCtx *s_ctx;
  Mat A;
  Vec r;
  PetscInt i;

  PetscCall(PCShellGetContext(pc, &s_ctx));
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCall(DMGetGlobalVector(s_ctx->dm, &r));
  // Need to zero y first!
  PetscCall(VecZeroEntries(y));

  if (s_ctx->use_W_cycle && !s_ctx->use_2level) {
    for (i = 0; i < s_ctx->smoothing_iters_lv1; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv1(s_ctx, r, y));
    }

    for (i = 0; i < s_ctx->smoothing_iters_lv2; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv2(s_ctx, r, y));
    }

    PetscCall(MatMult(A, y, r));
    PetscCall(VecAYPX(r, -1, x));
    PetscCall(_PC_apply_vec_lv3(s_ctx, r, y));

    for (i = 0; i < s_ctx->smoothing_iters_lv2; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv2(s_ctx, r, y));
    }

    PetscCall(MatMult(A, y, r));
    PetscCall(VecAYPX(r, -1, x));
    PetscCall(_PC_apply_vec_lv3(s_ctx, r, y));

    for (i = 0; i < s_ctx->smoothing_iters_lv2; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv2(s_ctx, r, y));
    }

    for (i = 0; i < s_ctx->smoothing_iters_lv1; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv1(s_ctx, r, y));
    }
  } else {
    for (i = 0; i < s_ctx->smoothing_iters_lv1; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv1(s_ctx, r, y));
    }

    if (!s_ctx->use_2level) {
      for (i = 0; i < s_ctx->smoothing_iters_lv2; ++i) {
        PetscCall(MatMult(A, y, r));
        PetscCall(VecAYPX(r, -1, x));
        PetscCall(_PC_apply_vec_lv2(s_ctx, r, y));
      }

      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv3(s_ctx, r, y));

      for (i = 0; i < s_ctx->smoothing_iters_lv2; ++i) {
        PetscCall(MatMult(A, y, r));
        PetscCall(VecAYPX(r, -1, x));
        PetscCall(_PC_apply_vec_lv2(s_ctx, r, y));
      }
    } else {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv2_2level(s_ctx, r, y));
    }

    for (i = 0; i < s_ctx->smoothing_iters_lv1; ++i) {
      PetscCall(MatMult(A, y, r));
      PetscCall(VecAYPX(r, -1, x));
      PetscCall(_PC_apply_vec_lv1(s_ctx, r, y));
    }
  }

  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &r));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV] -= time_tmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PC_get_range(const void *sendbuff, void *recvbuff,
                            MPI_Datatype datatype) {
  PetscFunctionBeginUser;
  if (datatype == MPI_INT) {
    PetscInt *recvbuf = (PetscInt *)recvbuff;
    PetscCallMPI(MPI_Reduce(sendbuff, &recvbuf[0], 1, MPI_INT, MPI_MIN, 0,
                            PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(sendbuff, &recvbuf[1], 1, MPI_INT, MPI_MAX, 0,
                            PETSC_COMM_WORLD));
  }
  if (datatype == MPI_DOUBLE) {
    PetscScalar *recvbuf = (PetscScalar *)recvbuff;
    PetscCallMPI(MPI_Reduce(sendbuff, &recvbuf[0], 1, MPI_DOUBLE, MPI_MIN, 0,
                            PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(sendbuff, &recvbuf[1], 1, MPI_DOUBLE, MPI_MAX, 0,
                            PETSC_COMM_WORLD));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PC_print_stat(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscInt i, size_mat_lv3,
      loc_eigen_num_lv1_range[2] = {0}, eigen_num_lv1_range[2] = {0},
      eigen_num_lv2_range[2] = {0}, coarse_elem_p,
      coarse_elem_p_num =
          s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  PetscScalar loc_eigen_max_lv1_range[2] = {0},
              loc_eigen_min_lv1_range[2] = {0}, eigen_max_lv1_range[2] = {0},
              eigen_min_lv1_range[2] = {0}, eigen_max_lv2_range[2] = {0},
              eigen_min_lv2_range[2] = {0};
  PetscScalar t_stages_range[MAX_LOG_STATES][2] = {{0}};

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    if (coarse_elem_p == 0) {
      loc_eigen_max_lv1_range[0] = s_ctx->eigen_max_lv1[0];
      loc_eigen_max_lv1_range[1] = s_ctx->eigen_max_lv1[0];
      loc_eigen_min_lv1_range[0] = s_ctx->eigen_min_lv1[0];
      loc_eigen_min_lv1_range[1] = s_ctx->eigen_min_lv1[0];
      loc_eigen_num_lv1_range[0] = s_ctx->eigen_num_lv1[0];
      loc_eigen_num_lv1_range[1] = s_ctx->eigen_num_lv1[0];
    } else {
      loc_eigen_max_lv1_range[0] = PetscMin(
          loc_eigen_max_lv1_range[0], s_ctx->eigen_max_lv1[coarse_elem_p]);
      loc_eigen_max_lv1_range[1] = PetscMax(
          loc_eigen_max_lv1_range[1], s_ctx->eigen_max_lv1[coarse_elem_p]);
      loc_eigen_min_lv1_range[0] = PetscMin(
          loc_eigen_min_lv1_range[0], s_ctx->eigen_min_lv1[coarse_elem_p]);
      loc_eigen_min_lv1_range[1] = PetscMax(
          loc_eigen_min_lv1_range[1], s_ctx->eigen_min_lv1[coarse_elem_p]);
      loc_eigen_num_lv1_range[0] = PetscMin(
          loc_eigen_num_lv1_range[0], s_ctx->eigen_num_lv1[coarse_elem_p]);
      loc_eigen_num_lv1_range[1] = PetscMax(
          loc_eigen_num_lv1_range[1], s_ctx->eigen_num_lv1[coarse_elem_p]);
    }
  }

  PetscCallMPI(MPI_Reduce(&loc_eigen_max_lv1_range[0], &eigen_max_lv1_range[0],
                          1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&loc_eigen_max_lv1_range[1], &eigen_max_lv1_range[1],
                          1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&loc_eigen_min_lv1_range[0], &eigen_min_lv1_range[0],
                          1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&loc_eigen_min_lv1_range[1], &eigen_min_lv1_range[1],
                          1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&loc_eigen_num_lv1_range[0], &eigen_num_lv1_range[0],
                          1, MPI_INT, MPI_MIN, 0, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Reduce(&loc_eigen_num_lv1_range[1], &eigen_num_lv1_range[1],
                          1, MPI_INT, MPI_MAX, 0, PETSC_COMM_WORLD));

  PetscCall(
      PC_get_range(&s_ctx->eigen_max_lv2, &eigen_max_lv2_range[0], MPI_DOUBLE));
  PetscCall(
      PC_get_range(&s_ctx->eigen_min_lv2, &eigen_min_lv2_range[0], MPI_DOUBLE));
  PetscCall(
      PC_get_range(&s_ctx->eigen_num_lv2, &eigen_num_lv2_range[0], MPI_INT));
  PetscCallMPI(MPI_Reduce(&s_ctx->eigen_num_lv2, &size_mat_lv3, 1, MPI_INT,
                          MPI_SUM, 0, PETSC_COMM_WORLD));

  for (i = 0; i < MAX_LOG_STATES; ++i)
    PetscCall(
        PC_get_range(&s_ctx->t_stages[i], &t_stages_range[i][0], MPI_DOUBLE));

  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "eigen_max_lv1_range=[%.5E, %.5E], \teigen_min_lv1_range=[%.5E, %.5E], "
      "\teigen_num_lv1_range=[%d, %d].\n",
      eigen_max_lv1_range[0], eigen_max_lv1_range[1], eigen_min_lv1_range[0],
      eigen_min_lv1_range[1], eigen_num_lv1_range[0], eigen_num_lv1_range[1]));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "eigen_max_lv2_range=[%.5E, %.5E], \teigen_min_lv2_range=[%.5E, %.5E], "
      "\teigen_num_lv2_range=[%d, %d], \tsize_mat_lv3=%d.\n",
      eigen_max_lv2_range[0], eigen_max_lv2_range[1], eigen_min_lv2_range[0],
      eigen_min_lv2_range[1], eigen_num_lv2_range[0], eigen_num_lv2_range[1],
      size_mat_lv3));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "su_lv1=[%.5f, %.5f], \tsu_lv2_1=[%.5f, %.5f], \tsu_lv2_2=[%.5f, %.5f], "
      "\tsu_lv3_1=[%.5f, %.5f], \tsu_lv3_2=[%.5f, %.5f], \tsu=[%.5f, %.5f].\n",
      t_stages_range[STAGE_SU_LV1][0], t_stages_range[STAGE_SU_LV1][1],
      t_stages_range[STAGE_SU_LV2_1][0], t_stages_range[STAGE_SU_LV2_1][1],
      t_stages_range[STAGE_SU_LV2_2][0], t_stages_range[STAGE_SU_LV2_2][1],
      t_stages_range[STAGE_SU_LV3_1][0], t_stages_range[STAGE_SU_LV3_1][1],
      t_stages_range[STAGE_SU_LV3_2][0], t_stages_range[STAGE_SU_LV3_2][1],
      t_stages_range[STAGE_SU][0], t_stages_range[STAGE_SU][1]));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "av_lv1=[%.5f, %.5f], \tav_lv2=[%.5f, %.5f], \tav_lv3=[%.5f, %.5f], "
      "\tav=[%.5f, %.5f].\n",
      t_stages_range[STAGE_AV_LV1][0], t_stages_range[STAGE_AV_LV1][1],
      t_stages_range[STAGE_AV_LV2][0], t_stages_range[STAGE_AV_LV2][1],
      t_stages_range[STAGE_AV_LV3][0], t_stages_range[STAGE_AV_LV3][1],
      t_stages_range[STAGE_AV][0], t_stages_range[STAGE_AV][1]));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscInt i, coarse_elem_num =
                  s_ctx->sub_domains * s_ctx->sub_domains * s_ctx->sub_domains;
  PetscInt dof_idx[coarse_elem_num + 1];

  _PC_get_dof_idx(s_ctx, &dof_idx[0]);

  for (i = 0; i < dof_idx[coarse_elem_num]; ++i)
    PetscCall(VecDestroy(&s_ctx->ms_bases_c[i]));

  if (!s_ctx->use_2level)
    for (i = 0; i < s_ctx->eigen_num_lv2; ++i)
      PetscCall(VecDestroy(&s_ctx->ms_bases_cc[i]));

  for (i = 0; i < coarse_elem_num; ++i)
    PetscCall(KSPDestroy(&s_ctx->ksp_lv1[i]));

  PetscCall(KSPDestroy(&s_ctx->ksp_lv2));

  if (!s_ctx->use_2level)
    PetscCall(KSPDestroy(&s_ctx->ksp_lv3));

  PetscCall(PC_final_default(s_ctx));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final_default(PCCtx *s_ctx) {
  PetscFunctionBeginUser;

  // PetscCall(DMDestroy(&s_ctx->dm));
  free(s_ctx->ms_bases_c);
  free(s_ctx->ms_bases_cc);
  free(s_ctx->ksp_lv1);
  PetscCall(PetscFree(s_ctx->coarse_startx));
  PetscCall(PetscFree(s_ctx->coarse_starty));
  PetscCall(PetscFree(s_ctx->coarse_startz));
  PetscCall(PetscFree(s_ctx->coarse_lenx));
  PetscCall(PetscFree(s_ctx->coarse_leny));
  PetscCall(PetscFree(s_ctx->coarse_lenz));
  PetscCall(PetscFree(s_ctx->eigen_num_lv1));
  PetscCall(PetscFree(s_ctx->eigen_max_lv1));
  PetscCall(PetscFree(s_ctx->eigen_min_lv1));

  PetscFunctionReturn(0);
}

/***********************************************************************
  Depreciated content.
***********************************************************************/

/*
void _D_PC_get_coarse_elem_p_corners(PCCtx *s_ctx, const PetscInt coarse_elem_p,
PetscInt *startx, PetscInt *nx, PetscInt *starty, PetscInt *ny, PetscInt
*startz, PetscInt *nz) { PetscInt coarse_elem_p_x, coarse_elem_p_y,
coarse_elem_p_z; _PC_get_coarse_elem_xyz(s_ctx, coarse_elem_p, &coarse_elem_p_x,
&coarse_elem_p_y, &coarse_elem_p_z); *startx =
s_ctx->coarse_p_startx[coarse_elem_p_x]; *nx =
s_ctx->coarse_p_lenx[coarse_elem_p_x]; *starty =
s_ctx->coarse_p_starty[coarse_elem_p_y]; *ny =
s_ctx->coarse_p_leny[coarse_elem_p_y]; *startz =
s_ctx->coarse_p_startz[coarse_elem_p_z]; *nz =
s_ctx->coarse_p_lenz[coarse_elem_p_z];
}

PetscErrorCode _D_PC_setup(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;

  PetscCall(PetscTime(&time_tmp));
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, i, j;
  PetscInt startx, starty, startz, nx, ny, nz;
  PetscScalar meas_elem, meas_face_yz, meas_face_zx, meas_face_xy;
  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
&proc_nx, &proc_ny, &proc_nz)); s_ctx->coarse_startx[0] = proc_startx;
  s_ctx->coarse_starty[0] = proc_starty;
  s_ctx->coarse_startz[0] = proc_startz;
  s_ctx->coarse_lenx[0] = (proc_nx / s_ctx->sub_domains) + (proc_nx %
s_ctx->sub_domains > 0); s_ctx->coarse_leny[0] = (proc_ny / s_ctx->sub_domains)
+ (proc_ny % s_ctx->sub_domains > 0); s_ctx->coarse_lenz[0] = (proc_nz /
s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > 0); for (i = 1; i <
s_ctx->sub_domains; ++i) { s_ctx->coarse_startx[i] = s_ctx->coarse_startx[i - 1]
+ s_ctx->coarse_lenx[i - 1]; s_ctx->coarse_lenx[i] = (proc_nx /
s_ctx->sub_domains) + (proc_nx % s_ctx->sub_domains > i);
    s_ctx->coarse_starty[i] = s_ctx->coarse_starty[i - 1] + s_ctx->coarse_leny[i
- 1]; s_ctx->coarse_leny[i] = (proc_ny / s_ctx->sub_domains) + (proc_ny %
s_ctx->sub_domains > i); s_ctx->coarse_startz[i] = s_ctx->coarse_startz[i - 1] +
s_ctx->coarse_lenz[i - 1]; s_ctx->coarse_lenz[i] = (proc_nz /
s_ctx->sub_domains) + (proc_nz % s_ctx->sub_domains > i);
  }
  for (i = 0; i < s_ctx->sub_domains; ++i) {
    s_ctx->coarse_p_startx[i] = s_ctx->coarse_startx[i] - s_ctx->over_sampling
>= 0 ? s_ctx->coarse_startx[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_lenx[i] = s_ctx->coarse_startx[i] + s_ctx->coarse_lenx[i] +
s_ctx->over_sampling <= s_ctx->M ? s_ctx->coarse_startx[i] +
s_ctx->coarse_lenx[i] + s_ctx->over_sampling - s_ctx->coarse_p_startx[i] :
s_ctx->M - s_ctx->coarse_p_startx[i]; s_ctx->coarse_p_starty[i] =
s_ctx->coarse_starty[i] - s_ctx->over_sampling >= 0 ? s_ctx->coarse_starty[i] -
s_ctx->over_sampling : 0; s_ctx->coarse_p_leny[i] = s_ctx->coarse_starty[i] +
s_ctx->coarse_leny[i] + s_ctx->over_sampling <= s_ctx->N ?
s_ctx->coarse_starty[i] + s_ctx->coarse_leny[i] + s_ctx->over_sampling -
s_ctx->coarse_p_starty[i] : s_ctx->N - s_ctx->coarse_p_starty[i];
    s_ctx->coarse_p_startz[i] = s_ctx->coarse_startz[i] - s_ctx->over_sampling
>= 0 ? s_ctx->coarse_startz[i] - s_ctx->over_sampling : 0;
    s_ctx->coarse_p_lenz[i] = s_ctx->coarse_startz[i] + s_ctx->coarse_lenz[i] +
s_ctx->over_sampling <= s_ctx->P ? s_ctx->coarse_startz[i] +
s_ctx->coarse_lenz[i] + s_ctx->over_sampling - s_ctx->coarse_p_startz[i] :
s_ctx->P - s_ctx->coarse_p_startz[i];
  }

  meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  meas_face_xy = s_ctx->H_x * s_ctx->H_y;

  Mat A_i;
  Vec kappa_loc[DIM]; // Destroy later.
  PetscScalar ***arr_kappa_3d[DIM], avg_kappa_e, val_A[2][2];
  PetscInt coarse_elem_p, row[2], col[2], coarse_elem_p_num = s_ctx->sub_domains
* s_ctx->sub_domains * s_ctx->sub_domains;

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
kappa_loc[i])); PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i],
&arr_kappa_3d[i]));
  }

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    _PC_get_coarse_elem_p_corners(s_ctx, coarse_elem_p, &startx, &nx, &starty,
&ny, &startz, &nz); PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF, 1, nx * ny *
nz, nx * ny * nz, 7, NULL, &A_i)); PetscCall(MatSetOption(A_i,
MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));

    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex) {
          // We first handle homogeneous Neumann BCs.
          if (ex >= startx + 1)
          // Inner x-direction edges.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx
- 1; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx; col[0]
= (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1; col[1] = (ez -
startz) * ny * nx + (ey - starty) * nx + ex - startx; avg_kappa_e = 2.0 / (1.0 /
arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
ADD_VALUES)); } else if (ex >= 1)
          // Left boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[0][ez][ey][ex];
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }
          if (ex + 1 == startx + nx && ex + 1 != s_ctx->M)
          // Right boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[0][ez][ey][ex];
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }

          if (ey >= starty + 1)
          // Inner y-direction edges.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] = meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e; val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem
* avg_kappa_e; val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
ADD_VALUES)); } else if (ey >= 1)
          // Down boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[1][ez][ey][ex];
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }
          if (ey + 1 == starty + ny && ey + 1 != s_ctx->N)
          // Up boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[1][ez][ey][ex];
            val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }

          if (ez >= startz + 1)
          // Inner z-direction edges.
          {
            row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] = meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e; val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem
* avg_kappa_e; val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
ADD_VALUES)); } else if (ez >= 1)
          // Back boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[2][ez][ey][ex];
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }
          if (ez + 1 == startz + nz && ez + 1 != s_ctx->P)
          // Front boundary on the coarse element.
          {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 * arr_kappa_3d[2][ez][ey][ex];
            val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
ADD_VALUES));
          }
        }
    PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &(s_ctx->ksp_lv1[coarse_elem_p])));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv1[coarse_elem_p], A_i, A_i));
    PetscCall(KSPSetType(s_ctx->ksp_lv1[coarse_elem_p], KSPPREONLY));
    {
      PC pc_;
      PetscCall(KSPGetPC(s_ctx->ksp_lv1[coarse_elem_p], &pc_));
      PetscCall(PCSetType(pc_, PCCHOLESKY));
      PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
      PetscCall(PCSetOptionsPrefix(pc_, "kspl1_"));
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv1[coarse_elem_p],
PETSC_TRUE)); PetscCall(PCSetFromOptions(pc_));
    }
    PetscCall(KSPSetUp(s_ctx->ksp_lv1[coarse_elem_p]));
    PetscCall(MatDestroy(&A_i));
  }
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV1] -= time_tmp;

  PetscCall(PetscTime(&time_tmp));
  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscScalar ***arr_ms_bases_loc, ***arr_M_i_3d;
  PetscInt max_eigen_num;
  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMCreateLocalVector(s_ctx->dm, &(s_ctx->ms_bases[i])));

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    _PC_get_coarse_elem_corners(s_ctx, coarse_elem_p, &startx, &nx, &starty,
&ny, &startz, &nz);
    // 7 point stencil.
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nx * ny * nz, &diag_M_i));
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz, 7,
NULL, &A_i_inner)); PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx
* ny * nz, 1, NULL, &M_i)); PetscCall(VecGetArray3d(diag_M_i, nz, ny, nx, 0, 0,
0, &arr_M_i_3d)); for (ez = startz; ez < startz + nz; ++ez) for (ey = starty; ey
< starty + ny; ++ey) { for (ex = startx; ex < startx + nx; ++ex) { arr_M_i_3d[ez
- startz][ey - starty][ex - startx] = 0.0; if (ex >= startx + 1) { row[0] = (ez
- startz) * ny * nx + (ey - starty) * nx + ex - startx - 1; row[1] = (ez -
startz) * ny * nx + (ey - starty) * nx + ex - startx; col[0] = (ez - startz) *
ny * nx + (ey - starty) * nx + ex - startx - 1; col[1] = (ez - startz) * ny * nx
+ (ey - starty) * nx + ex - startx; avg_kappa_e = 2.0 / (1.0 /
arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
            val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
            val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x; } else if (ex >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][ex]); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }
          if (ex + 1 != s_ctx->M) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); arr_M_i_3d[ez - startz][ey - starty][ex -
startx] += 2.0 * avg_kappa_e / s_ctx->H_x / s_ctx->H_x;
          }

          if (ey >= starty + 1) {
            row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] = meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e; val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem
* avg_kappa_e; val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y; } else if (ey >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }
          if (ey + 1 != s_ctx->N) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); arr_M_i_3d[ez - startz][ey - starty][ex -
startx] += 2.0 * avg_kappa_e / s_ctx->H_y / s_ctx->H_y;
          }

          if (ez >= startz + 1) {
            row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] = meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e; val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem
* avg_kappa_e; val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e; val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z; } else if (ez >= 1) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); arr_M_i_3d[ez - startz][ey - starty][ex - startx]
+= 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }
          if (ez + 1 != s_ctx->P) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); arr_M_i_3d[ez - startz][ey - starty][ex -
startx] += 2.0 * avg_kappa_e / s_ctx->H_z / s_ctx->H_z;
          }
        }
      }
    PetscCall(VecRestoreArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
    PetscCall(VecScale(diag_M_i, meas_elem * 0.25 * M_PI * M_PI / (nx * nx *
s_ctx->H_x * s_ctx->H_x + ny * ny * s_ctx->H_y * s_ctx->H_y + nz * nz *
s_ctx->H_z * s_ctx->H_z))); PetscCall(MatAssemblyBegin(A_i_inner,
MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));
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
PETSC_DEFAULT)); PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL)); ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= s_ctx->max_eigen_num_lv1, PETSC_COMM_WORLD,
PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d,
eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv1); PetscBool find_max_eigen =
PETSC_FALSE; for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem_p] = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv1) {
        s_ctx->eigen_num_lv1[coarse_elem_p] = j + 1;
        s_ctx->eigen_max_lv1[coarse_elem_p] = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (!find_max_eigen && j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_num_lv1[coarse_elem_p] = s_ctx->max_eigen_num_lv1;
        s_ctx->eigen_max_lv1[coarse_elem_p] = eig_val;
      }

      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases[j],
&arr_ms_bases_loc)); PetscCall(VecGetArray3d(eig_vec, nz, ny, nx, 0, 0, 0,
&arr_eig_vec)); for (ez = startz; ez < startz + nz; ++ez) for (ey = starty; ey <
starty + ny; ++ey) PetscCall(PetscArraycpy(&arr_ms_bases_loc[ez][ey][startx],
&arr_eig_vec[ez - startz][ey - starty][0], nx));
      PetscCall(VecRestoreArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases[j],
&arr_ms_bases_loc));
    }
    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }
  // We need to check the real maximum number of eigenvectors used, free what
those are not used. max_eigen_num = 0; for (i = 0; i < coarse_elem_p_num; ++i)
    max_eigen_num = PetscMax(max_eigen_num, s_ctx->eigen_num_lv1[i]);
  PetscCallMPI(MPI_Allreduce(&max_eigen_num, &s_ctx->max_eigen_num_lv1_upd, 1,
MPI_INT, MPI_MAX, PETSC_COMM_WORLD)); for (i = s_ctx->max_eigen_num_lv1_upd; i <
s_ctx->max_eigen_num_lv1; ++i) PetscCall(VecDestroy(&s_ctx->ms_bases[i]));

  Vec dummy_ms_bases_glo;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < s_ctx->max_eigen_num_lv1_upd; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, s_ctx->ms_bases[i], INSERT_VALUES,
dummy_ms_bases_glo)); PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo,
INSERT_VALUES, s_ctx->ms_bases[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));

  PetscInt dof_idx[coarse_elem_p_num + 1], coarse_elem_p_col, coarse_elem_x,
coarse_elem_y, coarse_elem_z; PetscScalar
***arr_ms_bases_array[s_ctx->max_eigen_num_lv1_upd]; if (s_ctx->sub_domains >=
2) { for (i = 0; i < s_ctx->max_eigen_num_lv1_upd; ++i)
      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases[i],
&arr_ms_bases_array[i])); _PC_get_dof_idx(s_ctx, &dof_idx[0]);
    PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF, 1, dof_idx[coarse_elem_p_num],
dof_idx[coarse_elem_p_num], 7 * s_ctx->max_eigen_num_lv1_upd, NULL, &A_i));
    PetscCall(MatSetOption(A_i, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p)
{ _PC_get_coarse_elem_xyz(s_ctx, coarse_elem_p, &coarse_elem_x, &coarse_elem_y,
&coarse_elem_z); _PC_get_coarse_elem_corners(s_ctx, coarse_elem_p, &startx, &nx,
&starty, &ny, &startz, &nz); for (i = 0; i <
s_ctx->eigen_num_lv1[coarse_elem_p]; ++i) { row[0] = dof_idx[coarse_elem_p] + i;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p]; ++j) {
          col[0] = dof_idx[coarse_elem_p] + j;
          val_A[0][0] = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                if (ex >= startx + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1]
+ 1.0 / arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] += meas_face_yz * meas_face_yz
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey][ex - 1] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey][ex - 1] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ex >= 1) { avg_kappa_e = 2.0 /
(1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                  val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ex + 1 == startx + nx && ex + 1 != s_ctx->M) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }

                if (ey >= starty + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex]
+ 1.0 / arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] += meas_face_zx * meas_face_zx
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey - 1][ex] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey - 1][ex] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ey >= 1) { avg_kappa_e = 2.0 /
(1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                  val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ey + 1 == starty + ny && ey + 1 != s_ctx->N) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }

                if (ez >= startz + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex]
+ 1.0 / arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez - 1][ey][ex] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez - 1][ey][ex] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ez >= 1) { avg_kappa_e = 2.0 /
(1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                  val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ez + 1 == startz + nz && ez + 1 != s_ctx->P) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
              }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }

        if (coarse_elem_x != 0) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1; for
(j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ez = startz; ez < startz
+ nz; ++ez) for (ey = starty; ey < starty + ny; ++ey) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[0][ez][ey][startx - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx]);
                val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx] *
arr_ms_bases_array[j][ez][ey][startx - 1];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }

        if (coarse_elem_x != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x + 1; for
(j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ez = startz; ez < startz
+ nz; ++ez) for (ey = starty; ey < starty + ny; ++ey) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[0][ez][ey][startx + nx - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][startx + nx]); val_A[0][0] -= meas_face_yz *
meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx +
nx - 1] * arr_ms_bases_array[j][ez][ey][startx + nx];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }

        if (coarse_elem_y != 0) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
          for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
            col[0] = dof_idx[coarse_elem_p_col] + j;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty - 1][ex]
+ 1.0 / arr_kappa_3d[1][ez][starty][ex]); val_A[0][0] -= meas_face_zx *
meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty][ex] *
arr_ms_bases_array[j][ez][starty - 1][ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }

        if (coarse_elem_y != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + (coarse_elem_y + 1) * s_ctx->sub_domains + coarse_elem_x;
          for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
            col[0] = dof_idx[coarse_elem_p_col] + j;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty + ny -
1][ex] + 1.0 / arr_kappa_3d[1][ez][starty + ny][ex]); val_A[0][0] -=
meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e *
arr_ms_bases_array[i][ez][starty + ny - 1][ex] *
arr_ms_bases_array[j][ez][starty + ny][ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }

        if (coarse_elem_z != 0) {
          coarse_elem_p_col = (coarse_elem_z - 1) * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x; for (j
= 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ey = starty; ey < starty
+ ny; ++ey) for (ex = startx; ex < startx + nx; ++ex) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[2][startz - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz][ey][ex]);
                val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][startz][ey][ex] *
arr_ms_bases_array[j][startz - 1][ey][ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }

        if (coarse_elem_z != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = (coarse_elem_z + 1) * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x; for (j
= 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ey = starty; ey < starty
+ ny; ++ey) for (ex = startx; ex < startx + nx; ++ex) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[2][startz + nz - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz +
nz][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][startz + nz - 1][ey][ex] *
arr_ms_bases_array[j][startz + nz][ey][ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i, MAT_SPD, PETSC_TRUE));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &s_ctx->ksp_lv2));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv2, A_i, A_i));
    PetscCall(KSPSetType(s_ctx->ksp_lv2, KSPPREONLY));
    {
      PC pc_;
      PetscCall(KSPGetPC(s_ctx->ksp_lv2, &pc_));
      PetscCall(PCSetType(pc_, PCCHOLESKY));
      PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERCHOLMOD));
      PetscCall(PCSetOptionsPrefix(pc_, "kspl2_"));
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv2, PETSC_TRUE));
      PetscCall(PCSetFromOptions(pc_));
    }
    PetscCall(KSPSetUp(s_ctx->ksp_lv2));
    PetscCall(MatDestroy(&A_i));
  }
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2] -= time_tmp;

  PetscCall(PetscTime(&time_tmp));
  // PetscScalar ***arr_ms_bases_tmp;
  if (s_ctx->sub_domains >= 2) {
    PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, dof_idx[coarse_elem_p_num],
dof_idx[coarse_elem_p_num], 7 * s_ctx->max_eigen_num_lv1_upd, NULL,
&A_i_inner)); for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num;
++coarse_elem_p) { _PC_get_coarse_elem_xyz(s_ctx, coarse_elem_p, &coarse_elem_x,
&coarse_elem_y, &coarse_elem_z); _PC_get_coarse_elem_corners(s_ctx,
coarse_elem_p, &startx, &nx, &starty, &ny, &startz, &nz); for (i = 0; i <
s_ctx->eigen_num_lv1[coarse_elem_p]; ++i) { row[0] = dof_idx[coarse_elem_p] + i;
        for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p]; ++j) {
          col[0] = dof_idx[coarse_elem_p] + j;
          val_A[0][0] = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                if (ex >= startx + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1]
+ 1.0 / arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] += meas_face_yz * meas_face_yz
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey][ex - 1] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey][ex - 1] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ex != proc_startx) { avg_kappa_e
= 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ex + 1 == startx + nx && ex + 1 != proc_startx + proc_nx) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }

                if (ey >= starty + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex]
+ 1.0 / arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] += meas_face_zx * meas_face_zx
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez][ey - 1][ex] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez][ey - 1][ex] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ey != proc_starty) { avg_kappa_e
= 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ey + 1 == starty + ny && ey + 1 != proc_starty + proc_ny) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }

                if (ez >= startz + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex]
+ 1.0 / arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy
/ meas_elem * avg_kappa_e * (arr_ms_bases_array[i][ez - 1][ey][ex] -
arr_ms_bases_array[i][ez][ey][ex]) * (arr_ms_bases_array[j][ez - 1][ey][ex] -
arr_ms_bases_array[j][ez][ey][ex]); } else if (ez != proc_startz) { avg_kappa_e
= 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
                if (ez + 1 == startz + nz && ez + 1 != proc_startz + proc_nz) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][ex] *
arr_ms_bases_array[j][ez][ey][ex];
                }
              }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
        }

        if (coarse_elem_x != 0) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x - 1; for
(j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ez = startz; ez < startz
+ nz; ++ez) for (ey = starty; ey < starty + ny; ++ey) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[0][ez][ey][startx - 1] + 1.0 / arr_kappa_3d[0][ez][ey][startx]);
                val_A[0][0] -= meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx] *
arr_ms_bases_array[j][ez][ey][startx - 1];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_x != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x + 1; for
(j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ez = startz; ez < startz
+ nz; ++ez) for (ey = starty; ey < starty + ny; ++ey) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[0][ez][ey][startx + nx - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][startx + nx]); val_A[0][0] -= meas_face_yz *
meas_face_yz / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][ey][startx +
nx - 1] * arr_ms_bases_array[j][ez][ey][startx + nx];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_y != 0) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + (coarse_elem_y - 1) * s_ctx->sub_domains + coarse_elem_x;
          for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
            col[0] = dof_idx[coarse_elem_p_col] + j;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty - 1][ex]
+ 1.0 / arr_kappa_3d[1][ez][starty][ex]); val_A[0][0] -= meas_face_zx *
meas_face_zx / meas_elem * avg_kappa_e * arr_ms_bases_array[i][ez][starty][ex] *
arr_ms_bases_array[j][ez][starty - 1][ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_y != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = coarse_elem_z * s_ctx->sub_domains *
s_ctx->sub_domains + (coarse_elem_y + 1) * s_ctx->sub_domains + coarse_elem_x;
          for (j = 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) {
            col[0] = dof_idx[coarse_elem_p_col] + j;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][starty + ny -
1][ex] + 1.0 / arr_kappa_3d[1][ez][starty + ny][ex]); val_A[0][0] -=
meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e *
arr_ms_bases_array[i][ez][starty + ny - 1][ex] *
arr_ms_bases_array[j][ez][starty + ny][ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_z != 0) {
          coarse_elem_p_col = (coarse_elem_z - 1) * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x; for (j
= 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ey = starty; ey < starty
+ ny; ++ey) for (ex = startx; ex < startx + nx; ++ex) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[2][startz - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz][ey][ex]);
                val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][startz][ey][ex] *
arr_ms_bases_array[j][startz - 1][ey][ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_z != s_ctx->sub_domains - 1) {
          coarse_elem_p_col = (coarse_elem_z + 1) * s_ctx->sub_domains *
s_ctx->sub_domains + coarse_elem_y * s_ctx->sub_domains + coarse_elem_x; for (j
= 0; j < s_ctx->eigen_num_lv1[coarse_elem_p_col]; ++j) { col[0] =
dof_idx[coarse_elem_p_col] + j; val_A[0][0] = 0.0; for (ey = starty; ey < starty
+ ny; ++ey) for (ex = startx; ex < startx + nx; ++ex) { avg_kappa_e = 2.0 / (1.0
/ arr_kappa_3d[2][startz + nz - 1][ey][ex] + 1.0 / arr_kappa_3d[2][startz +
nz][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_array[i][startz + nz - 1][ey][ex] *
arr_ms_bases_array[j][startz + nz][ey][ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
&val_A[0][0], INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));

    for (i = 0; i < s_ctx->max_eigen_num_lv2; ++i)
      PetscCall(DMCreateLocalVector(s_ctx->dm, &s_ctx->ms_bases_c[i]));

    EPS eps;
    PetscInt nconv;
    PetscScalar eig_val, *arr_eig_vec, ***arr_ms_bases_c;
    Vec eig_vec;

    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, A_i_inner, NULL));
    PetscCall(EPSSetProblemType(eps, EPS_HEP));
    PetscCall(EPSSetDimensions(eps, s_ctx->max_eigen_num_lv2, PETSC_DEFAULT,
PETSC_DEFAULT)); ST st; PetscCall(EPSGetST(eps, &st)); PetscCall(STSetType(st,
STSINVERT)); PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl2_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= s_ctx->max_eigen_num_lv2, PETSC_COMM_WORLD,
PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors for level-2! (nconv=%d,
eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv2);
    PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));
    PetscBool find_max_eigen = PETSC_FALSE;
    for (j = 0; j < s_ctx->max_eigen_num_lv2; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (!find_max_eigen && j == 0)
        s_ctx->eigen_min_lv2 = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv2) {
        s_ctx->eigen_num_lv2 = j + 1;
        s_ctx->eigen_max_lv2 = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (j == s_ctx->max_eigen_num_lv2 - 1) {
        s_ctx->eigen_num_lv2 = s_ctx->max_eigen_num_lv2;
        s_ctx->eigen_max_lv2 = eig_val;
      }
      PetscCall(VecGetArray(eig_vec, &arr_eig_vec));
      // Construct coarse-coarse bases in the original P space.
      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases_c[j],
&arr_ms_bases_c));

      for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num;
++coarse_elem_p) { _PC_get_coarse_elem_corners(s_ctx, coarse_elem_p, &startx,
&nx, &starty, &ny, &startz, &nz);
        // Insert data into ms_bases_c.
        // PetscCall(VecZeroEntries(ms_bases_tmp));
        // PetscCall(VecMAXPY(ms_bases_tmp, s_ctx->eigen_num_lv1[coarse_elem_p],
&arr_eig_vec[dof_idx[coarse_elem_p]], &s_ctx->ms_bases[0]));
        // PetscCall(DMDAVecGetArray(s_ctx->dm, ms_bases_tmp,
&arr_ms_bases_tmp)); for (ez = startz; ez < startz + nz; ++ez) for (ey = starty;
ey < starty + ny; ++ey) for (ex = startx; ex < startx + nx; ++ex) for (i =
dof_idx[coarse_elem_p]; i < dof_idx[coarse_elem_p + 1]; ++i)
                arr_ms_bases_c[ez][ey][ex] += arr_eig_vec[i] *
arr_ms_bases_array[i - dof_idx[coarse_elem_p]][ez][ey][ex];
        // PetscCall(PetscArraycpy(&arr_ms_bases_c[ez][ey][startx],
&arr_ms_bases_tmp[ez][ey][startx], nx));
        // PetscCall(DMDAVecRestoreArray(s_ctx->dm, ms_bases_tmp,
&arr_ms_bases_tmp));
      }
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases_c[j],
&arr_ms_bases_c)); PetscCall(VecRestoreArray(eig_vec, &arr_eig_vec));
    }

    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    // PetscCall(DMRestoreLocalVector(s_ctx->dm, &ms_bases_tmp));

    for (i = 0; i < s_ctx->max_eigen_num_lv1_upd; ++i)
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases[i],
&arr_ms_bases_array[i]));

    PetscCallMPI(MPI_Allreduce(&s_ctx->eigen_num_lv2,
&s_ctx->max_eigen_num_lv2_upd, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD)); for (i =
s_ctx->max_eigen_num_lv2_upd; i < s_ctx->max_eigen_num_lv2; ++i)
      PetscCall(VecDestroy(&s_ctx->ms_bases_c[i]));

    PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
    for (i = 0; i < s_ctx->max_eigen_num_lv2_upd; ++i) {
      PetscCall(DMLocalToGlobal(s_ctx->dm, s_ctx->ms_bases_c[i], INSERT_VALUES,
dummy_ms_bases_glo)); PetscCall(DMGlobalToLocal(s_ctx->dm, dummy_ms_bases_glo,
INSERT_VALUES, s_ctx->ms_bases_c[i]));
    }
    PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  }

  Mat A_cc;
  PetscInt A_cc_range[NEIGH + 1][2];
  const PetscMPIInt *ng_ranks;
  PetscScalar ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv2_upd];
  if (s_ctx->sub_domains >= 2) {
    for (i = 0; i < s_ctx->max_eigen_num_lv2_upd; ++i)
      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases_c[i],
&arr_ms_bases_c_array[i]));
    // Create a SBAIJ matrix for MUMPS' Cholesky solver.
    PetscCall(MatCreateSBAIJ(PETSC_COMM_WORLD, 1, s_ctx->eigen_num_lv2,
s_ctx->eigen_num_lv2, PETSC_DETERMINE, PETSC_DETERMINE, s_ctx->eigen_num_lv2,
NULL, 6 * s_ctx->max_eigen_num_lv2_upd, NULL, &A_cc));
    PetscCall(MatSetOption(A_cc, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
    PetscCall(DMDAGetNeighbors(s_ctx->dm, &ng_ranks));
    // Send dof_idx first.
    if (proc_startx != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[12],
ng_ranks[13], PETSC_COMM_WORLD)); // Left. if (proc_startx + proc_nx !=
s_ctx->M) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[14],
ng_ranks[13], PETSC_COMM_WORLD)); // Right. if (proc_starty != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[10],
ng_ranks[13], PETSC_COMM_WORLD)); // Down. if (proc_starty + proc_ny !=
s_ctx->N) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[16],
ng_ranks[13], PETSC_COMM_WORLD)); // Up. if (proc_startz != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[4],
ng_ranks[13], PETSC_COMM_WORLD)); // Back. if (proc_startz + proc_nz !=
s_ctx->P) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[26],
ng_ranks[13], PETSC_COMM_WORLD)); // Front.
    // Receive dof_idx then.
    MPI_Status ista;
    if (proc_startx != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[1][0], 2, MPI_INT, ng_ranks[12],
ng_ranks[12], PETSC_COMM_WORLD, &ista)); // Left. if (proc_startx + proc_nx !=
s_ctx->M) PetscCallMPI(MPI_Recv(&A_cc_range[2][0], 2, MPI_INT, ng_ranks[14],
ng_ranks[14], PETSC_COMM_WORLD, &ista)); // Right. if (proc_starty != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[3][0], 2, MPI_INT, ng_ranks[10],
ng_ranks[10], PETSC_COMM_WORLD, &ista)); // Down. if (proc_starty + proc_ny !=
s_ctx->N) PetscCallMPI(MPI_Recv(&A_cc_range[4][0], 2, MPI_INT, ng_ranks[16],
ng_ranks[16], PETSC_COMM_WORLD, &ista)); // Up. if (proc_startz != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[5][0], 2, MPI_INT, ng_ranks[4],
ng_ranks[4], PETSC_COMM_WORLD, &ista)); // Back. if (proc_startz + proc_nz !=
s_ctx->P) PetscCallMPI(MPI_Recv(&A_cc_range[6][0], 2, MPI_INT, ng_ranks[22],
ng_ranks[22], PETSC_COMM_WORLD, &ista)); // Front. for (i = A_cc_range[0][0]; i
< A_cc_range[0][1]; ++i) { row[0] = i; for (j = A_cc_range[0][0]; j <
A_cc_range[0][1]; ++j) { col[0] = j; val_A[0][0] = 0.0; for (ez = proc_startz;
ez < proc_startz + proc_nz; ++ez) for (ey = proc_starty; ey < proc_starty +
proc_ny; ++ey) for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) { if (ex
>= proc_startx + 1) { avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1]
+ 1.0 / arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] += meas_face_yz * meas_face_yz
/ meas_elem * avg_kappa_e * (arr_ms_bases_c_array[i][ez][ey][ex - 1] -
arr_ms_bases_c_array[i][ez][ey][ex]) * (arr_ms_bases_c_array[j][ez][ey][ex - 1]
- arr_ms_bases_c_array[j][ez][ey][ex]); } else if (ex != 0) { avg_kappa_e = 2.0
/ (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 / arr_kappa_3d[0][ez][ey][ex]);
                val_A[0][0] += meas_face_yz * meas_face_yz / meas_elem *
avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ex + 1 == proc_startx + proc_nx && ex + 1 != s_ctx->M) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }

              if (ey >= proc_starty + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0
/ arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * (arr_ms_bases_c_array[i][ez][ey - 1][ex] -
arr_ms_bases_c_array[i][ez][ey][ex]) * (arr_ms_bases_c_array[j][ez][ey - 1][ex]
- arr_ms_bases_c_array[j][ez][ey][ex]); } else if (ey != 0) { avg_kappa_e = 2.0
/ (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 / arr_kappa_3d[1][ez][ey][ex]);
                val_A[0][0] += meas_face_zx * meas_face_zx / meas_elem *
avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ey + 1 == proc_starty + proc_ny && ey + 1 != s_ctx->N) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }

              if (ez >= proc_startz + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0
/ arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * (arr_ms_bases_c_array[i][ez - 1][ey][ex] -
arr_ms_bases_c_array[i][ez][ey][ex]) * (arr_ms_bases_c_array[j][ez - 1][ey][ex]
- arr_ms_bases_c_array[j][ez][ey][ex]); } else if (ez != 0) { avg_kappa_e = 2.0
/ (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 / arr_kappa_3d[2][ez][ey][ex]);
                val_A[0][0] += meas_face_xy * meas_face_xy / meas_elem *
avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }
              if (ez + 1 == proc_startz + proc_nz && ez + 1 != s_ctx->P) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i][ez][ey][ex] *
arr_ms_bases_c_array[j][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
      }

      if (proc_startx != 0)
        for (j = A_cc_range[1][0]; j < A_cc_range[1][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] -= meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[1][0]][ez][ey][ex - 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startx + proc_nx != s_ctx->M)
        for (j = A_cc_range[2][0]; j < A_cc_range[2][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx + proc_nx - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] -= meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[2][0]][ez][ey][ex + 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_starty != 0)
        for (j = A_cc_range[3][0]; j < A_cc_range[3][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] -= meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[3][0]][ez][ey - 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_starty + proc_ny != s_ctx->N)
        for (j = A_cc_range[4][0]; j < A_cc_range[4][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty + proc_ny - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] -= meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[4][0]][ez][ey + 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startz != 0)
        for (j = A_cc_range[5][0]; j < A_cc_range[5][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[5][0]][ez - 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startz + proc_nz != s_ctx->P)
        for (j = A_cc_range[6][0]; j < A_cc_range[6][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz + proc_nz - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_c_array[i - A_cc_range[0][0]][ez][ey][ex]
* arr_ms_bases_c_array[j - A_cc_range[6][0]][ez + 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
    }
    for (i = 0; i < s_ctx->max_eigen_num_lv2_upd; ++i)
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases_c[i],
&arr_ms_bases_c_array[i])); PetscCall(MatAssemblyBegin(A_cc,
MAT_FINAL_ASSEMBLY)); PetscCall(MatAssemblyEnd(A_cc, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(A_cc, NIL));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &s_ctx->ksp_lv3));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv3, A_cc, A_cc));
    PetscCall(KSPSetType(s_ctx->ksp_lv3, KSPPREONLY));
    {
      PC pc_;
      PetscCall(KSPGetPC(s_ctx->ksp_lv3, &pc_));
      PetscCall(PCSetType(pc_, PCCHOLESKY));
      PetscCall(PCFactorSetShiftType(pc_, MAT_SHIFT_NONZERO));
      PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMUMPS));
      PetscCall(PCSetOptionsPrefix(pc_, "kspl3_"));
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv3, PETSC_TRUE));
      PetscCall(PCSetFromOptions(pc_));
    }
    PetscCall(MatView(A_cc, 0));
    PetscCall(KSPSetUp(s_ctx->ksp_lv3));
    PetscCall(MatDestroy(&A_cc));
  }

  // If there is a single subdomain per process, will skip the level-2 routine.
  if (s_ctx->sub_domains == 1) {
    Mat A_cc;
    PetscInt A_cc_range[NEIGH + 1][2];
    const PetscMPIInt *ng_ranks;
    PetscScalar ***arr_ms_bases_array[s_ctx->max_eigen_num_lv1_upd];
    for (i = 0; i < s_ctx->max_eigen_num_lv1_upd; ++i)
      PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->ms_bases[i],
&arr_ms_bases_array[i])); PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,
s_ctx->eigen_num_lv1[0], s_ctx->eigen_num_lv1[0], PETSC_DETERMINE,
PETSC_DETERMINE, s_ctx->eigen_num_lv1[0], NULL, 6 *
s_ctx->max_eigen_num_lv1_upd, NULL, &A_cc));
    PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
    PetscCall(DMDAGetNeighbors(s_ctx->dm, &ng_ranks));
    // Send dof_idx first.
    if (proc_startx != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[12],
ng_ranks[13], PETSC_COMM_WORLD)); // Left. if (proc_startx + proc_nx !=
s_ctx->M) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[14],
ng_ranks[13], PETSC_COMM_WORLD)); // Right. if (proc_starty != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[10],
ng_ranks[13], PETSC_COMM_WORLD)); // Down. if (proc_starty + proc_ny !=
s_ctx->N) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[16],
ng_ranks[13], PETSC_COMM_WORLD)); // Up. if (proc_startz != 0)
      PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[4],
ng_ranks[13], PETSC_COMM_WORLD)); // Back. if (proc_startz + proc_nz !=
s_ctx->P) PetscCallMPI(MPI_Send(&A_cc_range[0][0], 2, MPI_INT, ng_ranks[26],
ng_ranks[13], PETSC_COMM_WORLD)); // Front.
    // Receive dof_idx then.
    MPI_Status ista;
    if (proc_startx != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[1][0], 2, MPI_INT, ng_ranks[12],
ng_ranks[12], PETSC_COMM_WORLD, &ista)); // Left. if (proc_startx + proc_nx !=
s_ctx->M) PetscCallMPI(MPI_Recv(&A_cc_range[2][0], 2, MPI_INT, ng_ranks[14],
ng_ranks[14], PETSC_COMM_WORLD, &ista)); // Right. if (proc_starty != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[3][0], 2, MPI_INT, ng_ranks[10],
ng_ranks[10], PETSC_COMM_WORLD, &ista)); // Down. if (proc_starty + proc_ny !=
s_ctx->N) PetscCallMPI(MPI_Recv(&A_cc_range[4][0], 2, MPI_INT, ng_ranks[16],
ng_ranks[16], PETSC_COMM_WORLD, &ista)); // Up. if (proc_startz != 0)
      PetscCallMPI(MPI_Recv(&A_cc_range[5][0], 2, MPI_INT, ng_ranks[4],
ng_ranks[4], PETSC_COMM_WORLD, &ista)); // Back. if (proc_startz + proc_nz !=
s_ctx->P) PetscCallMPI(MPI_Recv(&A_cc_range[6][0], 2, MPI_INT, ng_ranks[22],
ng_ranks[22], PETSC_COMM_WORLD, &ista)); // Front. for (i = A_cc_range[0][0]; i
< A_cc_range[0][1]; ++i) { row[0] = i; for (j = A_cc_range[0][0]; j <
A_cc_range[0][1]; ++j) { col[0] = j; val_A[0][0] = 0.0; if (proc_startx != 0)
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        if (proc_startx + proc_nx != s_ctx->M)
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx + proc_nx - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] += meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        if (proc_starty != 0)
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        if (proc_starty + proc_ny != s_ctx->N)
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty + proc_ny - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] += meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        if (proc_startz != 0)
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        if (proc_startz + proc_nz != s_ctx->P)
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz + proc_nz - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] += meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[0][0]][ez][ey][ex];
            }
        PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
      }

      if (proc_startx != 0)
        for (j = A_cc_range[1][0]; j < A_cc_range[1][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] -= meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[1][0]][ez][ey][ex - 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startx + proc_nx != s_ctx->M)
        for (j = A_cc_range[2][0]; j < A_cc_range[2][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx + proc_nx - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex] + 1.0 /
arr_kappa_3d[0][ez][ey][ex + 1]); val_A[0][0] -= meas_face_yz * meas_face_yz /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[2][0]][ez][ey][ex + 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_starty != 0)
        for (j = A_cc_range[3][0]; j < A_cc_range[3][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] + 1.0 /
arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] -= meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[3][0]][ez][ey - 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_starty + proc_ny != s_ctx->N)
        for (j = A_cc_range[4][0]; j < A_cc_range[4][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty + proc_ny - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey][ex] + 1.0 /
arr_kappa_3d[1][ez][ey + 1][ex]); val_A[0][0] -= meas_face_zx * meas_face_zx /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[4][0]][ez][ey + 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startz != 0)
        for (j = A_cc_range[5][0]; j < A_cc_range[5][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] + 1.0 /
arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[5][0]][ez - 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
      if (proc_startz + proc_nz != s_ctx->P)
        for (j = A_cc_range[6][0]; j < A_cc_range[6][1]; ++j) {
          col[0] = j;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz + proc_nz - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez][ey][ex] + 1.0 /
arr_kappa_3d[2][ez + 1][ey][ex]); val_A[0][0] -= meas_face_xy * meas_face_xy /
meas_elem * avg_kappa_e * arr_ms_bases_array[i - A_cc_range[0][0]][ez][ey][ex] *
arr_ms_bases_array[j - A_cc_range[6][0]][ez + 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
INSERT_VALUES));
        }
    }
    for (i = 0; i < s_ctx->max_eigen_num_lv1_upd; ++i)
      PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->ms_bases[i],
&arr_ms_bases_array[i])); PetscCall(MatAssemblyBegin(A_cc, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_cc, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPCreate(PETSC_COMM_SELF, &s_ctx->ksp_lv3));
    PetscCall(KSPSetOperators(s_ctx->ksp_lv3, A_cc, A_cc));
    {
      PC pc_;
      PetscCall(KSPSetType(s_ctx->ksp_lv3, KSPPREONLY));
      PetscCall(KSPGetPC(s_ctx->ksp_lv3, &pc_));
      PetscCall(PCSetType(pc_, PCLU));
      PetscCall(PCFactorSetShiftType(pc_, MAT_SHIFT_NONZERO));
      PetscCall(PCFactorSetMatSolverType(pc_, MATSOLVERMUMPS));
      PetscCall(PCSetOptionsPrefix(pc_, "kspl3_"));
      PetscCall(KSPSetErrorIfNotConverged(s_ctx->ksp_lv3, PETSC_TRUE));
      PetscCall(PCSetFromOptions(pc_));
    }
    PetscCall(KSPSetUp(s_ctx->ksp_lv3));
    PetscCall(MatDestroy(&A_cc));
  }

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i],
&arr_kappa_3d[i])); PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
  }
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV3] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y) {
  // Input x, return y.
  PetscFunctionBeginUser;
  PetscLogDouble time_tmp;

  PetscCall(PetscTime(&time_tmp));
  PCCtx *s_ctx;
  PetscCall(PCShellGetContext(pc, &s_ctx));
  Mat A;
  PetscCall(PCGetOperators(pc, &A, NULL));

  PetscCall(VecZeroEntries(y));
  PetscCall(_PC_apply_vec_lv3(s_ctx, x, y));
  PetscCall(_PC_apply_vec_lv2(s_ctx, x, y));
  Vec temp_1;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &temp_1));
  PetscCall(MatMult(A, y, temp_1));
  PetscCall(VecAYPX(temp_1, -1, x));
  Vec temp_2;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &temp_2));
  PetscCall(VecZeroEntries(temp_2));
  PetscCall(_PC_apply_vec_lv1(s_ctx, temp_1, temp_2));
  PetscCall(VecAXPY(y, 1.0, temp_2));
  PetscCall(MatMult(A, temp_2, temp_1));
  PetscCall(VecZeroEntries(temp_2));
  PetscCall(_PC_apply_vec_lv3(s_ctx, temp_1, temp_2));
  PetscCall(_PC_apply_vec_lv2(s_ctx, temp_1, temp_2));
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &temp_1));
  PetscCall(VecAXPY(y, -1.0, temp_2));
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &temp_2));
  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_AV] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv2_eigen_mean(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscScalar ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv1], ***arr_M_i_3d,
val_A[2][2], avg_kappa_e; PetscInt max_eigen_num, i, j, coarse_elem, startx, nx,
ex, starty, ny, ey, startz, nz, ez, row[2], col[2];

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
NULL, &A_i_inner)); PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx
* ny * nz, 1, NULL, &M_i)); PetscCall(VecGetArray3d(diag_M_i, nz, ny, nx, 0, 0,
0, &arr_M_i_3d)); for (ez = startz; ez < startz + nz; ++ez) for (ey = starty; ey
< starty + ny; ++ey) { for (ex = startx; ex < startx + nx; ++ex) {

          if (ex >= startx + 1) {
            row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx
- 1; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx; col[0]
= (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx - 1; col[1] = (ez -
startz) * ny * nx + (ey - starty) * nx + ex - startx; avg_kappa_e = 2.0 / (1.0 /
int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
int_ctx->arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] = int_ctx->meas_face_yz *
int_ctx->meas_face_yz / int_ctx->meas_elem * avg_kappa_e; val_A[0][1] =
-int_ctx->meas_face_yz * int_ctx->meas_face_yz / int_ctx->meas_elem *
avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_yz * int_ctx->meas_face_yz /
int_ctx->meas_elem * avg_kappa_e; val_A[1][1] = int_ctx->meas_face_yz *
int_ctx->meas_face_yz / int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES));
          }

          if (ey >= starty + 1) {
            row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex]
+ 1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] =
int_ctx->meas_face_zx * int_ctx->meas_face_zx / int_ctx->meas_elem *
avg_kappa_e; val_A[0][1] = -int_ctx->meas_face_zx * int_ctx->meas_face_zx /
int_ctx->meas_elem * avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_zx *
int_ctx->meas_face_zx / int_ctx->meas_elem * avg_kappa_e; val_A[1][1] =
int_ctx->meas_face_zx * int_ctx->meas_face_zx / int_ctx->meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES));
          }

          if (ez >= startz + 1) {
            row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex]
+ 1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] =
int_ctx->meas_face_xy * int_ctx->meas_face_xy / int_ctx->meas_elem *
avg_kappa_e; val_A[0][1] = -int_ctx->meas_face_xy * int_ctx->meas_face_xy /
int_ctx->meas_elem * avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_xy *
int_ctx->meas_face_xy / int_ctx->meas_elem * avg_kappa_e; val_A[1][1] =
int_ctx->meas_face_xy * int_ctx->meas_face_xy / int_ctx->meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES));
          }

          arr_M_i_3d[ez - startz][ey - starty][ex - startx] =
int_ctx->arr_kappa_3d[0][ez][ey][ex] + int_ctx->arr_kappa_3d[1][ez][ey][ex] +
int_ctx->arr_kappa_3d[2][ez][ey][ex];
        }
      }
    PetscCall(VecRestoreArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
    // PetscCall(VecScale(diag_M_i, int_ctx->meas_elem * 0.25 * M_PI * M_PI /
(nx * nx * s_ctx->H_x * s_ctx->H_x + ny * ny * s_ctx->H_y * s_ctx->H_y + nz * nz
* s_ctx->H_z * s_ctx->H_z))); PetscCall(VecScale(diag_M_i, int_ctx->meas_elem));
    PetscScalar vec_min;
    PetscCall(VecMax(diag_M_i, NULL, &vec_min));
    if (vec_min < 1.0e-9)
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "something wrong! vec_min=%.5E.\n",
vec_min)); PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));

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
PETSC_DEFAULT)); PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL)); ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= s_ctx->max_eigen_num_lv1, PETSC_COMM_WORLD,
PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d,
eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv1); PetscBool find_max_eigen =
PETSC_FALSE; for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem] = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv1) {
        s_ctx->eigen_num_lv1[coarse_elem] = j + 1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (!find_max_eigen && j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_num_lv1[coarse_elem] = s_ctx->max_eigen_num_lv1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
      }

      PetscCall(VecGetArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_c_array[j][ez][ey][startx],
&arr_eig_vec[ez - startz][ey - starty][0], nx));
      PetscCall(VecRestoreArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
    }
    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }

  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
&arr_ms_bases_c_array[i]));
  // We need to check the real maximum number of eigenvectors used, free what
those are not used.
  // The first time clean ms_bases_c_tmp.
  max_eigen_num = 0;
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem)
    max_eigen_num = PetscMax(max_eigen_num, s_ctx->eigen_num_lv1[coarse_elem]);
  PetscCallMPI(MPI_Allreduce(&max_eigen_num, &int_ctx->max_eigen_num_lv1_upd, 1,
MPI_INT, MPI_MAX, PETSC_COMM_WORLD)); for (i = int_ctx->max_eigen_num_lv1_upd; i
< s_ctx->max_eigen_num_lv1; ++i) PetscCall(DMRestoreLocalVector(s_ctx->dm,
&int_ctx->ms_bases_c_tmp[i]));

  Vec dummy_ms_bases_glo;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
INSERT_VALUES, dummy_ms_bases_glo)); PetscCall(DMGlobalToLocal(s_ctx->dm,
dummy_ms_bases_glo, INSERT_VALUES, int_ctx->ms_bases_c_tmp[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2_1] -= time_tmp;

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_setup_lv2_eigen_diag(PCCtx *s_ctx, _IntCtx *int_ctx) {
  PetscFunctionBeginUser;

  PetscLogDouble time_tmp;
  PetscCall(PetscTime(&time_tmp));

  Mat A_i_inner, M_i;
  Vec diag_M_i;
  PetscScalar ***arr_ms_bases_c_array[s_ctx->max_eigen_num_lv1], ***arr_M_i_3d,
val_A[2][2], avg_kappa_e; PetscInt max_eigen_num, i, j, coarse_elem, startx, nx,
ex, starty, ny, ey, startz, nz, ez, row[2], col[2];

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
NULL, &A_i_inner)); PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx
* ny * nz, 1, NULL, &M_i)); PetscCall(VecGetArray3d(diag_M_i, nz, ny, nx, 0, 0,
0, &arr_M_i_3d)); for (ez = startz; ez < startz + nz; ++ez) for (ey = starty; ey
< starty + ny; ++ey) { for (ex = startx; ex < startx + nx; ++ex) { arr_M_i_3d[ez
- startz][ey - starty][ex - startx] = 0.0; if (ex >= startx + 1) { row[0] = (ez
- startz) * ny * nx + (ey - starty) * nx + ex - startx - 1; row[1] = (ez -
startz) * ny * nx + (ey - starty) * nx + ex - startx; col[0] = (ez - startz) *
ny * nx + (ey - starty) * nx + ex - startx - 1; col[1] = (ez - startz) * ny * nx
+ (ey - starty) * nx + ex - startx; avg_kappa_e = 2.0 / (1.0 /
int_ctx->arr_kappa_3d[0][ez][ey][ex - 1] + 1.0 /
int_ctx->arr_kappa_3d[0][ez][ey][ex]); val_A[0][0] = int_ctx->meas_face_yz *
int_ctx->meas_face_yz / int_ctx->meas_elem * avg_kappa_e; val_A[0][1] =
-int_ctx->meas_face_yz * int_ctx->meas_face_yz / int_ctx->meas_elem *
avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_yz * int_ctx->meas_face_yz /
int_ctx->meas_elem * avg_kappa_e; val_A[1][1] = int_ctx->meas_face_yz *
int_ctx->meas_face_yz / int_ctx->meas_elem * avg_kappa_e;
            PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz][ey - starty][ex - startx -
1] += val_A[0][0]; arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
val_A[1][1];
          }

          if (ey >= starty + 1) {
            row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / int_ctx->arr_kappa_3d[1][ez][ey - 1][ex]
+ 1.0 / int_ctx->arr_kappa_3d[1][ez][ey][ex]); val_A[0][0] =
int_ctx->meas_face_zx * int_ctx->meas_face_zx / int_ctx->meas_elem *
avg_kappa_e; val_A[0][1] = -int_ctx->meas_face_zx * int_ctx->meas_face_zx /
int_ctx->meas_elem * avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_zx *
int_ctx->meas_face_zx / int_ctx->meas_elem * avg_kappa_e; val_A[1][1] =
int_ctx->meas_face_zx * int_ctx->meas_face_zx / int_ctx->meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz][ey - starty - 1][ex -
startx] += val_A[0][0]; arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
val_A[1][1];
          }

          if (ez >= startz + 1) {
            row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; row[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
startx; col[1] = (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
            avg_kappa_e = 2.0 / (1.0 / int_ctx->arr_kappa_3d[2][ez - 1][ey][ex]
+ 1.0 / int_ctx->arr_kappa_3d[2][ez][ey][ex]); val_A[0][0] =
int_ctx->meas_face_xy * int_ctx->meas_face_xy / int_ctx->meas_elem *
avg_kappa_e; val_A[0][1] = -int_ctx->meas_face_xy * int_ctx->meas_face_xy /
int_ctx->meas_elem * avg_kappa_e; val_A[1][0] = -int_ctx->meas_face_xy *
int_ctx->meas_face_xy / int_ctx->meas_elem * avg_kappa_e; val_A[1][1] =
int_ctx->meas_face_xy * int_ctx->meas_face_xy / int_ctx->meas_elem *
avg_kappa_e; PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
&val_A[0][0], ADD_VALUES)); arr_M_i_3d[ez - startz - 1][ey - starty][ex -
startx] += val_A[0][0]; arr_M_i_3d[ez - startz][ey - starty][ex - startx] +=
val_A[1][1];
          }
        }
      }

    PetscCall(MatAssemblyBegin(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i_inner, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A_i_inner, MAT_SYMMETRIC, PETSC_TRUE));

    PetscCall(VecRestoreArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
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
PETSC_DEFAULT)); PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL)); ST st;
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= s_ctx->max_eigen_num_lv1, PETSC_COMM_WORLD,
PETSC_ERR_USER, "SLEPc cannot find enough eigenvectors! (nconv=%d,
eigen_num=%d)\n", nconv, s_ctx->max_eigen_num_lv1); PetscBool find_max_eigen =
PETSC_FALSE; for (j = 0; j < s_ctx->max_eigen_num_lv1; ++j) {
      PetscCall(EPSGetEigenpair(eps, j, &eig_val, NULL, eig_vec, NULL));
      if (j == 0)
        s_ctx->eigen_min_lv1[coarse_elem] = eig_val;

      if (!find_max_eigen && eig_val >= s_ctx->eigen_bd_lv1) {
        s_ctx->eigen_num_lv1[coarse_elem] = j + 1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
        find_max_eigen = PETSC_TRUE;
      }

      if (!find_max_eigen && j == s_ctx->max_eigen_num_lv1 - 1) {
        s_ctx->eigen_num_lv1[coarse_elem] = s_ctx->max_eigen_num_lv1;
        s_ctx->eigen_max_lv1[coarse_elem] = eig_val;
      }

      PetscCall(VecGetArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          PetscCall(PetscArraycpy(&arr_ms_bases_c_array[j][ez][ey][startx],
&arr_eig_vec[ez - startz][ey - starty][0], nx));
      PetscCall(VecRestoreArray3d(eig_vec, nz, ny, nx, 0, 0, 0, &arr_eig_vec));
    }
    // Do some cleaning.
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));
    PetscCall(MatDestroy(&M_i));
  }

  for (i = 0; i < s_ctx->max_eigen_num_lv1; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
&arr_ms_bases_c_array[i]));
  // We need to check the real maximum number of eigenvectors used, free what
those are not used.
  // The first time clean ms_bases_c_tmp.
  max_eigen_num = 0;
  for (coarse_elem = 0; coarse_elem < int_ctx->coarse_elem_num; ++coarse_elem)
    max_eigen_num = PetscMax(max_eigen_num, s_ctx->eigen_num_lv1[coarse_elem]);
  PetscCallMPI(MPI_Allreduce(&max_eigen_num, &int_ctx->max_eigen_num_lv1_upd, 1,
MPI_INT, MPI_MAX, PETSC_COMM_WORLD)); for (i = int_ctx->max_eigen_num_lv1_upd; i
< s_ctx->max_eigen_num_lv1; ++i) PetscCall(DMRestoreLocalVector(s_ctx->dm,
&int_ctx->ms_bases_c_tmp[i]));

  Vec dummy_ms_bases_glo;
  PetscCall(DMGetGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));
  for (i = 0; i < int_ctx->max_eigen_num_lv1_upd; ++i) {
    PetscCall(DMLocalToGlobal(s_ctx->dm, int_ctx->ms_bases_c_tmp[i],
INSERT_VALUES, dummy_ms_bases_glo)); PetscCall(DMGlobalToLocal(s_ctx->dm,
dummy_ms_bases_glo, INSERT_VALUES, int_ctx->ms_bases_c_tmp[i]));
  }
  PetscCall(DMRestoreGlobalVector(s_ctx->dm, &dummy_ms_bases_glo));

  PetscCall(PetscTimeSubtract(&time_tmp));
  s_ctx->t_stages[STAGE_SU_LV2_1] -= time_tmp;

  PetscFunctionReturn(0);
}

*/
