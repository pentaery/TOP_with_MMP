#ifndef YCQ_MIXFEM_3L3D_H
#define YCQ_MIXFEM_3L3D_H

#include <mpi.h>
#include <petscksp.h>

#define DIM 3
#define NEIGH 6
#define LEFT 1
#define RIGHT 2
#define DOWN 3
#define UP 4
#define BACK 5
#define FRONT 6
#define NIL 1.0e-12
#define MAX_LOG_STATES 10
#define STAGE_SU_LV1 0
#define STAGE_SU_LV2_1 1
#define STAGE_SU_LV2_2 2
#define STAGE_SU_LV3_1 3
#define STAGE_SU_LV3_2 4
#define STAGE_AV_LV1 5
#define STAGE_AV_LV2 6
#define STAGE_AV_LV3 7
#define STAGE_SU 8
#define STAGE_AV 9

#define EIG_OP_MOD 0
#define EIG_OP_MEAN 1
#define EIG_OP_DIAG 2

typedef struct preconditioner_context {
  DM dm;
  Vec kappa[DIM], *ms_bases_c, *ms_bases_cc;
  Vec cost;
  Vec boundary;
  KSP *ksp_lv1, ksp_lv2, ksp_lv3;
  PetscInt *coarse_startx, *coarse_lenx, *coarse_starty, *coarse_leny,
      *coarse_startz, *coarse_lenz;
  PetscInt smoothing_iters_lv1, smoothing_iters_lv2, sub_domains, lv2_eigen_op;
  PetscInt max_eigen_num_lv1, *eigen_num_lv1;
  PetscInt max_eigen_num_lv2, eigen_num_lv2;
  PetscScalar H_x, H_y, H_z, L, W, H;
  // 总的长宽高，网格的长宽高
  PetscScalar *eigen_max_lv1, *eigen_min_lv1, eigen_bd_lv1, eigen_max_lv2,
      eigen_min_lv2, eigen_bd_lv2;
  PetscScalar t_stages[MAX_LOG_STATES];
  PetscBool use_W_cycle, no_shift_A_cc, use_full_Cholesky_lv1, use_2level;
  PetscInt M, N, P;
  // 网格数
  PetscScalar widthportion, lengthportion;
} PCCtx;

PetscErrorCode PC_init(PCCtx *s_ctx, PetscScalar *dom, PetscInt *mesh);
/*
    dom[0], the length; dom[1], the width; dom[2], the height.
    mesh[0], partions in x-direction; mesh[1], partions in y-direction; mesh[2],
   partions in z-direction.
*/

PetscErrorCode PC_print_info(PCCtx *s_ctx);

PetscErrorCode PC_setup(PC pc);

PetscErrorCode PC_create_A(PCCtx *s_ctx, Mat *A);

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y);

PetscErrorCode PC_get_range(const void *sendbuff, void *recvbuff,
                            MPI_Datatype datatype);

PetscErrorCode PC_print_stat(PCCtx *s_ctx);

PetscErrorCode PC_final_default(PCCtx *s_ctx);

PetscErrorCode PC_final(PCCtx *s_ctx);

/* According to Wiki, it may help... */
int mkl_serv_intel_cpu_true();

#endif