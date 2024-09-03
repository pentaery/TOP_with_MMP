#include "mpi.h"
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>
#include <stdio.h>

static char help[] = "This is a fork of our code of PreMixFEM. We remove "
                     "several unnecessary properties.\n";

#define DEBUG_LOCAL 0
#define ROOT_PATH ""
#define DIM 3
#define NEIGH 6
#define LEFT 1
#define RIGHT 2
#define DOWN 3
#define UP 4
#define BACK 5
#define FRONT 6
#define NIL 1.0e-12
#define MAX_FILENAME_LEN 72
#define MAX_TIMESTEP 200.0
#define SIMU_STAGES_NUM 6
#define LOG_STAGES_NUM 7
#define LOG_STAGE_PRE 0
#define LOG_STAGE_SIMU 1
#define LOG_STAGE_ASSE 2
#define LOG_STAGE_SOL 3
#define LOG_STAGE_OUT 4
#define LOG_STAGE_UPD 5
#define LOG_STAGE_POST 6
#define PORO_MIN 0.05
#define S_W_BOUND_MIN 0.2
#define S_W_BOUND_MAX 0.8

typedef struct spe10_context {
  DM dm;
  Vec perm[DIM], poro;
  PetscScalar H_x, H_y, H_z;
  PetscScalar B_w, rho_w, mu_w, B_o, mu_o, rho_o, injection_rate,
      bottomhole_pressure, wellbore_radius, well_index;
  PetscScalar S_wc, S_or, S_wi;
  PetscInt M, N, P, total_active_blocks_proc_well;
} SPE10Ctx;

// Throw all junks into this struct...
typedef struct internal_context {
  Vec perm_loc[DIM], kappa_loc[DIM];
  PetscScalar ***arr_perm_loc[DIM], ***arr_poro;
  PetscScalar meas_elem, meas_face_xy, meas_face_yz, meas_face_zx;
} INTERCtx;

typedef struct preconditioner_context {
  Vec *ms_bases_c, *ms_bases_cc, *ms_bases_c_tmp;
  KSP *ksp_lv1, ksp_lv2, ksp_lv3;
  PetscInt *coarse_startx, *coarse_lenx, *coarse_starty, *coarse_leny,
      *coarse_startz, *coarse_lenz;
  PetscInt smoothing_iters_lv1, smoothing_iters_lv2, sub_domains;
  PetscInt max_eigen_num_lv1, max_eigen_num_lv2;
  PetscScalar *eigen_max_lv1, *eigen_min_lv1, eigen_max_lv2, eigen_min_lv2;
  Vec S_w;
  SPE10Ctx *spe10_ctx;
  INTERCtx *inter_ctx;
} PCCtx;

PetscErrorCode SPE10_init(SPE10Ctx *self);
PetscErrorCode SPE10_final(SPE10Ctx *self);
PetscErrorCode PC_init(PCCtx *self, SPE10Ctx *spe10_ctx);
PetscErrorCode PC_print_info(PCCtx *self);
PetscErrorCode PC_final(PCCtx *self);
PetscErrorCode PC_setup(PC pc);
PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y);
PetscErrorCode _PC_apply_vec_lv1(PCCtx *self, Vec x, Vec y);
PetscErrorCode _PC_apply_vec_lv2(PCCtx *self, Vec x, Vec y);
PetscErrorCode _PC_apply_vec_lv3(PCCtx *self, Vec x, Vec y);
void _PC_get_coarse_elem_xyz(PCCtx *self, const PetscInt coarse_elem,
                             PetscInt *coarse_elem_x, PetscInt *coarse_elem_y,
                             PetscInt *coarse_elem_z);
void _PC_get_coarse_elem_corners(PCCtx *self, const PetscInt coarse_elem,
                                 PetscInt *startx, PetscInt *nx,
                                 PetscInt *starty, PetscInt *ny,
                                 PetscInt *startz, PetscInt *nz);
PetscErrorCode INTER_init(INTERCtx *self, SPE10Ctx *spe10_ctx);
PetscErrorCode INTER_final(INTERCtx *self, SPE10Ctx *spe10_ctx);
PetscErrorCode INTER_upd_kappa(INTERCtx *self, SPE10Ctx *spe10_ctx, Vec S_w);
PetscErrorCode _load_into_dm_vec(Vec load_from_h5_file, Vec load_into_dm_vec,
                                 DM dm);
PetscErrorCode save_vec_into_vtr(Vec v, const char *vtr_name);
PetscErrorCode get_linear_system(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx,
                                 Vec S_w, Mat A, Vec rhs);
void get_mobi(SPE10Ctx *spe10_ctx, PetscScalar S_w_loc, PetscScalar *lambda_w,
              PetscScalar *lambda_o, PetscScalar *lambda);
PetscErrorCode get_next_S_w(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec P,
                            PetscScalar DS_max, PetscScalar *delta_t, Vec S_w);
PetscErrorCode get_next_S_w_sig(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec P,
                                PetscScalar delta_t, Vec S_w,
                                PetscScalar *delta_S_w);
PetscErrorCode get_next_S_w_multisteps(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx,
                                       Vec P, PetscInt step_num,
                                       PetscScalar DS_max, PetscScalar *delta_t,
                                       Vec S_w);
PetscErrorCode get_init_S_w(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec S_w);
PetscErrorCode get_modified_poro(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx);
PetscErrorCode create_cross_kappa(PCCtx *s_ctx, PetscInt cr);

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, (char *)0, help));
  PetscLogStage stage[LOG_STAGES_NUM];
  PetscCall(PetscLogStageRegister("Pre", &stage[LOG_STAGE_PRE]));
  // PetscCall(PetscLogStageRegister("Simu", &stage[LOG_STAGE_SIMU]));
  PetscCall(PetscLogStageRegister("Assemble", &stage[LOG_STAGE_ASSE]));
  PetscCall(PetscLogStageRegister("Solve P", &stage[LOG_STAGE_SOL]));
  PetscCall(PetscLogStageRegister("Output", &stage[LOG_STAGE_OUT]));
  PetscCall(PetscLogStageRegister("Update S_w", &stage[LOG_STAGE_UPD]));
  PetscCall(PetscLogStageRegister("Post", &stage[LOG_STAGE_POST]));

  // >>>>
  PetscLogStagePush(stage[LOG_STAGE_PRE]);
  PetscBool output_realtime = PETSC_FALSE, petsc_default = PETSC_FALSE,
            output_P = PETSC_FALSE;
  PetscInt max_simu_steps = 50, save_files_freq = 10;
  PetscScalar simu_time_stages[SIMU_STAGES_NUM + 1] = {
      0.0, 99.99, 199.99, 399.99, 899.99, 1599.99, 1999.99};
  PetscScalar DS_max_stages[SIMU_STAGES_NUM + 1] = {0.0,   0.001, 0.001, 0.001,
                                                    0.001, 0.001, 0.001};
  PetscInt S_w_steps_stages[SIMU_STAGES_NUM + 1] = {0, 50, 50, 50, 50, 50, 50};
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-output_realtime", &output_realtime));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-output_P", &output_P));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-max_simu_steps", &max_simu_steps, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-save_files_freq", &save_files_freq,
                               NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_1", &DS_max_stages[1], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_2", &DS_max_stages[2], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_3", &DS_max_stages[3], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_4", &DS_max_stages[4], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_5", &DS_max_stages[5], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-DS_max_6", &DS_max_stages[6], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_1", &simu_time_stages[1], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_2", &simu_time_stages[2], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_3", &simu_time_stages[3], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_4", &simu_time_stages[4], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_5", &simu_time_stages[5], NULL));
  PetscCall(
      PetscOptionsGetScalar(NULL, NULL, "-simu_6", &simu_time_stages[6], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_1", &S_w_steps_stages[1], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_2", &S_w_steps_stages[2], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_3", &S_w_steps_stages[3], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_4", &S_w_steps_stages[4], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_5", &S_w_steps_stages[5], NULL));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-step_6", &S_w_steps_stages[6], NULL));

  if (DEBUG_LOCAL) {
    output_realtime = PETSC_TRUE;
  }

  int world_rank = 0;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &world_rank));

  SPE10Ctx spe10_ctx;
  PetscCall(SPE10_init(&spe10_ctx));

  PetscCall(save_vec_into_vtr(spe10_ctx.perm[0], "perm_x.vtr"));
  PetscCall(save_vec_into_vtr(spe10_ctx.perm[1], "perm_y.vtr"));
  PetscCall(save_vec_into_vtr(spe10_ctx.perm[2], "perm_z.vtr"));
  PetscCall(save_vec_into_vtr(spe10_ctx.poro, "poro.vtr"));

  Vec ratio;
  PetscCall(VecDuplicate(spe10_ctx.poro, &ratio));
  PetscCall(PetscObjectSetName((PetscObject)ratio, "perm_x,y / perm_z"));
  PetscCall(VecPointwiseDivide(ratio, spe10_ctx.perm[0], spe10_ctx.perm[2]));
  PetscCall(save_vec_into_vtr(ratio, "ratio.vtr"));
  PetscCall(VecDestroy(&ratio));

  INTERCtx inter_ctx;
  PetscCall(INTER_init(&inter_ctx, &spe10_ctx));
  PetscCall(get_modified_poro(&spe10_ctx, &inter_ctx));

  PCCtx pc_ctx;
  if (!petsc_default) {
    PetscCall(PC_init(&pc_ctx, &spe10_ctx));
    pc_ctx.inter_ctx = &inter_ctx;
    PC_print_info(&pc_ctx);
  }

  Vec S_w, P;
  PetscCall(DMCreateGlobalVector(spe10_ctx.dm, &S_w));
  PetscCall(PetscObjectSetName((PetscObject)S_w, "S_w"));
  PetscCall(DMCreateGlobalVector(spe10_ctx.dm, &P));
  PetscCall(PetscObjectSetName((PetscObject)P, "P"));
  PetscCall(get_init_S_w(&spe10_ctx, &inter_ctx, S_w));

  Mat A;
  Vec rhs;
  KSP ksp;
  PetscCall(DMCreateMatrix(spe10_ctx.dm, &A));
  PetscCall(DMCreateGlobalVector(spe10_ctx.dm, &rhs));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetFromOptions(ksp));
  PetscLogStagePop();

  // >>>>
  PetscLogStagePush(stage[LOG_STAGE_OUT]);
  PetscInt simu_time_ind, stage_ind = 1, iter_count = 0;
  PetscScalar simu_time = 0.0, delta_t = 0.0, S_w_min = 0.0, S_w_max = 0.0;

  PetscLogDouble time_tmp = 0.0;

  char file_name[MAX_FILENAME_LEN], vtr_name[MAX_FILENAME_LEN];
  FILE *P_solver_history = NULL, *S_w_vtr_series = NULL, *P_vtr_series = NULL;

  if (world_rank == 0) {
    sprintf(file_name, "%s%s", ROOT_PATH, "data/spe10_output/P_solver.csv");
    P_solver_history = fopen(file_name, "w");
    PetscCall(
        PetscFPrintf(PETSC_COMM_SELF, P_solver_history,
                     "simu_time_ind,simu_time,stage_ind,DS_max,S_w_steps,wall_"
                     "time,iter_count,iter_count_base,S_w_min,S_w_max\n"));
    sprintf(file_name, "%s%s", ROOT_PATH, "data/spe10_output/S_w.vtr.series");
    S_w_vtr_series = fopen(file_name, "w");
    PetscCall(
        PetscFPrintf(PETSC_COMM_SELF, S_w_vtr_series,
                     "{\n\"file-series-version\" : \"1.0\",\n\"files\" : [\n"));
    if (output_P) {
      sprintf(file_name, "%s%s", ROOT_PATH, "data/spe10_output/P.vtr.series");
      P_vtr_series = fopen(file_name, "w");
      PetscCall(PetscFPrintf(
          PETSC_COMM_SELF, P_vtr_series,
          "{\n\"file-series-version\" : \"1.0\",\n\"files\" : [\n"));
    }
  }
  PetscLogStagePop();

  for (simu_time_ind = 0; simu_time_ind < max_simu_steps; ++simu_time_ind) {
    // >>>>
    PetscLogStagePush(stage[LOG_STAGE_ASSE]);
    PetscCall(INTER_upd_kappa(&inter_ctx, &spe10_ctx, S_w));
    PetscCall(get_linear_system(&spe10_ctx, &inter_ctx, S_w, A, rhs));
    PetscLogStagePop();

    // >>>>
    PetscLogStagePush(stage[LOG_STAGE_SOL]);
    PetscCall(PetscTime(&time_tmp));
    PetscCall(KSPSetOperators(ksp, A, A));
    if (!petsc_default) {
      PC pc;
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &pc_ctx));
      pc_ctx.S_w = S_w;
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    }

    PetscCall(KSPSetUp(ksp));
    PetscCall(KSPSolve(ksp, rhs, P));
    PetscCall(PetscTimeSubtract(&time_tmp));
    PetscLogStagePop();

    // >>>>
    PetscLogStagePush(stage[LOG_STAGE_OUT]);
    PetscCall(KSPGetIterationNumber(ksp, &iter_count));
    PetscCall(VecMin(S_w, NULL, &S_w_min));
    PetscCall(VecMax(S_w, NULL, &S_w_max));
    PetscCheck(S_W_BOUND_MIN - NIL < S_w_min && S_W_BOUND_MAX + NIL > S_w_max,
               PETSC_COMM_WORLD, PETSC_ERR_USER,
               "S_w=[%.5E, %.5E] exceeds the bounds.\n", S_w_min, S_w_max);
    if (output_realtime)
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "%d, simu_time=%.5E, stage_ind=%d, DS_max=%.5f, S_w_steps=%d, "
          "wall_time=%.5f, iter_count=%d, S_w=[%.5E, %.5E].\n",
          simu_time_ind, simu_time, stage_ind, DS_max_stages[stage_ind],
          S_w_steps_stages[stage_ind], -time_tmp, iter_count, S_w_min,
          S_w_max));
    if (world_rank == 0)
      PetscCall(PetscFPrintf(
          PETSC_COMM_SELF, P_solver_history,
          "%d,%.5E,%d,%.5f,%d,%.5f,%d,%.5E,%.5E\n", simu_time_ind, simu_time,
          stage_ind, DS_max_stages[stage_ind], S_w_steps_stages[stage_ind],
          -time_tmp, iter_count, S_w_min, S_w_max));

    if (simu_time_ind % save_files_freq == 0) {
      sprintf(vtr_name, "S_w%d.vtr", simu_time_ind);
      PetscCall(save_vec_into_vtr(S_w, vtr_name));
      if (world_rank == 0) {
        if (simu_time_ind == 0)
          PetscCall(PetscFPrintf(PETSC_COMM_SELF, S_w_vtr_series,
                                 "{\"name\" :\"%s\", \"time\" :%.9f}", vtr_name,
                                 simu_time));
        else
          PetscCall(PetscFPrintf(PETSC_COMM_SELF, S_w_vtr_series,
                                 ",\n{\"name\" :\"%s\", \"time\" :%.9f}",
                                 vtr_name, simu_time));
      }

      if (output_P) {
        sprintf(vtr_name, "P%d.vtr", simu_time_ind);
        PetscCall(save_vec_into_vtr(P, vtr_name));
        if (world_rank == 0) {
          if (simu_time_ind == 0)
            PetscCall(PetscFPrintf(PETSC_COMM_SELF, P_vtr_series,
                                   "{\"name\" :\"%s\", \"time\" :%.9f}",
                                   vtr_name, simu_time));
          else
            PetscCall(PetscFPrintf(PETSC_COMM_SELF, P_vtr_series,
                                   ",\n{\"name\" :\"%s\", \"time\" :%.9f}",
                                   vtr_name, simu_time));
        }
      }
    }
    if (simu_time > simu_time_stages[SIMU_STAGES_NUM])
      break;
    PetscLogStagePop();

    // >>>>
    PetscLogStagePush(stage[LOG_STAGE_UPD]);
    // delta_t = DS_max_stages[stage_ind];
    // PetscCall(get_next_S_w_sig(&spe10_ctx, &inter_ctx, P, delta_t, S_w,
    // &delta_S_w)); PetscCall(get_next_S_w(&spe10_ctx, &inter_ctx, P,
    // DS_max_stages[stage_ind], &delta_t, S_w));
    PetscCall(get_next_S_w_multisteps(&spe10_ctx, &inter_ctx, P,
                                      S_w_steps_stages[stage_ind],
                                      DS_max_stages[stage_ind], &delta_t, S_w));
    simu_time += delta_t;
    if (simu_time > simu_time_stages[stage_ind]) {
      stage_ind++;
      stage_ind = PetscMin(stage_ind, SIMU_STAGES_NUM);
    }

    PetscLogStagePop();
  }

  PetscLogStagePush(stage[LOG_STAGE_OUT]);
  if (world_rank == 0) {
    fclose(P_solver_history);
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, S_w_vtr_series, "\n]\n}"));
    fclose(S_w_vtr_series);
    if (output_P) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, P_vtr_series, "\n]\n}"));
      fclose(P_vtr_series);
    }
  }
  PetscLogStagePop();

  PetscLogStagePush(stage[LOG_STAGE_POST]);
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(KSPDestroy(&ksp));

  if (!petsc_default)
    PetscCall(PC_final(&pc_ctx));
  PetscCall(VecDestroy(&S_w));
  PetscCall(VecDestroy(&P));
  PetscCall(INTER_final(&inter_ctx, &spe10_ctx));
  PetscCall(SPE10_final(&spe10_ctx));
  PetscLogStagePop();

  PetscCall(SlepcFinalize());
  return 0;
}

PetscErrorCode SPE10_init(SPE10Ctx *self) {
  PetscFunctionBeginUser;

  if (DEBUG_LOCAL) {
    self->M = 16;
    self->N = 18;
    self->P = 15;
  } else {
    self->M = 60;
    self->N = 220;
    self->P = 85;
  }
  self->H_x = 20.0; // [ft]
  self->H_y = 10.0; // [ft]
  self->H_z = 2.0;  // [ft]
  self->mu_w =
      0.3 * 1.6786805555555556E-12; // 1.0 cP = 1.0E-3 [pa * s] = 1.0E-3
                                    // * 1.45038E-4 / 8.64E+4 [psi * Day]
  self->mu_o =
      3.0 * 1.6786805555555556E-12; // 1.0 cP = 1.0E-3 [pa * s] = 1.0E-3
                                    // * 1.45038E-4 / 8.64E+4 [psi * Day]
  self->rho_w = 64.0;               // [lb / ft^3]
  self->rho_o = 53.0;               // [lb / ft^3]
  self->B_w = 1.01;                 // 1.0
  self->B_o = 1.01;                 // 1.0
  self->injection_rate =
      5.0E+3 * 5.61458;               // 1.0 [bbl / Day] = 5.61458 [ft^3 / Day]
  self->bottomhole_pressure = 4.0E+3; // [psi]
  self->S_wc = 0.2;
  self->S_or = 0.2;
  self->S_wi = 0.2;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, self->M, self->N,
                         self->P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1,
                         1, NULL, NULL, NULL, &self->dm));
  PetscCall(DMSetUp(self->dm));

  PetscCall(DMCreateGlobalVector(self->dm, &self->perm[0]));
  PetscCall(PetscObjectSetName((PetscObject)self->perm[0], "perm_x"));
  PetscCall(DMCreateGlobalVector(self->dm, &self->perm[1]));
  PetscCall(PetscObjectSetName((PetscObject)self->perm[1], "perm_y"));
  PetscCall(DMCreateGlobalVector(self->dm, &self->perm[2]));
  PetscCall(PetscObjectSetName((PetscObject)self->perm[2], "perm_z"));
  PetscCall(DMCreateGlobalVector(self->dm, &self->poro));
  PetscCall(PetscObjectSetName((PetscObject)self->poro, "poro"));

  Vec load_from_h5_file;
  PetscViewer h5_viewer;
  if (DEBUG_LOCAL)
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 60 * 220 * 85, &load_from_h5_file));
  else
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, self->M * self->N * self->P,
                           &load_from_h5_file));

  char file_name[MAX_FILENAME_LEN];
  sprintf(file_name, "%s%s", ROOT_PATH, "data/spe10/SPE10_rock.h5");
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_SELF, file_name, FILE_MODE_READ,
                                &h5_viewer));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_x"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, self->perm[0], self->dm));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_y"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, self->perm[1], self->dm));

  PetscCall(PetscObjectSetName((PetscObject)load_from_h5_file, "perm_z"));
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(VecScale(load_from_h5_file, 10.7639)); // 1 [m^2] = 10.7639 [ft^2].
  PetscCall(_load_into_dm_vec(load_from_h5_file, self->perm[2], self->dm));

  PetscObjectSetName((PetscObject)load_from_h5_file, "poro");
  PetscCall(VecLoad(load_from_h5_file, h5_viewer));
  PetscCall(_load_into_dm_vec(load_from_h5_file, self->poro, self->dm));

  PetscCall(PetscViewerDestroy(&h5_viewer));
  PetscCall(VecDestroy(&load_from_h5_file));

  self->wellbore_radius = 1.0; // [ft]
  PetscScalar r_1 =
                  sqrt(9.0 * self->H_x * self->H_x + self->H_y * self->H_y) / 2,
              r_2 =
                  sqrt(self->H_x * self->H_x + 9.0 * self->H_y * self->H_y) / 2;
  PetscScalar temp;
  temp = self->H_y / self->H_x * log(r_1 / self->wellbore_radius) +
         self->H_x / self->H_y * log(r_2 / self->wellbore_radius);
  temp *= (2.0 / PETSC_PI);
  temp -= 1.0;
  self->well_index =
      (1.0 / self->H_x / self->H_x + 1.0 / self->H_y / self->H_y) / temp;

  self->injection_rate *= self->B_w;
  self->total_active_blocks_proc_well = 4 * self->P;

  PetscFunctionReturn(0);
}

PetscErrorCode SPE10_final(SPE10Ctx *self) {
  PetscFunctionBeginUser;

  PetscCall(DMDestroy(&self->dm));
  PetscCall(VecDestroy(&self->perm[0]));
  PetscCall(VecDestroy(&self->perm[1]));
  PetscCall(VecDestroy(&self->perm[2]));
  PetscCall(VecDestroy(&self->poro));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_init(PCCtx *self, SPE10Ctx *spe10_ctx) {
  PetscFunctionBeginUser;

  self->smoothing_iters_lv1 = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-si_lv1",
                               &self->smoothing_iters_lv1, NULL));
  self->smoothing_iters_lv1 =
      self->smoothing_iters_lv1 >= 1 ? self->smoothing_iters_lv1 : 1;

  self->smoothing_iters_lv2 = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-si_lv2",
                               &self->smoothing_iters_lv2, NULL));
  self->smoothing_iters_lv2 =
      self->smoothing_iters_lv2 >= 1 ? self->smoothing_iters_lv2 : 1;

  self->sub_domains = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-sd", &self->sub_domains, NULL));
  PetscCheck(self->sub_domains >= 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "Error in sub_domains=%d.\n", self->sub_domains);

  self->max_eigen_num_lv1 = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv1", &self->max_eigen_num_lv1,
                               NULL));
  PetscCheck(self->max_eigen_num_lv1 >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Error in max_eigen_num_lv1=%d for the level-1 problem.\n",
             self->max_eigen_num_lv1);

  self->max_eigen_num_lv2 = 4;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-en_lv2", &self->max_eigen_num_lv2,
                               NULL));
  PetscCheck(self->max_eigen_num_lv2 >= 1, PETSC_COMM_WORLD,
             PETSC_ERR_ARG_WRONG,
             "Error in max_eigen_num_lv2=%d for the level-2 problem.\n",
             self->max_eigen_num_lv2);

  self->spe10_ctx = spe10_ctx;
  PetscInt m, n, p, coarse_elem_num;
  // Users should be responsible for constructing kappa.
  PetscCall(DMDAGetInfo(self->spe10_ctx->dm, NULL, NULL, NULL, NULL, &m, &n, &p,
                        NULL, NULL, NULL, NULL, NULL, NULL));
  if (self->sub_domains == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "The subdomain may not be proper (too "
                          "long/wide/high), reset sub_domains from %d to 2.\n",
                          self->sub_domains));
    self->sub_domains = 2;
  }

  coarse_elem_num = self->sub_domains * self->sub_domains * self->sub_domains;
  self->ms_bases_c =
      (Vec *)malloc(self->max_eigen_num_lv1 * coarse_elem_num * sizeof(Vec));
  self->ms_bases_cc = (Vec *)malloc(self->max_eigen_num_lv2 * sizeof(Vec));
  self->ms_bases_c_tmp = (Vec *)malloc(self->max_eigen_num_lv1 * sizeof(Vec));
  self->ksp_lv1 = (KSP *)malloc(coarse_elem_num * sizeof(KSP));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_startx));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_starty));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_startz));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_lenx));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_leny));
  PetscCall(PetscMalloc1(self->sub_domains, &self->coarse_lenz));
  PetscCall(PetscMalloc1(coarse_elem_num, &self->eigen_max_lv1));
  PetscCall(PetscMalloc1(coarse_elem_num, &self->eigen_min_lv1));

  /*********************************************************************
  Get the boundaries of each coarse subdomain.
  *********************************************************************/
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz;

  PetscCall(DMDAGetCorners(self->spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));
  self->coarse_startx[0] = proc_startx;
  self->coarse_starty[0] = proc_starty;
  self->coarse_startz[0] = proc_startz;
  self->coarse_lenx[0] =
      (proc_nx / self->sub_domains) + (proc_nx % self->sub_domains > 0);
  self->coarse_leny[0] =
      (proc_ny / self->sub_domains) + (proc_ny % self->sub_domains > 0);
  self->coarse_lenz[0] =
      (proc_nz / self->sub_domains) + (proc_nz % self->sub_domains > 0);
  for (PetscInt sd = 1; sd < self->sub_domains; ++sd) {
    self->coarse_startx[sd] =
        self->coarse_startx[sd - 1] + self->coarse_lenx[sd - 1];
    self->coarse_lenx[sd] =
        (proc_nx / self->sub_domains) + (proc_nx % self->sub_domains > sd);
    self->coarse_starty[sd] =
        self->coarse_starty[sd - 1] + self->coarse_leny[sd - 1];
    self->coarse_leny[sd] =
        (proc_ny / self->sub_domains) + (proc_ny % self->sub_domains > sd);
    self->coarse_startz[sd] =
        self->coarse_startz[sd - 1] + self->coarse_lenz[sd - 1];
    self->coarse_lenz[sd] =
        (proc_nz / self->sub_domains) + (proc_nz % self->sub_domains > sd);
  }

  PetscInt coarse_elem, eigen_ind_lv1, nx, ny, nz, dof;
  PC pc;
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    PetscCall(KSPCreate(PETSC_COMM_SELF, &(self->ksp_lv1[coarse_elem])));
    PetscCall(KSPSetType(self->ksp_lv1[coarse_elem], KSPPREONLY));
    PetscCall(
        KSPSetErrorIfNotConverged(self->ksp_lv1[coarse_elem], PETSC_TRUE));
    PetscCall(KSPGetPC(self->ksp_lv1[coarse_elem], &pc));
    PetscCall(PCSetType(pc, PCICC));
    PetscCall(PCSetOptionsPrefix(pc, "kspl1_"));
    PetscCall(PCSetFromOptions(pc));
    _PC_get_coarse_elem_corners(self, coarse_elem, NULL, &nx, NULL, &ny, NULL,
                                &nz);
    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1) {
      dof = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1;
      PetscCall(
          VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &self->ms_bases_c[dof]));
    }
  }

  for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
       ++eigen_ind_lv1)
    PetscCall(DMCreateLocalVector(self->spe10_ctx->dm,
                                  &self->ms_bases_c_tmp[eigen_ind_lv1]));

  PetscCall(KSPCreate(PETSC_COMM_SELF, &self->ksp_lv2));
  PetscCall(KSPSetType(self->ksp_lv2, KSPPREONLY));
  PetscCall(KSPSetErrorIfNotConverged(self->ksp_lv2, PETSC_TRUE));
  PetscCall(KSPGetPC(self->ksp_lv2, &pc));
  PetscCall(PCSetType(pc, PCICC));
  // PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  PetscCall(PCSetOptionsPrefix(pc, "kspl2_"));
  PetscCall(PCSetFromOptions(pc));

  PetscInt eigen_ind_lv2;
  for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
       ++eigen_ind_lv2)
    PetscCall(DMCreateLocalVector(self->spe10_ctx->dm,
                                  &self->ms_bases_cc[eigen_ind_lv2]));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &self->ksp_lv3));
  PetscCall(KSPSetType(self->ksp_lv3, KSPPREONLY));
  PetscCall(KSPSetErrorIfNotConverged(self->ksp_lv3, PETSC_TRUE));
  PetscCall(KSPGetPC(self->ksp_lv3, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
  // PetscCall(PCSetType(pc, PCCHOLESKY));
  // PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  PetscCall(PCSetOptionsPrefix(pc, "kspl3_"));
  PetscCall(PCSetFromOptions(pc));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_print_info(PCCtx *self) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DIM=%d, M=%d, N=%d, P=%d.\n", 3,
                        self->spe10_ctx->M, self->spe10_ctx->N,
                        self->spe10_ctx->P));
  PetscInt m, n, p;
  PetscCall(DMDAGetInfo(self->spe10_ctx->dm, NULL, NULL, NULL, NULL, &m, &n, &p,
                        NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "MPI Processes in each direction: X=%d, Y=%d, Z=%d.\n",
                        m, n, p));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "smoothing_iters_lv1=%d, smoothing_iters_lv2=%d, sub_domains=%d.\n",
      self->smoothing_iters_lv1, self->smoothing_iters_lv2, self->sub_domains));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "max_eigen_num_lv1=%d, max_eigen_num_lv2=%d.\n",
                        self->max_eigen_num_lv1, self->max_eigen_num_lv2));

  PetscFunctionReturn(0);
}

PetscErrorCode PC_final(PCCtx *self) {
  PetscFunctionBeginUser;

  PetscInt coarse_elem_num =
               self->sub_domains * self->sub_domains * self->sub_domains,
           coarse_elem, eigen_ind_lv1, eigen_ind_lv2;
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(
          VecDestroy(&self->ms_bases_c[coarse_elem * self->max_eigen_num_lv1 +
                                       eigen_ind_lv1]));
    PetscCall(KSPDestroy(&self->ksp_lv1[coarse_elem]));
  }

  for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
       ++eigen_ind_lv1)
    PetscCall(VecDestroy(&self->ms_bases_c_tmp[eigen_ind_lv1]));

  PetscCall(KSPDestroy(&self->ksp_lv2));
  for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
       ++eigen_ind_lv2)
    PetscCall(VecDestroy(&self->ms_bases_cc[eigen_ind_lv2]));
  PetscCall(KSPDestroy(&self->ksp_lv3));

  free(self->ms_bases_c);
  free(self->ms_bases_cc);
  free(self->ms_bases_c_tmp);
  free(self->ksp_lv1);
  PetscCall(PetscFree(self->coarse_startx));
  PetscCall(PetscFree(self->coarse_starty));
  PetscCall(PetscFree(self->coarse_startz));
  PetscCall(PetscFree(self->coarse_lenx));
  PetscCall(PetscFree(self->coarse_leny));
  PetscCall(PetscFree(self->coarse_lenz));
  PetscCall(PetscFree(self->eigen_max_lv1));
  PetscCall(PetscFree(self->eigen_min_lv1));

  PetscFunctionReturn(0);
}

PetscErrorCode INTER_init(INTERCtx *self, SPE10Ctx *spe10_ctx) {
  PetscFunctionBeginUser;

  for (PetscInt xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMCreateLocalVector(spe10_ctx->dm, &self->perm_loc[xyz_ind]));
    PetscCall(DMGlobalToLocal(spe10_ctx->dm, spe10_ctx->perm[xyz_ind],
                              INSERT_VALUES, self->perm_loc[xyz_ind]));
    PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, self->perm_loc[xyz_ind],
                                  &(self->arr_perm_loc[xyz_ind])));
    PetscCall(DMCreateLocalVector(spe10_ctx->dm, &self->kappa_loc[xyz_ind]));
  }

  PetscCall(DMDAVecGetArray(spe10_ctx->dm, spe10_ctx->poro, &(self->arr_poro)));
  // We may need to modify poro.

  self->meas_elem = spe10_ctx->H_x * spe10_ctx->H_y * spe10_ctx->H_z;
  self->meas_face_xy = spe10_ctx->H_x * spe10_ctx->H_y;
  self->meas_face_yz = spe10_ctx->H_y * spe10_ctx->H_z;
  self->meas_face_zx = spe10_ctx->H_z * spe10_ctx->H_x;

  PetscFunctionReturn(0);
}

PetscErrorCode INTER_final(INTERCtx *self, SPE10Ctx *spe10_ctx) {
  PetscFunctionBeginUser;

  for (PetscInt xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, self->perm_loc[xyz_ind],
                                      &(self->arr_perm_loc[xyz_ind])));
    PetscCall(VecDestroy(&self->perm_loc[xyz_ind]));
    PetscCall(VecDestroy(&self->kappa_loc[xyz_ind]));
  }
  PetscCall(
      DMDAVecRestoreArray(spe10_ctx->dm, spe10_ctx->poro, &(self->arr_poro)));

  PetscFunctionReturn(0);
}

PetscErrorCode INTER_upd_kappa(INTERCtx *self, SPE10Ctx *spe10_ctx, Vec S_w) {
  PetscFunctionBeginUser;

  Vec kappa[DIM];
  PetscScalar ***arr_kappa[DIM], ***arr_S_w, lambda;
  PetscInt xyz_ind, proc_startx, proc_starty, proc_startz, proc_nx, proc_ny,
      proc_nz, ex, ey, ez;
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMGetGlobalVector(spe10_ctx->dm, &kappa[xyz_ind]));
    PetscCall(
        DMDAVecGetArray(spe10_ctx->dm, kappa[xyz_ind], &arr_kappa[xyz_ind]));
  }
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, S_w, &arr_S_w));
  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        // Formulas from the SPE10 model.
        get_mobi(spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL, &lambda);
        for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind)
          arr_kappa[xyz_ind][ez][ey][ex] =
              self->arr_perm_loc[xyz_ind][ez][ey][ex] * lambda;
      }
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, kappa[xyz_ind],
                                  &arr_kappa[xyz_ind]));
    PetscCall(DMGlobalToLocal(spe10_ctx->dm, kappa[xyz_ind], INSERT_VALUES,
                              self->kappa_loc[xyz_ind]));
    PetscCall(DMRestoreGlobalVector(spe10_ctx->dm, &kappa[xyz_ind]));
  }
  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, S_w, &arr_S_w));

  PetscFunctionReturn(0);
}

PetscErrorCode _load_into_dm_vec(Vec load_from_h5_file, Vec load_into_dm_vec,
                                 DM dm) {
  PetscFunctionBeginUser;

  PetscInt startx, starty, startz, nx, ny, nz, ey, ez, M, N, P, size;
  PetscScalar ***arr_load_from_h5_file, ***load_into_dm_vec_arr;

  PetscCall(DMDAGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAGetInfo(dm, NULL, &M, &N, &P, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL));
  PetscCall(VecGetSize(load_from_h5_file, &size));

  if (DEBUG_LOCAL)
    PetscCall(VecGetArray3dRead(load_from_h5_file, 85, 220, 60, 0, 0, 0,
                                &arr_load_from_h5_file));
  else {
    PetscCheck(size == M * N * P, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT,
               "Something wrong! the vector loaded from the h5 file with the "
               "size=%d while we need size=%d.\n",
               size, M * N * P);
    PetscCall(VecGetArray3dRead(load_from_h5_file, P, N, M, 0, 0, 0,
                                &arr_load_from_h5_file));
  }

  PetscCall(DMDAVecGetArray(dm, load_into_dm_vec, &load_into_dm_vec_arr));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ++ey) {
      PetscCall(PetscArraycpy(&load_into_dm_vec_arr[ez][ey][startx],
                              &arr_load_from_h5_file[ez][ey][startx], nx));
    }

  if (DEBUG_LOCAL)
    PetscCall(VecRestoreArray3dRead(load_from_h5_file, 85, 220, 60, 0, 0, 0,
                                    &arr_load_from_h5_file));
  else
    PetscCall(VecRestoreArray3dRead(load_from_h5_file, P, N, M, 0, 0, 0,
                                    &arr_load_from_h5_file));

  PetscCall(DMDAVecRestoreArray(dm, load_into_dm_vec, &load_into_dm_vec_arr));

  PetscFunctionReturn(0);
}

PetscErrorCode save_vec_into_vtr(Vec v, const char *vtr_name) {
  PetscFunctionBeginUser;

  char file_name[MAX_FILENAME_LEN];
  PetscViewer vtr_viewer;

  sprintf(file_name, "%sdata/spe10_output/%s", ROOT_PATH, vtr_name);
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, file_name, FILE_MODE_WRITE,
                               &vtr_viewer));
  PetscCall(VecView(v, vtr_viewer));
  PetscCall(PetscViewerDestroy(&vtr_viewer));

  PetscFunctionReturn(0);
}

void _PC_get_coarse_elem_corners(PCCtx *self, const PetscInt coarse_elem,
                                 PetscInt *startx, PetscInt *nx,
                                 PetscInt *starty, PetscInt *ny,
                                 PetscInt *startz, PetscInt *nz) {
  PetscInt coarse_elem_x, coarse_elem_y, coarse_elem_z;
  _PC_get_coarse_elem_xyz(self, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                          &coarse_elem_z);
  if (startx != NULL)
    *startx = self->coarse_startx[coarse_elem_x];
  if (nx != NULL)
    *nx = self->coarse_lenx[coarse_elem_x];
  if (starty != NULL)
    *starty = self->coarse_starty[coarse_elem_y];
  if (ny != NULL)
    *ny = self->coarse_leny[coarse_elem_y];
  if (startz != NULL)
    *startz = self->coarse_startz[coarse_elem_z];
  if (nz != NULL)
    *nz = self->coarse_lenz[coarse_elem_z];
}

void _PC_get_coarse_elem_xyz(PCCtx *self, const PetscInt coarse_elem,
                             PetscInt *coarse_elem_x, PetscInt *coarse_elem_y,
                             PetscInt *coarse_elem_z) {
  if (coarse_elem_x != NULL)
    *coarse_elem_x = coarse_elem % self->sub_domains;
  if (coarse_elem_y != NULL)
    *coarse_elem_y = (coarse_elem / self->sub_domains) % self->sub_domains;
  if (coarse_elem_z != NULL)
    *coarse_elem_z = coarse_elem / (self->sub_domains * self->sub_domains);
}

PetscErrorCode get_linear_system(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx,
                                 Vec S_w, Mat A, Vec rhs) {
  PetscFunctionBeginUser;

  PetscInt xyz_ind, proc_startx, proc_starty, proc_startz, proc_nx, proc_ny,
      proc_nz, ex, ey, ez;
  PetscScalar ***arr_kappa[DIM], ***arr_rhs, ***arr_S_w, val_A[2][2],
      avg_kappa_e, lambda;
  MatStencil row[2], col[2];
  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind)
    PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind],
                                  &arr_kappa[xyz_ind]));
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMDAVecGetArray(spe10_ctx->dm, rhs, &arr_rhs));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, S_w, &arr_S_w));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if (ex >= 1) {
          row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                               1.0 / arr_kappa[0][ez][ey][ex]);
          val_A[0][0] = inter_ctx->meas_face_yz * inter_ctx->meas_face_yz /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[0][1] = -inter_ctx->meas_face_yz * inter_ctx->meas_face_yz /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][0] = -inter_ctx->meas_face_yz * inter_ctx->meas_face_yz /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][1] = inter_ctx->meas_face_yz * inter_ctx->meas_face_yz /
                        inter_ctx->meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }

        if (ey >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                               1.0 / arr_kappa[1][ez][ey][ex]);
          val_A[0][0] = inter_ctx->meas_face_zx * inter_ctx->meas_face_zx /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[0][1] = -inter_ctx->meas_face_zx * inter_ctx->meas_face_zx /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][0] = -inter_ctx->meas_face_zx * inter_ctx->meas_face_zx /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][1] = inter_ctx->meas_face_zx * inter_ctx->meas_face_zx /
                        inter_ctx->meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }

        if (ez >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                               1.0 / arr_kappa[2][ez][ey][ex]);
          val_A[0][0] = inter_ctx->meas_face_xy * inter_ctx->meas_face_xy /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[0][1] = -inter_ctx->meas_face_xy * inter_ctx->meas_face_xy /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][0] = -inter_ctx->meas_face_xy * inter_ctx->meas_face_xy /
                        inter_ctx->meas_elem * avg_kappa_e;
          val_A[1][1] = inter_ctx->meas_face_xy * inter_ctx->meas_face_xy /
                        inter_ctx->meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }

        // Process well conditions.
        get_mobi(spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL, &lambda);
        if ((ex == 0 && ey == 0) || (ex == 0 && ey == spe10_ctx->N - 1) ||
            (ex == spe10_ctx->M - 1 && ey == 0) ||
            (ex == spe10_ctx->M - 1 && ey == spe10_ctx->N - 1)) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = inter_ctx->meas_elem *
                        inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda *
                        spe10_ctx->well_index;
          PetscCall(MatSetValuesStencil(A, 1, &row[0], 1, &col[0], &val_A[0][0],
                                        ADD_VALUES));
          arr_rhs[ez][ey][ex] +=
              inter_ctx->meas_elem * inter_ctx->arr_perm_loc[0][ez][ey][ex] *
              lambda * spe10_ctx->well_index * spe10_ctx->bottomhole_pressure;
          // arr_rhs[ez][ey][ex] -= spe10_ctx->injection_rate /
          // spe10_ctx->total_active_blocks_proc_well;
        }
        if ((ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2)) {
          arr_rhs[ez][ey][ex] += spe10_ctx->injection_rate /
                                 spe10_ctx->total_active_blocks_proc_well;
        }
      }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, rhs, &arr_rhs));
  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, S_w, &arr_S_w));
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind)
    PetscCall(DMDAVecRestoreArrayRead(
        spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind], &arr_kappa[xyz_ind]));

  PetscFunctionReturn(0);
}

void get_mobi(SPE10Ctx *spe10_ctx, PetscScalar S_w_loc, PetscScalar *lambda_w,
              PetscScalar *lambda_o, PetscScalar *lambda) {
  PetscScalar S_star, K_rw_temp, K_ro_temp;
  // Formulas from the SPE10 model.
  S_star =
      (S_w_loc - spe10_ctx->S_wc) / (1.0 - spe10_ctx->S_wc - spe10_ctx->S_or);
  K_rw_temp = S_star * S_star;
  K_ro_temp = (1.0 - S_star) * (1.0 - S_star);
  if (lambda_w != NULL)
    *lambda_w = K_rw_temp / spe10_ctx->mu_w;
  if (lambda_o != NULL)
    *lambda_o = K_ro_temp / spe10_ctx->mu_o;
  if (lambda != NULL)
    *lambda = K_rw_temp / spe10_ctx->mu_w + K_ro_temp / spe10_ctx->mu_o;
}

PetscErrorCode PC_setup(PC pc) {
  PetscFunctionBeginUser;

  PCCtx *self;
  PetscCall(PCShellGetContext(pc, &self));
  if (self->S_w != NULL) {
    PetscInt xyz_ind;
    PetscScalar ***arr_kappa[DIM], ***arr_S_w;

    for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind)
      PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm,
                                    self->inter_ctx->kappa_loc[xyz_ind],
                                    &arr_kappa[xyz_ind]));
    PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm, self->S_w, &arr_S_w));

    /*******************************************************************
      Get level-1 block Jacobi.
    *******************************************************************/
    Mat A_i;
    PetscScalar avg_kappa_e, val_A[2][2];
    PetscScalar lambda;
    PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, coarse_elem,
        row[2], col[2];
    PetscInt coarse_elem_num =
        self->sub_domains * self->sub_domains * self->sub_domains;
    for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
      _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                  &startz, &nz);
      PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF, 1, nx * ny * nz,
                                  nx * ny * nz, (NEIGH + 1), NULL, &A_i));
      PetscCall(MatSetOption(A_i, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey)
          for (ex = startx; ex < startx + nx; ++ex) {
            if (ex >= startx + 1) {
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex -
                       startx - 1;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex -
                       startx - 1;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                   1.0 / arr_kappa[0][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            } else if (ex >= 1) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                   1.0 / arr_kappa[0][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }
            if (ex + 1 == startx + nx && ex + 1 != self->spe10_ctx->M) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex] +
                                   1.0 / arr_kappa[0][ez][ey][ex + 1]);
              val_A[0][0] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }

            if (ey >= starty + 1) {
              row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
                       startx;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
                       startx;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                   1.0 / arr_kappa[1][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            } else if (ey >= 1) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                   1.0 / arr_kappa[1][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }
            if (ey + 1 == starty + ny && ey + 1 != self->spe10_ctx->N) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey][ex] +
                                   1.0 / arr_kappa[1][ez][ey + 1][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }

            if (ez >= startz + 1) {
              row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
                       startx;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
                       startx;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                   1.0 / arr_kappa[2][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 2, &row[0], 2, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            } else if (ez >= 1) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                   1.0 / arr_kappa[2][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }
            if (ez + 1 == startz + nz && ez + 1 != self->spe10_ctx->P) {
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez][ey][ex] +
                                   1.0 / arr_kappa[2][ez + 1][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }
            if ((ex == 0 && ey == 0) ||
                (ex == 0 && ey == self->spe10_ctx->N - 1) ||
                (ex == self->spe10_ctx->M - 1 && ey == 0) ||
                (ex == self->spe10_ctx->M - 1 &&
                 ey == self->spe10_ctx->N - 1)) {
              get_mobi(self->spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL,
                       &lambda);
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = self->inter_ctx->meas_elem *
                            self->inter_ctx->arr_perm_loc[0][ez][ey][ex] *
                            lambda * self->spe10_ctx->well_index;
              PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                     ADD_VALUES));
            }
          }
      PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
      PetscCall(KSPSetOperators(self->ksp_lv1[coarse_elem], A_i, A_i));
      PetscCall(KSPSetUp(self->ksp_lv1[coarse_elem]));
      PetscCall(MatDestroy(&A_i));
    }

    /*******************************************************************
      Get eigenvectors from level-1 for level-2.
    *******************************************************************/
    Mat A_i_inner, M_i;
    Vec diag_M_i, dummy_ms_bases_glo;
    PetscScalar ***arr_ms_bases_c_tmp_array[self->max_eigen_num_lv1],
        ***arr_M_i_3d, ***arr_eig_vec, eig_val;
    PetscInt eigen_ind_lv1, dof, nconv;
    EPS eps;
    ST st;

    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecGetArray(self->spe10_ctx->dm,
                                self->ms_bases_c_tmp[eigen_ind_lv1],
                                &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));

    for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
      _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                  &startz, &nz);
      // 7 point stencil.
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nx * ny * nz, &diag_M_i));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz,
                                (NEIGH + 1), NULL, &A_i_inner));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nx * ny * nz, nx * ny * nz, 1,
                                NULL, &M_i));
      PetscCall(VecGetArray3d(diag_M_i, nz, ny, nx, 0, 0, 0, &arr_M_i_3d));
      for (ez = startz; ez < startz + nz; ++ez)
        for (ey = starty; ey < starty + ny; ++ey) {
          for (ex = startx; ex < startx + nx; ++ex) {
            arr_M_i_3d[ez - startz][ey - starty][ex - startx] = 0.0;
            if (ex >= startx + 1) {
              row[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex -
                       startx - 1;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty) * nx + ex -
                       startx - 1;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                   1.0 / arr_kappa[0][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_yz *
                            self->inter_ctx->meas_face_yz /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                     &val_A[0][0], ADD_VALUES));
              arr_M_i_3d[ez - startz][ey - starty][ex - startx - 1] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }

            if (ey >= starty + 1) {
              row[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
                       startx;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz) * ny * nx + (ey - starty - 1) * nx + ex -
                       startx;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                   1.0 / arr_kappa[1][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_zx *
                            self->inter_ctx->meas_face_zx /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                     &val_A[0][0], ADD_VALUES));
              arr_M_i_3d[ez - startz][ey - starty - 1][ex - startx] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }

            if (ez >= startz + 1) {
              row[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
                       startx;
              row[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] = (ez - startz - 1) * ny * nx + (ey - starty) * nx + ex -
                       startx;
              col[1] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                   1.0 / arr_kappa[2][ez][ey][ex]);
              val_A[0][0] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[0][1] = -self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][0] = -self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              val_A[1][1] = self->inter_ctx->meas_face_xy *
                            self->inter_ctx->meas_face_xy /
                            self->inter_ctx->meas_elem * avg_kappa_e;
              PetscCall(MatSetValues(A_i_inner, 2, &row[0], 2, &col[0],
                                     &val_A[0][0], ADD_VALUES));
              arr_M_i_3d[ez - startz - 1][ey - starty][ex - startx] +=
                  val_A[0][0];
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[1][1];
            }

            if ((ex == 0 && ey == 0) ||
                (ex == 0 && ey == self->spe10_ctx->N - 1) ||
                (ex == self->spe10_ctx->M - 1 && ey == 0) ||
                (ex == self->spe10_ctx->M - 1 &&
                 ey == self->spe10_ctx->N - 1)) {
              get_mobi(self->spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL,
                       &lambda);
              row[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              col[0] =
                  (ez - startz) * ny * nx + (ey - starty) * nx + ex - startx;
              val_A[0][0] = self->inter_ctx->meas_elem *
                            self->inter_ctx->arr_perm_loc[0][ez][ey][ex] *
                            lambda * self->spe10_ctx->well_index;
              PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                     &val_A[0][0], ADD_VALUES));
              arr_M_i_3d[ez - startz][ey - starty][ex - startx] += val_A[0][0];
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

      PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
      PetscCall(EPSSetOperators(eps, A_i_inner, M_i));
      PetscCall(EPSSetProblemType(eps, EPS_GHEP));
      PetscCall(EPSSetDimensions(eps, self->max_eigen_num_lv1, PETSC_DEFAULT,
                                 PETSC_DEFAULT));

      PetscCall(EPSGetST(eps, &st));
      PetscCall(STSetType(st, STSINVERT));
      PetscCall(EPSSetTarget(eps, -NIL));
      PetscCall(EPSSetOptionsPrefix(eps, "epsl1_"));
      PetscCall(EPSSetFromOptions(eps));
      PetscCall(EPSSolve(eps));
      PetscCall(EPSGetConverged(eps, &nconv));
      PetscCheck(
          nconv >= self->max_eigen_num_lv1, PETSC_COMM_WORLD, PETSC_ERR_USER,
          "SLEPc cannot find enough eigenvectors! (nconv=%d, eigen_num=%d)\n",
          nconv, self->max_eigen_num_lv1);
      for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
           ++eigen_ind_lv1) {
        dof = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1;
        PetscCall(EPSGetEigenpair(eps, eigen_ind_lv1, &eig_val, NULL,
                                  self->ms_bases_c[dof], NULL));
        if (eigen_ind_lv1 == 0)
          self->eigen_min_lv1[coarse_elem] = eig_val;

        if (eigen_ind_lv1 == self->max_eigen_num_lv1 - 1)
          self->eigen_max_lv1[coarse_elem] = eig_val;

        PetscCall(VecGetArray3d(self->ms_bases_c[dof], nz, ny, nx, 0, 0, 0,
                                &arr_eig_vec));

        for (ez = startz; ez < startz + nz; ++ez)
          for (ey = starty; ey < starty + ny; ++ey)
            PetscCall(PetscArraycpy(
                &arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][startx],
                &arr_eig_vec[ez - startz][ey - starty][0], nx));
        PetscCall(VecRestoreArray3d(self->ms_bases_c[dof], nz, ny, nx, 0, 0, 0,
                                    &arr_eig_vec));
      }
      PetscCall(EPSDestroy(&eps));
      PetscCall(MatDestroy(&A_i_inner));
      PetscCall(MatDestroy(&M_i));
    }

    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm,
                                    self->ms_bases_c_tmp[eigen_ind_lv1],
                                    &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));

    PetscCall(DMGetGlobalVector(self->spe10_ctx->dm, &dummy_ms_bases_glo));
    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1) {
      PetscCall(DMLocalToGlobal(self->spe10_ctx->dm,
                                self->ms_bases_c_tmp[eigen_ind_lv1],
                                INSERT_VALUES, dummy_ms_bases_glo));
      PetscCall(DMGlobalToLocal(self->spe10_ctx->dm, dummy_ms_bases_glo,
                                INSERT_VALUES,
                                self->ms_bases_c_tmp[eigen_ind_lv1]));
    }
    PetscCall(DMRestoreGlobalVector(self->spe10_ctx->dm, &dummy_ms_bases_glo));

    /*******************************************************************
      Get lever-2 block Jacobi.
    *******************************************************************/
    PetscInt eigen_ind_lv1_col, coarse_elem_col, coarse_elem_x, coarse_elem_y,
        coarse_elem_z;

    PetscCall(MatCreateSeqSBAIJ(
        PETSC_COMM_SELF, 1, coarse_elem_num * self->max_eigen_num_lv1,
        coarse_elem_num * self->max_eigen_num_lv1,
        (NEIGH + 1) * self->max_eigen_num_lv1, NULL, &A_i));
    PetscCall(MatSetOption(A_i, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm,
                                    self->ms_bases_c_tmp[eigen_ind_lv1],
                                    &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));
    for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
      _PC_get_coarse_elem_xyz(self, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                              &coarse_elem_z);
      _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                  &startz, &nz);
      for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
           ++eigen_ind_lv1) {
        row[0] = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1;
        for (eigen_ind_lv1_col = 0; eigen_ind_lv1_col < self->max_eigen_num_lv1;
             ++eigen_ind_lv1_col) {
          col[0] = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
          val_A[0][0] = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                if (ex >= startx + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                       1.0 / arr_kappa[0][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex - 1] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey]
                                               [ex - 1] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ex >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                       1.0 / arr_kappa[0][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ex + 1 == startx + nx && ex + 1 != self->spe10_ctx->M) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex] +
                                       1.0 / arr_kappa[0][ez][ey][ex + 1]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if (ey >= starty + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                       1.0 / arr_kappa[1][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey - 1][ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey - 1]
                                               [ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ey >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                       1.0 / arr_kappa[1][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ey + 1 == starty + ny && ey + 1 != self->spe10_ctx->N) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey][ex] +
                                       1.0 / arr_kappa[1][ez][ey + 1][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if (ez >= startz + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                       1.0 / arr_kappa[2][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez - 1][ey][ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez - 1][ey]
                                               [ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ez >= 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                       1.0 / arr_kappa[2][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ez + 1 == startz + nz && ez + 1 != self->spe10_ctx->P) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez][ey][ex] +
                                       1.0 / arr_kappa[2][ez + 1][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if ((ex == 0 && ey == 0) ||
                    (ex == 0 && ey == self->spe10_ctx->N - 1) ||
                    (ex == self->spe10_ctx->M - 1 && ey == 0) ||
                    (ex == self->spe10_ctx->M - 1 &&
                     ey == self->spe10_ctx->N - 1)) {
                  get_mobi(self->spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL,
                           &lambda);
                  val_A[0][0] +=
                      self->inter_ctx->meas_elem *
                      self->inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda *
                      self->spe10_ctx->well_index *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
              }
          PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }

        if (coarse_elem_x != 0) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x - 1;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ey = starty; ey < starty + ny; ++ey) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][startx - 1] +
                                     1.0 / arr_kappa[0][ez][ey][startx]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_yz *
                    self->inter_ctx->meas_face_yz / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][startx] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey]
                                            [startx - 1];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }

        if (coarse_elem_x != self->sub_domains - 1) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x + 1;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ey = starty; ey < starty + ny; ++ey) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[0][ez][ey][startx + nx - 1] +
                           1.0 / arr_kappa[0][ez][ey][startx + nx]);
                val_A[0][0] -= self->inter_ctx->meas_face_yz *
                               self->inter_ctx->meas_face_yz /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey]
                                                       [startx + nx - 1] *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez]
                                                       [ey][startx + nx];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }

        if (coarse_elem_y != 0) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              (coarse_elem_y - 1) * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][starty - 1][ex] +
                                     1.0 / arr_kappa[1][ez][starty][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_zx *
                    self->inter_ctx->meas_face_zx / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][starty][ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][starty - 1]
                                            [ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }

        if (coarse_elem_y != self->sub_domains - 1) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              (coarse_elem_y + 1) * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[1][ez][starty + ny - 1][ex] +
                           1.0 / arr_kappa[1][ez][starty + ny][ex]);
                val_A[0][0] -= self->inter_ctx->meas_face_zx *
                               self->inter_ctx->meas_face_zx /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez]
                                                       [starty + ny - 1][ex] *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez]
                                                       [starty + ny][ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }

        if (coarse_elem_z != 0) {
          coarse_elem_col =
              (coarse_elem_z - 1) * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][startz - 1][ey][ex] +
                                     1.0 / arr_kappa[2][startz][ey][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_xy *
                    self->inter_ctx->meas_face_xy / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][startz][ey][ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][startz - 1][ey]
                                            [ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }

        if (coarse_elem_z != self->sub_domains - 1) {
          coarse_elem_col =
              (coarse_elem_z + 1) * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[2][startz + nz - 1][ey][ex] +
                           1.0 / arr_kappa[2][startz + nz][ey][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_xy *
                    self->inter_ctx->meas_face_xy / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][startz + nz - 1][ey]
                                            [ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][startz + nz][ey]
                                            [ex];
              }
            PetscCall(MatSetValues(A_i, 1, &row[0], 1, &col[0], &val_A[0][0],
                                   INSERT_VALUES));
          }
        }
      }
    }
    PetscCall(MatAssemblyBegin(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_i, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSetOperators(self->ksp_lv2, A_i, A_i));
    PetscCall(KSPSetUp(self->ksp_lv2));
    PetscCall(MatDestroy(&A_i));

    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecRestoreArrayRead(
          self->spe10_ctx->dm, self->ms_bases_c_tmp[eigen_ind_lv1],
          &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));

    /*******************************************************************
      Get eigenvectors from lever-2 for level-3.
    *******************************************************************/
    PetscInt proc_startx, proc_nx, proc_starty, proc_ny, proc_startz, proc_nz,
        eigen_ind_lv2;
    PetscScalar ***arr_ms_bases_cc, *arr_eig_vec_1d;
    Vec eig_vec;
    PetscCall(DMDAGetCorners(self->spe10_ctx->dm, &proc_startx, &proc_starty,
                             &proc_startz, &proc_nx, &proc_ny, &proc_nz));
    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm,
                                    self->ms_bases_c_tmp[eigen_ind_lv1],
                                    &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));
    PetscCall(MatCreateSeqAIJ(
        PETSC_COMM_SELF, coarse_elem_num * self->max_eigen_num_lv1,
        coarse_elem_num * self->max_eigen_num_lv1,
        (NEIGH + 1) * self->max_eigen_num_lv1, NULL, &A_i_inner));
    for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
      _PC_get_coarse_elem_xyz(self, coarse_elem, &coarse_elem_x, &coarse_elem_y,
                              &coarse_elem_z);
      _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                  &startz, &nz);
      for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
           ++eigen_ind_lv1) {
        row[0] = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1;
        for (eigen_ind_lv1_col = 0; eigen_ind_lv1_col < self->max_eigen_num_lv1;
             ++eigen_ind_lv1_col) {
          col[0] = coarse_elem * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
          val_A[0][0] = 0.0;
          for (ez = startz; ez < startz + nz; ++ez)
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                if (ex >= startx + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                       1.0 / arr_kappa[0][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex - 1] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey]
                                               [ex - 1] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ex != proc_startx) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                       1.0 / arr_kappa[0][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ex + 1 == startx + nx && ex + 1 != proc_startx + proc_nx) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex] +
                                       1.0 / arr_kappa[0][ez][ey][ex + 1]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_yz *
                      self->inter_ctx->meas_face_yz /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if (ey >= starty + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                       1.0 / arr_kappa[1][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey - 1][ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey - 1]
                                               [ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ey != proc_starty) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                       1.0 / arr_kappa[1][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ey + 1 == starty + ny && ey + 1 != proc_starty + proc_ny) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey][ex] +
                                       1.0 / arr_kappa[1][ez][ey + 1][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_zx *
                      self->inter_ctx->meas_face_zx /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if (ez >= startz + 1) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                       1.0 / arr_kappa[2][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez - 1][ey][ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex]) *
                      (arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez - 1][ey]
                                               [ex] -
                       arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex]);
                } else if (ez != proc_startz) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                       1.0 / arr_kappa[2][ez][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
                if (ez + 1 == startz + nz && ez + 1 != proc_startz + proc_nz) {
                  avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez][ey][ex] +
                                       1.0 / arr_kappa[2][ez + 1][ey][ex]);
                  val_A[0][0] +=
                      self->inter_ctx->meas_face_xy *
                      self->inter_ctx->meas_face_xy /
                      self->inter_ctx->meas_elem * avg_kappa_e *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }

                if ((ex == 0 && ey == 0) ||
                    (ex == 0 && ey == self->spe10_ctx->N - 1) ||
                    (ex == self->spe10_ctx->M - 1 && ey == 0) ||
                    (ex == self->spe10_ctx->M - 1 &&
                     ey == self->spe10_ctx->N - 1)) {
                  get_mobi(self->spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL,
                           &lambda);
                  val_A[0][0] +=
                      self->inter_ctx->meas_elem *
                      self->inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda *
                      self->spe10_ctx->well_index *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex] *
                      arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey][ex];
                }
              }
          PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                 &val_A[0][0], INSERT_VALUES));
        }

        if (coarse_elem_x != 0) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x - 1;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ey = starty; ey < starty + ny; ++ey) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][startx - 1] +
                                     1.0 / arr_kappa[0][ez][ey][startx]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_yz *
                    self->inter_ctx->meas_face_yz / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][startx] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][ey]
                                            [startx - 1];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                   &val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_x != self->sub_domains - 1) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x + 1;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ey = starty; ey < starty + ny; ++ey) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[0][ez][ey][startx + nx - 1] +
                           1.0 / arr_kappa[0][ez][ey][startx + nx]);
                val_A[0][0] -= self->inter_ctx->meas_face_yz *
                               self->inter_ctx->meas_face_yz /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey]
                                                       [startx + nx - 1] *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez]
                                                       [ey][startx + nx];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                   &val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_y != 0) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              (coarse_elem_y - 1) * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][starty - 1][ex] +
                                     1.0 / arr_kappa[1][ez][starty][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_zx *
                    self->inter_ctx->meas_face_zx / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][starty][ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez][starty - 1]
                                            [ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                   &val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_y != self->sub_domains - 1) {
          coarse_elem_col =
              coarse_elem_z * self->sub_domains * self->sub_domains +
              (coarse_elem_y + 1) * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ez = startz; ez < startz + nz; ++ez)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[1][ez][starty + ny - 1][ex] +
                           1.0 / arr_kappa[1][ez][starty + ny][ex]);
                val_A[0][0] -= self->inter_ctx->meas_face_zx *
                               self->inter_ctx->meas_face_zx /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez]
                                                       [starty + ny - 1][ex] *
                               arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][ez]
                                                       [starty + ny][ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                   &val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_z != 0) {
          coarse_elem_col =
              (coarse_elem_z - 1) * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][startz - 1][ey][ex] +
                                     1.0 / arr_kappa[2][startz][ey][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_xy *
                    self->inter_ctx->meas_face_xy / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][startz][ey][ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][startz - 1][ey]
                                            [ex];
              }
            PetscCall(MatSetValues(A_i_inner, 1, &row[0], 1, &col[0],
                                   &val_A[0][0], INSERT_VALUES));
          }
        }

        if (coarse_elem_z != self->sub_domains - 1) {
          coarse_elem_col =
              (coarse_elem_z + 1) * self->sub_domains * self->sub_domains +
              coarse_elem_y * self->sub_domains + coarse_elem_x;
          for (eigen_ind_lv1_col = 0;
               eigen_ind_lv1_col < self->max_eigen_num_lv1;
               ++eigen_ind_lv1_col) {
            col[0] =
                coarse_elem_col * self->max_eigen_num_lv1 + eigen_ind_lv1_col;
            val_A[0][0] = 0.0;
            for (ey = starty; ey < starty + ny; ++ey)
              for (ex = startx; ex < startx + nx; ++ex) {
                avg_kappa_e =
                    2.0 / (1.0 / arr_kappa[2][startz + nz - 1][ey][ex] +
                           1.0 / arr_kappa[2][startz + nz][ey][ex]);
                val_A[0][0] -=
                    self->inter_ctx->meas_face_xy *
                    self->inter_ctx->meas_face_xy / self->inter_ctx->meas_elem *
                    avg_kappa_e *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][startz + nz - 1][ey]
                                            [ex] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1_col][startz + nz][ey]
                                            [ex];
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

    PetscCall(EPSCreate(PETSC_COMM_SELF, &eps));
    PetscCall(EPSSetOperators(eps, A_i_inner, NULL));
    PetscCall(EPSSetProblemType(eps, EPS_HEP));
    PetscCall(EPSSetDimensions(eps, self->max_eigen_num_lv2, PETSC_DEFAULT,
                               PETSC_DEFAULT));

    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(EPSSetTarget(eps, -NIL));
    PetscCall(EPSSetOptionsPrefix(eps, "epsl2_"));
    PetscCall(EPSSetFromOptions(eps));
    PetscCall(EPSSolve(eps));
    PetscCall(EPSGetConverged(eps, &nconv));
    PetscCheck(nconv >= self->max_eigen_num_lv2, PETSC_COMM_WORLD,
               PETSC_ERR_USER,
               "SLEPc cannot find enough eigenvectors for level-2! (nconv=%d, "
               "eigen_num=%d)\n",
               nconv, self->max_eigen_num_lv2);
    PetscCall(MatCreateVecs(A_i_inner, &eig_vec, NULL));

    for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
         ++eigen_ind_lv2) {
      PetscCall(
          EPSGetEigenpair(eps, eigen_ind_lv2, &eig_val, NULL, eig_vec, NULL));
      if (eigen_ind_lv2 == 0)
        self->eigen_min_lv2 = eig_val;

      if (eigen_ind_lv2 == self->max_eigen_num_lv2 - 1)
        self->eigen_max_lv2 = eig_val;

      PetscCall(VecGetArray(eig_vec, &arr_eig_vec_1d));
      // Construct coarse-coarse bases in the original P space.
      PetscCall(VecZeroEntries(self->ms_bases_cc[eigen_ind_lv2]));
      PetscCall(DMDAVecGetArray(self->spe10_ctx->dm,
                                self->ms_bases_cc[eigen_ind_lv2],
                                &arr_ms_bases_cc));

      for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
        _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty,
                                    &ny, &startz, &nz);
        for (ez = startz; ez < startz + nz; ++ez)
          for (ey = starty; ey < starty + ny; ++ey)
            for (ex = startx; ex < startx + nx; ++ex)
              for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
                   ++eigen_ind_lv1)
                arr_ms_bases_cc[ez][ey][ex] +=
                    arr_eig_vec_1d[coarse_elem * self->max_eigen_num_lv1 +
                                   eigen_ind_lv1] *
                    arr_ms_bases_c_tmp_array[eigen_ind_lv1][ez][ey][ex];
      }
      PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm,
                                    self->ms_bases_cc[eigen_ind_lv2],
                                    &arr_ms_bases_cc));
      PetscCall(VecRestoreArray(eig_vec, &arr_eig_vec_1d));
    }
    PetscCall(VecDestroy(&eig_vec));
    PetscCall(EPSDestroy(&eps));
    PetscCall(MatDestroy(&A_i_inner));

    PetscCall(DMGetGlobalVector(self->spe10_ctx->dm, &dummy_ms_bases_glo));
    for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
         ++eigen_ind_lv2) {
      PetscCall(DMLocalToGlobal(self->spe10_ctx->dm,
                                self->ms_bases_cc[eigen_ind_lv2], INSERT_VALUES,
                                dummy_ms_bases_glo));
      PetscCall(DMGlobalToLocal(self->spe10_ctx->dm, dummy_ms_bases_glo,
                                INSERT_VALUES,
                                self->ms_bases_cc[eigen_ind_lv2]));
    }
    PetscCall(DMRestoreGlobalVector(self->spe10_ctx->dm, &dummy_ms_bases_glo));

    for (eigen_ind_lv1 = 0; eigen_ind_lv1 < self->max_eigen_num_lv1;
         ++eigen_ind_lv1)
      PetscCall(DMDAVecRestoreArrayRead(
          self->spe10_ctx->dm, self->ms_bases_c_tmp[eigen_ind_lv1],
          &arr_ms_bases_c_tmp_array[eigen_ind_lv1]));

    /*******************************************************************
      Get level-3.
    *******************************************************************/
    Mat A_cc;
    PetscScalar ***arr_ms_bases_cc_array[self->max_eigen_num_lv2];
    PetscInt A_cc_range[NEIGH + 1][2], row_off, col_off;
    const PetscMPIInt *ng_ranks;
    for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
         ++eigen_ind_lv2)
      PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm,
                                    self->ms_bases_cc[eigen_ind_lv2],
                                    &arr_ms_bases_cc_array[eigen_ind_lv2]));
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, self->max_eigen_num_lv2,
                           self->max_eigen_num_lv2, PETSC_DETERMINE,
                           PETSC_DETERMINE, self->max_eigen_num_lv2, NULL,
                           NEIGH * self->max_eigen_num_lv2, NULL, &A_cc));
    // PetscCall(MatSetOption(A_cc, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
    PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
    PetscCall(DMDAGetNeighbors(self->spe10_ctx->dm, &ng_ranks));
    if (proc_startx != 0) {
      A_cc_range[LEFT][0] = ng_ranks[12] * self->max_eigen_num_lv2;
      A_cc_range[LEFT][1] = A_cc_range[LEFT][0] + self->max_eigen_num_lv2;
    }
    if (proc_startx + proc_nx != self->spe10_ctx->M) {
      A_cc_range[RIGHT][0] = ng_ranks[14] * self->max_eigen_num_lv2;
      A_cc_range[RIGHT][1] = A_cc_range[RIGHT][0] + self->max_eigen_num_lv2;
    }
    if (proc_starty != 0) {
      A_cc_range[DOWN][0] = ng_ranks[10] * self->max_eigen_num_lv2;
      A_cc_range[DOWN][1] = A_cc_range[DOWN][0] + self->max_eigen_num_lv2;
    }
    if (proc_starty + proc_ny != self->spe10_ctx->N) {
      A_cc_range[UP][0] = ng_ranks[16] * self->max_eigen_num_lv2;
      A_cc_range[UP][1] = A_cc_range[UP][0] + self->max_eigen_num_lv2;
    }
    if (proc_startz != 0) {
      A_cc_range[BACK][0] = ng_ranks[4] * self->max_eigen_num_lv2;
      A_cc_range[BACK][1] = A_cc_range[BACK][0] + self->max_eigen_num_lv2;
    }
    if (proc_startz + proc_nz != self->spe10_ctx->P) {
      A_cc_range[FRONT][0] = ng_ranks[22] * self->max_eigen_num_lv2;
      A_cc_range[FRONT][1] = A_cc_range[FRONT][0] + self->max_eigen_num_lv2;
    }
    for (row_off = 0; row_off < self->max_eigen_num_lv2; ++row_off) {
      row[0] = A_cc_range[0][0] + row_off;
      for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
        col[0] = A_cc_range[0][0] + col_off;
        val_A[0][0] = 0.0;
        for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              if (ex >= proc_startx + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                     1.0 / arr_kappa[0][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_yz *
                               self->inter_ctx->meas_face_yz /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               (arr_ms_bases_cc_array[row_off][ez][ey][ex - 1] -
                                arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                               (arr_ms_bases_cc_array[col_off][ez][ey][ex - 1] -
                                arr_ms_bases_cc_array[col_off][ez][ey][ex]);
              } else if (ex != 0) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                     1.0 / arr_kappa[0][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_yz *
                               self->inter_ctx->meas_face_yz /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }
              if (ex + 1 == proc_startx + proc_nx &&
                  ex + 1 != self->spe10_ctx->M) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex] +
                                     1.0 / arr_kappa[0][ez][ey][ex + 1]);
                val_A[0][0] += self->inter_ctx->meas_face_yz *
                               self->inter_ctx->meas_face_yz /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }

              if (ey >= proc_starty + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                     1.0 / arr_kappa[1][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_zx *
                               self->inter_ctx->meas_face_zx /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               (arr_ms_bases_cc_array[row_off][ez][ey - 1][ex] -
                                arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                               (arr_ms_bases_cc_array[col_off][ez][ey - 1][ex] -
                                arr_ms_bases_cc_array[col_off][ez][ey][ex]);
              } else if (ey != 0) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                     1.0 / arr_kappa[1][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_zx *
                               self->inter_ctx->meas_face_zx /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }
              if (ey + 1 == proc_starty + proc_ny &&
                  ey + 1 != self->spe10_ctx->N) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey][ex] +
                                     1.0 / arr_kappa[1][ez][ey + 1][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_zx *
                               self->inter_ctx->meas_face_zx /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }

              if (ez >= proc_startz + 1) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                     1.0 / arr_kappa[2][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_xy *
                               self->inter_ctx->meas_face_xy /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               (arr_ms_bases_cc_array[row_off][ez - 1][ey][ex] -
                                arr_ms_bases_cc_array[row_off][ez][ey][ex]) *
                               (arr_ms_bases_cc_array[col_off][ez - 1][ey][ex] -
                                arr_ms_bases_cc_array[col_off][ez][ey][ex]);
              } else if (ez != 0) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                     1.0 / arr_kappa[2][ez][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_xy *
                               self->inter_ctx->meas_face_xy /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }
              if (ez + 1 == proc_startz + proc_nz &&
                  ez + 1 != self->spe10_ctx->P) {
                avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez][ey][ex] +
                                     1.0 / arr_kappa[2][ez + 1][ey][ex]);
                val_A[0][0] += self->inter_ctx->meas_face_xy *
                               self->inter_ctx->meas_face_xy /
                               self->inter_ctx->meas_elem * avg_kappa_e *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }

              if ((ex == 0 && ey == 0) ||
                  (ex == 0 && ey == self->spe10_ctx->N - 1) ||
                  (ex == self->spe10_ctx->M - 1 && ey == 0) ||
                  (ex == self->spe10_ctx->M - 1 &&
                   ey == self->spe10_ctx->N - 1)) {
                get_mobi(self->spe10_ctx, arr_S_w[ez][ey][ex], NULL, NULL,
                         &lambda);
                val_A[0][0] += self->inter_ctx->meas_elem *
                               self->inter_ctx->arr_perm_loc[0][ez][ey][ex] *
                               lambda * self->spe10_ctx->well_index *
                               arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                               arr_ms_bases_cc_array[col_off][ez][ey][ex];
              }
            }
        PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                               INSERT_VALUES));
      }

      if (proc_startx != 0)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[LEFT][0] + col_off;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                   1.0 / arr_kappa[0][ez][ey][ex]);
              val_A[0][0] -= self->inter_ctx->meas_face_yz *
                             self->inter_ctx->meas_face_yz /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez][ey][ex - 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
      if (proc_startx + proc_nx != self->spe10_ctx->M)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[RIGHT][0] + col_off;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey) {
              ex = proc_startx + proc_nx - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex] +
                                   1.0 / arr_kappa[0][ez][ey][ex + 1]);
              val_A[0][0] -= self->inter_ctx->meas_face_yz *
                             self->inter_ctx->meas_face_yz /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez][ey][ex + 1];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
      if (proc_starty != 0)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[DOWN][0] + col_off;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                   1.0 / arr_kappa[1][ez][ey][ex]);
              val_A[0][0] -= self->inter_ctx->meas_face_zx *
                             self->inter_ctx->meas_face_zx /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez][ey - 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
      if (proc_starty + proc_ny != self->spe10_ctx->N)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[UP][0] + col_off;
          val_A[0][0] = 0.0;
          for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ey = proc_starty + proc_ny - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey][ex] +
                                   1.0 / arr_kappa[1][ez][ey + 1][ex]);
              val_A[0][0] -= self->inter_ctx->meas_face_zx *
                             self->inter_ctx->meas_face_zx /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez][ey + 1][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
      if (proc_startz != 0)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[BACK][0] + col_off;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                   1.0 / arr_kappa[2][ez][ey][ex]);
              val_A[0][0] -= self->inter_ctx->meas_face_xy *
                             self->inter_ctx->meas_face_xy /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez - 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
      if (proc_startz + proc_nz != self->spe10_ctx->P)
        for (col_off = 0; col_off < self->max_eigen_num_lv2; ++col_off) {
          col[0] = A_cc_range[FRONT][0] + col_off;
          val_A[0][0] = 0.0;
          for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
            for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
              ez = proc_startz + proc_nz - 1;
              avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez][ey][ex] +
                                   1.0 / arr_kappa[2][ez + 1][ey][ex]);
              val_A[0][0] -= self->inter_ctx->meas_face_xy *
                             self->inter_ctx->meas_face_xy /
                             self->inter_ctx->meas_elem * avg_kappa_e *
                             arr_ms_bases_cc_array[row_off][ez][ey][ex] *
                             arr_ms_bases_cc_array[col_off][ez + 1][ey][ex];
            }
          PetscCall(MatSetValues(A_cc, 1, &row[0], 1, &col[0], &val_A[0][0],
                                 INSERT_VALUES));
        }
    }
    for (eigen_ind_lv2 = 0; eigen_ind_lv2 < self->max_eigen_num_lv2;
         ++eigen_ind_lv2)
      PetscCall(DMDAVecRestoreArrayRead(self->spe10_ctx->dm,
                                        self->ms_bases_cc[eigen_ind_lv2],
                                        &arr_ms_bases_cc_array[eigen_ind_lv2]));
    PetscCall(MatAssemblyBegin(A_cc, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_cc, MAT_FINAL_ASSEMBLY));

    PetscCall(KSPSetOperators(self->ksp_lv3, A_cc, A_cc));
    PetscCall(KSPSetUp(self->ksp_lv3));
    PetscCall(MatDestroy(&A_cc));

    /*******************************************************************
      Final cleaning.
    *******************************************************************/
    for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind)
      PetscCall(DMDAVecRestoreArrayRead(self->spe10_ctx->dm,
                                        self->inter_ctx->kappa_loc[xyz_ind],
                                        &arr_kappa[xyz_ind]));
    PetscCall(
        DMDAVecRestoreArrayRead(self->spe10_ctx->dm, self->S_w, &arr_S_w));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PC_apply_vec(PC pc, Vec x, Vec y) {
  PetscFunctionBeginUser;

  PCCtx *self;
  Mat A;
  Vec r;
  PetscInt iter_ind_lv1, iter_ind_lv2;

  PetscCall(PCShellGetContext(pc, &self));
  PetscCall(PCGetOperators(pc, &A, NULL));
  PetscCall(DMGetGlobalVector(self->spe10_ctx->dm, &r));
  // Need to zero y first!
  PetscCall(VecZeroEntries(y));

  for (iter_ind_lv1 = 0; iter_ind_lv1 < self->smoothing_iters_lv1;
       ++iter_ind_lv1) {
    // PetscCall(MatMult(A, y, r));
    // PetscCall(VecAYPX(r, -1, x));
    PetscCall(MatResidual(A, x, y, r));
    PetscCall(_PC_apply_vec_lv1(self, r, y));
  }

  for (iter_ind_lv2 = 0; iter_ind_lv2 < self->smoothing_iters_lv2;
       ++iter_ind_lv2) {
    // PetscCall(MatMult(A, y, r));
    // PetscCall(VecAYPX(r, -1, x));
    PetscCall(MatResidual(A, x, y, r));
    PetscCall(_PC_apply_vec_lv2(self, r, y));
  }

  // PetscCall(MatMult(A, y, r));
  // PetscCall(VecAYPX(r, -1, x));
  PetscCall(MatResidual(A, x, y, r));
  PetscCall(_PC_apply_vec_lv3(self, r, y));

  for (iter_ind_lv2 = 0; iter_ind_lv2 < self->smoothing_iters_lv2;
       ++iter_ind_lv2) {
    // PetscCall(MatMult(A, y, r));
    // PetscCall(VecAYPX(r, -1, x));
    PetscCall(MatResidual(A, x, y, r));
    PetscCall(_PC_apply_vec_lv2(self, r, y));
  }

  for (iter_ind_lv1 = 0; iter_ind_lv1 < self->smoothing_iters_lv1;
       ++iter_ind_lv1) {
    // PetscCall(MatMult(A, y, r));
    // PetscCall(VecAYPX(r, -1, x));
    PetscCall(MatResidual(A, x, y, r));
    PetscCall(_PC_apply_vec_lv1(self, r, y));
  }

  PetscCall(DMRestoreGlobalVector(self->spe10_ctx->dm, &r));

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv1(PCCtx *self, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;

  PetscInt ey, ez, ex, startx, starty, startz, nx, ny, nz;
  PetscInt coarse_elem_p,
      coarse_elem_p_num =
          self->sub_domains * self->sub_domains * self->sub_domains;
  Vec x_loc, y_loc, rhs, sol;
  PetscScalar ***arr_x, ***arr_y, ***arr_rhs, ***arr_sol;

  PetscCall(DMGetLocalVector(self->spe10_ctx->dm, &x_loc));
  PetscCall(DMGlobalToLocal(self->spe10_ctx->dm, x, INSERT_VALUES, x_loc));
  PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm, x_loc, &arr_x));
  PetscCall(DMGetLocalVector(self->spe10_ctx->dm, &y_loc));
  PetscCall(VecZeroEntries(y_loc));
  PetscCall(DMDAVecGetArray(self->spe10_ctx->dm, y_loc, &arr_y));

  for (coarse_elem_p = 0; coarse_elem_p < coarse_elem_p_num; ++coarse_elem_p) {
    _PC_get_coarse_elem_corners(self, coarse_elem_p, &startx, &nx, &starty, &ny,
                                &startz, &nz);

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &rhs));
    PetscCall(VecDuplicate(rhs, &sol));
    PetscCall(VecGetArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_rhs[ez - startz][ey - starty][0],
                                &arr_x[ez][ey][startx], nx));
    PetscCall(VecRestoreArray3d(rhs, nz, ny, nx, 0, 0, 0, &arr_rhs));
    PetscCall(KSPSolve(self->ksp_lv1[coarse_elem_p], rhs, sol));
    VecDestroy(&rhs);

    PetscCall(VecGetArray3d(sol, nz, ny, nx, 0, 0, 0, &arr_sol));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        for (ex = startx; ex < startx + nx; ++ex)
          arr_y[ez][ey][ex] += arr_sol[ez - startz][ey - starty][ex - startx];

    PetscCall(VecRestoreArray3d(sol, nz, ny, nx, 0, 0, 0, &arr_sol));
    VecDestroy(&sol);
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(self->spe10_ctx->dm, x_loc, &arr_x));
  PetscCall(DMRestoreLocalVector(self->spe10_ctx->dm, &x_loc));
  PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm, y_loc, &arr_y));

  PetscCall(DMLocalToGlobal(self->spe10_ctx->dm, y_loc, ADD_VALUES, y));
  PetscCall(DMRestoreLocalVector(self->spe10_ctx->dm, &y_loc));

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv2(PCCtx *self, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;

  Vec rhs, sol, temp_loc;
  PetscScalar ***arr_x, ***arr_y, *arr_rhs, *arr_sol, ***arr_temp_loc;
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz, coarse_elem,
      coarse_elem_num =
          self->sub_domains * self->sub_domains * self->sub_domains;

  PetscCall(VecCreateSeq(PETSC_COMM_SELF,
                         coarse_elem_num * self->max_eigen_num_lv1, &rhs));
  PetscCall(VecDuplicate(rhs, &sol));

  PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm, x, &arr_x));
  PetscCall(VecGetArray(rhs, &arr_rhs));
  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(VecGetArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    for (ez = startz; ez < startz + nz; ++ez)
      for (ey = starty; ey < starty + ny; ++ey)
        PetscCall(PetscArraycpy(&arr_temp_loc[ez - startz][ey - starty][0],
                                &arr_x[ez][ey][startx], nx));
    PetscCall(VecRestoreArray3d(temp_loc, nz, ny, nx, 0, 0, 0, &arr_temp_loc));
    PetscCall(VecMDot(temp_loc, self->max_eigen_num_lv1,
                      &self->ms_bases_c[coarse_elem * self->max_eigen_num_lv1],
                      &arr_rhs[coarse_elem * self->max_eigen_num_lv1]));
    PetscCall(VecDestroy(&temp_loc));
  }
  PetscCall(VecRestoreArray(rhs, &arr_rhs));
  PetscCall(DMDAVecRestoreArrayRead(self->spe10_ctx->dm, x, &arr_x));

  PetscCall(KSPSolve(self->ksp_lv2, rhs, sol));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecGetArray(sol, &arr_sol));
  PetscCall(DMDAVecGetArray(self->spe10_ctx->dm, y, &arr_y));

  for (coarse_elem = 0; coarse_elem < coarse_elem_num; ++coarse_elem) {
    _PC_get_coarse_elem_corners(self, coarse_elem, &startx, &nx, &starty, &ny,
                                &startz, &nz);
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nz * ny * nx, &temp_loc));
    PetscCall(
        VecMAXPY(temp_loc, self->max_eigen_num_lv1,
                 &arr_sol[coarse_elem * self->max_eigen_num_lv1],
                 &self->ms_bases_c[coarse_elem * self->max_eigen_num_lv1]));
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

  PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm, y, &arr_y));

  PetscFunctionReturn(0);
}

PetscErrorCode _PC_apply_vec_lv3(PCCtx *self, Vec x, Vec y) {
  // Input x, return y, additive.
  PetscFunctionBeginUser;

  Vec rhs, sol, temp_loc;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez;
  PetscScalar *arr_rhs, *arr_sol, ***arr_x, ***arr_y, ***arr_temp_loc;

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, self->max_eigen_num_lv2,
                         PETSC_DETERMINE, &rhs));
  PetscCall(VecDuplicate(rhs, &sol));
  PetscCall(DMDAGetCorners(self->spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  PetscCall(DMGetLocalVector(self->spe10_ctx->dm, &temp_loc));
  PetscCall(VecZeroEntries(temp_loc));
  PetscCall(DMDAVecGetArray(self->spe10_ctx->dm, temp_loc, &arr_temp_loc));

  PetscCall(DMDAVecGetArrayRead(self->spe10_ctx->dm, x, &arr_x));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      PetscCall(PetscArraycpy(&arr_temp_loc[ez][ey][proc_startx],
                              &arr_x[ez][ey][proc_startx], proc_nx));
  PetscCall(DMDAVecRestoreArrayRead(self->spe10_ctx->dm, x, &arr_x));
  PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm, temp_loc, &arr_temp_loc));

  PetscCall(VecGetArray(rhs, &arr_rhs));
  PetscCall(VecMDot(temp_loc, self->max_eigen_num_lv2, &self->ms_bases_cc[0],
                    &arr_rhs[0]));
  PetscCall(VecRestoreArray(rhs, &arr_rhs));
  PetscCall(KSPSolve(self->ksp_lv3, rhs, sol));
  PetscCall(VecDestroy(&rhs));

  PetscCall(VecGetArray(sol, &arr_sol));
  PetscCall(VecZeroEntries(temp_loc));
  PetscCall(VecMAXPY(temp_loc, self->max_eigen_num_lv2, &arr_sol[0],
                     &self->ms_bases_cc[0]));
  PetscCall(VecRestoreArray(sol, &arr_sol));
  PetscCall(VecDestroy(&sol));

  PetscCall(DMDAVecGetArray(self->spe10_ctx->dm, temp_loc, &arr_temp_loc));
  PetscCall(DMDAVecGetArray(self->spe10_ctx->dm, y, &arr_y));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex)
        arr_y[ez][ey][ex] += arr_temp_loc[ez][ey][ex];
  PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm, temp_loc, &arr_temp_loc));
  PetscCall(DMRestoreLocalVector(self->spe10_ctx->dm, &temp_loc));

  PetscCall(DMDAVecRestoreArray(self->spe10_ctx->dm, y, &arr_y));

  PetscFunctionReturn(0);
}

PetscErrorCode get_next_S_w(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec P,
                            PetscScalar DS_max, PetscScalar *delta_t, Vec S_w) {
  PetscFunctionBeginUser;

  Vec temp_rhs, P_loc, S_w_loc;
  PetscScalar ***arr_temp_rhs, ***arr_P, ***arr_S_w, ***arr_kappa[DIM], lambda,
      lambda_w, f_w, lambda_, lambda_w_, f_w_, avg_kappa_e;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez, xyz_ind;

  PetscCall(DMGetGlobalVector(spe10_ctx->dm, &temp_rhs));
  PetscCall(VecZeroEntries(temp_rhs));
  PetscCall(DMDAVecGetArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));

  PetscCall(DMGetLocalVector(spe10_ctx->dm, &P_loc));
  PetscCall(DMGlobalToLocal(spe10_ctx->dm, P, INSERT_VALUES, P_loc));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, P_loc, &arr_P));

  PetscCall(DMGetLocalVector(spe10_ctx->dm, &S_w_loc));
  PetscCall(DMGlobalToLocal(spe10_ctx->dm, S_w, INSERT_VALUES, S_w_loc));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));

  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind],
                                  &arr_kappa[xyz_ind]));
  }

  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if ((ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2))
          arr_temp_rhs[ez][ey][ex] += spe10_ctx->injection_rate /
                                      spe10_ctx->total_active_blocks_proc_well /
                                      inter_ctx->meas_elem;

        get_mobi(spe10_ctx, arr_S_w[ez][ey][ex], &lambda_w, NULL, &lambda);
        f_w = lambda_w / lambda;
        if ((ex == 0 && ey == 0) || (ex == 0 && ey == spe10_ctx->N - 1) ||
            (ex == spe10_ctx->M - 1 && ey == 0) ||
            (ex == spe10_ctx->M - 1 && ey == spe10_ctx->N - 1))
          arr_temp_rhs[ez][ey][ex] +=
              inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda_w *
              spe10_ctx->well_index *
              (spe10_ctx->bottomhole_pressure - arr_P[ez][ey][ex]);

        if (ex != 0) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                               1.0 / arr_kappa[0][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey][ex - 1], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex - 1])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
        }
        if (ex + 1 != spe10_ctx->M) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex + 1] +
                               1.0 / arr_kappa[0][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey][ex + 1], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex + 1])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
        }

        if (ey != 0) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                               1.0 / arr_kappa[1][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey - 1][ex], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez][ey - 1][ex])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
        }
        if (ey + 1 != spe10_ctx->N) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey + 1][ex] +
                               1.0 / arr_kappa[1][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey + 1][ex], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez][ey + 1][ex])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
        }

        if (ez != 0) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                               1.0 / arr_kappa[2][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez - 1][ey][ex], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez - 1][ey][ex])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
        }
        if (ez + 1 != spe10_ctx->P) {
          avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez + 1][ey][ex] +
                               1.0 / arr_kappa[2][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez + 1][ey][ex], &lambda_w_, NULL,
                   &lambda_);
          f_w_ = lambda_w_ / lambda_;
          if (arr_P[ez][ey][ex] > arr_P[ez + 1][ey][ex])
            arr_temp_rhs[ez][ey][ex] +=
                f_w * avg_kappa_e *
                (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
          else
            arr_temp_rhs[ez][ey][ex] +=
                f_w_ * avg_kappa_e *
                (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
        }

        if (inter_ctx->arr_poro[ez][ey][ex] > 0.0)
          arr_temp_rhs[ez][ey][ex] /= inter_ctx->arr_poro[ez][ey][ex];
        else
          arr_temp_rhs[ez][ey][ex] = 0.0;
      }

  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, P_loc, &arr_P));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &P_loc));

  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &S_w_loc));

  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecRestoreArrayRead(
        spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind], &arr_kappa[xyz_ind]));
  }

  PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));
  PetscCall(VecMax(temp_rhs, NULL, delta_t));
  *delta_t = DS_max / *delta_t;
  if (*delta_t >= MAX_TIMESTEP)
    *delta_t = MAX_TIMESTEP;
  if (*delta_t < 0.0) {
    PetscCall(PetscErrorPrintf(
        "Water saturation is decreasing everywhere, this is impossible!\n"));
    PetscFunctionReturn(PETSC_ERR_USER);
  }

  // *delta_t = DS_max;
  PetscCall(VecAXPY(S_w, *delta_t, temp_rhs));
  PetscCall(DMRestoreGlobalVector(spe10_ctx->dm, &temp_rhs));

  PetscFunctionReturn(0);
}

PetscErrorCode get_next_S_w_multisteps(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx,
                                       Vec P, PetscInt step_num,
                                       PetscScalar DS_max, PetscScalar *delta_t,
                                       Vec S_w) {
  PetscFunctionBeginUser;

  Vec temp_rhs, P_loc, S_w_loc;
  PetscScalar ***arr_temp_rhs, ***arr_P, ***arr_S_w, ***arr_kappa[DIM], lambda,
      lambda_w, f_w, lambda_, lambda_w_, f_w_, avg_kappa_e, rhs_max = 0.0;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez, xyz_ind, step_ind;

  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  PetscCall(DMGetGlobalVector(spe10_ctx->dm, &temp_rhs));
  PetscCall(DMGetLocalVector(spe10_ctx->dm, &S_w_loc));
  PetscCall(DMGetLocalVector(spe10_ctx->dm, &P_loc));
  PetscCall(DMGlobalToLocal(spe10_ctx->dm, P, INSERT_VALUES, P_loc));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, P_loc, &arr_P));
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind],
                                  &arr_kappa[xyz_ind]));
  }

  *delta_t = 0.0;

  for (step_ind = 0; step_ind < step_num; ++step_ind) {
    PetscCall(DMGlobalToLocal(spe10_ctx->dm, S_w, INSERT_VALUES, S_w_loc));
    PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));
    PetscCall(VecZeroEntries(temp_rhs));
    PetscCall(DMDAVecGetArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));

    for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
      for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
        for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
          if ((ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2 - 1) ||
              (ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2) ||
              (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2 - 1) ||
              (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2))
            arr_temp_rhs[ez][ey][ex] +=
                spe10_ctx->injection_rate /
                spe10_ctx->total_active_blocks_proc_well / inter_ctx->meas_elem;

          get_mobi(spe10_ctx, arr_S_w[ez][ey][ex], &lambda_w, NULL, &lambda);
          f_w = lambda_w / lambda;
          if ((ex == 0 && ey == 0) || (ex == 0 && ey == spe10_ctx->N - 1) ||
              (ex == spe10_ctx->M - 1 && ey == 0) ||
              (ex == spe10_ctx->M - 1 && ey == spe10_ctx->N - 1))
            arr_temp_rhs[ez][ey][ex] +=
                inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda_w *
                spe10_ctx->well_index *
                (spe10_ctx->bottomhole_pressure - arr_P[ez][ey][ex]);

          if (ex != 0) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex - 1] +
                                 1.0 / arr_kappa[0][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez][ey][ex - 1], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex - 1])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                  spe10_ctx->H_x;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                  spe10_ctx->H_x;
          }
          if (ex + 1 != spe10_ctx->M) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[0][ez][ey][ex + 1] +
                                 1.0 / arr_kappa[0][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez][ey][ex + 1], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex + 1])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                  spe10_ctx->H_x;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                  spe10_ctx->H_x;
          }

          if (ey != 0) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey - 1][ex] +
                                 1.0 / arr_kappa[1][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez][ey - 1][ex], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez][ey - 1][ex])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                  spe10_ctx->H_y;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                  spe10_ctx->H_y;
          }
          if (ey + 1 != spe10_ctx->N) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[1][ez][ey + 1][ex] +
                                 1.0 / arr_kappa[1][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez][ey + 1][ex], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez][ey + 1][ex])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                  spe10_ctx->H_y;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                  spe10_ctx->H_y;
          }

          if (ez != 0) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez - 1][ey][ex] +
                                 1.0 / arr_kappa[2][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez - 1][ey][ex], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez - 1][ey][ex])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                  spe10_ctx->H_z;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                  spe10_ctx->H_z;
          }
          if (ez + 1 != spe10_ctx->P) {
            avg_kappa_e = 2.0 / (1.0 / arr_kappa[2][ez + 1][ey][ex] +
                                 1.0 / arr_kappa[2][ez][ey][ex]);
            get_mobi(spe10_ctx, arr_S_w[ez + 1][ey][ex], &lambda_w_, NULL,
                     &lambda_);
            f_w_ = lambda_w_ / lambda_;
            if (arr_P[ez][ey][ex] > arr_P[ez + 1][ey][ex])
              arr_temp_rhs[ez][ey][ex] +=
                  f_w * avg_kappa_e *
                  (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                  spe10_ctx->H_z;
            else
              arr_temp_rhs[ez][ey][ex] +=
                  f_w_ * avg_kappa_e *
                  (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                  spe10_ctx->H_z;
          }

          if (inter_ctx->arr_poro[ez][ey][ex] > 0.0)
            arr_temp_rhs[ez][ey][ex] /= inter_ctx->arr_poro[ez][ey][ex];
          else
            arr_temp_rhs[ez][ey][ex] = 0.0;
        }

    PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));
    PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));

    PetscCall(VecMax(temp_rhs, NULL, &rhs_max));
    if (rhs_max < 0.0) {
      PetscCall(PetscErrorPrintf(
          "Water saturation is decreasing everywhere, this is impossible!\n"));
      PetscFunctionReturn(PETSC_ERR_USER);
    }
    PetscCall(VecAXPY(S_w, DS_max / rhs_max, temp_rhs));
    *delta_t += DS_max / rhs_max;
  }

  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, P_loc, &arr_P));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &P_loc));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &S_w_loc));
  for (xyz_ind = 0; xyz_ind < DIM; ++xyz_ind) {
    PetscCall(DMDAVecRestoreArrayRead(
        spe10_ctx->dm, inter_ctx->kappa_loc[xyz_ind], &arr_kappa[xyz_ind]));
  }
  PetscCall(DMRestoreGlobalVector(spe10_ctx->dm, &temp_rhs));

  PetscFunctionReturn(0);
}

PetscErrorCode get_init_S_w(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec S_w) {
  PetscFunctionBeginUser;

  PetscScalar ***arr_S_w;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez;
  PetscCall(DMDAVecGetArray(spe10_ctx->dm, S_w, &arr_S_w));

  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        arr_S_w[ez][ey][ex] = spe10_ctx->S_wi;
      }

  PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, S_w, &arr_S_w));

  PetscFunctionReturn(0);
}

PetscErrorCode get_modified_poro(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx) {
  PetscFunctionBeginUser;

  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez;

  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if (inter_ctx->arr_poro[ez][ey][ex] < PORO_MIN)
          inter_ctx->arr_poro[ez][ey][ex] = PORO_MIN;
      }

  PetscFunctionReturn(0);
}

#define CELL_LEN 16
PetscErrorCode create_cross_kappa(PCCtx *s_ctx, PetscInt cr) {
  PetscFunctionBeginUser;
  PetscInt startx, nx, ex, ex_r, starty, ny, ey, ey_r, startz, nz, ez, ez_r, i;
  PetscCall(DMDAGetCorners(s_ctx->spe10_ctx->dm, &startx, &starty, &startz, &nx,
                           &ny, &nz));
  PetscScalar ***arr_kappa_array[DIM];
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecGetArray(s_ctx->spe10_ctx->dm,
                              s_ctx->inter_ctx->kappa_loc[i],
                              &arr_kappa_array[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    ez_r = ez % CELL_LEN;
    for (ey = starty; ey < starty + ny; ++ey) {
      ey_r = ey % CELL_LEN;
      for (ex = startx; ex < startx + nx; ++ex) {
        ex_r = ex % CELL_LEN;
        if ((ex_r >= 3 && ex_r < 5 && ey_r >= 3 && ey_r < 5) ||
            (ex_r >= 3 && ex_r < 5 && ez_r >= 3 && ez_r < 5) ||
            (ey_r >= 3 && ey_r < 5 && ez_r >= 3 && ez_r < 5)) {
          arr_kappa_array[0][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[1][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[2][ez][ey][ex] = PetscPowInt(10.0, cr);
        } else {
          arr_kappa_array[0][ez][ey][ex] = 1.0;
          arr_kappa_array[1][ez][ey][ex] = 1.0;
          arr_kappa_array[2][ez][ey][ex] = 1.0;
        }
      }
    }
  }
  for (i = 0; i < DIM; ++i)
    PetscCall(DMDAVecRestoreArray(s_ctx->spe10_ctx->dm,
                                  s_ctx->inter_ctx->kappa_loc[i],
                                  &arr_kappa_array[i]));

  PetscFunctionReturn(0);
}

PetscErrorCode get_next_S_w_sig(SPE10Ctx *spe10_ctx, INTERCtx *inter_ctx, Vec P,
                                PetscScalar delta_t, Vec S_w,
                                PetscScalar *delta_S_w) {
  PetscFunctionBeginUser;

  Vec temp_rhs, P_loc, S_w_loc;
  PetscScalar ***arr_temp_rhs, ***arr_P, ***arr_S_w, lambda_w, lambda_w_,
      avg_kappa_e;
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez;

  PetscCall(DMGetGlobalVector(spe10_ctx->dm, &temp_rhs));
  PetscCall(VecZeroEntries(temp_rhs));
  PetscCall(DMDAVecGetArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));

  PetscCall(DMGetLocalVector(spe10_ctx->dm, &P_loc));
  PetscCall(DMGlobalToLocal(spe10_ctx->dm, P, INSERT_VALUES, P_loc));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, P_loc, &arr_P));

  PetscCall(DMGetLocalVector(spe10_ctx->dm, &S_w_loc));
  PetscCall(DMGlobalToLocal(spe10_ctx->dm, S_w, INSERT_VALUES, S_w_loc));
  PetscCall(DMDAVecGetArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));

  PetscCall(DMDAGetCorners(spe10_ctx->dm, &proc_startx, &proc_starty,
                           &proc_startz, &proc_nx, &proc_ny, &proc_nz));

  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if ((ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 - 1 && ey == spe10_ctx->N / 2) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2 - 1) ||
            (ex == spe10_ctx->M / 2 && ey == spe10_ctx->N / 2))
          arr_temp_rhs[ez][ey][ex] += spe10_ctx->injection_rate /
                                      spe10_ctx->total_active_blocks_proc_well /
                                      inter_ctx->meas_elem;

        get_mobi(spe10_ctx, arr_S_w[ez][ey][ex], &lambda_w, NULL, NULL);
        if ((ex == 0 && ey == 0) || (ex == 0 && ey == spe10_ctx->N - 1) ||
            (ex == spe10_ctx->M - 1 && ey == 0) ||
            (ex == spe10_ctx->M - 1 && ey == spe10_ctx->N - 1))
          arr_temp_rhs[ez][ey][ex] +=
              inter_ctx->arr_perm_loc[0][ez][ey][ex] * lambda_w *
              spe10_ctx->well_index *
              (spe10_ctx->bottomhole_pressure - arr_P[ez][ey][ex]);

        if (ex != 0) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[0][ez][ey][ex - 1] +
                     1.0 / inter_ctx->arr_perm_loc[0][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey][ex - 1], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex - 1])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez][ey][ex - 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
        }
        if (ex + 1 != spe10_ctx->M) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[0][ez][ey][ex + 1] +
                     1.0 / inter_ctx->arr_perm_loc[0][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey][ex + 1], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez][ey][ex + 1])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez][ey][ex + 1] - arr_P[ez][ey][ex]) / spe10_ctx->H_x /
                spe10_ctx->H_x;
        }

        if (ey != 0) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[1][ez][ey - 1][ex] +
                     1.0 / inter_ctx->arr_perm_loc[1][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey - 1][ex], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez][ey - 1][ex])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez][ey - 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
        }
        if (ey + 1 != spe10_ctx->N) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[1][ez][ey + 1][ex] +
                     1.0 / inter_ctx->arr_perm_loc[1][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez][ey + 1][ex], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez][ey + 1][ex])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez][ey + 1][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_y /
                spe10_ctx->H_y;
        }

        if (ez != 0) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[2][ez - 1][ey][ex] +
                     1.0 / inter_ctx->arr_perm_loc[2][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez - 1][ey][ex], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez - 1][ey][ex])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez - 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
        }
        if (ez + 1 != spe10_ctx->P) {
          avg_kappa_e =
              2.0 / (1.0 / inter_ctx->arr_perm_loc[2][ez + 1][ey][ex] +
                     1.0 / inter_ctx->arr_perm_loc[2][ez][ey][ex]);
          get_mobi(spe10_ctx, arr_S_w[ez + 1][ey][ex], &lambda_w_, NULL, NULL);
          if (arr_P[ez][ey][ex] > arr_P[ez + 1][ey][ex])
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w *
                (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
          else
            arr_temp_rhs[ez][ey][ex] +=
                avg_kappa_e * lambda_w_ *
                (arr_P[ez + 1][ey][ex] - arr_P[ez][ey][ex]) / spe10_ctx->H_z /
                spe10_ctx->H_z;
        }

        if (inter_ctx->arr_poro[ez][ey][ex] > 0.0)
          arr_temp_rhs[ez][ey][ex] /= inter_ctx->arr_poro[ez][ey][ex];
        else
          arr_temp_rhs[ez][ey][ex] = 0.0;
      }

  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, P_loc, &arr_P));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &P_loc));

  PetscCall(DMDAVecRestoreArrayRead(spe10_ctx->dm, S_w_loc, &arr_S_w));
  PetscCall(DMRestoreLocalVector(spe10_ctx->dm, &S_w_loc));

  PetscCall(DMDAVecRestoreArray(spe10_ctx->dm, temp_rhs, &arr_temp_rhs));
  PetscCall(VecMax(temp_rhs, NULL, delta_S_w));
  *delta_S_w *= delta_t;

  PetscCall(VecAXPY(S_w, delta_t, temp_rhs));
  PetscCall(DMRestoreGlobalVector(spe10_ctx->dm, &temp_rhs));

  PetscFunctionReturn(0);
}
