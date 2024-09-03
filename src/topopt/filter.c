#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "hello";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  PetscInt startx, starty, nx, ny, ex, ey, M = 9, N = 9;

  PetscReal ***arraydc, ***arrayx, ***arraydcn;
  Vec dcn, dc, x, localx, localdcn, localdc;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 1,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &dcn));
  PetscCall(DMCreateGlobalVector(dm, &dc));

  PetscCall(DMCreateLocalVector(dm, &localx));
  PetscCall(DMCreateLocalVector(dm, &localdcn));
  PetscCall(DMCreateLocalVector(dm, &localdc));

  PetscCall(VecSet(x, 0.5));
  PetscCall(VecSet(dc, 0.7));
  PetscCall(VecSet(dcn, 0.0));

  PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, localx));
  PetscCall(DMGlobalToLocal(dm, dcn, INSERT_VALUES, localdcn));
  PetscCall(DMGlobalToLocal(dm, dc, INSERT_VALUES, localdc));

  PetscCall(DMDAVecGetArrayDOF(dm, localdc, &arraydc));
  PetscCall(DMDAVecGetArrayDOF(dm, localdcn, &arraydcn));
  PetscCall(DMDAVecGetArrayDOF(dm, localx, &arrayx));

  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ex++) {
      if (ex < M - 2 && ey < N - 2 && ex > 0 && ey > 0) {
        arraydcn[ey][ex][0] =
            (0.6 * arraydc[ey][ex][0] * arrayx[ey][ex][0] +
             0.1 * arraydc[ey - 1][ex][0] * arrayx[ey - 1][ex][0] +
             0.1 * arraydc[ey + 1][ex][0] * arrayx[ey + 1][ex][0] +
             0.1 * arraydc[ey][ex - 1][0] * arrayx[ey][ex - 1][0] +
             0.1 * arraydc[ey][ex + 1][0] * arrayx[ey][ex + 1][0]) /
            arrayx[ey][ex][0];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(dm, localdc, &arraydc));
  PetscCall(DMDAVecRestoreArrayDOF(dm, localx, &arrayx));
  PetscCall(DMDAVecRestoreArrayDOF(dm, localdcn, &arraydcn));

  PetscCall(DMLocalToGlobal(dm, localdcn, INSERT_VALUES, dcn));
  PetscCall(VecDuplicate(dcn, &dc));

  PetscCall(VecView(dcn, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&dcn));
  PetscCall(VecDestroy(&localx));
  PetscCall(VecDestroy(&localdc));
  PetscCall(VecDestroy(&localdcn));

  PetscCall(PetscFinalize());
}