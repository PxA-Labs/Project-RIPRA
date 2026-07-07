#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rippra/recon.h"

static int ntests = 0, npass = 0;

static void test_check(const char *label, int cond)
{
    ntests++;
    if (cond) { npass++; printf("  PASS %s\n", label); }
    else      { printf("  FAIL %s\n", label); }
}

/*
 * Set up a minimal calibration + zonal mesh for closed-loop tests.
 * Uses a single sub-aperture at the pupil centre.
 */
static int setup_mesh(rippra_zonal_mesh *mesh, rippa_config *cfg)
{
    cfg->camera_pixsize = 7.4e-6;
    cfg->frame_width = 648;
    cfg->frame_height = 492;
    cfg->totlenses = 140;
    cfg->flength = 18e-3;
    cfg->pitch = 300e-6;
    cfg->sa_radius = 150e-6;
    cfg->pupil_radius = 2e-3;
    cfg->wavelength = 632.8e-9;
    cfg->thresh_binary = 0.3;
    cfg->centroid_percent = 0.5;
    cfg->coarse_grid_radius = 12;
    cfg->zernike_nmax = 2;

    rippra_subap sa;
    sa.col_min = 312; sa.col_max = 336;
    sa.row_min = 234; sa.row_max = 258;
    sa.ref_cx = 324.0; sa.ref_cy = 246.0;

    rippra_calibration cal;
    cal.nspots = 1;
    cal.subaps = &sa;
    cal.pupil_cx = 324.0;
    cal.pupil_cy = 246.0;
    cal.width = 648;
    cal.height = 492;
    cal.pitch_px = 40.5;

    return rippra_zonal_setup(&cal, cfg, mesh);
}

static void test_closed_loop_convergent(void)
{
    printf("\ntest_closed_loop_convergent:\n");
    rippa_config cfg = {0};
    rippra_zonal_mesh mesh;
    int ret = setup_mesh(&mesh, &cfg);
    test_check("setup_ok", ret == 0);
    if (ret != 0) return;

    int nnodes = mesh.nnodes;
    double *initial_phase = (double *)calloc(nnodes, sizeof(double));
    double *dm_commands = (double *)calloc(nnodes, sizeof(double));

    /* Create a phase with a single peak at node 0 */
    for (int i = 0; i < nnodes; i++)
        initial_phase[i] = (i == 0) ? 1.0e-6 : 0.0;

    int out_iters = 0;
    double out_rms = 0.0;
    ret = rippra_closed_loop_run_impl(initial_phase, nnodes, &mesh, &cfg,
                                      dm_commands, 0.5, 20, 1e-10,
                                      &out_iters, &out_rms);
    test_check("converged", ret == 0);
    test_check("iters_within_max", out_iters < 20);
    test_check("rms_below_target", out_rms <= 1e-10);
    test_check("dm_commands_finite", isfinite(dm_commands[0]));

    /* Residual should be near zero */
    double *residual = (double *)malloc(nnodes * sizeof(double));
    rippra_dm_apply_impl(dm_commands, nnodes, &mesh, &cfg, initial_phase, residual);
    double rmax = 0.0;
    for (int i = 0; i < nnodes; i++)
        if (fabs(residual[i]) > rmax) rmax = fabs(residual[i]);
    test_check("residual_near_zero", rmax < 1e-8);

    free(initial_phase);
    free(dm_commands);
    free(residual);
    rippra_zonal_free(&mesh);
}

static void test_closed_loop_divergent(void)
{
    printf("\ntest_closed_loop_divergent:\n");
    rippa_config cfg = {0};
    rippra_zonal_mesh mesh;
    int ret = setup_mesh(&mesh, &cfg);
    test_check("setup_ok", ret == 0);
    if (ret != 0) return;

    int nnodes = mesh.nnodes;
    double *initial_phase = (double *)calloc(nnodes, sizeof(double));
    double *dm_commands = (double *)calloc(nnodes, sizeof(double));

    for (int i = 0; i < nnodes; i++)
        initial_phase[i] = (i == 0) ? 1.0e-6 : 0.0;

    /* Over-amplified gain should prevent convergence */
    int out_iters = 0;
    double out_rms = 0.0;
    ret = rippra_closed_loop_run_impl(initial_phase, nnodes, &mesh, &cfg,
                                      dm_commands, 2.0, 20, 1e-10,
                                      &out_iters, &out_rms);
    test_check("returned_max_iter", ret == 1);
    test_check("iters_equals_max", out_iters == 20);
    test_check("rms_not_converged", out_rms > 1e-10);

    free(initial_phase);
    free(dm_commands);
    rippra_zonal_free(&mesh);
}

int main(void)
{
    printf("=== Closed-Loop Convergence Tests ===\n");

    test_closed_loop_convergent();
    test_closed_loop_divergent();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
