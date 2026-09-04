/*
 * test_dm_safety.c — Unit tests for Deformable Mirror safety clamping & park
 *
 * Verifies:
 *   1. Disabled clamping (dm_max_stroke = 0) passes commands through unchanged.
 *   2. Commands within bounds return RIPPRA_DM_OK.
 *   3. Out-of-bounds commands with saturation <= park_threshold clamp to ±max_stroke
 *      and return RIPPRA_DM_SATURATED.
 *   4. Severe saturation exceeding park_threshold triggers rippra_dm_park()
 *      and returns RIPPRA_DM_PARKED (all zeros).
 *   5. Exact boundary behavior for stroke limit and park threshold.
 *   6. Closed-loop integrator clamp prevents runaway accumulation.
 *   7. Direct rippra_dm_park() zeroes actuator commands.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/recon.h"
#include "rippra/io.h"

static int ntests = 0, npass = 0;

static void test_check(const char *label, int cond)
{
    ntests++;
    if (cond) { npass++; printf("  PASS %s\n", label); }
    else      { printf("  FAIL %s\n", label); }
}

/* 1. Direct DM park test */
static void test_dm_park_direct(void)
{
    printf("\ntest_dm_park_direct:\n");
    double cmds[10] = {1.0, -2.5, 3.14, 0.42, -9.9, 8.0, 7.0, -0.1, 0.0, 5.5};

    int ret = rippra_dm_park(cmds, 10);
    test_check("park_return_code", ret == 0);

    int all_zero = 1;
    for (int i = 0; i < 10; ++i) {
        if (cmds[i] != 0.0) { all_zero = 0; break; }
    }
    test_check("park_all_zeroes", all_zero);

    test_check("park_null_guard", rippra_dm_park(NULL, 10) == -1);
    test_check("park_zero_len_guard", rippra_dm_park(cmds, 0) == -1);
}

/* 2. Disabled clamping (dm_max_stroke = 0.0) */
static void test_dm_saturate_disabled(void)
{
    printf("\ntest_dm_saturate_disabled:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 0.0; /* disabled */
    cfg.dm_park_threshold = 0.30;

    double cmds[5] = {10.5, -99.9, 0.0, 500.0, -250.0};
    double orig[5];
    memcpy(orig, cmds, sizeof(cmds));

    int status = rippra_dm_saturate(cmds, 5, &cfg);
    test_check("status_ok_when_disabled", status == RIPPRA_DM_OK);

    int matches = 1;
    for (int i = 0; i < 5; ++i) {
        if (cmds[i] != orig[i]) { matches = 0; break; }
    }
    test_check("commands_unmodified_when_disabled", matches);
}

/* 3. Commands strictly within bounds */
static void test_dm_saturate_within_bounds(void)
{
    printf("\ntest_dm_saturate_within_bounds:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 2.0;
    cfg.dm_park_threshold = 0.30;

    double cmds[4] = {0.5, -1.2, 1.99, -0.01};
    double orig[4];
    memcpy(orig, cmds, sizeof(cmds));

    int status = rippra_dm_saturate(cmds, 4, &cfg);
    test_check("status_ok_within_bounds", status == RIPPRA_DM_OK);

    int matches = 1;
    for (int i = 0; i < 4; ++i) {
        if (cmds[i] != orig[i]) { matches = 0; break; }
    }
    test_check("commands_unmodified_within_bounds", matches);
}

/* 4. Mild saturation: clamped to limits */
static void test_dm_saturate_clamped(void)
{
    printf("\ntest_dm_saturate_clamped:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 2.0;
    cfg.dm_park_threshold = 0.30;

    /* 10 actuators: 2 out of bounds (20% <= 30%) */
    double cmds[10] = {0.5, 3.5, -0.8, -4.2, 0.1, 1.0, -1.5, 0.0, 0.7, -0.3};

    int status = rippra_dm_saturate(cmds, 10, &cfg);
    test_check("status_saturated", status == RIPPRA_DM_SATURATED);
    test_check("positive_clamped_to_max", fabs(cmds[1] - 2.0) < 1e-9);
    test_check("negative_clamped_to_min", fabs(cmds[3] - (-2.0)) < 1e-9);
    test_check("in_range_actuator_preserved", fabs(cmds[0] - 0.5) < 1e-9);
}

/* 5. Severe saturation: park triggered */
static void test_dm_saturate_parked(void)
{
    printf("\ntest_dm_saturate_parked:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 1.5;
    cfg.dm_park_threshold = 0.25;

    /* 10 actuators: 3 out of bounds (30% > 25% threshold) */
    double cmds[10] = {0.2, 5.0, -0.3, -2.5, 0.1, 3.0, -0.5, 0.0, 0.1, -0.2};

    int status = rippra_dm_saturate(cmds, 10, &cfg);
    test_check("status_parked", status == RIPPRA_DM_PARKED);

    int all_zero = 1;
    for (int i = 0; i < 10; ++i) {
        if (cmds[i] != 0.0) { all_zero = 0; break; }
    }
    test_check("all_actuators_zeroed_on_park", all_zero);
}

/* 6. Exact boundary conditions */
static void test_dm_exact_boundaries(void)
{
    printf("\ntest_dm_exact_boundaries:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 2.0;
    cfg.dm_park_threshold = 0.30;

    /* Exact boundary strokes: exactly +2.0 and -2.0 should NOT saturate */
    double cmds[4] = {2.0, -2.0, 1.5, -1.0};
    int status = rippra_dm_saturate(cmds, 4, &cfg);
    test_check("exact_stroke_limit_is_ok", status == RIPPRA_DM_OK);
    test_check("exact_stroke_val_preserved", cmds[0] == 2.0 && cmds[1] == -2.0);

    /* Exact threshold fraction: exactly 3/10 (30%) <= 30% should clamp, NOT park (strict >) */
    double cmds10[10] = {5.0, 5.0, -5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
    int status_thresh = rippra_dm_saturate(cmds10, 10, &cfg);
    test_check("exact_threshold_clamps_not_parks", status_thresh == RIPPRA_DM_SATURATED);
    test_check("commands_clamped_not_zeroed", cmds10[0] == 2.0 && cmds10[3] == 0.1);
}

/* 7. Closed loop accumulation clamp */
static void test_closed_loop_clamp(void)
{
    printf("\ntest_closed_loop_clamp:\n");
    rippa_config cfg = {0};
    cfg.camera_pixsize = 7.4e-6;
    cfg.frame_width = 648;
    cfg.frame_height = 492;
    cfg.totlenses = 140;
    cfg.flength = 18e-3;
    cfg.pitch = 300e-6;
    cfg.sa_radius = 150e-6;
    cfg.pupil_radius = 2e-3;
    cfg.wavelength = 632.8e-9;
    cfg.thresh_binary = 0.3;
    cfg.centroid_percent = 0.5;
    cfg.coarse_grid_radius = 12;
    cfg.zernike_nmax = 2;
    cfg.coupling = 0.15;
    cfg.dm_max_stroke = 1.0;
    cfg.dm_park_threshold = 0.80;

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

    rippra_zonal_mesh mesh;
    int ret = rippra_zonal_setup(&cal, &cfg, &mesh);
    test_check("mesh_setup_ok", ret == 0);
    if (ret != 0) return;

    double *cmds = (double *)calloc(mesh.nnodes, sizeof(double));
    double *huge_phase = (double *)malloc(mesh.nnodes * sizeof(double));
    for (int i = 0; i < mesh.nnodes; ++i) huge_phase[i] = 1000.0;

    /* Repeated steps with massive input phase would cause runaway divergence without clamp */
    for (int step = 0; step < 5; ++step) {
        rippra_closed_loop_step_impl(huge_phase, mesh.nnodes, &mesh, &cfg, cmds, 0.5);
    }

    int within_stroke = 1;
    for (int i = 0; i < mesh.nnodes; ++i) {
        if (fabs(cmds[i]) > cfg.dm_max_stroke + 1e-9) {
            within_stroke = 0;
            break;
        }
    }
    test_check("closed_loop_commands_strictly_clamped", within_stroke);

    free(cmds);
    free(huge_phase);
    free(mesh.node_u);
    free(mesh.node_v);
    free(mesh.G);
    free(mesh.Gpinv);
}

int main(void)
{
    printf("=== Test Suite: DM Safety Clamping & Park ===\n");
    test_dm_park_direct();
    test_dm_saturate_disabled();
    test_dm_saturate_within_bounds();
    test_dm_saturate_clamped();
    test_dm_saturate_parked();
    test_dm_exact_boundaries();
    test_closed_loop_clamp();

    printf("\nResults: %d / %d tests passed\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
