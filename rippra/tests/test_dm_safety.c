#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "rippra/recon.h"
#include "rippra/rippra_api.h"

static int ntests = 0, npass = 0;

static void test_check(const char *label, int cond)
{
    ntests++;
    if (cond) { npass++; printf("  PASS %s\n", label); }
    else      { printf("  FAIL %s\n", label); }
}

/*
 * Set up a minimal calibration + zonal mesh for DM safety tests.
 * Single sub-aperture produces 4 nodes (2x2 grid).
 */
static int setup_mesh(rippra_zonal_mesh *mesh, rippa_config *cfg)
{
    cfg->camera_pixsize     = 7.4e-6;
    cfg->frame_width        = 648;
    cfg->frame_height       = 492;
    cfg->totlenses          = 140;
    cfg->flength            = 18e-3;
    cfg->pitch              = 300e-6;
    cfg->sa_radius          = 150e-6;
    cfg->pupil_radius       = 2e-3;
    cfg->wavelength         = 632.8e-9;
    cfg->thresh_binary      = 0.3;
    cfg->centroid_percent   = 0.5;
    cfg->coarse_grid_radius = 12;
    cfg->zernike_nmax       = 2;
    cfg->dm_nact_x          = 12;
    cfg->dm_nact_y          = 12;
    cfg->coupling           = 0.15;
    cfg->dm_max_stroke      = 0.0;
    cfg->dm_park_threshold  = 0.30;

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

/* Test 1: rippra_dm_park zeroes all actuator commands */
static void test_dm_park(void)
{
    printf("\ntest_dm_park:\n");
    double cmds[6] = {1.2, -3.4, 0.5, 9.1, -0.01, 100.0};
    int ret = rippra_dm_park(cmds, 6);
    test_check("park_return_ok", ret == 0);
    int all_zero = 1;
    for (int i = 0; i < 6; i++) {
        if (cmds[i] != 0.0) all_zero = 0;
    }
    test_check("all_commands_zero", all_zero);

    test_check("park_null_guard", rippra_dm_park(NULL, 6) == -1);
    test_check("park_invalid_n_guard", rippra_dm_park(cmds, 0) == -1);
}

/* Test 2: Clamping when actuators exceed ±v_max and saturated fraction <= threshold */
static void test_dm_saturate_clamping(void)
{
    printf("\ntest_dm_saturate_clamping:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 2.0e-6;      /* +/- 2.0 um */
    cfg.dm_park_threshold = 0.50;    /* 50% threshold */

    /* 4 actuators: only 1 exceeds max_stroke (25% <= 50% -> clamps, does not park) */
    double cmds[4] = {1.0e-6, -2.5e-6, 0.5e-6, 1.8e-6};
    int status = -1;
    int n_sat = rippra_dm_saturate_impl(cmds, 4, &cfg, &status);

    test_check("n_sat_is_1", n_sat == 1);
    test_check("status_saturated", status == RIPPRA_DM_SATURATED);
    test_check("cmd0_unchanged", fabs(cmds[0] - 1.0e-6) < 1e-12);
    test_check("cmd1_clamped_neg", fabs(cmds[1] - (-2.0e-6)) < 1e-12);
    test_check("cmd2_unchanged", fabs(cmds[2] - 0.5e-6) < 1e-12);
    test_check("cmd3_unchanged", fabs(cmds[3] - 1.8e-6) < 1e-12);
}

/* Test 3: Automatic parking when saturated actuators fraction > threshold */
static void test_dm_saturate_park_trigger(void)
{
    printf("\ntest_dm_saturate_park_trigger:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 1.0e-6;
    cfg.dm_park_threshold = 0.30; /* 30% */

    /* 4 actuators: 2 exceed bound (50% > 30% -> trigger park) */
    double cmds[4] = {1.5e-6, -1.2e-6, 0.2e-6, 0.4e-6};
    int status = -1;
    int n_sat = rippra_dm_saturate_impl(cmds, 4, &cfg, &status);

    test_check("n_sat_is_2", n_sat == 2);
    test_check("status_parked", status == RIPPRA_DM_PARKED);
    int all_zero = 1;
    for (int i = 0; i < 4; i++) {
        if (cmds[i] != 0.0) all_zero = 0;
    }
    test_check("all_commands_parked_to_zero", all_zero);
}

/* Test 4: Boundary conditions (exactly v_max, exactly threshold) */
static void test_dm_saturate_boundary(void)
{
    printf("\ntest_dm_saturate_boundary:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 2.0e-6;
    cfg.dm_park_threshold = 0.50; /* 50% */

    /* Exactly +v_max and -v_max should not count as saturated */
    double cmds[4] = {2.0e-6, -2.0e-6, 1.0e-6, 0.0};
    int status = -1;
    int n_sat = rippra_dm_saturate_impl(cmds, 4, &cfg, &status);
    test_check("exact_vmax_no_saturation", n_sat == 0);
    test_check("exact_vmax_status_ok", status == RIPPRA_DM_OK);
    test_check("exact_vmax_values_preserved", cmds[0] == 2.0e-6 && cmds[1] == -2.0e-6);

    /* Exactly equal to threshold (2 out of 4 = 50% == 50%): strict > must NOT park */
    double cmds2[4] = {3.0e-6, -3.0e-6, 0.5e-6, 0.0};
    status = -1;
    n_sat = rippra_dm_saturate_impl(cmds2, 4, &cfg, &status);
    test_check("exact_threshold_n_sat_2", n_sat == 2);
    test_check("exact_threshold_does_not_park", status == RIPPRA_DM_SATURATED);
    test_check("exact_threshold_clamped_0", fabs(cmds2[0] - 2.0e-6) < 1e-12);
    test_check("exact_threshold_clamped_1", fabs(cmds2[1] - (-2.0e-6)) < 1e-12);
}

/* Test 5: Disabled mode (dm_max_stroke = 0.0) preserves unconstrained behaviour */
static void test_dm_saturate_disabled(void)
{
    printf("\ntest_dm_saturate_disabled:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 0.0;
    cfg.dm_park_threshold = 0.30;

    double cmds[4] = {100.0, -500.0, 20.0, 0.0};
    int status = -1;
    int n_sat = rippra_dm_saturate_impl(cmds, 4, &cfg, &status);

    test_check("disabled_n_sat_0", n_sat == 0);
    test_check("disabled_status_ok", status == RIPPRA_DM_OK);
    test_check("disabled_unmodified_cmd0", cmds[0] == 100.0);
    test_check("disabled_unmodified_cmd1", cmds[1] == -500.0);
}

/* Test 6: Non-finite commands are never allowed through, even when disabled. */
static void test_dm_saturate_nonfinite(void)
{
    printf("\ntest_dm_saturate_nonfinite:\n");
    rippa_config cfg = {0};
    cfg.dm_max_stroke = 0.0;
    cfg.dm_park_threshold = 0.30;

    double cmds[4] = {1.0, NAN, INFINITY, -2.0};
    int status = -1;
    int n_unsafe = rippra_dm_saturate_impl(cmds, 4, &cfg, &status);

    test_check("nonfinite_counted", n_unsafe == 2);
    test_check("nonfinite_status_parked", status == RIPPRA_DM_PARKED);
    int all_zero = 1;
    for (int i = 0; i < 4; i++) {
        if (cmds[i] != 0.0) all_zero = 0;
    }
    test_check("nonfinite_commands_parked", all_zero);

    double invalid_cfg_commands[2] = {1.0, -1.0};
    cfg.dm_max_stroke = NAN;
    status = RIPPRA_DM_OK;
    n_unsafe = rippra_dm_saturate_impl(invalid_cfg_commands, 2, &cfg, &status);
    test_check("nonfinite_config_returns_error", n_unsafe == -1);
    test_check("nonfinite_config_status_error", status == RIPPRA_DM_ERROR);
    test_check("nonfinite_config_commands_parked",
               invalid_cfg_commands[0] == 0.0 && invalid_cfg_commands[1] == 0.0);
}

/* Test 7: Public C API wrapper preserves the internal safety contract. */
static void test_public_dm_saturate(void)
{
    printf("\ntest_public_dm_saturate:\n");
    rippra_api_config cfg = rippra_default_config();
    cfg.dm_max_stroke = 2.0;
    cfg.dm_park_threshold = 0.5;
    double commands[4] = {1.0, -3.0, 0.5, 1.8};
    int status = RIPPRA_DM_ERROR;
    int n_unsafe = rippra_dm_saturate(commands, 4, &cfg, &status);
    test_check("public_count_is_1", n_unsafe == 1);
    test_check("public_status_is_saturated", status == RIPPRA_DM_SATURATED);
    test_check("public_command_is_clamped", fabs(commands[1] + 2.0) < 1e-12);
}

/* Test 8: Mapping applies safety once and preserves its resulting status. */
static void test_dm_map_status(void)
{
    printf("\ntest_dm_map_status:\n");
    rippa_config cfg = {0};
    rippra_zonal_mesh mesh;
    int ret = setup_mesh(&mesh, &cfg);
    test_check("map_setup_ok", ret == 0);
    if (ret != 0) return;

    cfg.dm_max_stroke = 5.0e-7;
    cfg.dm_park_threshold = 1.0; /* clamp but never park at exactly 100% */
    double target[4] = {1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5};
    double commands[4] = {0.0, 0.0, 0.0, 0.0};
    int status = RIPPRA_DM_ERROR;
    ret = rippra_dm_map_ex_impl(target, 4, &mesh, &cfg, commands, &status);
    test_check("map_returns_success", ret == 0);
    test_check("map_reports_saturation", status == RIPPRA_DM_SATURATED);
    for (int i = 0; i < 4; ++i)
        test_check("map_command_is_bounded", fabs(commands[i]) <= cfg.dm_max_stroke + 1e-12);

    rippra_zonal_free(&mesh);
}

/* Test 9: Closed-loop integration respects dm_max_stroke bounds */
static void test_dm_closed_loop_clamping(void)
{
    printf("\ntest_dm_closed_loop_clamping:\n");
    rippa_config cfg = {0};
    rippra_zonal_mesh mesh;
    int ret = setup_mesh(&mesh, &cfg);
    test_check("setup_ok", ret == 0);
    if (ret != 0) return;

    cfg.dm_max_stroke = 5.0e-7; /* 0.5 um max */
    cfg.dm_park_threshold = 0.80;

    int nnodes = mesh.nnodes;
    double *initial_phase = (double *)calloc(nnodes, sizeof(double));
    double *dm_commands = (double *)calloc(nnodes, sizeof(double));

    /* Very high initial phase */
    for (int i = 0; i < nnodes; i++) initial_phase[i] = 1.0e-5;

    int out_iters = 0;
    double out_rms = 0.0;
    ret = rippra_closed_loop_run_impl(initial_phase, nnodes, &mesh, &cfg,
                                      dm_commands, 1.0, 10, 1e-10,
                                      &out_iters, &out_rms);

    /* Commands must never exceed cfg.dm_max_stroke */
    double max_cmd = 0.0;
    for (int i = 0; i < nnodes; i++) {
        if (fabs(dm_commands[i]) > max_cmd) max_cmd = fabs(dm_commands[i]);
    }
    test_check("cl_commands_bounded", max_cmd <= cfg.dm_max_stroke + 1e-12);

    /* Already unsafe state must be sanitized before early convergence. */
    memset(dm_commands, 0, (size_t)nnodes * sizeof(double));
    for (int i = 0; i < nnodes; ++i) dm_commands[i] = 2.0e-6;
    memset(initial_phase, 0, (size_t)nnodes * sizeof(double));
    cfg.dm_park_threshold = 0.80;
    ret = rippra_closed_loop_run_impl(initial_phase, nnodes, &mesh, &cfg,
                                      dm_commands, 1.0, 1, 1e-10,
                                      &out_iters, &out_rms);
    int all_zero = 1;
    for (int i = 0; i < nnodes; ++i) {
        if (dm_commands[i] != 0.0) all_zero = 0;
    }
    test_check("early_convergence_sanitizes_existing_commands", ret == 0 && all_zero);

    free(initial_phase);
    free(dm_commands);
    rippra_zonal_free(&mesh);
}

/* Test 10: Public closed-loop wrappers use the current config, not the
 * calibration-time safety defaults, while retaining cached geometry. */
static void test_public_closed_loop_current_config(void)
{
    printf("\ntest_public_closed_loop_current_config:\n");

    rippra_api_config calibration_cfg = rippra_default_config();
    test_check("calibration_default_stroke_disabled",
               calibration_cfg.dm_max_stroke == 0.0);

    int width = calibration_cfg.frame_width;
    int height = calibration_cfg.frame_height;
    size_t npix = (size_t)width * (size_t)height;
    double *flat = (double *)calloc(npix, sizeof(double));
    if (!flat) {
        test_check("public_calibration_allocation", 0);
        return;
    }

    /* One bright 9x9 spot is sufficient for the public calibration path. */
    for (int y = height / 2 - 4; y <= height / 2 + 4; ++y) {
        for (int x = width / 2 - 4; x <= width / 2 + 4; ++x)
            flat[(size_t)y * (size_t)width + (size_t)x] = 1.0;
    }

    void *cal = rippra_calibrate(flat, width, height, &calibration_cfg);
    free(flat);
    test_check("public_calibration_succeeds", cal != NULL);
    if (!cal) return;

    /* Force the zonal geometry to be cached before changing runtime config. */
    int nnodes = rippra_calibration_nnodes(cal);
    test_check("public_geometry_cached", nnodes == 4);
    if (nnodes != 4) {
        rippra_calibration_free(cal);
        return;
    }

    rippra_api_config current_cfg = calibration_cfg;
    current_cfg.dm_max_stroke = 1.0e-6;
    current_cfg.dm_park_threshold = 1.0; /* clamp, never park at 100% */
    current_cfg.coupling = 0.75;

    double phase[4] = {1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3};
    double commands[4] = {0.0, 0.0, 0.0, 0.0};
    double residual[4] = {0.0, 0.0, 0.0, 0.0};

    /* dm_apply must use the current coupling with the cached mesh. */
    double apply_commands[4] = {0.5e-6, 0.0, 0.0, 0.0};
    double zero_phase[4] = {0.0, 0.0, 0.0, 0.0};
    int ret = rippra_dm_apply(apply_commands, nnodes, cal, &current_cfg,
                              zero_phase, residual);
    test_check("public_apply_uses_current_config", ret == 0 &&
               fabs(residual[1] - current_cfg.coupling * apply_commands[0]) < 1e-12);

    /* The single-step wrapper must bound newly accumulated commands. */
    ret = rippra_closed_loop_step(cal, phase, nnodes, &current_cfg,
                                  commands, 1.0);
    int step_bounded = (ret >= 0);
    for (int i = 0; i < nnodes; ++i) {
        if (!isfinite(commands[i]) || fabs(commands[i]) > current_cfg.dm_max_stroke + 1e-12)
            step_bounded = 0;
    }
    test_check("public_step_commands_bounded", step_bounded);

    /* The run wrapper must sanitize an unsafe caller-provided state before
     * its early-convergence return. */
    for (int i = 0; i < nnodes; ++i) commands[i] = 2.0e-3;
    double initial_phase[4] = {0.0, 0.0, 0.0, 0.0};
    int out_iters = -1;
    double out_rms = -1.0;
    ret = rippra_closed_loop_run(cal, initial_phase, nnodes, &current_cfg,
                                 commands, 1.0, 1, 1.0e9,
                                 &out_iters, &out_rms);
    int run_bounded = (ret == 0);
    for (int i = 0; i < nnodes; ++i) {
        if (!isfinite(commands[i]) || fabs(commands[i]) > current_cfg.dm_max_stroke + 1e-12)
            run_bounded = 0;
    }
    test_check("public_run_sanitizes_commands", run_bounded);

    /* Invalid public calls must fail without allowing a command buffer
     * through unsanitized. */
    for (int i = 0; i < nnodes; ++i) commands[i] = 2.0e-3;
    ret = rippra_closed_loop_step(cal, phase, nnodes, NULL, commands, 1.0);
    int step_invalid_parked = (ret == -1);
    for (int i = 0; i < nnodes; ++i) {
        if (commands[i] != 0.0) step_invalid_parked = 0;
    }
    test_check("public_step_null_config_parks", step_invalid_parked);

    for (int i = 0; i < nnodes; ++i) commands[i] = 2.0e-3;
    ret = rippra_closed_loop_run(cal, initial_phase, nnodes, NULL,
                                 commands, 1.0, 1, 1.0e9,
                                 &out_iters, &out_rms);
    int run_invalid_parked = (ret == -1);
    for (int i = 0; i < nnodes; ++i) {
        if (commands[i] != 0.0) run_invalid_parked = 0;
    }
    test_check("public_run_null_config_parks", run_invalid_parked);

    rippra_calibration_free(cal);
}

int main(void)
{
    printf("=== DM Safety Layer Tests ===\n");

    test_dm_park();
    test_dm_saturate_clamping();
    test_dm_saturate_park_trigger();
    test_dm_saturate_boundary();
    test_dm_saturate_disabled();
    test_dm_saturate_nonfinite();
    test_public_dm_saturate();
    test_dm_map_status();
    test_dm_closed_loop_clamping();
    test_public_closed_loop_current_config();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
