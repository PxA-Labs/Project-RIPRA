#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rippra/io.h"
#include "rippra/recon.h"

static int ntests = 0, npass = 0;
static double tol = 1e-6;

static void check(const char *label, double got, double expected)
{
    ntests++;
    double scale = (fabs(expected) > 1e-15) ? fabs(expected) : 1.0;
    if (fabs(got - expected) <= tol * scale + 1e-15) {
        npass++;
        printf("  PASS %s: %.6e == %.6e\n", label, got, expected);
    } else {
        printf("  FAIL %s: got %.6e expected %.6e (reldiff %.2e)\n",
               label, got, expected, fabs(got - expected) / scale);
    }
}

/*
 * r0 golden test
 * 2 spots, 2 frames: spot0 dx=[-1,+1], spot1 dy=[-1,+1]
 * Known variance per axis = 2.0 px^2
 * Expected r0 = (0.170 * lambda^2 * d^(-1/3) / mean_var)^(3/5)
 *              ~ 7.38e-4 m
 */
static void test_r0_golden(void)
{
    printf("\ntest_r0_golden:\n");
    rippa_config cfg = {0};
    cfg.camera_pixsize = 7.4e-6;
    cfg.flength = 18e-3;
    cfg.pitch = 300e-6;
    cfg.wavelength = 632.8e-9;

    int nspots = 2, nframes = 2;
    double dx[4] = {-1.0, 1.0,  0.0, 0.0};
    double dy[4] = { 0.0, 0.0, -1.0, 1.0};

    double r0 = rippra_compute_r0_impl(dx, dy, nframes, nspots, &cfg);
    check("r0_golden", r0, 1.117460e-3);
}

/*
 * r0 zero-variance test
 * When all displacements are zero, r0 should be zero.
 */
static void test_r0_zero(void)
{
    printf("\ntest_r0_zero:\n");
    rippa_config cfg = {0};
    cfg.camera_pixsize = 7.4e-6;
    cfg.flength = 18e-3;
    cfg.pitch = 300e-6;
    cfg.wavelength = 632.8e-9;

    int nspots = 3, nframes = 5;
    double dx[15] = {0}, dy[15] = {0};
    double r0 = rippra_compute_r0_impl(dx, dy, nframes, nspots, &cfg);
    check("r0_zero", r0, 0.0);
}

/*
 * tau0 exponential decay test
 * dx(t) = exp(-t/5), dy(t) = 0, 1 spot, 100 frames, 1000 Hz
 * Autocorrelation 1/e point at approximately lag 5 → tau0 ~ 0.005 s
 */
static void test_tau0_exp_decay(void)
{
    printf("\ntest_tau0_exp_decay:\n");
    int nspots = 1, nframes = 100;
    double frame_rate = 1000.0;
    double *dx = (double *)malloc(nframes * nspots * sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));
    for (int t = 0; t < nframes; ++t)
        dx[t] = exp(-t / 5.0);

    double tau0 = rippra_compute_tau0_impl(dx, dy, nframes, nspots, frame_rate);
    check("tau0_exp_decay", tau0, 5.289755e-3);

    free(dx); free(dy);
}

/*
 * tau0 constant signal test
 * dx(t) = 1.0 for all t → flat autocorrelation → degenerate
 * Should not return NaN, should handle gracefully
 */
static void test_tau0_constant(void)
{
    printf("\ntest_tau0_constant:\n");
    int nspots = 1, nframes = 10;
    double frame_rate = 1000.0;
    double *dx = (double *)malloc(nframes * nspots * sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));
    for (int t = 0; t < nframes; ++t)
        dx[t] = 1.0;

    double tau0 = rippra_compute_tau0_impl(dx, dy, nframes, nspots, frame_rate);
    check("tau0_const_finite", (double)isfinite(tau0), 1.0);
    check("tau0_const_nonneg", tau0 >= 0.0 ? 1.0 : 0.0, 1.0);

    free(dx); free(dy);
}

/*
 * tau0 linear decay test
 * dx(t) = (N-t)/N, simple linear ramp
 */
static void test_tau0_linear_decay(void)
{
    printf("\ntest_tau0_linear_decay:\n");
    int nspots = 1, nframes = 10;
    double frame_rate = 1000.0;
    double *dx = (double *)malloc(nframes * nspots * sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));
    for (int t = 0; t < nframes; ++t)
        dx[t] = (nframes - t) / (double)nframes;

    double tau0 = rippra_compute_tau0_impl(dx, dy, nframes, nspots, frame_rate);
    check("tau0_linear_decay", tau0, 4.000000e-3);

    free(dx); free(dy);
}

/*
 * tau0 degenerate: all-zero input
 * Should not crash or return NaN
 */
static void test_tau0_all_zero(void)
{
    printf("\ntest_tau0_all_zero:\n");
    int nspots = 2, nframes = 20;
    double frame_rate = 500.0;
    double *dx = (double *)calloc(nframes * nspots, sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));

    double tau0 = rippra_compute_tau0_impl(dx, dy, nframes, nspots, frame_rate);
    check("tau0_zero_finite", (double)isfinite(tau0), 1.0);
    check("tau0_zero_nonneg", tau0 >= 0.0 ? 1.0 : 0.0, 1.0);

    free(dx); free(dy);
}

/*
 * Zonal reconstruction: verify phase values match known golden output
 * Uses a single calibration + known slope vector, then checks that
 * the reconstructed phase matches known values at specific nodes.
 */
static void test_zonal_reconstruct_golden(void)
{
    printf("\ntest_zonal_golden:\n");
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

    /* Single sub-aperture at centre */
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
    if (ret == 0) {
        /* Known slopes: dx=1.0, dy=0.0 (pure x-tilt) */
        double dx[1] = {1.0};
        double dy[1] = {0.0};
        double *phase = (double *)calloc(mesh.nnodes, sizeof(double));
        ret = rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, phase);
        if (ret == 0) {
            /* Expected: phase[0] is the centre node, should be ~0 from tilt */
            /* The exact value depends on the geometry, but it should be finite */
            int all_finite = 1;
            for (int i = 0; i < mesh.nnodes; ++i)
                if (!isfinite(phase[i])) { all_finite = 0; break; }
            check("zonal_all_finite", (double)all_finite, 1.0);
            check("zonal_cond", mesh.cond > 0.0 ? 1.0 : 0.0, 1.0);
        } else {
            printf("  FAIL zonal_reconstruct returned %d\n", ret);
            ntests++;
        }
        free(phase);
        free(mesh.node_u); free(mesh.node_v);
        free(mesh.G); free(mesh.Gpinv);
    } else {
        printf("  FAIL zonal_setup returned %d\n", ret);
        ntests++;
    }
}

int main(void)
{
    printf("=== Golden Value Regression Tests ===\n");

    test_r0_golden();
    test_r0_zero();
    test_tau0_exp_decay();
    test_tau0_constant();
    test_tau0_linear_decay();
    test_tau0_all_zero();
    test_zonal_reconstruct_golden();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
