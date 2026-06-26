/*
 * tests/test_full_pipeline.c - End-to-end integration test
 *
 * Loads a reference flat frame and an aberrated frame,
 * runs calibration -> centroiding -> zonal/modal reconstruction ->
 * turbulence characterization -> DM mapping, and validates all outputs.
 *
 * Run from the rippra/ directory (same as test_recon).
 */

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/recon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int test_count = 0, pass_count = 0;

#define TEST(cond, msg) do { \
    test_count++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [%d] %s\n", test_count, msg); \
    } else { \
        pass_count++; \
        printf("PASS [%d] %s\n", test_count, msg); \
    } \
} while(0)

int main(void)
{
    rippa_config cfg;
    double *flat_frame = NULL, *img_frame = NULL;
    rippra_calibration cal;
    rippra_zonal_mesh zmesh;
    rippra_modal_model mmodel;
    double *dx = NULL, *dy = NULL, *phase = NULL, *coeffs = NULL;
    double *dm_cmds = NULL;
    int ret, nspots, nmodes, i;

    printf("=== RIPRA Full Pipeline Integration Test ===\n\n");

    /* ---- Step 0: Load config ---- */
    ret = rippa_config_load(&cfg, "config/system.conf");
    TEST(ret == 0, "Load config");
    if (ret != 0) return 1;
    printf("Config: %dx%d, %d lenses, nmax=%d\n",
           cfg.frame_width, cfg.frame_height,
           cfg.totlenses, cfg.zernike_nmax);

    /* ---- Step 1: Load frames ---- */
    ret = rippa_load_raw("data_raw/sh_flat.raw", cfg.frame_width, cfg.frame_height, &flat_frame);
    TEST(ret == 0, "Load flat frame");
    if (ret != 0) goto cleanup;

    ret = rippa_load_raw("data_raw/img.raw", cfg.frame_width, cfg.frame_height, &img_frame);
    TEST(ret == 0, "Load aberrated frame");
    if (ret != 0) goto cleanup;

    double fsum = 0.0, fmin = flat_frame[0], fmax = flat_frame[0];
    for (i = 0; i < cfg.frame_width * cfg.frame_height; i++) {
        fsum += flat_frame[i];
        if (flat_frame[i] < fmin) fmin = flat_frame[i];
        if (flat_frame[i] > fmax) fmax = flat_frame[i];
    }
    TEST(fsum > 0.0, "Flat frame has non-zero sum");
    TEST(fmax > fmin, "Flat frame has dynamic range");
    printf("  Flat frame: [%.4f, %.4f], sum=%.1f\n", fmin, fmax, fsum);

    /* ---- Step 2: Calibrate grid ---- */
    memset(&cal, 0, sizeof(cal));
    ret = rippa_calibrate_grid(flat_frame, cfg.frame_width, cfg.frame_height, &cfg, &cal);
    TEST(ret == 0, "Grid calibration");
    if (ret != 0) goto cleanup;
    nspots = cal.nspots;
    TEST(nspots > 0, "Spots detected");
    TEST(nspots <= cfg.totlenses, "nspots <= totlenses");
    printf("  Detected %d sub-apertures\n", nspots);

    int cx_ok = 1, cy_ok = 1;
    for (i = 0; i < nspots; i++) {
        if (cal.subaps[i].ref_cx < 0 || cal.subaps[i].ref_cx >= cfg.frame_width) cx_ok = 0;
        if (cal.subaps[i].ref_cy < 0 || cal.subaps[i].ref_cy >= cfg.frame_height) cy_ok = 0;
    }
    TEST(cx_ok, "Ref centroids within X bounds");
    TEST(cy_ok, "Ref centroids within Y bounds");

    /* ---- Step 3: Compute centroids on aberrated frame ---- */
    dx = (double*)malloc(nspots * sizeof(double));
    dy = (double*)malloc(nspots * sizeof(double));
    double *tmp_cx = (double*)malloc(nspots * sizeof(double));
    double *tmp_cy = (double*)malloc(nspots * sizeof(double));
    ret = rippa_compute_centroids(img_frame, cfg.frame_width, cfg.frame_height, &cal, &cfg, tmp_cx, tmp_cy);
    TEST(ret == 0, "Centroid computation");
    if (ret == 0) {
        rippa_compute_deltas(tmp_cx, tmp_cy, &cal, nspots, dx, dy);
        int deltas_ok = 1;
        for (i = 0; i < nspots; i++)
            if (!isfinite(dx[i]) || !isfinite(dy[i])) deltas_ok = 0;
        TEST(deltas_ok, "Spot deltas are finite");
        double dx_max = 0, dy_max = 0;
        for (i = 0; i < nspots; i++) {
            if (fabs(dx[i]) > dx_max) dx_max = fabs(dx[i]);
            if (fabs(dy[i]) > dy_max) dy_max = fabs(dy[i]);
        }
        printf("  Max |dx| = %.3f px, max |dy| = %.3f px\n", dx_max, dy_max);
    }
    free(tmp_cx); free(tmp_cy);

    /* ---- Step 4: Zonal reconstruction ---- */
    memset(&zmesh, 0, sizeof(zmesh));
    ret = rippra_zonal_setup(&cal, &cfg, &zmesh);
    TEST(ret == 0, "Zonal mesh setup");
    if (ret == 0) {
        phase = (double*)malloc(zmesh.nnodes * sizeof(double));
        ret = rippra_zonal_reconstruct(&zmesh, dx, dy, &cfg, phase);
        TEST(ret == 0, "Zonal reconstruction");
        double p_min = phase[0], p_max = phase[0];
        for (i = 0; i < zmesh.nnodes; i++) {
            if (phase[i] < p_min) p_min = phase[i];
            if (phase[i] > p_max) p_max = phase[i];
        }
        TEST(p_max - p_min > 0.0, "Phase has non-zero variation");
        printf("  Zonal phase: [%.4e, %.4e] rad, PV=%.4e rad\n", p_min, p_max, p_max - p_min);
    }

    /* ---- Step 5: Modal reconstruction ---- */
    memset(&mmodel, 0, sizeof(mmodel));
    ret = rippra_modal_setup(&cal, &cfg, &mmodel);
    TEST(ret == 0, "Modal model setup");
    nmodes = mmodel.nmodes;
    if (ret == 0 && nmodes > 0) {
        coeffs = (double*)malloc(nmodes * sizeof(double));
        ret = rippra_modal_reconstruct(&mmodel, dx, dy, &cfg, coeffs);
        TEST(ret == 0, "Modal reconstruction");
        int coeffs_ok = 1;
        for (i = 0; i < nmodes; i++)
            if (!isfinite(coeffs[i])) coeffs_ok = 0;
        TEST(coeffs_ok, "Coefficients are finite");
        double c_max = 0;
        for (i = 0; i < nmodes; i++)
            if (fabs(coeffs[i]) > c_max) c_max = fabs(coeffs[i]);
        TEST(c_max < 50.0, "Coef magnitude < 50 rad");
        printf("  Max |coef| = %.4f rad, nmodes=%d\n", c_max, nmodes);
    }

    /* ---- Step 6: Turbulence ---- */
    {
        int nf = 50;
        double *dx_s = (double*)malloc(nf * nspots * sizeof(double));
        double *dy_s = (double*)malloc(nf * nspots * sizeof(double));
        for (i = 0; i < nf * nspots; i++) {
            dx_s[i] = dx[i % nspots] * (1.0 + 0.1 * (double)(i / nspots));
            dy_s[i] = dy[i % nspots] * (1.0 + 0.1 * (double)(i / nspots));
        }
        double r0 = rippra_compute_r0(dx_s, dy_s, nf, nspots, &cfg);
        double tau0 = rippra_compute_tau0(dx_s, dy_s, nf, nspots, 1000.0);
        TEST(isfinite(r0) && r0 > 0.0, "r0 > 0");
        TEST(isfinite(tau0) && tau0 > 0.0, "tau0 > 0");
        double D_r0 = cfg.pupil_radius * 2.0 / r0;
        printf("  r0 = %.6f m, tau0 = %.6f s, D/r0 = %.2f\n", r0, tau0, D_r0);
        free(dx_s); free(dy_s);
    }

    /* ---- Step 7: DM mapping ---- */
    if (phase && zmesh.nnodes > 0) {
        int nnodes = zmesh.nnodes;
        dm_cmds = (double*)calloc(nnodes, sizeof(double));
        ret = rippra_dm_map(phase, nnodes, &zmesh, &cfg, dm_cmds);
        TEST(ret == 0, "DM map computation");
        double cmd_max = 0;
        for (i = 0; i < nnodes; i++)
            if (fabs(dm_cmds[i]) > cmd_max) cmd_max = fabs(dm_cmds[i]);
        TEST(isfinite(cmd_max), "DM commands finite");
        printf("  Max |DM cmd| = %.6f\n", cmd_max);
    }

    /* ---- Summary ---- */
    printf("\n=== Results: %d / %d tests passed ===\n", pass_count, test_count);

cleanup:
    free(flat_frame); free(img_frame); free(dx); free(dy);
    free(phase); free(coeffs); free(dm_cmds);
    rippa_calibration_free(&cal);
    rippra_zonal_free(&zmesh);
    rippra_modal_free(&mmodel);
    return (pass_count == test_count) ? 0 : 1;
}
