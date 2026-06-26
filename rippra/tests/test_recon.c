/*
 * tests/test_recon.c - test zonal/modal reconstruction, turbulence params, and DM mapping
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx, *cy, *dx, *dy;
    int rc;

    printf("=== RIPRA Reconstruction Integration Test ===\n\n");
    
    /* Load configuration */
    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) {
        printf("ERROR: Failed to load configuration (rc=%d)\n", rc);
        return 1;
    }
    
    /* Load raw data files */
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) {
        printf("ERROR: Failed to load sh_flat.raw\n");
        return 1;
    }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) {
        printf("ERROR: Failed to load img.raw\n");
        free(sh_flat);
        return 1;
    }

    /* 1. Calibrate grid and compute spot deviations */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) {
        printf("ERROR: Calibration failed (rc=%d)\n", rc);
        free(sh_flat); free(img);
        return 1;
    }
    printf("1. Grid Calibration:\n");
    printf("   Active spots: %d\n", cal.nspots);
    printf("   Estimated pitch: %.2f px\n", cal.pitch_px);
    
    cx = (double *)malloc(cal.nspots * sizeof(double));
    cy = (double *)malloc(cal.nspots * sizeof(double));
    dx = (double *)malloc(cal.nspots * sizeof(double));
    dy = (double *)malloc(cal.nspots * sizeof(double));
    
    rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy);
    
    /* 2. Zonal Phase Reconstruction (Fried Geometry) */
    printf("\n2. Zonal Reconstruction Setup (Fried Geometry)...\n");
    rippra_zonal_mesh mesh;
    rc = rippra_zonal_setup(&cal, &cfg, &mesh);
    if (rc != 0) {
        printf("ERROR: Zonal setup failed (rc=%d)\n", rc);
        free(sh_flat); free(img); free(cx); free(cy); free(dx); free(dy);
        return 1;
    }
    printf("   Unique phase nodes generated: %d\n", mesh.nnodes);
    
    double *W = (double *)calloc(mesh.nnodes, sizeof(double));
    rc = rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W);
    if (rc != 0) {
        printf("ERROR: Zonal reconstruction failed (rc=%d)\n", rc);
    } else {
        /* Compute statistics of zonal phase map */
        double w_sum = 0.0, w_min = 1e18, w_max = -1e18;
        for (int i = 0; i < mesh.nnodes; ++i) {
            w_sum += W[i];
            if (W[i] < w_min) w_min = W[i];
            if (W[i] > w_max) w_max = W[i];
        }
        double w_mean = w_sum / mesh.nnodes;
        
        double w_var = 0.0;
        for (int i = 0; i < mesh.nnodes; ++i) {
            double dev = W[i] - w_mean;
            w_var += dev * dev;
        }
        double w_rms = sqrt(w_var / mesh.nnodes);
        
        printf("   Zonal phase statistics (in meters):\n");
        printf("     Mean: %.4e m\n", w_mean);
        printf("     RMS:  %.4e m\n", w_rms);
        printf("     Min:  %.4e m\n", w_min);
        printf("     Max:  %.4e m\n", w_max);
    }
    
    /* 3. Modal Phase Reconstruction (Zernike Polynomials) */
    printf("\n3. Modal Reconstruction Setup (Zernike Polynomials)...\n");
    rippra_modal_model model;
    rc = rippra_modal_setup(&cal, &cfg, &model);
    if (rc != 0) {
        printf("ERROR: Modal setup failed (rc=%d)\n", rc);
    } else {
        printf("   Zernike modes to estimate (radial order <= %d): %d\n", cfg.zernike_nmax, model.nmodes);
        
        double *coeffs = (double *)calloc(model.nmodes, sizeof(double));
        rc = rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs);
        if (rc != 0) {
            printf("ERROR: Modal reconstruction failed (rc=%d)\n", rc);
        } else {
            printf("   Reconstructed Zernike coefficients (in radians):\n");
            printf("     Mode ID | Noll Index |  (n, m)  | Value (rad)\n");
            printf("     --------|------------|----------|------------\n");
            for (int i = 0; i < model.nmodes; ++i) {
                printf("     %7d | %10d | (%2d, %3d) | %+.6f\n",
                       i + 1, model.mode_j[i], model.mode_n[i], model.mode_m[i], coeffs[i]);
            }
        }
        free(coeffs);
    }
    
    /* 4. Turbulence Characterization (r0, tau0) */
    printf("\n4. Turbulence Characterization...\n");
    /* Simulate a time-series of 100 frames with wind advection */
    int nframes = 100;
    double *dx_series = (double *)malloc(nframes * cal.nspots * sizeof(double));
    double *dy_series = (double *)malloc(nframes * cal.nspots * sizeof(double));
    
    for (int t = 0; t < nframes; ++t) {
        /* Introduce physical wind drift using Taylor Frozen-Flow: drift ~ 1 pixel/frame in X and Y */
        double t_phase = 2.0 * M_PI * 5.0 * t / 100.0; /* 5 Hz modulation */
        for (int k = 0; k < cal.nspots; ++k) {
            /* Modulated signal + small Gaussian-like noise */
            dx_series[t * cal.nspots + k] = dx[k] * cos(t_phase) + 0.15 * ((double)rand() / RAND_MAX - 0.5);
            dy_series[t * cal.nspots + k] = dy[k] * cos(t_phase) + 0.15 * ((double)rand() / RAND_MAX - 0.5);
        }
    }
    
    double r0 = rippra_compute_r0_impl(dx_series, dy_series, nframes, cal.nspots, &cfg);
    double tau0 = rippra_compute_tau0_impl(dx_series, dy_series, nframes, cal.nspots, 1000.0); /* 1000 Hz frame rate */
    
    printf("   Fried parameter (r0):   %.4e m\n", r0);
    printf("   Coherence time (tau0):  %.4f ms\n", tau0 * 1000.0);
    
    free(dx_series);
    free(dy_series);

    /* 5. DM Command Mapping */
    printf("\n5. Deformable Mirror Actuator Mapping...\n");
    double *dm_cmds = (double *)calloc(mesh.nnodes, sizeof(double));
    rc = rippra_dm_map_impl(W, mesh.nnodes, &mesh, &cfg, dm_cmds);
    if (rc != 0) {
        printf("ERROR: DM Mapping failed (rc=%d)\n", rc);
    } else {
        double max_cmd = 0.0, min_cmd = 1e18, sum_cmd = 0.0;
        for (int i = 0; i < mesh.nnodes; ++i) {
            sum_cmd += dm_cmds[i];
            if (fabs(dm_cmds[i]) > max_cmd) max_cmd = fabs(dm_cmds[i]);
            if (dm_cmds[i] < min_cmd) min_cmd = dm_cmds[i];
        }
        printf("   DM Actuator stroke commands computed successfully!\n");
        printf("     Total Actuators: %d\n", mesh.nnodes);
        printf("     Peak Stroke Command: %.4e m\n", max_cmd);
        printf("     Average Stroke Command: %.4e m\n", sum_cmd / mesh.nnodes);
        
        printf("\n     First 10 DM actuator command strokes (meters):\n");
        for (int i = 0; i < 10 && i < mesh.nnodes; ++i) {
            printf("       Actuator %d at (%2d, %2d): Target Phase = %+.4e m, DM Stroke = %+.4e m\n",
                   i, mesh.node_u[i], mesh.node_v[i], W[i], dm_cmds[i]);
        }
    }
    
    /* Cleanup */
    free(dm_cmds);
    free(W);
    rippra_zonal_free(&mesh);
    rippra_modal_free(&model);
    
    free(cx); free(cy); free(dx); free(dy);
    free(sh_flat); free(img);
    rippa_calibration_free(&cal);
    
    printf("\n=== Reconstruction Test Complete ===\n");
    return 0;
}
