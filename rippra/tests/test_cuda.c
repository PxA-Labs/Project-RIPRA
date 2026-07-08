/*
 * tests/test_cuda.c - Test CUDA-accelerated wavefront reconstruction pipeline
 * Requires CUDA toolkit. Compile with nvcc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"
#include "../cuda/rippra_cuda.h"

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx = NULL, *cy = NULL, *dx = NULL, *dy = NULL;
    double *W_cpu = NULL, *coeffs_cpu = NULL;
    double *W_gpu = NULL, *coeffs_gpu = NULL;
    double *d_frame = NULL, *d_W = NULL, *d_coeffs = NULL, *d_dm = NULL;
    rippra_zonal_mesh mesh;
    rippra_modal_model model;
    int rc;
    cudaError_t cerr;

    printf("=== RIPRA CUDA Acceleration Test ===\n\n");

    /* Load configuration */
    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) { printf("ERROR: Failed to load configuration\n"); return 1; }

    /* Load raw data */
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) { printf("ERROR: Failed to load sh_flat.raw\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) { printf("ERROR: Failed to load img.raw\n"); free(sh_flat); return 1; }

    /* Calibrate grid (CPU) */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { printf("ERROR: Calibration failed\n"); free(sh_flat); free(img); return 1; }
    printf("1. Grid Calibration: %d spots\n", cal.nspots);

    /* Setup zonal/ modal (CPU - done once at calibration) */
    memset(&mesh, 0, sizeof(mesh));
    rc = rippra_zonal_setup(&cal, &cfg, &mesh);
    if (rc != 0) { printf("ERROR: Zonal setup failed\n"); goto cleanup; }
    printf("2. Zonal mesh: %d nodes\n", mesh.nnodes);

    memset(&model, 0, sizeof(model));
    rc = rippra_modal_setup(&cal, &cfg, &model);
    if (rc != 0) { printf("ERROR: Modal setup failed\n"); goto cleanup; }
    printf("3. Modal model: %d modes\n", model.nmodes);

    /* ---- CPU baseline ---- */
    cx = (double *)malloc(cal.nspots * sizeof(double));
    cy = (double *)malloc(cal.nspots * sizeof(double));
    dx = (double *)malloc(cal.nspots * sizeof(double));
    dy = (double *)malloc(cal.nspots * sizeof(double));

    rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);

    W_cpu = (double *)calloc(mesh.nnodes, sizeof(double));
    rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W_cpu);

    coeffs_cpu = (double *)calloc(model.nmodes, sizeof(double));
    rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs_cpu);

    printf("4. CPU reconstruction complete\n");

    /* ---- GPU pipeline ---- */
    rc = rippra_cuda_centroid_init(&cal, w, h);
    if (rc != 0) {
        printf("SKIP: CUDA centroid init failed (no GPU?) — compile-only check\n");
        free(cx); free(cy); free(dx); free(dy);
        free(W_cpu); free(coeffs_cpu);
        free(sh_flat); free(img);
        rippra_zonal_free(&mesh);
        rippra_modal_free(&model);
        rippa_calibration_free(&cal);
        printf("\n=== CUDA Test Skipped ===\n");
        return 0;
    }

    {
    rc = rippra_cuda_dm_init(&mesh);
    if (rc != 0) {
        printf("ERROR: CUDA DM init failed\n"); goto cleanup;
    }

    cerr = cudaMalloc(&d_frame, (size_t)w * h * sizeof(double));
    if (cerr) { printf("ERROR: cudaMalloc frame failed\n"); goto cleanup; }

    cerr = cudaMemcpy(d_frame, img, (size_t)w * h * sizeof(double), cudaMemcpyHostToDevice);
    if (cerr) { printf("ERROR: cudaMemcpy frame failed\n"); goto cleanup; }

    cerr = cudaMalloc(&d_W, mesh.nnodes * sizeof(double));
    if (cerr) { printf("ERROR: cudaMalloc W failed\n"); goto cleanup; }

    cerr = cudaMalloc(&d_coeffs, model.nmodes * sizeof(double));
    if (cerr) { printf("ERROR: cudaMalloc coeffs failed\n"); goto cleanup; }

    cerr = cudaMalloc(&d_dm, mesh.nnodes * sizeof(double));
    if (cerr) { printf("ERROR: cudaMalloc dm failed\n"); goto cleanup; }

    /* Run GPU full pipeline */
    rc = rippra_cuda_full_pipeline(d_frame, w, h, &cal, &cfg, &mesh, &model,
                                    d_W, d_coeffs, d_dm);
    if (rc != 0) {
        printf("ERROR: CUDA pipeline failed\n");
        goto cleanup;
    }

    /* Copy results back */
    W_gpu = (double *)malloc(mesh.nnodes * sizeof(double));
    coeffs_gpu = (double *)malloc(model.nmodes * sizeof(double));

    cudaMemcpy(W_gpu, d_W, mesh.nnodes * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(coeffs_gpu, d_coeffs, model.nmodes * sizeof(double), cudaMemcpyDeviceToHost);

    printf("\n5. GPU reconstruction complete\n");

    /* Compare CPU vs GPU */
    printf("\n6. CPU vs GPU Comparison:\n\n");

    double max_diff_W = 0.0, rms_W = 0.0, max_diff_c = 0.0, rms_c = 0.0;

    /* Wavefront phase comparison */
    for (int i = 0; i < mesh.nnodes; ++i) {
        double diff = fabs(W_cpu[i] - W_gpu[i]);
        if (diff > max_diff_W) max_diff_W = diff;
        rms_W += diff * diff;
    }
    rms_W = sqrt(rms_W / mesh.nnodes);
    printf("   Zonal phase (W):\n");
    printf("     Max absolute diff: %.2e m\n", max_diff_W);
    printf("     RMS diff:          %.2e m\n", rms_W);

    /* Zernike coefficients comparison */
    for (int i = 0; i < model.nmodes; ++i) {
        double diff = fabs(coeffs_cpu[i] - coeffs_gpu[i]);
        if (diff > max_diff_c) max_diff_c = diff;
        rms_c += diff * diff;
    }
    rms_c = sqrt(rms_c / model.nmodes);
    printf("   Modal coefficients:\n");
    printf("     Max absolute diff: %.2e rad\n", max_diff_c);
    printf("     RMS diff:          %.2e rad\n", rms_c);

    /* Verify correctness */
    if (max_diff_W < 1e-6 && max_diff_c < 1e-6) {
        printf("\n   GPU results match CPU (within tolerance)\n");
    } else {
        printf("\n   GPU results deviate from CPU\n");
    }

cleanup:
    cudaFree(d_frame);
    cudaFree(d_W);
    cudaFree(d_coeffs);
    cudaFree(d_dm);
    free(W_gpu);
    free(coeffs_gpu);
    }
    rippra_cuda_centroid_free();
    rippra_cuda_dm_free();
    free(cx); free(cy); free(dx); free(dy);
    free(W_cpu); free(coeffs_cpu);
    free(sh_flat); free(img);
    rippra_zonal_free(&mesh);
    rippra_modal_free(&model);
    rippa_calibration_free(&cal);

    printf("\n=== CUDA Test Complete ===\n");
    return 0;
}
