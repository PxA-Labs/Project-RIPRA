/*
 * tests/benchmark_cuda_vs_cpu.cu — CUDA vs CPU/OpenMP benchmark
 *
 * Runs identical synthetic input through CPU (with OpenMP) and GPU (CUDA)
 * pipelines, reports per-stage timing, speedup, and crossover estimate.
 *
 * Compile with nvcc, requires CUDA toolkit and synthetic test data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/recon.h"
#include "cuda/rippra_cuda.h"

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#define WARMUP 5
#define ITERS  50

static int run_cpu_pipeline(const double *img, int w, int h,
                             const rippra_calibration *cal,
                             const rippa_config *cfg,
                             const rippra_zonal_mesh *mesh,
                             const rippra_modal_model *model,
                             double *t_centroid_ms,
                             double *t_recon_ms)
{
    const int n = cal->nspots;
    double *cx = (double *)malloc(n * sizeof(double));
    double *cy = (double *)malloc(n * sizeof(double));
    double *dx = (double *)malloc(n * sizeof(double));
    double *dy = (double *)malloc(n * sizeof(double));
    double *W = (double *)malloc(mesh->nnodes * sizeof(double));
    double *c = (double *)malloc(model->nmodes * sizeof(double));
    if (!cx || !cy || !dx || !dy || !W || !c) { free(cx); free(cy); free(dx); free(dy); free(W); free(c); return -1; }

    for (int i = 0; i < WARMUP; i++) {
        rippa_compute_centroids(img, w, h, cal, cfg, cx, cy);
        rippa_compute_deltas(cx, cy, cal, n, dx, dy, NULL);
        rippra_zonal_reconstruct(mesh, dx, dy, cfg, W);
        rippra_modal_reconstruct(model, dx, dy, cfg, c);
    }

    double t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippa_compute_centroids(img, w, h, cal, cfg, cx, cy);
    *t_centroid_ms = (get_time_ms() - t0) / ITERS;

    t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippa_compute_deltas(cx, cy, cal, n, dx, dy, NULL);
    double t_deltas = (get_time_ms() - t0) / ITERS;

    t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++) {
        rippra_zonal_reconstruct(mesh, dx, dy, cfg, W);
        rippra_modal_reconstruct(model, dx, dy, cfg, c);
    }
    *t_recon_ms = (get_time_ms() - t0) / ITERS;

    printf("  Centroid:         %.3f ms\n", *t_centroid_ms);
    printf("  Deltas:           %.3f ms\n", t_deltas);
    printf("  Recon (z+m):      %.3f ms\n", *t_recon_ms);
    printf("  Total per frame:  %.3f ms\n", *t_centroid_ms + t_deltas + *t_recon_ms);

    free(cx); free(cy); free(dx); free(dy); free(W); free(c);
    return 0;
}

static int run_gpu_pipeline(double *d_frame, int w, int h,
                             const rippra_calibration *cal,
                             const rippa_config *cfg,
                             const rippra_zonal_mesh *mesh,
                             const rippra_modal_model *model,
                             double *t_transfer_ms,
                             double *t_centroid_ms,
                             double *t_recon_ms)
{
    const int nspots = cal->nspots;
    double *d_cx, *d_cy, *d_dx, *d_dy, *d_W, *d_c;
    cudaError_t err;

    #define CU(x) do { err = (x); if (err) { printf("CUDA err: %s:%d\n", __FILE__, __LINE__); return -1; } } while(0)
    CU(cudaMalloc(&d_cx, nspots * sizeof(double)));
    CU(cudaMalloc(&d_cy, nspots * sizeof(double)));
    CU(cudaMalloc(&d_dx, nspots * sizeof(double)));
    CU(cudaMalloc(&d_dy, nspots * sizeof(double)));
    CU(cudaMalloc(&d_W,  mesh->nnodes * sizeof(double)));
    CU(cudaMalloc(&d_c,  model->nmodes * sizeof(double)));
    #undef CU

    for (int i = 0; i < WARMUP; i++) {
        rippra_cuda_compute_centroids(d_frame, w, h, cal, cfg, d_cx, d_cy);
        rippra_cuda_compute_deltas(d_cx, d_cy, nspots, d_dx, d_dy);
        rippra_cuda_zonal_reconstruct(d_dx, d_dy, mesh, cfg, d_W);
        rippra_cuda_modal_reconstruct(d_dx, d_dy, model, cfg, d_c);
    }

    *t_transfer_ms = 0.0;
    *t_centroid_ms = 0.0;
    *t_recon_ms = 0.0;
    double t_deltas = 0.0;

    double t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippra_cuda_compute_centroids(d_frame, w, h, cal, cfg, d_cx, d_cy);
    *t_centroid_ms = (get_time_ms() - t0) / ITERS;

    t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippra_cuda_compute_deltas(d_cx, d_cy, nspots, d_dx, d_dy);
    t_deltas = (get_time_ms() - t0) / ITERS;

    t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++) {
        rippra_cuda_zonal_reconstruct(d_dx, d_dy, mesh, cfg, d_W);
        rippra_cuda_modal_reconstruct(d_dx, d_dy, model, cfg, d_c);
    }
    *t_recon_ms = (get_time_ms() - t0) / ITERS;

    printf("  H2D transfer:     %.3f ms\n", *t_transfer_ms);
    printf("  Centroid:         %.3f ms\n", *t_centroid_ms);
    printf("  Deltas:           %.3f ms\n", t_deltas);
    printf("  Recon (z+m):      %.3f ms\n", *t_recon_ms);
    printf("  Compute total:    %.3f ms\n", *t_centroid_ms + t_deltas + *t_recon_ms);

    cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_dx); cudaFree(d_dy);
    cudaFree(d_W); cudaFree(d_c);
    return 0;
}

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    int rc;

    printf("=== RIPRA CUDA vs CPU/OpenMP Benchmark ===\n\n");

    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) { printf("ERROR: config load\n"); return 1; }

    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) { printf("ERROR: sh_flat.raw\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) { printf("ERROR: img.raw\n"); free(sh_flat); return 1; }

    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { printf("ERROR: calibrate\n"); return 1; }

    rippra_zonal_mesh mesh;
    rippra_modal_model model;
    rc = rippra_zonal_setup(&cal, &cfg, &mesh);  if (rc) { printf("ERROR: zonal setup\n");  return 1; }
    rc = rippra_modal_setup(&cal, &cfg, &model); if (rc) { printf("ERROR: modal setup\n"); return 1; }

    printf("Input:  %dx%d px, spots=%d, nnodes=%d, nmodes=%d\n\n",
           w, h, cal.nspots, mesh.nnodes, model.nmodes);

    /* ---- CPU ---- */
#ifdef _OPENMP
    printf("CPU (%d OpenMP threads):\n", omp_get_max_threads());
#else
    printf("CPU (1 thread, no OpenMP):\n");
#endif
    double t_cpu_cent, t_cpu_recon;
    if (run_cpu_pipeline(img, w, h, &cal, &cfg, &mesh, &model,
                          &t_cpu_cent, &t_cpu_recon) != 0)
        { printf("CPU benchmark failed\n"); return 1; }

    /* ---- GPU ---- */
    rc = rippra_cuda_centroid_init(&cal, w, h);
    if (rc != 0) {
        printf("\nGPU pipeline: SKIP — CUDA init failed (no GPU?)\n");
        goto done;
    }

    double *d_frame = NULL;
    cudaError_t err = cudaMalloc(&d_frame, (size_t)w * h * sizeof(double));
    if (err) { printf("\nGPU pipeline: SKIP — cudaMalloc failed\n"); goto done; }
    err = cudaMemcpy(d_frame, img, (size_t)w * h * sizeof(double), cudaMemcpyHostToDevice);
    if (err) { printf("\nGPU pipeline: SKIP — cudaMemcpy failed\n"); goto done; }

    /* Measure single-frame H2D transfer cost */
    double t_transfer = 0;
    double t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        cudaMemcpy(d_frame, img, (size_t)w * h * sizeof(double), cudaMemcpyHostToDevice);
    t_transfer = (get_time_ms() - t0) / ITERS;

    printf("\nGPU (CUDA):\n");
    double t_gpu_cent, t_gpu_recon;
    if (run_gpu_pipeline(d_frame, w, h, &cal, &cfg, &mesh, &model,
                          &t_transfer, &t_gpu_cent, &t_gpu_recon) != 0)
        { printf("GPU benchmark failed\n"); cudaFree(d_frame); goto done; }

    /* ---- Speedup ---- */
    double cpu_total = t_cpu_cent + t_cpu_recon;
    double gpu_comp  = t_gpu_cent + t_gpu_recon;
    double gpu_e2e   = t_transfer + gpu_comp;

    printf("\nSpeedup (GPU vs CPU):\n");
    printf("  Centroid:          %.2f×\n", t_cpu_cent / t_gpu_cent);
    printf("  Recon (z+m):       %.2f×\n", t_cpu_recon / t_gpu_recon);
    printf("  Compute-only:      %.2f×\n", cpu_total / gpu_comp);
    printf("  End-to-end:        %.2f×\n", cpu_total / gpu_e2e);
    printf("  (E2E incl. H2D transfer)\n");

    /* Crossover: N frames where GPU_total < CPU_total
       GPU_total(N) = transfer_ms + N * gpu_compute_ms
       CPU_total(N) = N * cpu_compute_ms
       N_crossover > transfer_ms / (cpu_compute_ms - gpu_compute_ms)  */
    if (gpu_comp < cpu_total) {
        double N = t_transfer / (cpu_total - gpu_comp) + 1.0;
        printf("\n  Crossover: ~%.0f frames (GPU faster for batched processing)\n", N);
        printf("  Amortizes transfer overhead across N frames\n");
    } else {
        printf("\n  GPU compute is slower than CPU for this problem size\n");
    }

    /* ---- Correctness ---- */
    printf("\nCorrectness check:\n");
    double *d_cx, *d_cy;
    cudaMalloc(&d_cx, cal.nspots * sizeof(double));
    cudaMalloc(&d_cy, cal.nspots * sizeof(double));
    rippra_cuda_compute_centroids(d_frame, w, h, &cal, &cfg, d_cx, d_cy);
    double *gcx = (double *)malloc(cal.nspots * sizeof(double));
    double *gcy = (double *)malloc(cal.nspots * sizeof(double));
    cudaMemcpy(gcx, d_cx, cal.nspots * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gcy, d_cy, cal.nspots * sizeof(double), cudaMemcpyDeviceToHost);

    double *ccx = (double *)malloc(cal.nspots * sizeof(double));
    double *ccy = (double *)malloc(cal.nspots * sizeof(double));
    rippa_compute_centroids(img, w, h, &cal, &cfg, ccx, ccy);

    double max_err = 0.0;
    for (int i = 0; i < cal.nspots; i++) {
        double e = fabs(ccx[i] - gcx[i]) + fabs(ccy[i] - gcy[i]);
        if (e > max_err) max_err = e;
    }
    printf("  Centroid max |CPU−GPU|: %.2e px  %s\n",
           max_err, max_err < 1e-6 ? "✓" : "⚠ (check numerical accuracy)");

    cudaFree(d_cx); cudaFree(d_cy);
    free(gcx); free(gcy); free(ccx); free(ccy);
    cudaFree(d_frame);

done:
    rippra_cuda_centroid_free();
    free(sh_flat); free(img);
    rippra_zonal_free(&mesh);
    rippra_modal_free(&model);
    rippa_calibration_free(&cal);

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
