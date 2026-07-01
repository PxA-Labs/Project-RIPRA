#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"
#include "rippra/rippra_api.h"

#define WARMUP_ITERS 5
#define BENCH_ITERS 100

#ifdef _WIN32
#include <windows.h>
double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx, *cy, *dx, *dy;
    int rc;
    
    printf("=== RIPRA OpenMP Benchmark ===\n\n");
    
    /* Load configuration */
    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) {
        printf("ERROR: Failed to load configuration\n");
        return 1;
    }
    
    /* Load raw data files */
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) { printf("ERROR: Failed to load sh_flat.raw\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) { printf("ERROR: Failed to load img.raw\n"); free(sh_flat); return 1; }
    
    /* Calibrate grid */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { printf("ERROR: Calibration failed\n"); free(sh_flat); free(img); return 1; }
    
    cx = (double *)malloc(cal.nspots * sizeof(double));
    cy = (double *)malloc(cal.nspots * sizeof(double));
    dx = (double *)malloc(cal.nspots * sizeof(double));
    dy = (double *)malloc(cal.nspots * sizeof(double));
    
    /* Setup zonal mesh */
    rippra_zonal_mesh mesh;
    rc = rippra_zonal_setup(&cal, &cfg, &mesh);
    if (rc != 0) { free(cx); free(cy); free(dx); free(dy); printf("ERROR: Zonal setup failed\n"); return 1; }
    
    double *W = (double *)calloc(mesh.nnodes, sizeof(double));
    
    /* Setup modal model */
    rippra_modal_model model;
    rc = rippra_modal_setup(&cal, &cfg, &model);
    if (rc != 0) { free(cx); free(cy); free(dx); free(dy); free(W); printf("ERROR: Modal setup failed\n"); return 1; }
    
    double *coeffs = (double *)calloc(model.nmodes, sizeof(double));
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
        rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);
        rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W);
        rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs);
    }
    
    /* Benchmark centroiding */
    double t_start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    }
    double t_centroid = (get_time_ms() - t_start) / BENCH_ITERS;
    
    /* Benchmark deltas + zonal + modal */
    t_start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);
        rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W);
        rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs);
    }
    double t_reconstruct = (get_time_ms() - t_start) / BENCH_ITERS;
    
    /* Benchmark turbulence params (simulate 100 frames) */
    int nframes = 100;
    double *dx_series = (double *)malloc(nframes * cal.nspots * sizeof(double));
    double *dy_series = (double *)malloc(nframes * cal.nspots * sizeof(double));
    
    for (int t = 0; t < nframes; ++t) {
        for (int k = 0; k < cal.nspots; ++k) {
            dx_series[t * cal.nspots + k] = dx[k] * cos(2.0 * M_PI * 5.0 * t / 100.0);
            dy_series[t * cal.nspots + k] = dy[k] * cos(2.0 * M_PI * 5.0 * t / 100.0);
        }
    }
    
    t_start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rippra_compute_r0_impl(dx_series, dy_series, nframes, cal.nspots, &cfg);
        rippra_compute_tau0(dx_series, dy_series, nframes, cal.nspots, 1000.0);
    }
    double t_turbulence = (get_time_ms() - t_start) / BENCH_ITERS;
    
    /* Benchmark DM mapping */
    double *dm_cmds = (double *)calloc(mesh.nnodes, sizeof(double));
    t_start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        rippra_dm_map_impl(W, mesh.nnodes, &mesh, &cfg, dm_cmds);
    }
    double t_dm = (get_time_ms() - t_start) / BENCH_ITERS;
    
    printf("Results (average over %d iterations):\n\n", BENCH_ITERS);
    printf("  Centroiding (127 spots):      %.3f ms\n", t_centroid);
    printf("  Deltas + Zonal + Modal:       %.3f ms\n", t_reconstruct);
    printf("  Turbulence (r0, tau0, 100 fr):  %.3f ms\n", t_turbulence);
    printf("  DM Mapping (140 actuators):   %.3f ms\n", t_dm);
    printf("  ------------------------------------\n");
    printf("  Total per frame (est):          %.3f ms\n", t_centroid + t_reconstruct + t_turbulence/100 + t_dm);
    
#ifdef _OPENMP
    printf("\nOpenMP enabled: YES (%d threads max)\n", omp_get_max_threads());
#else
    printf("\nOpenMP enabled: NO\n");
#endif
    
    /* Cleanup */
    free(dm_cmds);
    free(dx_series);
    free(dy_series);
    free(coeffs);
    free(W);
    rippra_modal_free(&model);
    rippra_zonal_free(&mesh);
    free(cx); free(cy); free(dx); free(dy);
    free(sh_flat); free(img);
    rippa_calibration_free(&cal);
    
    return 0;
}