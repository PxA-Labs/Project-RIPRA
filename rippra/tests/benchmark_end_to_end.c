#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"
#include "rippra/rippra_api.h"

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

#define WARMUP 3
#define BENCH_ITERS 30

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    if (n % 2 == 0) return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    return arr[n/2];
}

static double percentile(double *arr, int n, double p) {
    qsort(arr, n, sizeof(double), cmp_double);
    int idx = (int)(p / 100.0 * (n - 1));
    return arr[idx];
}

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx, *cy, *dx, *dy;

    printf("=== RIPRA End-to-End Latency Benchmark ===\n\n");

    /* ===== Stage 0: I/O (load from disk) ===== */
    double t_io_start = get_time_ms();
    int rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) { printf("ERROR: Failed to load configuration\n"); return 1; }
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) { printf("ERROR: Failed to load sh_flat.raw\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) { printf("ERROR: Failed to load img.raw\n"); free(sh_flat); return 1; }
    double t_io = get_time_ms() - t_io_start;

    /* Setup (one-time, excluded from per-frame timing) */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { printf("ERROR: Calibration failed\n"); free(sh_flat); free(img); return 1; }
    cx = (double *)malloc(cal.nspots * sizeof(double));
    cy = (double *)malloc(cal.nspots * sizeof(double));
    dx = (double *)malloc(cal.nspots * sizeof(double));
    dy = (double *)malloc(cal.nspots * sizeof(double));

    rippra_zonal_mesh mesh;
    rc = rippra_zonal_setup(&cal, &cfg, &mesh);
    if (rc != 0) { free(cx); free(cy); free(dx); free(dy); printf("ERROR: Zonal setup failed\n"); return 1; }
    double *W = (double *)calloc(mesh.nnodes, sizeof(double));

    rippra_modal_model model;
    rc = rippra_modal_setup(&cal, &cfg, &model);
    if (rc != 0) { free(cx); free(cy); free(dx); free(dy); free(W); printf("ERROR: Modal setup failed\n"); return 1; }
    double *coeffs = (double *)calloc(model.nmodes, sizeof(double));
    double *dm_cmds = (double *)calloc(mesh.nnodes, sizeof(double));

    printf("Detection: %d spots, %d zonal nodes, %d modes\n\n", cal.nspots, mesh.nnodes, model.nmodes);

    /* Warmup */
    for (int i = 0; i < WARMUP; ++i) {
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
        rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);
        rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W);
        rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs);
        rippra_dm_map_impl(W, mesh.nnodes, &mesh, &cfg, dm_cmds);
    }

    /* Per-frame benchmark */
    double total_ms[BENCH_ITERS];
    double cent_ms[BENCH_ITERS];
    double recon_ms[BENCH_ITERS];
    double dm_ms[BENCH_ITERS];

    for (int i = 0; i < BENCH_ITERS; ++i) {
        double t0 = get_time_ms();

        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
        double t1 = get_time_ms();

        rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);
        rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, W);
        rippra_modal_reconstruct(&model, dx, dy, &cfg, coeffs);
        double t2 = get_time_ms();

        rippra_dm_map_impl(W, mesh.nnodes, &mesh, &cfg, dm_cmds);
        double t3 = get_time_ms();

        cent_ms[i] = t1 - t0;
        recon_ms[i] = t2 - t1;
        dm_ms[i] = t3 - t2;
        total_ms[i] = t3 - t0;
    }

    /* Report */
    printf("Results (over %d iterations):\n\n", BENCH_ITERS);
    printf("  I/O (config + frame load):        %8.3f ms (one-time)\n\n", t_io);
    printf("  Per-frame breakdown:\n");
    printf("    Centroiding (%d spots):          %8.3f ms (mean)  %8.3f ms (p99)\n",
           cal.nspots, cent_ms[0], percentile(cent_ms, BENCH_ITERS, 99));
    printf("    Deltas + Zonal + Modal:          %8.3f ms (mean)  %8.3f ms (p99)\n",
           recon_ms[0], percentile(recon_ms, BENCH_ITERS, 99));
    printf("    DM Mapping (%d actuators):       %8.3f ms (mean)  %8.3f ms (p99)\n",
           mesh.nnodes, dm_ms[0], percentile(dm_ms, BENCH_ITERS, 99));
    printf("  --------------------------------------\n");

    double mean_total = 0;
    for (int i = 0; i < BENCH_ITERS; ++i) mean_total += total_ms[i];
    mean_total /= BENCH_ITERS;

    printf("  HOT-PATH TOTAL (cent+recon+dm):    %8.3f ms (mean)  %8.3f ms (p99)\n",
           mean_total, percentile(total_ms, BENCH_ITERS, 99));
    printf("  END-TO-END (I/O + hot-path):      %8.3f ms (mean)\n",
           t_io + mean_total);
    printf("  HOT-PATH MEDIAN:                   %8.3f ms\n", median(total_ms, BENCH_ITERS));
    printf("\n");

    /* Print individual runs */
    printf("Per-iteration hot-path (ms):\n");
    for (int i = 0; i < BENCH_ITERS; ++i) {
        printf("  %2d:  %7.3f  (cent=%6.3f  recon=%6.3f  dm=%6.3f)\n",
               i+1, total_ms[i], cent_ms[i], recon_ms[i], dm_ms[i]);
    }

#ifdef _OPENMP
    printf("\nOpenMP enabled: YES (%d threads max)\n", omp_get_max_threads());
#else
    printf("\nOpenMP enabled: NO\n");
#endif

    /* Cleanup */
    free(dm_cmds);
    free(coeffs);
    free(W);
    rippra_modal_free(&model);
    rippra_zonal_free(&mesh);
    free(cx); free(cy); free(dx); free(dy);
    free(sh_flat); free(img);
    rippa_calibration_free(&cal);
    return 0;
}
