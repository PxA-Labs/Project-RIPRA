#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/simd.h"

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

#define WARMUP 10
#define ITERS  200

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    int rc;

    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) { printf("ERROR: config\n"); return 1; }
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc != 0) { printf("ERROR: sh_flat\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) { printf("ERROR: img\n"); free(sh_flat); return 1; }

    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { printf("ERROR: calibrate\n"); return 1; }

    double *cx = (double *)malloc(cal.nspots * sizeof(double));
    double *cy = (double *)malloc(cal.nspots * sizeof(double));

    rippra_simd_level level = rippra_simd_detect();
    printf("CPU SIMD support: %s\n", rippra_simd_level_name(level));

    /* ---- Scalar baseline ---- */
    rippra_simd_force_level(RIPPRA_SIMD_NONE);
    for (int i = 0; i < WARMUP; i++)
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    double t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    double t_scalar = (get_time_ms() - t0) / ITERS;
    printf("Scalar:  %.3f ms/frame (%d spots)\n", t_scalar, cal.nspots);

    /* ---- AVX2 (separate invocation to avoid cache interaction) ---- */
    rippra_simd_force_level(RIPPRA_SIMD_AVX2);
    for (int i = 0; i < WARMUP; i++)
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    t0 = get_time_ms();
    for (int i = 0; i < ITERS; i++)
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    double t_avx2 = (get_time_ms() - t0) / ITERS;
    printf("AVX2:    %.3f ms/frame (%d spots)\n", t_avx2, cal.nspots);

    rippra_simd_force_level(-1);

    printf("\nSpeedup (AVX2 vs scalar): %.2f×\n", t_scalar / t_avx2);

    /* Verify correctness (force_level is checked every call now, so order doesn't matter) */
    double *cx_ref = (double *)malloc(cal.nspots * sizeof(double));
    double *cy_ref = (double *)malloc(cal.nspots * sizeof(double));
    double *cx_avx = (double *)malloc(cal.nspots * sizeof(double));
    double *cy_avx = (double *)malloc(cal.nspots * sizeof(double));

    rippra_simd_force_level(RIPPRA_SIMD_NONE);
    rippa_compute_centroids(img, w, h, &cal, &cfg, cx_ref, cy_ref);
    rippra_simd_force_level(RIPPRA_SIMD_AVX2);
    rippa_compute_centroids(img, w, h, &cal, &cfg, cx_avx, cy_avx);
    rippra_simd_force_level(-1);

    double max_err = 0.0;
    for (int i = 0; i < cal.nspots; i++) {
        double e = fabs(cx_ref[i] - cx_avx[i]) + fabs(cy_ref[i] - cy_avx[i]);
        if (e > max_err) max_err = e;
    }
    printf("Max centroid error (scalar vs AVX2): %.2e px  %s\n",
           max_err, max_err < 1e-12 ? "PASS" : "FAIL");

    free(cx); free(cy); free(cx_ref); free(cy_ref); free(cx_avx); free(cy_avx);
    rippa_calibration_free(&cal);
    free(sh_flat); free(img);
    return 0;
}
