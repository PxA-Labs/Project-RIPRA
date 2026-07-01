/* Benchmark centroid + wavefront RMS lambda */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/recon.h"

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

int main(void) {
    rippa_config cfg;
    double *flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx, *cy, *dx, *dy, *phase;
    int i, nspots, nnodes;
    int rc;

    rc = rippa_config_load(&cfg, "config/system.conf");
    if (rc != 0) return 1;

    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &flat);
    if (rc != 0) return 1;
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc != 0) return 1;

    memset(&cal, 0, sizeof(cal));
    rc = rippa_calibrate_grid(flat, w, h, &cfg, &cal);
    if (rc != 0) return 1;
    nspots = cal.nspots;
    printf("Spots: %d\n", nspots);

    cx = (double*)malloc(nspots * sizeof(double));
    cy = (double*)malloc(nspots * sizeof(double));
    dx = (double*)malloc(nspots * sizeof(double));
    dy = (double*)malloc(nspots * sizeof(double));

    /* ---- Benchmark centroid ---- */
    int N = 100;
    double t0 = now_ms();
    for (i = 0; i < N; i++) {
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
        rippa_compute_deltas(cx, cy, &cal, nspots, dx, dy, NULL);
    }
    double t1 = now_ms();
    printf("Centroid (fast): %.3f ms/frame (%d runs)\n", (t1-t0)/N, N);

    /* ---- Benchmark refined centroid ---- */
    t0 = now_ms();
    for (i = 0; i < N; i++) {
        rippa_compute_centroids_refined(img, w, h, &cal, &cfg, cx, cy, dx, dy);
    }
    t1 = now_ms();
    printf("Centroid (refined): %.3f ms/frame (%d runs)\n", (t1-t0)/N, N);

    /* ---- Zonal reconstruction + wavefront RMS ---- */
    rippra_zonal_mesh mesh;
    memset(&mesh, 0, sizeof(mesh));
    rippra_zonal_setup(&cal, &cfg, &mesh);
    nnodes = mesh.nnodes;
    phase = (double*)malloc(nnodes * sizeof(double));

    rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    rippa_compute_deltas(cx, cy, &cal, nspots, dx, dy, NULL);
    rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, phase);

    double rms_lambda = rippra_wavefront_rms_lambda(phase, nnodes, &cfg);
    printf("Wavefront RMS (turbulence): %.6f lambda\n", rms_lambda);

    /* Stability: repeat measurement on same frame, compute std of RMS */
    double *rms_vals = (double*)malloc(N * sizeof(double));
    for (i = 0; i < N; i++) {
        rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
        rippa_compute_deltas(cx, cy, &cal, nspots, dx, dy, NULL);
        rippra_zonal_reconstruct(&mesh, dx, dy, &cfg, phase);
        rms_vals[i] = rippra_wavefront_rms_lambda(phase, nnodes, &cfg);
    }
    double mean = 0.0;
    for (i = 0; i < N; i++) mean += rms_vals[i];
    mean /= N;
    double var = 0.0;
    for (i = 0; i < N; i++) var += (rms_vals[i] - mean) * (rms_vals[i] - mean);
    var /= N;
    double sigma_lambda = sqrt(var);
    printf("Stability (sigma over %d runs): %.6f lambda\n", N, sigma_lambda);
    printf("Stability threshold (0.05 lambda): %s\n",
           sigma_lambda < 0.05 ? "PASS" : "FAIL");
    free(rms_vals);

    free(flat); free(img); free(cx); free(cy); free(dx); free(dy); free(phase);
    rippa_calibration_free(&cal);
    rippra_zonal_free(&mesh);
    return 0;
}
