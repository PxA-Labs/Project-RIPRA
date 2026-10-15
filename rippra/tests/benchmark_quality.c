/* benchmark_quality.c - repeatable frame-quality gate timing. */
#include "rippra/io.h"
#include "rippra/centroid.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define WARMUP 10
#define ITERS 200

static double now_us(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1.0e6 / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1.0e6 + (double)ts.tv_nsec / 1.0e3;
#endif
}

static int compare_double(const void *a, const void *b)
{
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void report(const char *name, double *samples)
{
    double sum = 0.0;
    for (int i = 0; i < ITERS; ++i) sum += samples[i];
    qsort(samples, ITERS, sizeof(double), compare_double);
    printf("  %-28s mean=%8.3f us p99=%8.3f us\n",
           name, sum / ITERS, samples[(int)(0.99 * (ITERS - 1))]);
}

int main(void)
{
    const int width = 648, height = 492;
    rippa_config cfg;
    rippra_calibration cal = {0};
    double *flat = NULL, *cx = NULL, *cy = NULL, *dx = NULL, *dy = NULL;
    double *mass = NULL;
    int *mask = NULL;
    rippra_spot_metrics *metrics = NULL;
    double quality_metrics[4];
    double quality_samples[ITERS], path_samples[ITERS];

    if (rippa_config_load(&cfg, "config/system.conf") != 0 ||
        rippa_load_raw("data_raw/sh_flat.raw", width, height, &flat) != 0 ||
        rippa_calibrate_grid(flat, width, height, &cfg, &cal) != 0) {
        fprintf(stderr, "ERROR: quality benchmark fixture/calibration failed\n");
        free(flat);
        return 1;
    }
    int nspots = cal.nspots;
    cx = (double *)malloc((size_t)nspots * sizeof(double));
    cy = (double *)malloc((size_t)nspots * sizeof(double));
    dx = (double *)malloc((size_t)nspots * sizeof(double));
    dy = (double *)malloc((size_t)nspots * sizeof(double));
    mass = (double *)malloc((size_t)nspots * sizeof(double));
    mask = (int *)malloc((size_t)nspots * sizeof(int));
    metrics = (rippra_spot_metrics *)malloc((size_t)nspots * sizeof(*metrics));
    if (!cx || !cy || !dx || !dy || !mass || !mask || !metrics) return 1;

    rippa_compute_centroids_with_metrics(flat, width, height, &cal, &cfg,
                                         cx, cy, metrics);
    for (int i = 0; i < nspots; ++i) mass[i] = metrics[i].mass;
    rippa_compute_deltas(cx, cy, &cal, nspots, dx, dy, mask);
    for (int i = 0; i < WARMUP; ++i)
        (void)rippra_frame_quality_with_metrics(
            dx, dy, mass, mask, metrics, nspots, &cfg, &cal, quality_metrics);

    for (int i = 0; i < ITERS; ++i) {
        double start = now_us();
        (void)rippra_frame_quality_with_metrics(
            dx, dy, mass, mask, metrics, nspots, &cfg, &cal, quality_metrics);
        quality_samples[i] = now_us() - start;
    }
    for (int i = 0; i < ITERS; ++i) {
        double start = now_us();
        rippa_compute_centroids_with_metrics(flat, width, height, &cal, &cfg,
                                             cx, cy, metrics);
        for (int j = 0; j < nspots; ++j) mass[j] = metrics[j].mass;
        rippa_compute_deltas(cx, cy, &cal, nspots, dx, dy, mask);
        (void)rippra_frame_quality_with_metrics(
            dx, dy, mass, mask, metrics, nspots, &cfg, &cal, quality_metrics);
        path_samples[i] = now_us() - start;
    }

    printf("=== Frame-quality gate benchmark (%d spots, %d iterations) ===\n", nspots, ITERS);
    report("quality reduction only", quality_samples);
    report("centroid + quality path", path_samples);
    printf("  final Q=%.6f terms=[%.3f %.3f %.3f %.3f]\n",
           rippra_frame_quality_with_metrics(dx, dy, mass, mask, metrics,
                                             nspots, &cfg, &cal, quality_metrics),
           quality_metrics[0], quality_metrics[1], quality_metrics[2],
           quality_metrics[3]);

    free(flat); free(cx); free(cy); free(dx); free(dy); free(mass); free(mask);
    free(metrics);
    rippa_calibration_free(&cal);
    return 0;
}
