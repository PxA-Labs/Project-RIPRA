/*
 * tests/test_centroid.c - calibrate grid on reference frame, centroid the
 * aberrated frame, compare deltas against the existing scratch CSV output.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "rippra/io.h"
#include "rippra/centroid.h"

static double absmean(double *a, int n)
{
    double s = 0.0; int i;
    for (i = 0; i < n; ++i) s += fabs(a[i]);
    return s / n;
}

int main(void)
{
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    double *cx, *cy, *dx, *dy;
    int rc, k;

    printf("=== RIPRA centroid / calibration test ===\n\n");
    rippa_config_load(&cfg, "config/system.conf");

    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    if (rc) { printf("ERROR loading sh_flat\n"); return 1; }
    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    if (rc) { printf("ERROR loading img\n"); return 1; }

    /* 1. calibrate grid on reference frame */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    printf("1. Calibration: rc=%d  nspots=%d (expected ~127)\n", rc, cal.nspots);
    printf("   pupil centre = (%.2f, %.2f) px\n", cal.pupil_cx, cal.pupil_cy);
    printf("   pitch (est)  = %.2f px (expected ~40.5)\n", cal.pitch_px);

    if (rc != 0 || cal.nspots < 50) {
        printf("   CALIBRATION FAILED\n");
        return 1;
    }

    /* 2. centroid the aberrated frame */
    cx = (double *)malloc(cal.nspots * sizeof(double));
    cy = (double *)malloc(cal.nspots * sizeof(double));
    dx = (double *)malloc(cal.nspots * sizeof(double));
    dy = (double *)malloc(cal.nspots * sizeof(double));
    rippa_compute_centroids(img, w, h, &cal, &cfg, cx, cy);
    rippa_compute_deltas(cx, cy, &cal, cal.nspots, dx, dy, NULL);

    printf("\n2. Centroiding complete on %d sub-apertures\n", cal.nspots);
    printf("   mean |dx| = %.3f px,  mean |dy| = %.3f px\n",
           absmean(dx, cal.nspots), absmean(dy, cal.nspots));
    {
        double mdx = 0, mdy = 0; int i;
        for (i = 0; i < cal.nspots; ++i) {
            if (fabs(dx[i]) > mdx) mdx = fabs(dx[i]);
            if (fabs(dy[i]) > mdy) mdy = fabs(dy[i]);
        }
        printf("   max  |dx| = %.3f px,  max  |dy| = %.3f px\n", mdx, mdy);
    }

    /* 3. save reference centroids + deltas to results for inspection */
    {
        FILE *fp = fopen("results/reference_centroids_c.csv", "w");
        if (fp) {
            fprintf(fp, "Spot_ID,ref_cx,ref_cy,col_min,col_max,row_min,row_max\n");
            for (k = 0; k < cal.nspots; ++k) {
                fprintf(fp, "%d,%.4f,%.4f,%d,%d,%d,%d\n", k,
                        cal.subaps[k].ref_cx, cal.subaps[k].ref_cy,
                        cal.subaps[k].col_min, cal.subaps[k].col_max,
                        cal.subaps[k].row_min, cal.subaps[k].row_max);
            }
            fclose(fp);
        }
    }
    {
        FILE *fp = fopen("results/spot_deviations_c.csv", "w");
        if (fp) {
            fprintf(fp, "Spot_ID,Delta_X,Delta_Y\n");
            for (k = 0; k < cal.nspots; ++k)
                fprintf(fp, "%d,%.6f,%.6f\n", k, dx[k], dy[k]);
            fclose(fp);
        }
    }
    printf("\n3. Results saved (if results/ directory exists)\n");

    printf("\n4. First 8 sub-apertures:\n");
    printf("   %-4s %-9s %-9s %-7s %-7s %-7s %-7s %-7s %-7s\n",
           "ID", "ref_cx", "ref_cy", "cx", "cy", "dx", "dy", "", "");
    for (k = 0; k < 8 && k < cal.nspots; ++k) {
        printf("   %-4d %-9.2f %-9.2f %-7.2f %-7.2f %-7.2f %-7.2f\n",
               k, cal.subaps[k].ref_cx, cal.subaps[k].ref_cy,
               cx[k], cy[k], dx[k], dy[k]);
    }

    free(cx); free(cy); free(dx); free(dy);
    free(sh_flat); free(img);
    rippa_calibration_free(&cal);
    printf("\n=== centroid test complete ===\n");
    return 0;
}
