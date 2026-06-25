/*
 * tests/test_io.c - test I/O: load config, load raw frames, verify data
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "rippra/io.h"

int main(void)
{
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL, *bg = NULL;
    int w = 648, h = 492;
    int rc;

    printf("=== RIPPA I/O test ===\n\n");

    /* 1. Load config */
    rc = rippa_config_load(&cfg, "config/system.conf");
    printf("1. Config loaded (rc=%d)\n", rc);
    printf("   camera_pixsize = %.2e m\n", cfg.camera_pixsize);
    printf("   pitch          = %.2e m  (%.1f px)\n",
           cfg.pitch, cfg.pitch / cfg.camera_pixsize);
    printf("   flength        = %.2e m\n", cfg.flength);
    printf("   wavelength     = %.2e m\n", cfg.wavelength);
    printf("   pupil_radius   = %.2e m\n", cfg.pupil_radius);
    printf("   zernike_nmax   = %d\n", cfg.zernike_nmax);

    /* 2. Load raw frames */
    rc = rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    printf("\n2. sh_flat loaded (rc=%d)\n", rc);

    rc = rippa_load_raw("data_raw/img.raw", w, h, &img);
    printf("   img loaded (rc=%d)\n", rc);

    rc = rippa_load_raw("data_raw/sh_flat_bg.raw", w, h, &bg);
    printf("   sh_flat_bg loaded (rc=%d)\n", rc);

    /* 3. Verify ranges match known values */
    if (sh_flat && img && bg) {
        size_t n = (size_t)w * (size_t)h;
        double mn, mx;
        int i;
        mn = sh_flat[0]; mx = sh_flat[0];
        for (i = 1; i < (int)n; ++i) {
            if (sh_flat[i] < mn) mn = sh_flat[i];
            if (sh_flat[i] > mx) mx = sh_flat[i];
        }
        printf("\n3. Data verification:\n");
        printf("   sh_flat: min=%.6f max=%.6f (expected ~0.007 .. ~0.843)\n", mn, mx);

        mn = img[0]; mx = img[0];
        for (i = 1; i < (int)n; ++i) {
            if (img[i] < mn) mn = img[i];
            if (img[i] > mx) mx = img[i];
        }
        printf("   img:     min=%.6f max=%.6f (expected ~0.005 .. ~0.778)\n", mn, mx);

        mn = bg[0]; mx = bg[0];
        for (i = 1; i < (int)n; ++i) {
            if (bg[i] < mn) mn = bg[i];
            if (bg[i] > mx) mx = bg[i];
        }
        printf("   bg:      min=%.6f max=%.6f\n", mn, mx);

        /* 4. Save a CSV snapshot of the first 10 rows of img for spot-check */
        printf("\n4. Saving first 10 rows of img as results/img_head.csv ...\n");
        /* save just a small crop */
        double crop[10 * w];
        for (i = 0; i < 10; ++i)
            memcpy(crop + i * w, img + i * w, w * sizeof(double));
        rippa_save_csv("results/img_head.csv", crop, w, 10);
        printf("   Done.\n");
    }

    /* cleanup */
    free(sh_flat); free(img); free(bg);
    printf("\n=== I/O test complete ===\n");
    return 0;
}
