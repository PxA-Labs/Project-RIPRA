/*
 * tools/gen_test_data.c — Generate synthetic SHWFS test data for CI
 * Compile: gcc -O2 gen_test_data.c -o gen_test_data -lm
 * Run: ./gen_test_data
 *
 * rippa_load_raw() reads double-precision floats, so we write doubles.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void gen_raw(const char *path, int w, int h, int dx, int dy)
{
    double *buf = (double*)malloc(w * h * sizeof(double));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    /* lenslet pitch ≈ 300um / 7.4um ≈ 40.5 px → use 40-px grid */
    int pitch = 40, half = pitch / 2;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int cx = (x + half) % pitch, cy = (y + half) % pitch;
            int sx = cx - half, sy = cy - half;
            double d = sqrt((double)(sx-dx)*(sx-dx) + (double)(sy-dy)*(sy-dy));
            buf[y * w + x] = d < 3.5 ? 600.0 * exp(-d*d/2.5) : 20.0 + (rand() % 10);
        }
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", path); free(buf); exit(1); }
    fwrite(buf, sizeof(double), w * h, fp);
    fclose(fp);
    free(buf);
}

int main(void)
{
    int w = 648, h = 492;
    printf("Generating synthetic test data (%dx%d)...\n", w, h);
    gen_raw("data_raw/sh_flat.raw", w, h, 0, 0);
    gen_raw("data_raw/img.raw", w, h, 1, 1);
    printf("Done.\n");
    return 0;
}
