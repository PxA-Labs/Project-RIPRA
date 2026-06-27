/*
 * tools/gen_test_data.c — Generate synthetic SHWFS test data for CI
 * Compile: gcc -O2 gen_test_data.c -o gen_test_data -lm
 * Run: ./gen_test_data
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void gen_raw(const char *path, int w, int h, int dx, int dy)
{
    uint16_t *buf = (uint16_t*)malloc(w * h * sizeof(uint16_t));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int cx = (x + w/2) % 30, cy = (y + h/2) % 30;
            int sx = cx - 15, sy = cy - 15;
            double d = sqrt((double)(sx-dx)*(sx-dx) + (double)(sy-dy)*(sy-dy));
            int val = d < 3.5 ? (int)(600 * exp(-d*d/2.5)) : 20 + (rand() % 10);
            buf[y * w + x] = val < 65535 ? (uint16_t)val : 65535;
        }
    }
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", path); free(buf); exit(1); }
    fwrite(buf, sizeof(uint16_t), w * h, fp);
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
