/*
 * rippa/io.c - file I/O implementation
 */
#include "rippra/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

/* ---- helpers ------------------------------------------------------------ */

static char *strip(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s) - 1;
    while (e > s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

static void default_config(rippa_config *cfg)
{
    cfg->camera_pixsize     = 7.4e-6;
    cfg->frame_width        = 648;
    cfg->frame_height       = 492;
    cfg->totlenses          = 140;
    cfg->flength            = 18e-3;
    cfg->pitch              = 300e-6;
    cfg->sa_radius          = 150e-6;
    cfg->pupil_radius       = 2e-3;
    cfg->wavelength         = 632.8e-9;
    cfg->thresh_binary      = 0.08;
    cfg->centroid_percent   = 0.2;
    cfg->coarse_grid_radius = 12;
    cfg->zernike_nmax       = 5;
    cfg->dm_nact_x          = 12;
    cfg->dm_nact_y          = 12;
    cfg->coupling           = 0.15;
}

/* ---- config ------------------------------------------------------------ */

int rippa_config_load(rippa_config *cfg, const char *path)
{
    FILE *fp;
    char line[512];

    default_config(cfg);

    fp = fopen(path, "r");
    if (!fp) return 0; /* missing file -> use defaults, not an error */

    while (fgets(line, sizeof(line), fp)) {
        char *s = strip(line);
        if (s[0] == '#' || s[0] == '\0') continue;
        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = strip(s);
        char *val = strip(eq + 1);

        if      (!strcmp(key, "camera_pixsize"))    cfg->camera_pixsize = atof(val);
        else if (!strcmp(key, "frame_width"))       cfg->frame_width = atoi(val);
        else if (!strcmp(key, "frame_height"))      cfg->frame_height = atoi(val);
        else if (!strcmp(key, "totlenses"))         cfg->totlenses = atoi(val);
        else if (!strcmp(key, "flength"))           cfg->flength = atof(val);
        else if (!strcmp(key, "pitch"))             cfg->pitch = atof(val);
        else if (!strcmp(key, "sa_radius"))         cfg->sa_radius = atof(val);
        else if (!strcmp(key, "pupil_radius"))      cfg->pupil_radius = atof(val);
        else if (!strcmp(key, "wavelength"))        cfg->wavelength = atof(val);
        else if (!strcmp(key, "thresh_binary"))     cfg->thresh_binary = atof(val);
        else if (!strcmp(key, "centroid_percent"))  cfg->centroid_percent = atof(val);
        else if (!strcmp(key, "coarse_grid_radius"))cfg->coarse_grid_radius = atoi(val);
        else if (!strcmp(key, "zernike_nmax"))      cfg->zernike_nmax = atoi(val);
        else if (!strcmp(key, "dm_nact_x"))         cfg->dm_nact_x = atoi(val);
        else if (!strcmp(key, "dm_nact_y"))         cfg->dm_nact_y = atoi(val);
        else if (!strcmp(key, "coupling"))          cfg->coupling = atof(val);
    }
    fclose(fp);
    return 0;
}

/* ---- raw loader -------------------------------------------------------- */

int rippa_load_raw(const char *path, int width, int height, double **data)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }

    size_t n = (size_t)width * (size_t)height;
    *data = (double *)malloc(n * sizeof(double));
    if (!*data) { fclose(fp); return -2; }

    size_t nr = fread(*data, sizeof(double), n, fp);
    fclose(fp);
    if (nr != n) { free(*data); return -3; }
    return 0;
}

/* ---- BMP loader -------------------------------------------------------- */
/*
 * Minimal BMP reader: handles 8-bit and 16-bit uncompressed (BI_RGB).
 * Reads the pixel data rows (bottom-up in BMP, flipped to top-left origin).
 * Returns pixel values normalised to [0, 1] double.
 */
int rippa_load_bmp(const char *path, int *out_width, int *out_height,
                   double **data)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }

    unsigned char hdr[54];
    if (fread(hdr, 1, 54, fp) != 54) { fclose(fp); return -3; }
    /* check "BM" signature */
    if (hdr[0] != 'B' || hdr[1] != 'M') { fclose(fp); return -3; }

    int offset = *(int *)(hdr + 10);
    int w = *(int *)(hdr + 18);
    int h = *(int *)(hdr + 22);
    short bpp = *(short *)(hdr + 28);
    unsigned int comp = *(unsigned int *)(hdr + 30);

    /* Validate offset and image dimensions to protect against buffer issues */
    if (offset < 54 || offset > 10000000) {
        fprintf(stderr, "ERROR: invalid BMP offset %d\n", offset);
        fclose(fp); return -3;
    }
    int abs_h = (h < 0) ? -h : h;
    if (w <= 0 || abs_h == 0 || w > 16384 || abs_h > 16384) {
        fprintf(stderr, "ERROR: invalid BMP dimensions %dx%d\n", w, h);
        fclose(fp); return -3;
    }

    if (comp != 0) {
        fprintf(stderr, "ERROR: only uncompressed BMP supported\n");
        fclose(fp); return -3;
    }
    if (bpp != 8 && bpp != 16 && bpp != 24 && bpp != 32) {
        fprintf(stderr, "ERROR: unsupported bpp=%d\n", bpp);
        fclose(fp); return -3;
    }

    int top_down = (h < 0);
    int actual_h = abs_h;

    int channels = bpp / 8;
    if (channels < 1) channels = 1;

    /* row stride with 4-byte alignment */
    int rowbytes = w * channels;
    int stride = (rowbytes + 3) & ~3;

    unsigned char *rowbuf = (unsigned char *)malloc(stride);
    if (!rowbuf) { fclose(fp); return -2; }

    size_t n = (size_t)w * (size_t)actual_h;
    *data = (double *)malloc(n * sizeof(double));
    if (!*data) { free(rowbuf); fclose(fp); return -2; }

    double maxval = (bpp == 16) ? 65535.0 : 255.0;

    if (fseek(fp, offset, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: failed to seek to BMP offset\n");
        free(rowbuf); free(*data); *data = NULL;
        fclose(fp); return -3;
    }

    int y;
    for (y = 0; y < actual_h; ++y) {
        int destrow = top_down ? y : (actual_h - 1 - y);
        if (fread(rowbuf, 1, stride, fp) != (size_t)stride) {
            fprintf(stderr, "ERROR: truncated/corrupted BMP file during pixel read at row %d\n", y);
            free(rowbuf); free(*data); *data = NULL;
            fclose(fp); return -4;
        }
        int x;
        for (x = 0; x < w; ++x) {
            double val;
            if (bpp >= 24) {
                /* use first channel (blue in BGR) or luminance approx */
                val = rowbuf[x * channels + 0];
            } else if (bpp == 16) {
                val = (double)rowbuf[2 * x] + (double)rowbuf[2 * x + 1] * 256.0;
            } else {
                val = rowbuf[x];
            }
            (*data)[destrow * w + x] = val / maxval;
        }
    }

    *out_width = w;
    *out_height = actual_h;
    free(rowbuf);
    fclose(fp);
    return 0;
}

/* ---- BMP loader with config validation ---------------------------------- */

int rippa_load_bmp_with_config(const char *path, const rippa_config *cfg,
                               double **data)
{
    int w, h;
    int ret = rippa_load_bmp(path, &w, &h, data);
    if (ret != 0) return ret;

    if (w != cfg->frame_width) {
        fprintf(stderr, "ERROR: BMP width %d does not match config frame_width %d\n",
                w, cfg->frame_width);
        free(*data); *data = NULL;
        return -5;
    }
    if (h != cfg->frame_height) {
        fprintf(stderr, "ERROR: BMP height %d does not match config frame_height %d\n",
                h, cfg->frame_height);
        free(*data); *data = NULL;
        return -5;
    }
    return 0;
}

/* ---- raw writer -------------------------------------------------------- */

int rippa_save_raw(const char *path, const double *data, int width, int height)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t n = (size_t)width * (size_t)height;
    fwrite(data, sizeof(double), n, fp);
    fclose(fp);
    return 0;
}

/* ---- CSV writer -------------------------------------------------------- */

int rippa_save_csv(const char *path, const double *data, int width, int height)
{
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    int y, x;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            if (x > 0) fprintf(fp, ",");
            fprintf(fp, "%.10f", data[y * width + x]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}
