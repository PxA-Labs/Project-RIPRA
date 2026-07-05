/*
 * rippa/io.h - file I/O: raw float loader, BMP reader, config parser
 *
 * All WFS frames are stored internally as row-major doubles in a
 * contiguous array of width*height elements (image[row*width + col]).
 *
 * Supported inputs:
 *   - .raw  : raw IEEE-754 doubles, row-major, width*height*8 bytes
 *   - .bmp  : uncompressed 8-bit or 16-bit BMP (no palette for 16-bit)
 *   - .conf : key=value text config file (see config/system.conf)
 */
#ifndef RIPRA_IO_H
#define RIPRA_IO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- config ------------------------------------------------------------- */

typedef struct {
    /* camera */
    double  camera_pixsize;      /* m          */
    int     frame_width;
    int     frame_height;
    /* MLA / lenslet */
    int     totlenses;
    double  flength;             /* m          */
    double  pitch;               /* m          */
    double  sa_radius;           /* m  (= pitch/2) */
    /* pupil */
    double  pupil_radius;        /* m          */
    /* wavelength */
    double  wavelength;          /* m          */
    /* centroiding */
    double  thresh_binary;        /* 0..1 relative */
    double  centroid_percent;     /* 0..1 relative */
    int     coarse_grid_radius;   /* pixels     */
    /* reconstruction */
    int     zernike_nmax;         /* max radial order */
    /* DM */
    int     dm_nact_x;            /* actuators across X */
    int     dm_nact_y;            /* actuators across Y */
    double  coupling;             /* inter-actuator coupling coeff */
} rippa_config;

/*
 * Parse a .conf file. Returns 0 on success. Unknown keys are ignored.
 * Defaults are filled for every field so a partial/empty file is OK.
 */
int rippa_config_load(rippa_config *cfg, const char *path);

/* ---- raw float loader --------------------------------------------------- */

/*
 * Load a raw file of width*height doubles (row-major, IEEE-754, native endian).
 * Caller must free(*data) when done. Returns 0 on success.
 */
int rippa_load_raw(const char *path, int width, int height, double **data);

/* ---- BMP loader --------------------------------------------------------- */

/*
 * Load an uncompressed 8-bit or 16-bit BMP file.
 * Image is flipped bottom-to-top to match row-major top-left origin.
 * Pixel values are normalized to [0,1] double.
 * Caller must free(*data). Returns 0 on success.
 */
int rippa_load_bmp(const char *path, int *out_width, int *out_height,
                   double **data);

/*
 * Load a BMP with config validation.
 * Same as rippa_load_bmp, but also validates that the image dimensions
 * match cfg->frame_width and cfg->frame_height before decoding.
 * Returns 0 on success, negative on error.
 */
int rippa_load_bmp_with_config(const char *path, const rippa_config *cfg,
                               double **data);

/* ---- raw writer --------------------------------------------------------- */

/*
 * Write width*height doubles to a raw binary file.
 * Returns 0 on success.
 */
int rippa_save_raw(const char *path, const double *data, int width, int height);

/* ---- CSV writer --------------------------------------------------------- */

/*
 * Write width*height doubles as CSV (comma-separated, one row per line).
 * Returns 0 on success.
 */
int rippa_save_csv(const char *path, const double *data, int width, int height);

#ifdef __cplusplus
}
#endif
#endif /* RIPRA_IO_H */
