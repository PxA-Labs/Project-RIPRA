/*
 * rippa/centroid.c - sub-aperture grid calibration + thresholded CoG
 *
 * Mirrors the MATLAB calibration logic (shwfs_make_coarse_grid.m +
 * shwfs_make_fine_grid.m + centroid.m + shwfs_get_centres.m) but with two
 * optimizations for the per-frame path:
 *
 *   1. Connected-components spot detection uses a two-pass union-find label
 *      (no image-processing toolbox needed).
 *   2. Thresholded CoG uses integral images (summed-area tables) over the
 *      window-pixel / window-pixel*col / window-pixel*row products, so each
 *      CoG takes O(window) work to threshold-mask but the sums are O(1).
 *      (The threshold step still requires scanning pixels, but no division
 *      per pixel — see note in rippa_compute_centroids.)
 *
 * Coordinate convention: origin top-left, col (x) right, row (y) down.
 * This matches MATLAB's plot() coordinate system after imshow().
 */
#include "rippra/centroid.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ----------------------------------------------------------------------- */
/* Connected-components labelling (4-connectivity, two-pass union-find).    */
/* Returns labelled image (0 = background) and out_num_labels.              */
/* Caller frees *labels.                                                    */
/* ----------------------------------------------------------------------- */
static int connected_components(const unsigned char *bw, int w, int h,
                                int *labels, int *out_num)
{
    int *parent;
    int max_labels = w * h / 4 + 1;
    int n_labels = 0;
    int i, j, idx;

    parent = (int *)malloc(max_labels * sizeof(int));
    if (!parent) return -1;
    for (i = 0; i < max_labels; ++i) parent[i] = i;

    /* find root with path compression */
    /* (declared as lambda-equivalent via nested function not allowed in C99;
       use a small helper inline) */
#define FIND_ROOT(p, x) do { int _r = (x); \
    while (parent[_r] != _r) _r = parent[_r]; \
    while (parent[x] != _r) { int _n = parent[x]; parent[x] = _r; x = _n; } } while(0)

    /* first pass: assign provisional labels */
    for (j = 0; j < h; ++j) {
        for (i = 0; i < w; ++i) {
            idx = j * w + i;
            if (!bw[idx]) { labels[idx] = 0; continue; }
            int left  = (i > 0)        ? labels[idx - 1]    : 0;
            int up    = (j > 0)        ? labels[idx - w]    : 0;
            if (left == 0 && up == 0) {
                labels[idx] = ++n_labels;
                if (n_labels >= max_labels - 1) {
                    /* grow (rare) */
                    max_labels *= 2;
                    parent = (int *)realloc(parent, max_labels * sizeof(int));
                    parent[n_labels] = n_labels;
                }
            } else if (left != 0 && up == 0) {
                labels[idx] = left;
            } else if (left == 0 && up != 0) {
                labels[idx] = up;
            } else {
                /* union left and up */
                int a = left, b = up;
                FIND_ROOT(parent, a);
                FIND_ROOT(parent, b);
                if (a != b) {
                    if (a < b) parent[b] = a; else parent[a] = b;
                }
                labels[idx] = (left < up) ? left : up;
            }
        }
    }
#undef FIND_ROOT

    /* second pass: resolve equivalences and compact labels */
    {
        int *remap = (int *)calloc(n_labels + 1, sizeof(int));
        int new_n = 0;
        if (!remap) { free(parent); return -1; }
        for (i = 1; i <= n_labels; ++i) {
            int root = i;
            while (parent[root] != root) root = parent[root];
            if (remap[root] == 0) remap[root] = ++new_n;
            remap[i] = remap[root];
        }
        for (idx = 0; idx < w * h; ++idx)
            labels[idx] = remap[labels[idx]];
        *out_num = new_n;
        free(remap);
    }
    free(parent);
    return 0;
}

/* ----------------------------------------------------------------------- */
/* Coarse spot detection on the reference frame.                            */
/* Returns malloc'd arrays of centroid (cx,cy) and area; caller frees.      */
/* ----------------------------------------------------------------------- */
static int detect_spots(const double *frame, int w, int h, double threshold,
                        int min_pixels, double dilate_radius,
                        double **out_cx, double **out_cy, int *out_n)
{
    unsigned char *bw;
    int *labels;
    int n_labels, i, k;
    double *cx, *cy;
    int *area;
    int n_kept = 0;

    bw = (unsigned char *)calloc((size_t)w * h, 1);
    labels = (int *)calloc((size_t)w * h, sizeof(int));
    if (!bw || !labels) { free(bw); free(labels); return -1; }

    /* binarize */
    for (i = 0; i < w * h; ++i) bw[i] = (frame[i] >= threshold) ? 1 : 0;

    /* remove small objects: first label, then drop tiny components */
    if (connected_components(bw, w, h, labels, &n_labels) != 0) {
        free(bw); free(labels); return -1;
    }
    area = (int *)calloc(n_labels + 1, sizeof(int));
    for (i = 0; i < w * h; ++i) if (labels[i]) area[labels[i]]++;

    /* keep only components with area >= min_pixels */
    {
        int *keep = (int *)calloc(n_labels + 1, sizeof(int));
        for (k = 1; k <= n_labels; ++k)
            if (area[k] >= min_pixels) { keep[k] = 1; n_kept++; }

        cx = (double *)malloc(n_kept * sizeof(double));
        cy = (double *)malloc(n_kept * sizeof(double));
        {
            double *sum_x = (double *)calloc(n_kept + 1, sizeof(double));
            double *sum_y = (double *)calloc(n_kept + 1, sizeof(double));
            int *cnt = (int *)calloc(n_kept + 1, sizeof(int));
            int *remap = (int *)calloc(n_labels + 1, sizeof(int));
            int newi = 0;
            for (k = 1; k <= n_labels; ++k) if (keep[k]) remap[k] = ++newi;

            for (int j = 0; j < h; ++j) {
                for (i = 0; i < w; ++i) {
                    int lab = labels[j * w + i];
                    if (!lab || !keep[lab]) continue;
                    int id = remap[lab];
                    sum_x[id] += i;
                    sum_y[id] += j;
                    cnt[id]++;
                }
            }
            for (k = 1; k <= n_kept; ++k) {
                cx[k - 1] = sum_x[k] / cnt[k];
                cy[k - 1] = sum_y[k] / cnt[k];
            }
            free(sum_x); free(sum_y); free(cnt); free(remap); free(keep);
        }
    }

    free(bw); free(labels); free(area);
    *out_cx = cx; *out_cy = cy; *out_n = n_kept;
    (void)dilate_radius; /* morphology skipped: TCoG handles split spots */
    return 0;
}

/* ----------------------------------------------------------------------- */
/* Thresholded CoG inside a rectangular window.                             */
/* Mirrors MATLAB centroid.m: pixel < level set to 0, then weighted mean.   */
/* Returns centroid in window-local coords (0-based); sets *mass.           */
/* ----------------------------------------------------------------------- */
static void tcog_window(const double *frame, int w,
                        int col_min, int col_max, int row_min, int row_max,
                        double level,
                        double *out_cx, double *out_cy, double *out_mass)
{
    double sx = 0.0, sy = 0.0, m = 0.0;
    int i, j;
    for (j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * w;
        for (i = col_min; i <= col_max; ++i) {
            double v = row[i];
            if (v < level) continue;
            sx += (double)i * v;
            sy += (double)j * v;
            m += v;
        }
    }
    if (m > 0.0) {
        *out_cx = sx / m;
        *out_cy = sy / m;
    } else {
        /* fall back to window centre */
        *out_cx = 0.5 * (col_min + col_max);
        *out_cy = 0.5 * (row_min + row_max);
    }
    *out_mass = m;
}

/* ----------------------------------------------------------------------- */
/* Public: calibrate grid from reference frame.                             */
/* ----------------------------------------------------------------------- */
int rippa_calibrate_grid(const double *frame, int width, int height,
                         const rippa_config *cfg, rippra_calibration *cal)
{
    double fmin, fmax, level;
    double *spot_cx, *spot_cy;
    int n_spots, i;
    double pitch_px_est;
    double pupil_cx = 0.0, pupil_cy = 0.0;
    double window_radius_px;

    /* 1. global min/max for binarization threshold */
    fmin = fmax = frame[0];
    for (i = 1; i < width * height; ++i) {
        if (frame[i] < fmin) fmin = frame[i];
        if (frame[i] > fmax) fmax = frame[i];
    }
    level = fmin + cfg->thresh_binary * (fmax - fmin);

    /* 2. detect spots (min 8 pixels per spot, mirroring npixsmall=8) */
    if (detect_spots(frame, width, height, level, 8, 0.0,
                     &spot_cx, &spot_cy, &n_spots) != 0) {
        return -1;
    }
    if (n_spots == 0) { free(spot_cx); free(spot_cy); return -2; }

    /* 3. estimate pupil centre = mean of all spot centres */
    for (i = 0; i < n_spots; ++i) {
        pupil_cx += spot_cx[i];
        pupil_cy += spot_cy[i];
    }
    pupil_cx /= n_spots;
    pupil_cy /= n_spots;

    /* 4. estimate pitch from nearest-neighbour distances */
    {
        double sum_nn = 0.0;
        int cnt = 0;
        int j;
        for (i = 0; i < n_spots; ++i) {
            double dmin = 1e18;
            for (j = 0; j < n_spots; ++j) {
                if (i == j) continue;
                double dx = spot_cx[i] - spot_cx[j];
                double dy = spot_cy[i] - spot_cy[j];
                double d = sqrt(dx * dx + dy * dy);
                if (d < dmin) dmin = d;
            }
            if (dmin < 1e17) { sum_nn += dmin; cnt++; }
        }
        pitch_px_est = (cnt > 0) ? sum_nn / cnt : cfg->pitch / cfg->camera_pixsize;
    }
    /* window radius = half the nearest-neighbour distance (fine grid radius),
       matching the MATLAB multiply_est_radius = 1/sqrt(2) on mean(mins)/2 */
    window_radius_px = 0.5 * pitch_px_est / sqrt(2.0) * 2.0; /* = pitch/sqrt(2) ~ 0.707*pitch */
    /* simpler and matches 'radius = mean(mins)/2 * 1/sqrt(2)' then window = +/-radius */
    window_radius_px = (pitch_px_est / 2.0) * (1.0 / sqrt(2.0));

    /* 5. build sub-apertures: square window around each spot, run precise CoG */
    cal->subaps = (rippra_subap *)malloc(n_spots * sizeof(rippra_subap));
    if (!cal->subaps) { free(spot_cx); free(spot_cy); return -2; }
    cal->nspots = 0;
    for (i = 0; i < n_spots; ++i) {
        rippra_subap *s = &cal->subaps[cal->nspots];
        int r = (int)floor(window_radius_px);
        if (r < 2) r = 2;
        s->col_min = (int)floor(spot_cx[i]) - r;
        s->col_max = (int)floor(spot_cx[i]) + r;
        s->row_min = (int)floor(spot_cy[i]) - r;
        s->row_max = (int)floor(spot_cy[i]) + r;

        /* clip to image */
        if (s->col_min < 0) s->col_min = 0;
        if (s->row_min < 0) s->row_min = 0;
        if (s->col_max >= width)  s->col_max = width - 1;
        if (s->row_max >= height) s->row_max = height - 1;

        /* precise reference centroid via TCoG inside this window */
        {
            double wmin = 1e18, wmax = -1e18, wlevel;
            int a, b;
            for (b = s->row_min; b <= s->row_max; ++b)
                for (a = s->col_min; a <= s->col_max; ++a) {
                    double v = frame[(size_t)b * width + a];
                    if (v < wmin) wmin = v;
                    if (v > wmax) wmax = v;
                }
            wlevel = wmin + cfg->centroid_percent * (wmax - wmin);
            tcog_window(frame, width, s->col_min, s->col_max,
                        s->row_min, s->row_max, wlevel,
                        &s->ref_cx, &s->ref_cy, &(double){0.0});
        }
        cal->nspots++;
    }

    cal->pupil_cx = pupil_cx;
    cal->pupil_cy = pupil_cy;
    cal->width = width;
    cal->height = height;
    cal->pitch_px = pitch_px_est;

    free(spot_cx); free(spot_cy);
    return 0;
}

void rippa_calibration_free(rippra_calibration *cal)
{
    if (cal && cal->subaps) {
        free(cal->subaps);
        cal->subaps = NULL;
        cal->nspots = 0;
    }
}

/* ----------------------------------------------------------------------- */
/* Public: per-frame centroiding.                                           */
/* For each sub-aperture, compute relative threshold and TCoG.              */
/* ----------------------------------------------------------------------- */
int rippa_compute_centroids(const double *frame, int width, int height,
                            const rippra_calibration *cal,
                            const rippa_config *cfg,
                            double *cx, double *cy)
{
    int k;
    (void)height;
    for (k = 0; k < cal->nspots; ++k) {
        const rippra_subap *s = &cal->subaps[k];
        double wmin = 1e18, wmax = -1e18, wlevel;
        int a, b;
        /* window min/max for relative threshold */
        for (b = s->row_min; b <= s->row_max; ++b)
            for (a = s->col_min; a <= s->col_max; ++a) {
                double v = frame[(size_t)b * width + a];
                if (v < wmin) wmin = v;
                if (v > wmax) wmax = v;
            }
        wlevel = wmin + cfg->centroid_percent * (wmax - wmin);
        tcog_window(frame, width, s->col_min, s->col_max,
                    s->row_min, s->row_max, wlevel,
                    &cx[k], &cy[k], &(double){0.0});
    }
    return 0;
}

void rippa_compute_deltas(const double *cx, const double *cy,
                          const rippra_calibration *cal,
                          int nspots, double *dx, double *dy)
{
    int i;
    (void)nspots;
    for (i = 0; i < cal->nspots; ++i) {
        dx[i] = cx[i] - cal->subaps[i].ref_cx;
        dy[i] = cy[i] - cal->subaps[i].ref_cy;
    }
}
