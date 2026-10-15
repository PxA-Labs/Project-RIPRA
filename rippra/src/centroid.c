/*
 * rippa/centroid.c - sub-aperture grid calibration + thresholded CoG
 *
 * Mirrors the MATLAB calibration logic (shwfs_make_coarse_grid.m +
 * shwfs_make_fine_grid.m + centroid.m + shwfs_get_centres.m) with an
 * optimization for the per-frame path:
 *
 *   1. Connected-components spot detection uses a two-pass union-find label
 *      (no image-processing toolbox needed).
 *
 * Per-frame TCoG uses direct window scanning, which is optimal for typical
 * SH-WFS window sizes (5-7 px): O(window) per sub-aperture totals ~3k
 * pixels vs O(width*height) ~300k for a summed-area table precomputation.
 *
 * Coordinate convention: origin top-left, col (x) right, row (y) down.
 * This matches MATLAB's plot() coordinate system after imshow().
 */
#include "rippra/centroid.h"
#include "rippra/simd.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define RIPPRA_QUALITY_MAGIC 0x51554c31u

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
                    int *tmp = (int *)realloc(parent, max_labels * sizeof(int));
                    if (!tmp) return -1;
                    parent = tmp;
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
/* Combined pass: compute min/max AND run TCoG in a single window scan.
   Dispatches to SIMD-accelerated path when AVX2 is available. */
static void tcog_window_fast(const double *frame, int w,
                             int col_min, int col_max, int row_min, int row_max,
                             double centroid_percent,
                             double *out_cx, double *out_cy, double *out_mass)
{
    rippra_simd_tcog_window_fast(frame, w, col_min, col_max, row_min, row_max,
                                 centroid_percent, out_cx, out_cy, out_mass);
}

/* Compute the non-centroid statistics for one already-bounded window.  The
 * threshold is deliberately the same relative min/max threshold used by the
 * TCoG kernel.  Background statistics use pixels below that threshold and
 * the shape statistic is the intensity-weighted radial second moment of the
 * accepted pixels. */
static void compute_window_metrics(const double *frame, int w,
                                   int col_min, int col_max,
                                   int row_min, int row_max,
                                   double centroid_percent,
                                   double cx, double cy,
                                   rippra_spot_metrics *out)
{
    double mn = 1e18, mx = -1e18;
    double bg_sum = 0.0, bg_sum2 = 0.0;
    double signal_sum = 0.0, moment_sum = 0.0;
    double moment_x = 0.0, moment_y = 0.0, signal_peak = -1e18;
    int bg_count = 0, signal_count = 0;

    memset(out, 0, sizeof(*out));
    for (int row = row_min; row <= row_max; ++row) {
        const double *pixels = frame + (size_t)row * w;
        for (int col = col_min; col <= col_max; ++col) {
            double value = pixels[col];
            if (!isfinite(value)) continue;
            if (value < mn) mn = value;
            if (value > mx) mx = value;
        }
    }

    if (mn == 1e18 || mx == -1e18) return;
    double level = mn + centroid_percent * (mx - mn);
    for (int row = row_min; row <= row_max; ++row) {
        const double *pixels = frame + (size_t)row * w;
        for (int col = col_min; col <= col_max; ++col) {
            double value = pixels[col];
            if (!isfinite(value)) continue;
            if (value >= level) {
                signal_sum += value;
                signal_count++;
                if (value > signal_peak) signal_peak = value;
                if (isfinite(cx) && isfinite(cy)) {
                    double dx = (double)col - cx;
                    double dy = (double)row - cy;
                    moment_sum += value * (dx * dx + dy * dy);
                    moment_x += value * dx * dx;
                    moment_y += value * dy * dy;
                }
            } else {
                bg_sum += value;
                bg_sum2 += value * value;
                bg_count++;
            }
        }
    }

    out->mass = signal_sum;
    if (signal_count > 0) {
        out->signal_mean = signal_sum / signal_count;
        out->signal_peak = signal_peak;
        if (signal_sum > 1e-12 && isfinite(cx) && isfinite(cy)) {
            out->second_moment = moment_sum / signal_sum;
            out->second_moment_x = moment_x / signal_sum;
            out->second_moment_y = moment_y / signal_sum;
        }
    }
    if (bg_count > 0) {
        out->background_mean = bg_sum / bg_count;
        double variance = bg_sum2 / bg_count -
                          out->background_mean * out->background_mean;
        out->background_std = sqrt(variance > 0.0 ? variance : 0.0);
    }
}

static int compare_quality_values(const void *a, const void *b)
{
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double quality_clamp(double value)
{
    if (!isfinite(value) || value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

static double quality_median(double *values, int count)
{
    if (count <= 0) return 0.0;
    qsort(values, (size_t)count, sizeof(double), compare_quality_values);
    if (count & 1) return values[count / 2];
    return 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double quality_spatial_coherence(const double *dx, const double *dy,
                                        const int *mask, int nspots,
                                        const rippra_calibration *cal)
{
    double residual_sum = 0.0, displacement_sum = 0.0;
    int residual_count = 0;
    double pitch = (cal && cal->pitch_px > 0.0) ? cal->pitch_px : 0.0;
    double neighbour_limit = pitch > 0.0 ? 1.51 * pitch : 1e30;

    if (!dx || !dy || !mask || !cal || !cal->subaps || pitch <= 0.0)
        return 1.0;

    for (int i = 0; i < nspots; ++i) {
        double mean_x = 0.0, mean_y = 0.0;
        int neighbours = 0;
        if (!mask[i]) continue;
        if (cal->quality_magic == RIPPRA_QUALITY_MAGIC &&
            cal->quality_neighbor_count && cal->quality_neighbors) {
            int count = cal->quality_neighbor_count[i];
            for (int k = 0; k < count; ++k) {
                int j = cal->quality_neighbors[i * 8 + k];
                if (j >= 0 && mask[j]) {
                    mean_x += dx[j];
                    mean_y += dy[j];
                    neighbours++;
                }
            }
        } else {
            for (int j = 0; j < nspots; ++j) {
                if (i == j || !mask[j]) continue;
                double px = cal->subaps[i].ref_cx - cal->subaps[j].ref_cx;
                double py = cal->subaps[i].ref_cy - cal->subaps[j].ref_cy;
                if (fabs(px) <= neighbour_limit && fabs(py) <= neighbour_limit) {
                    mean_x += dx[j];
                    mean_y += dy[j];
                    neighbours++;
                }
            }
        }
        displacement_sum += dx[i] * dx[i] + dy[i] * dy[i];
        if (neighbours > 0) {
            mean_x /= neighbours;
            mean_y /= neighbours;
            double ex = dx[i] - mean_x;
            double ey = dy[i] - mean_y;
            residual_sum += ex * ex + ey * ey;
            residual_count++;
        }
    }
    if (residual_count == 0) return 1.0;
    double chi = sqrt(residual_sum) / sqrt(displacement_sum + 1e-12);
    return quality_clamp(1.0 - chi);
}

static double quality_shape_term(const rippra_spot_metrics *metrics,
                                 const rippra_calibration *cal,
                                 const int *mask, int nspots,
                                 double *values)
{
    int count = 0;
    if (!metrics || !cal || !mask ||
        cal->quality_magic != RIPPRA_QUALITY_MAGIC ||
        !cal->ref_second_moment) return 1.0;
    for (int i = 0; i < nspots; ++i) {
        if (!mask[i]) continue;
        double reference;
        double current;
        if (cal->ref_second_moment_x && cal->ref_second_moment_y &&
            metrics[i].second_moment_x > 1e-12 &&
            metrics[i].second_moment_y > 1e-12) {
            reference = cal->ref_second_moment_x[i] *
                        cal->ref_second_moment_y[i];
            current = metrics[i].second_moment_x * metrics[i].second_moment_y;
        } else {
            reference = cal->ref_second_moment[i];
            current = metrics[i].second_moment;
        }
        if (reference > 1e-12 && current > 1e-12) {
            values[count++] = quality_clamp(1.0 -
                fabs(current - reference) / reference);
        }
    }
    return count > 0 ? quality_median(values, count) : 1.0;
}

double rippra_frame_quality_with_metrics(
    const double *dx, const double *dy, const double *mass, const int *mask,
    const rippra_spot_metrics *metrics, int nspots,
    const rippa_config *cfg, const rippra_calibration *cal,
    double *out_metrics)
{
    double values[256];
    double *heap_values = NULL;
    double *terms = values;
    int valid = 0, value_count = 0;
    double validity, mass_term, shape_term, coherence_term;

    if (!dx || !dy || !mass || !mask || !cfg || !cal || nspots <= 0 ||
        nspots != cal->nspots) {
        if (out_metrics) memset(out_metrics, 0, 4 * sizeof(double));
        return 0.0;
    }
    if (nspots > (int)(sizeof(values) / sizeof(values[0]))) {
        heap_values = (double *)malloc((size_t)nspots * sizeof(double));
        if (!heap_values) {
            if (out_metrics) memset(out_metrics, 0, 4 * sizeof(double));
            return 0.0;
        }
        terms = heap_values;
    }

    for (int i = 0; i < nspots; ++i) {
        if (!mask[i]) continue;
        valid++;
        if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_mass &&
            cal->ref_mass[i] > 1e-12 &&
            isfinite(mass[i]) && mass[i] >= 0.0) {
            terms[value_count++] = quality_clamp(mass[i] / cal->ref_mass[i]);
        }
    }
    validity = (double)valid / nspots;
    if (valid == 0) {
        if (out_metrics) memset(out_metrics, 0, 4 * sizeof(double));
        free(heap_values);
        return 0.0;
    }
    mass_term = (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_mass) ?
        (value_count > 0 ? quality_median(terms, value_count) : 0.0) : 1.0;

    /* Include current SNR in the flux term when the streaming path supplied
     * the per-spot metrics.  The public compact API has no frame pixels from
     * which to derive current background, so it uses the mass term only. */
    if (metrics) {
        value_count = 0;
        for (int i = 0; i < nspots; ++i) {
            if (!mask[i] || cal->quality_magic != RIPPRA_QUALITY_MAGIC ||
                !cal->ref_snr || cal->ref_snr[i] <= 1e-9)
                continue;
            double snr = (metrics[i].signal_peak - metrics[i].background_mean) /
                         (metrics[i].background_std + 1e-9);
            if (isfinite(snr) && snr >= 0.0)
                terms[value_count++] = quality_clamp(snr / cal->ref_snr[i]);
        }
        if (value_count > 0)
            mass_term = 0.75 * mass_term + 0.25 * quality_median(terms, value_count);
    }

    value_count = 0;
    shape_term = quality_shape_term(metrics, cal, mask, nspots, terms);
    coherence_term = quality_spatial_coherence(dx, dy, mask, nspots, cal);
    if (out_metrics) {
        out_metrics[0] = validity;
        out_metrics[1] = mass_term;
        out_metrics[2] = shape_term;
        out_metrics[3] = coherence_term;
    }
    free(heap_values);

    /* Weights sum to one; every term is explicitly clamped. */
    return quality_clamp(0.85 * quality_clamp(validity) +
                         0.05 * quality_clamp(mass_term) +
                         0.05 * quality_clamp(shape_term) +
                         0.05 * quality_clamp(coherence_term));
}

double rippra_frame_quality(const double *dx, const double *dy,
                            const double *mass, const int *mask,
                            int nspots, const rippa_config *cfg,
                            const rippra_calibration *cal,
                            double *out_metrics)
{
    return rippra_frame_quality_with_metrics(
        dx, dy, mass, mask, NULL, nspots, cfg, cal, out_metrics);
}

int rippra_quality_calibrate(const double *frame, int width, int height,
                            const rippra_calibration *cal,
                            const rippa_config *cfg,
                            rippra_quality_ref *ref)
{
    if (!frame || !cal || !cfg || !ref || width != cal->width || height != cal->height ||
        cal->quality_magic != RIPPRA_QUALITY_MAGIC || !cal->ref_mass ||
        !cal->ref_background_mean ||
        !cal->ref_background_std || !cal->ref_second_moment ||
        cal->nspots <= 0) return -1;

    memset(ref, 0, sizeof(*ref));
    ref->nspots = cal->nspots;
    ref->ref_mass = cal->ref_mass;
    ref->ref_second_moment = cal->ref_second_moment;
    ref->ref_second_moment_x = cal->ref_second_moment_x;
    ref->ref_second_moment_y = cal->ref_second_moment_y;
    ref->ref_background_mean = cal->ref_background_mean;
    ref->ref_background_sigma = cal->ref_background_std;
    ref->bg_mean = cal->bg_mean;
    ref->bg_sigma = cal->bg_sigma;
    return 0;
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

    if (!frame || !cfg || !cal || width <= 0 || height <= 0 ||
        !isfinite(cfg->thresh_binary) || cfg->thresh_binary < 0.0 ||
        cfg->thresh_binary > 1.0 || !isfinite(cfg->centroid_percent) ||
        cfg->centroid_percent < 0.0 || cfg->centroid_percent > 1.0)
        return -3;

    /* The caller owns any previous calibration and should free it first. */
    memset(cal, 0, sizeof(*cal));

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
    cal->ref_mass = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_background_mean = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_background_std = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_snr = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_second_moment = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_second_moment_x = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->ref_second_moment_y = (double *)calloc((size_t)n_spots, sizeof(double));
    cal->quality_neighbor_count = (int *)calloc((size_t)n_spots, sizeof(int));
    cal->quality_neighbors = (int *)malloc((size_t)n_spots * 8 * sizeof(int));
    cal->quality_magic = RIPPRA_QUALITY_MAGIC;
    if (!cal->subaps || !cal->ref_mass || !cal->ref_background_mean ||
        !cal->ref_background_std || !cal->ref_snr || !cal->ref_second_moment ||
        !cal->ref_second_moment_x || !cal->ref_second_moment_y ||
        !cal->quality_neighbor_count || !cal->quality_neighbors) {
        free(spot_cx); free(spot_cy);
        rippa_calibration_free(cal);
        return -2;
    }
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

        /* Precise reference centroid and quality statistics via TCoG inside
         * this window.  Invalid reference spots are not admitted into the
         * calibration, so the quality gate never compares against a made-up
         * baseline. */
        {
            rippra_spot_metrics metrics;
            tcog_window_fast(frame, width, s->col_min, s->col_max,
                             s->row_min, s->row_max, cfg->centroid_percent,
                             &s->ref_cx, &s->ref_cy, &metrics.mass);
            compute_window_metrics(frame, width, s->col_min, s->col_max,
                                   s->row_min, s->row_max,
                                   cfg->centroid_percent, s->ref_cx, s->ref_cy,
                                   &metrics);
            if (!isfinite(s->ref_cx) || !isfinite(s->ref_cy) ||
                !isfinite(metrics.mass) || metrics.mass <= 1e-9) {
                continue;
            }
            cal->ref_mass[cal->nspots] = metrics.mass;
            cal->ref_background_mean[cal->nspots] = metrics.background_mean;
            cal->ref_background_std[cal->nspots] = metrics.background_std;
            cal->ref_snr[cal->nspots] =
                (metrics.signal_peak - metrics.background_mean) /
                (metrics.background_std + 1e-9);
            cal->ref_second_moment[cal->nspots] = metrics.second_moment;
            cal->ref_second_moment_x[cal->nspots] = metrics.second_moment_x;
            cal->ref_second_moment_y[cal->nspots] = metrics.second_moment_y;
        }
        cal->nspots++;
    }

    cal->pupil_cx = pupil_cx;
    cal->pupil_cy = pupil_cy;
    cal->width = width;
    cal->height = height;
    cal->pitch_px = pitch_px_est;

    /* Precompute the bounded 3x3-pupil neighbourhood used by the coherence
     * term.  This moves the O(N^2) geometry search out of the frame path. */
    for (i = 0; i < cal->nspots; ++i) {
        double best_d2[8];
        int best_idx[8];
        int count = 0;
        for (int k = 0; k < 8; ++k) {
            best_d2[k] = 1e30;
            best_idx[k] = -1;
        }
        for (int j = 0; j < cal->nspots; ++j) {
            if (i == j) continue;
            double px = cal->subaps[i].ref_cx - cal->subaps[j].ref_cx;
            double py = cal->subaps[i].ref_cy - cal->subaps[j].ref_cy;
            if (fabs(px) > 1.51 * cal->pitch_px ||
                fabs(py) > 1.51 * cal->pitch_px) continue;
            double d2 = px * px + py * py;
            int pos = count < 8 ? count++ : 7;
            while (pos > 0 && d2 < best_d2[pos - 1]) {
                if (pos < 8) {
                    best_d2[pos] = best_d2[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                }
                --pos;
            }
            if (pos < 8) {
                best_d2[pos] = d2;
                best_idx[pos] = j;
            }
        }
        cal->quality_neighbor_count[i] = count;
        for (int k = 0; k < 8; ++k)
            cal->quality_neighbors[i * 8 + k] = best_idx[k];
    }

    /* Store a single calibration-wide background reference in addition to
     * the per-window values used by the spot-level SNR term. */
    {
        double sum = 0.0, sum2 = 0.0;
        for (i = 0; i < cal->nspots; ++i) {
            sum += cal->ref_background_mean[i];
            sum2 += cal->ref_background_std[i] * cal->ref_background_std[i];
        }
        if (cal->nspots > 0) {
            cal->bg_mean = sum / cal->nspots;
            cal->bg_sigma = sqrt(sum2 / cal->nspots);
        }
    }

    free(spot_cx); free(spot_cy);
    if (cal->nspots == 0) {
        rippa_calibration_free(cal);
        return -2;
    }
    return 0;
}

void rippa_calibration_free(rippra_calibration *cal)
{
    if (!cal) return;
    free(cal->subaps); cal->subaps = NULL;
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC) {
        free(cal->ref_mass); cal->ref_mass = NULL;
        free(cal->ref_background_mean); cal->ref_background_mean = NULL;
        free(cal->ref_background_std); cal->ref_background_std = NULL;
        free(cal->ref_snr); cal->ref_snr = NULL;
        free(cal->ref_second_moment); cal->ref_second_moment = NULL;
        free(cal->ref_second_moment_x); cal->ref_second_moment_x = NULL;
        free(cal->ref_second_moment_y); cal->ref_second_moment_y = NULL;
        free(cal->quality_neighbor_count); cal->quality_neighbor_count = NULL;
        free(cal->quality_neighbors); cal->quality_neighbors = NULL;
    }
    cal->bg_mean = 0.0;
    cal->bg_sigma = 0.0;
    cal->quality_magic = 0;
    cal->nspots = 0;
}

/* ----------------------------------------------------------------------- */
/* Public: per-frame centroiding.                                          */
/* For each sub-aperture, compute relative threshold and TCoG.             */
/* ----------------------------------------------------------------------- */
int rippa_compute_centroids_with_mass(
    const double *frame, int width, int height,
    const rippra_calibration *cal,
    const rippa_config *cfg,
    double *cx, double *cy, double *mass)
{
    int k;
    if (!frame || !cal || !cfg || !cx || !cy || width != cal->width ||
        height != cal->height || !isfinite(cfg->centroid_percent) ||
        cfg->centroid_percent < 0.0 || cfg->centroid_percent > 1.0) return -1;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (k = 0; k < cal->nspots; ++k) {
        const rippra_subap *s = &cal->subaps[k];
        double local_mass = 0.0;

        tcog_window_fast(
            frame, width,
            s->col_min, s->col_max,
            s->row_min, s->row_max,
            cfg->centroid_percent,
            &cx[k], &cy[k], &local_mass);

        if (mass) {
            mass[k] = local_mass;
        }
    }

    return 0;
}

int rippa_compute_centroids_with_metrics(
    const double *frame, int width, int height,
    const rippra_calibration *cal,
    const rippa_config *cfg,
    double *cx, double *cy, rippra_spot_metrics *metrics)
{
    int k;
    if (!frame || !cal || !cfg || !cx || !cy || width != cal->width ||
        height != cal->height || !isfinite(cfg->centroid_percent) ||
        cfg->centroid_percent < 0.0 || cfg->centroid_percent > 1.0) return -1;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (k = 0; k < cal->nspots; ++k) {
        const rippra_subap *s = &cal->subaps[k];
        double mass = 0.0;
        tcog_window_fast(frame, width, s->col_min, s->col_max,
                         s->row_min, s->row_max, cfg->centroid_percent,
                         &cx[k], &cy[k], &mass);
        if (metrics) {
            compute_window_metrics(frame, width, s->col_min, s->col_max,
                                   s->row_min, s->row_max,
                                   cfg->centroid_percent, cx[k], cy[k],
                                   &metrics[k]);
            /* Keep the SIMD/scalar TCoG mass as the authoritative integrated
             * mass even if a future metrics implementation changes the
             * accepted-pixel accounting. */
            metrics[k].mass = mass;
        }
    }
    return 0;
}

/* Backward-compatible API for callers that do not need spot mass. */
int rippa_compute_centroids(const double *frame, int width, int height,
                            const rippra_calibration *cal,
                            const rippa_config *cfg,
                            double *cx, double *cy)
{
    return rippa_compute_centroids_with_mass(
        frame, width, height, cal, cfg, cx, cy, NULL);
}



void rippa_compute_deltas(const double *cx, const double *cy,
                          const rippra_calibration *cal,
                          int nspots, double *dx, double *dy,
                          int *mask)
{
    int i;
    int limit;
    if (!cx || !cy || !cal || !dx || !dy || nspots <= 0) return;
    limit = nspots < cal->nspots ? nspots : cal->nspots;
    for (i = 0; i < limit; ++i) {
        if (!isfinite(cx[i]) || !isfinite(cy[i])) {
            dx[i] = 0.0;
            dy[i] = 0.0;
            if (mask) mask[i] = 0;
        } else {
            dx[i] = cx[i] - cal->subaps[i].ref_cx;
            dy[i] = cy[i] - cal->subaps[i].ref_cy;
            if (mask) mask[i] = 1;
        }
    }
}


int rippa_compute_centroids_refined_with_mass(
    const double *frame, int width, int height,
    const rippra_calibration *cal, const rippa_config *cfg,
    double *cx_out, double *cy_out, double *dx, double *dy, double *mass)
{
    int k, ret;
    int nspots = cal->nspots;
    int r = (int)(0.5 * cal->pitch_px / sqrt(2.0));
    if (r < 2) r = 2;

    /* First pass */
    ret = rippa_compute_centroids_with_mass(frame, width, height, cal, cfg,
                                            cx_out, cy_out, mass);
    if (ret != 0) return ret;

    /* Second pass: refine window around first-pass centroid */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (k = 0; k < nspots; ++k) {
        if (!isfinite(cx_out[k]) || !isfinite(cy_out[k])) {
            if (dx) dx[k] = 0.0;
            if (dy) dy[k] = 0.0;
            continue;
        }
        int cx_i = (int)round(cx_out[k]);
        int cy_i = (int)round(cy_out[k]);
        int cmin = cx_i - r;
        int cmax = cx_i + r;
        int rmin = cy_i - r;
        int rmax = cy_i + r;
        if (cmin < 0) cmin = 0;
        if (rmin < 0) rmin = 0;
        if (cmax >= width)  cmax = width - 1;
        if (rmax >= height) rmax = height - 1;

        tcog_window_fast(frame, width, cmin, cmax, rmin, rmax,
                         cfg->centroid_percent,
                         &cx_out[k], &cy_out[k], mass ? &mass[k] : &(double){0.0});
    }

    rippa_compute_deltas(cx_out, cy_out, cal, nspots, dx, dy, NULL);
    return 0;
}

int rippa_compute_centroids_refined(const double *frame, int width, int height,
                                     const rippra_calibration *cal,
                                     const rippa_config *cfg,
                                     double *cx_out, double *cy_out,
                                     double *dx, double *dy)
{
    return rippa_compute_centroids_refined_with_mass(
        frame, width, height, cal, cfg, cx_out, cy_out, dx, dy, NULL);
}
