/*
 * rippa/centroid.h - sub-aperture grid calibration and thresholded CoG centroiding
 *
 * Calibration (run once):
 *   rippa_calibrate_grid()  - detect spots on reference frame, build per-sub-aperture
 *                            window grid, compute reference centroids
 *
 * Per-frame centroiding:
 *   rippa_compute_centroids()  - TCoG in each sub-aperture window using integral images
 *
 * Data structures hold all calibration state so the per-frame path is stateless.
 */
#ifndef RIPPA_CENTROID_H
#define RIPPA_CENTROID_H

#include "rippra/io.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Per-sub-aperture descriptor: bounding box in the image + reference centroid.
 * Boxes use [col_min, col_max, row_min, row_max] convention matching
 * the MATLAB shstruct.ord_sqgrid.
 */
typedef struct rippra_subap {
    int col_min, col_max;   /* pixel column bounds (inclusive) */
    int row_min, row_max;   /* pixel row bounds (inclusive)    */
    double ref_cx, ref_cy;  /* reference centroid (pixels, origin top-left) */
} rippra_subap;

/*
 * Calibration result: everything needed for per-frame centroiding.
 */
typedef struct rippra_calibration {
    rippra_subap *subaps;    /* array of nspots sub-aperture descriptors */
    int nspots;             /* number of active sub-apertures          */
    /* Reference statistics captured from the calibration frame.  These are
     * owned by the calibration object and are optional for old callers that
     * construct rippra_calibration manually. */
    double *ref_mass;
    double *ref_background_mean;
    double *ref_background_std;
    double *ref_snr;
    double *ref_second_moment;
    double *ref_second_moment_x;
    double *ref_second_moment_y;
    int *quality_neighbor_count;
    int *quality_neighbors;       /* nspots x 8 precomputed 3x3 neighbours */
    double bg_mean;
    double bg_sigma;
    unsigned int quality_magic;   /* internal validity marker for optional arrays */
    double pupil_cx;        /* pupil centre X (pixels)                 */
    double pupil_cy;        /* pupil centre Y (pixels)                 */
    int width, height;      /* frame dimensions                        */
    double pitch_px;        /* estimated lenslet pitch in pixels        */
} rippra_calibration;

/* Per-frame statistics used by the streaming quality gate. */
typedef struct rippra_spot_metrics {
    double mass;
    double background_mean;
    double background_std;
    double second_moment;
    double second_moment_x;
    double second_moment_y;
    double signal_mean;
    double signal_peak;
} rippra_spot_metrics;

/* Quality reference view.  The arrays are owned by rippra_calibration and
 * remain valid until rippa_calibration_free() is called. */
typedef struct rippra_quality_ref {
    int nspots;
    const double *ref_mass;
    const double *ref_second_moment;
    const double *ref_second_moment_x;
    const double *ref_second_moment_y;
    const double *ref_background_mean;
    const double *ref_background_sigma;
    double bg_mean;
    double bg_sigma;
} rippra_quality_ref;

/*
 * Calibrate the sub-aperture grid from a reference (flat) frame.
 *
 * Steps:
 *   1. Binarize at threshold, close small gaps, find connected components
 *   2. Build coarse grid: square window of radius cfg->coarse_grid_radius
 *      around each spot centre
 *   3. Run thresholded CoG within each coarse window on the reference frame
 *      to get precise reference centroids
 *   4. Refine to fine grid: recompute windows using estimated pitch/2 radius
 *   5. Recompute reference centroids using fine windows
 *
 * All sub-apertures whose window falls outside the image or whose centroid
 * is too close to the pupil edge are discarded.
 *
 * The caller must rippra_calibration_free() the result when done.
 */
int rippa_calibrate_grid(const double *frame, int width, int height,
                         const rippa_config *cfg, rippra_calibration *cal);

void rippa_calibration_free(rippra_calibration *cal);

/*
 * Compute centroids for an aberrated frame using the pre-computed grid.
 *
 * For each sub-aperture, extracts the window, computes a relative threshold
 * (cfg->centroid_percent * range), then applies thresholded CoG using the
 * integral-image accelerated method.
 *
 * cx[nspots] and cy[nspots] store the output centroids (same ordering as
 * cal->subaps).
 */
int rippa_compute_centroids(const double *frame, int width, int height,
                            const rippra_calibration *cal,
                            const rippa_config *cfg,
                            double *cx, double *cy);

/*
 * Compute centroids and return integrated intensity for each spot.
 * mass may be NULL if the caller does not need intensity values.
 */
int rippa_compute_centroids_with_mass(
    const double *frame, int width, int height,
    const rippra_calibration *cal,
    const rippa_config *cfg,
    double *cx, double *cy, double *mass);

/* Compute centroids and the statistics used by the frame-quality gate.  The
 * metrics array may be NULL when only centroids are required. */
int rippa_compute_centroids_with_metrics(
    const double *frame, int width, int height,
    const rippra_calibration *cal,
    const rippa_config *cfg,
    double *cx, double *cy, rippra_spot_metrics *metrics);

/* Quality-calibrate an already detected grid against a known-good frame.
 * This publishes the reference statistics retained by cal without copying
 * their arrays.  The returned view is valid for the lifetime of cal. */
int rippra_quality_calibrate(const double *frame, int width, int height,
                            const rippra_calibration *cal,
                            const rippa_config *cfg,
                            rippra_quality_ref *ref);

/* Compute Q in [0, 1] from public per-spot centroid data.  out_metrics, when
 * non-NULL, receives validity, mass, shape, and spatial-coherence terms. */
double rippra_frame_quality(const double *dx, const double *dy,
                            const double *mass, const int *mask,
                            int nspots, const rippa_config *cfg,
                            const rippra_calibration *cal,
                            double *out_metrics);

/* Extended form used by the streaming path so SNR/background and shape
 * diagnostics are included without rescanning the frame. */
double rippra_frame_quality_with_metrics(
    const double *dx, const double *dy, const double *mass, const int *mask,
    const rippra_spot_metrics *metrics, int nspots,
    const rippa_config *cfg, const rippra_calibration *cal,
    double *out_metrics);

/*
 * Compute deltas (spot deviations) from reference centroids.
 * dx[nspots] = cx[i] - cal->subaps[i].ref_cx
 * dy[nspots] = cy[i] - cal->subaps[i].ref_cy
 */
void rippa_compute_deltas(const double *cx, const double *cy,
                          const rippra_calibration *cal,
                          int nspots, double *dx, double *dy,
                          int *mask);

/*
 * Refined (iterative) centroiding: two-pass TCoG.
 * First pass: standard centroid in reference window.
 * Second pass: re-center window around first-pass centroid, re-run TCoG.
 * Returns final centroids in cx_out, cy_out and deltas in dx, dy.
 */
int rippa_compute_centroids_refined(const double *frame, int width, int height,
                                     const rippra_calibration *cal,
                                     const rippa_config *cfg,
                                     double *cx_out, double *cy_out,
                                     double *dx, double *dy);

/* Refined centroiding variant with optional per-spot mass output. */
int rippa_compute_centroids_refined_with_mass(
    const double *frame, int width, int height,
    const rippra_calibration *cal, const rippa_config *cfg,
    double *cx_out, double *cy_out, double *dx, double *dy, double *mass);

#ifdef __cplusplus
}
#endif
#endif /* RIPRA_CENTROID_H */
