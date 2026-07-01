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
    double pupil_cx;        /* pupil centre X (pixels)                 */
    double pupil_cy;        /* pupil centre Y (pixels)                 */
    int width, height;      /* frame dimensions                        */
    double pitch_px;        /* estimated lenslet pitch in pixels        */
} rippra_calibration;

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

#ifdef __cplusplus
}
#endif
#endif /* RIPRA_CENTROID_H */
