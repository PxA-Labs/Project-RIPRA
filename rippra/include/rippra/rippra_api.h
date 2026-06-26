/*
 * rippra/rippra_api.h - Public C API for RIPPA shared library (DLL/.so)
 *
 * This header exposes the complete wavefront reconstruction pipeline:
 *   calibration load, centroiding, zonal/modal reconstruction,
 *   turbulence characterization, DM mapping, and pipeline control.
 *
 * Compile with BUILD_RIPRA_DLL defined to build the shared library,
 * or link against it to use as a consumer.
 */
#ifndef RIPRA_API_H
#define RIPRA_API_H

#include <stddef.h>

#ifdef BUILD_RIPRA_DLL
#  if defined(_WIN32) || defined(_WIN64)
#    define RIPRA_API __declspec(dllexport)
#  else
#    define RIPRA_API __attribute__((visibility("default")))
#  endif
#else
#  if defined(_WIN32) || defined(_WIN64)
#    define RIPRA_API __declspec(dllimport)
#  else
#    define RIPRA_API
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Version ----------------------------------------------------------- */
RIPRA_API const char* rippra_version(void);

/* ---- Config ------------------------------------------------------------ */
typedef struct rippra_api_config {
    double camera_pixsize;
    int    frame_width;
    int    frame_height;
    int    totlenses;
    double flength;
    double pitch;
    double sa_radius;
    double pupil_radius;
    double wavelength;
    double thresh_binary;
    double centroid_percent;
    int    coarse_grid_radius;
    int    zernike_nmax;
    int    dm_nact_x;
    int    dm_nact_y;
    double coupling;
} rippra_api_config;

RIPRA_API rippra_api_config rippra_default_config(void);
RIPRA_API int rippra_config_load(rippra_api_config *cfg, const char *path);

/* ---- Calibration ------------------------------------------------------- */
RIPRA_API void* rippra_calibrate(const double *flat_frame,
                                  int width, int height,
                                  const rippra_api_config *cfg);

RIPRA_API void rippra_calibration_free(void *cal);

RIPRA_API int  rippra_calibration_nspots(void *cal);
RIPRA_API void rippra_calibration_ref_centroids(void *cal,
                                                  double *out_cx,
                                                  double *out_cy);

/* ---- Centroiding ------------------------------------------------------- */
RIPRA_API int rippra_centroid(void *cal,
                               const double *frame,
                               int width, int height,
                               double *out_dx, double *out_dy);

/* ---- Zonal Reconstruction ---------------------------------------------- */
RIPRA_API int rippra_reconstruct_zonal(void *cal,
                                        const double *dx,
                                        const double *dy,
                                        const rippra_api_config *cfg,
                                        double *out_phase);

/* ---- Modal Reconstruction ---------------------------------------------- */
RIPRA_API int rippra_reconstruct_modal(void *cal,
                                        const double *dx,
                                        const double *dy,
                                        const rippra_api_config *cfg,
                                        double *out_coeffs);

/* ---- Full Pipeline (centroid + modal reconstruction) ------------------- */
RIPRA_API int rippra_process_frame(void *cal,
                                    const double *frame,
                                    int width, int height,
                                    const rippra_api_config *cfg,
                                    double *out_dx, double *out_dy,
                                    double *out_coeffs);

/* ---- Turbulence Characterization --------------------------------------- */
RIPRA_API double rippra_compute_r0(const double *dx_series,
                                    const double *dy_series,
                                    int nframes, int nspots,
                                    const rippra_api_config *cfg);

RIPRA_API double rippra_compute_tau0(const double *dx_series,
                                      const double *dy_series,
                                      int nframes, int nspots,
                                      double frame_rate);

/* ---- DM Mapping -------------------------------------------------------- */
RIPRA_API int rippra_dm_map(const double *target_phase,
                              int nnodes,
                              void *cal,
                              const rippra_api_config *cfg,
                              double *out_commands);

#ifdef __cplusplus
}
#endif

#endif /* RIPRA_API_H */
