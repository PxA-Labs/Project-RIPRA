/*
 * rippra/cuda/rippra_cuda.h - CUDA-accelerated RIPPA functions
 * Requires CUDA toolkit. Compile with nvcc.
 */
#ifndef RIPPRA_CUDA_H
#define RIPPRA_CUDA_H

#include "rippra/centroid.h"
#include "rippra/recon.h"

/* DM map status returned through rippra_cuda_dm_map_ex(). */
#ifndef RIPPRA_DM_OK
#define RIPPRA_DM_OK        0
#endif
#ifndef RIPPRA_DM_SATURATED
#define RIPPRA_DM_SATURATED 1
#endif
#ifndef RIPPRA_DM_PARKED
#define RIPPRA_DM_PARKED    2
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ---- Centroiding ---- */

/* Initialize GPU resources for centroiding (call once after calibration) */
int rippra_cuda_centroid_init(const rippra_calibration *cal, int width, int height);

/* Compute centroids on GPU */
int rippra_cuda_compute_centroids(const double *d_frame, int width, int height,
                                  const rippra_calibration *cal, const rippa_config *cfg,
                                  double *d_cx, double *d_cy);

/* Compute centroid deltas (dx, dy) from centroids on GPU — separated for benchmarking */
int rippra_cuda_compute_deltas(double *d_cx, double *d_cy, int nspots,
                               double *d_dx, double *d_dy);

/* Free GPU resources for centroiding */
void rippra_cuda_centroid_free(void);

/* ---- Matrix Operations ---- */

/* d_A: (m x n) row-major, d_x: (n), d_dst: (m) */
int rippra_cuda_matvec(const double *d_A, const double *d_x, double *d_dst,
                       size_t m, size_t n);

/* d_A: (m x k), d_B: (k x n), d_dst: (m x n) row-major */
int rippra_cuda_matmul(const double *d_A, const double *d_B, double *d_dst,
                       size_t m, size_t k, size_t n);

/* ---- Reconstruction Pipeline (combined) ---- */

/* Reconstruct zonal phase on GPU: W = Gpinv * s */
int rippra_cuda_zonal_reconstruct(const double *d_dx, const double *d_dy,
                                  const rippra_zonal_mesh *mesh,
                                  const rippa_config *cfg,
                                  double *d_W);

/* Reconstruct modal coefficients on GPU: coeffs = Zprime_pinv * s */
int rippra_cuda_modal_reconstruct(const double *d_dx, const double *d_dy,
                                  const rippra_modal_model *model,
                                  const rippa_config *cfg,
                                  double *d_coeffs);

/*
 * Compute DM commands using the authoritative CPU solver through an explicit
 * device-to-host/host-to-device fallback.  d_target_phase and
 * d_dm_commands are device pointers; out_status is a host pointer.  Return
 * zero for a valid map, including saturated or parked results, and a
 * negative value for an error.  Failed maps clear d_dm_commands when it can
 * be safely addressed.
 */
int rippra_cuda_dm_map_ex(const double *d_target_phase, int nnodes,
                          const rippra_zonal_mesh *mesh,
                          const rippa_config *cfg,
                          double *d_dm_commands, int *out_status);

/* Compatibility wrapper; equivalent to rippra_cuda_dm_map_ex(..., NULL). */
int rippra_cuda_dm_map(const double *d_target_phase, int nnodes,
                       const rippra_zonal_mesh *mesh,
                       const rippa_config *cfg,
                       double *d_dm_commands);

/* Initialize GPU resources for DM mapping (call once after zonal setup) */
int rippra_cuda_dm_init(const rippra_zonal_mesh *mesh);

/* Free GPU resources for DM mapping */
void rippra_cuda_dm_free(void);

/* ---- Combined Pipeline ---- */

/* Run full GPU pipeline without exposing the DM safety status. */
int rippra_cuda_full_pipeline(const double *d_frame, int width, int height,
                              const rippra_calibration *cal,
                              const rippa_config *cfg,
                              const rippra_zonal_mesh *mesh,
                              const rippra_modal_model *model,
                              double *d_W, double *d_coeffs,
                              double *d_dm_commands);

/* Status-bearing full-pipeline variant. */
int rippra_cuda_full_pipeline_ex(const double *d_frame, int width, int height,
                                 const rippra_calibration *cal,
                                 const rippa_config *cfg,
                                 const rippra_zonal_mesh *mesh,
                                 const rippra_modal_model *model,
                                 double *d_W, double *d_coeffs,
                                 double *d_dm_commands, int *out_status);

/* ---- Utility ---- */

/* Allocate device memory and copy from host */
int rippra_cuda_malloc_host_to_device(double **d_ptr, const double *h_ptr, size_t n);

/* Allocate device memory and copy to host */
int rippra_cuda_malloc_device_to_host(double **h_ptr, const double *d_ptr, size_t n);

/* Free device memory */
void rippra_cuda_free(void *d_ptr);

/* Check CUDA error */
int rippra_cuda_check_error(const char *file, int line);

#define CUDA_CHECK(call) rippra_cuda_check_error(__FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif /* RIPPRA_CUDA_H */
