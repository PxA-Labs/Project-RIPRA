/*
 * rippra/recon.h - Wavefront Reconstruction, Turbulence Characterization, and DM Mapping
 */
#ifndef RIPRA_RECON_H
#define RIPRA_RECON_H

#include "rippra/io.h"
#include "rippra/centroid.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zonal mesh definition (Fried Geometry)
 */
typedef struct rippra_zonal_mesh {
    int nnodes;
    int nspots;       /* number of sub-apertures used during setup */
    int *node_u;      /* u-coordinate (column) on the lenslet grid */
    int *node_v;      /* v-coordinate (row) on the lenslet grid */
    double *G;        /* Geometry matrix, size (2 * nspots) x nnodes, row-major */
    double *Gpinv;    /* Pseudo-inverse of G, size nnodes x (2 * nspots), row-major */
    double cond;      /* condition number of G (smax/smin of non-truncated singular values) */
} rippra_zonal_mesh;

/*
 * Modal reconstruction model definition (Zernike Polynomials)
 * Active modes exclude piston (Noll index 1).
 */
typedef struct rippra_modal_model {
    int nmodes;       /* Number of active modes (e.g. j = 2 to tot_modes) */
    int nspots;       /* number of sub-apertures used during setup */
    int *mode_j;      /* Noll index for each active mode */
    int *mode_n;      /* Radial order n for each active mode */
    int *mode_m;      /* Azimuthal order m for each active mode */
    double *Zprime;   /* Zernike derivative matrix, size (2 * nspots) x nmodes, row-major */
    double *Zprime_pinv; /* Pseudo-inverse, size nmodes x (2 * nspots), row-major */
    double cond;      /* condition number of Zprime (smax/smin of non-truncated singular values) */
} rippra_modal_model;

/*
 * Zonal reconstruction setup & execution
 */
int rippra_zonal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_zonal_mesh *mesh);
void rippra_zonal_free(rippra_zonal_mesh *mesh);
int rippra_zonal_reconstruct(const rippra_zonal_mesh *mesh, const double *dx, const double *dy, const rippa_config *cfg, double *W);

/*
 * Evaluate analytical Zernike derivatives at (x, y) for a given mode (n, m)
 * (x, y) are normalized pupil coordinates in [-1, 1]
 */
void evaluate_zernike_derivatives(int n, int m, double x, double y, double *dzdx, double *dzdy);

/*
 * Modal reconstruction setup & execution
 */
int rippra_modal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_modal_model *model);
void rippra_modal_free(rippra_modal_model *model);
int rippra_modal_reconstruct(const rippra_modal_model *model, const double *dx, const double *dy, const rippa_config *cfg, double *coeffs);

/*
 * Turbulence Characterization
 * dx_series and dy_series are flat contiguous arrays of size nframes * nspots
 */
double rippra_compute_r0_impl(const double *dx_series, const double *dy_series, int nframes, int nspots, const rippa_config *cfg);
double rippra_compute_tau0_impl(const double *dx_series, const double *dy_series, int nframes, int nspots, double frame_rate);

/*
 * DM Command Map (coupling matrix inversion)
 * target_phase is of size nnodes. dm_commands is of size nnodes.
 */
int rippra_dm_map_impl(const double *target_phase, int nnodes, const rippra_zonal_mesh *mesh, const rippa_config *cfg, double *dm_commands);
int rippra_dm_apply_impl(const double *dm_commands, int nnodes,
                          const rippra_zonal_mesh *mesh,
                          const rippa_config *cfg,
                          const double *input_phase,
                          double *output_residual);
int rippra_closed_loop_step_impl(const double *measured_phase, int nnodes,
                                  const rippra_zonal_mesh *mesh,
                                  const rippa_config *cfg,
                                  double *dm_commands, double gain);
int rippra_closed_loop_run_impl(const double *initial_phase, int nnodes,
                                 const rippra_zonal_mesh *mesh,
                                 const rippa_config *cfg,
                                 double *dm_commands, double gain,
                                 int max_iter, double target_rms,
                                 int *out_iters, double *out_residual_rms);

/*
 * Wavefront quality metrics
 * Returns wavefront RMS in units of wavelength (dimensionless).
 * sigma < 0.05 lambda is the typical Marechal criterion for diffraction-limited.
 */
double rippra_wavefront_rms_lambda(const double *phase, int nnodes, const rippa_config *cfg);

/*
 * Computes Strehl ratio using Marechal approximation: S = exp(-sigma^2)
 * where sigma is the spatial phase variance in radians.
 */
double rippra_compute_strehl(const double *phase_rad, int nnodes);

#ifdef __cplusplus
}
#endif
#endif /* RIPRA_RECON_H */
