/*
 * src/rippra_api.c - Public C API implementation for RIPPA shared library.
 *
 * NOTE: does NOT #include recon.h — rippra_api.h and recon.h declare the
 * same function names with incompatible types (rippra_api_config vs rippa_config,
 * void* vs rippra_zonal_mesh*).  The few recon types/functions needed here
 * are duplicated below so this file compiles cleanly.
 */
#define BUILD_RIPRA_DLL 1
#include "rippra/rippra_api.h"
#include "rippra/io.h"
#include "rippra/centroid.h"
#include <stdlib.h>
#include <string.h>

/* ---- recon types duplicated here (see rippra/recon.h) ---- */
typedef struct {
    int nnodes;
    int *node_u, *node_v;
    double *G, *Gpinv;
} rippra_zonal_mesh;

typedef struct {
    int nmodes;
    int *mode_j, *mode_n, *mode_m;
    double *Zprime, *Zprime_pinv;
} rippra_modal_model;

/* ---- recon functions needed internally (suffixed _impl) ---- */
int rippra_zonal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_zonal_mesh *mesh);
int rippra_zonal_reconstruct(const rippra_zonal_mesh *mesh, const double *dx, const double *dy, const rippa_config *cfg, double *phase);
void rippra_zonal_free(rippra_zonal_mesh *mesh);
int rippra_modal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_modal_model *model);
int rippra_modal_reconstruct(const rippra_modal_model *model, const double *dx, const double *dy, const rippa_config *cfg, double *coeffs);
void rippra_modal_free(rippra_modal_model *model);
double rippra_compute_r0_impl(const double *dx, const double *dy, int nf, int ns, const rippa_config *cfg);
double rippra_compute_tau0_impl(const double *dx, const double *dy, int nf, int ns, double fr);
int rippra_dm_map_impl(const double *phase, int nn, const rippra_zonal_mesh *m, const rippa_config *cfg, double *cmd);
int rippra_dm_apply_impl(const double *cmd, int nn, const rippra_zonal_mesh *m, const rippa_config *cfg, const double *in, double *out);
int rippra_closed_loop_step_impl(const double *ph, int nn, const rippra_zonal_mesh *m, const rippa_config *cfg, double *cmd, double g);
int rippra_closed_loop_run_impl(const double *ph, int nn, const rippra_zonal_mesh *m, const rippa_config *cfg, double *cmd, double g, int mi, double tr, int *oi, double *or);

#define RIPRA_VERSION "2.0.0"

static void api_cfg_to_internal(const rippra_api_config *src, rippa_config *dst)
{
    dst->camera_pixsize    = src->camera_pixsize;
    dst->frame_width       = src->frame_width;
    dst->frame_height      = src->frame_height;
    dst->totlenses         = src->totlenses;
    dst->flength           = src->flength;
    dst->pitch             = src->pitch;
    dst->sa_radius         = src->sa_radius;
    dst->pupil_radius      = src->pupil_radius;
    dst->wavelength        = src->wavelength;
    dst->thresh_binary     = src->thresh_binary;
    dst->centroid_percent  = src->centroid_percent;
    dst->coarse_grid_radius= src->coarse_grid_radius;
    dst->zernike_nmax      = src->zernike_nmax;
    dst->dm_nact_x         = src->dm_nact_x;
    dst->dm_nact_y         = src->dm_nact_y;
    dst->coupling          = src->coupling;
}

RIPRA_API const char* rippra_version(void) { return RIPRA_VERSION; }

RIPRA_API rippra_api_config rippra_default_config(void)
{
    rippra_api_config c;
    c.camera_pixsize     = 5.3e-6;
    c.frame_width        = 640;
    c.frame_height       = 480;
    c.totlenses          = 127;
    c.flength            = 1e-3;
    c.pitch              = 300e-6;
    c.sa_radius          = 150e-6;
    c.pupil_radius       = 2e-3;
    c.wavelength         = 635e-9;
    c.thresh_binary      = 0.5;
    c.centroid_percent   = 0.3;
    c.coarse_grid_radius = 3;
    c.zernike_nmax       = 5;
    c.dm_nact_x          = 12;
    c.dm_nact_y          = 12;
    c.coupling           = 0.2;
    return c;
}

RIPRA_API int rippra_config_load(rippra_api_config *cfg, const char *path)
{
    rippa_config internal;
    int ret = rippa_config_load(&internal, path);
    if (ret != 0) return ret;
    cfg->camera_pixsize     = internal.camera_pixsize;
    cfg->frame_width        = internal.frame_width;
    cfg->frame_height       = internal.frame_height;
    cfg->totlenses          = internal.totlenses;
    cfg->flength            = internal.flength;
    cfg->pitch              = internal.pitch;
    cfg->sa_radius          = internal.sa_radius;
    cfg->pupil_radius       = internal.pupil_radius;
    cfg->wavelength         = internal.wavelength;
    cfg->thresh_binary      = internal.thresh_binary;
    cfg->centroid_percent   = internal.centroid_percent;
    cfg->coarse_grid_radius = internal.coarse_grid_radius;
    cfg->zernike_nmax       = internal.zernike_nmax;
    cfg->dm_nact_x          = internal.dm_nact_x;
    cfg->dm_nact_y          = internal.dm_nact_y;
    cfg->coupling           = internal.coupling;
    return 0;
}

/* ---- Calibration state ------------------------------------------------- */
typedef struct api_calibration {
    rippra_calibration base;
    rippa_config       cfg;
    rippra_zonal_mesh  zmesh;
    rippra_modal_model mmodel;
    int                zmesh_ready;
    int                mmodel_ready;
} api_calibration;

RIPRA_API void* rippra_calibrate(const double *flat_frame,
                                  int width, int height,
                                  const rippra_api_config *cfg)
{
    api_calibration *cal = (api_calibration*)calloc(1, sizeof(api_calibration));
    if (!cal) return NULL;
    api_cfg_to_internal(cfg, &cal->cfg);
    int ret = rippa_calibrate_grid(flat_frame, width, height, &cal->cfg, &cal->base);
    if (ret != 0) { free(cal); return NULL; }
    cal->zmesh_ready = 0;
    cal->mmodel_ready = 0;
    return (void*)cal;
}

RIPRA_API void rippra_calibration_free(void *cal_ptr)
{
    if (!cal_ptr) return;
    api_calibration *cal = (api_calibration*)cal_ptr;
    rippa_calibration_free(&cal->base);
    if (cal->zmesh_ready) rippra_zonal_free(&cal->zmesh);
    if (cal->mmodel_ready) rippra_modal_free(&cal->mmodel);
    free(cal);
}

RIPRA_API int  rippra_calibration_nspots(void *cal_ptr)
    { return (int)((api_calibration*)cal_ptr)->base.nspots; }

RIPRA_API void rippra_calibration_ref_centroids(void *cal_ptr,
                                                  double *out_cx, double *out_cy)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    int n = (int)cal->base.nspots;
    for (int i = 0; i < n; i++) {
        out_cx[i] = cal->base.subaps[i].ref_cx;
        out_cy[i] = cal->base.subaps[i].ref_cy;
    }
}

/* ---- Centroiding ------------------------------------------------------- */
RIPRA_API int rippra_centroid(void *cal_ptr,
                               const double *frame,
                               int width, int height,
                               double *out_dx, double *out_dy)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    double *cx = (double*)malloc(cal->base.nspots * sizeof(double));
    double *cy = (double*)malloc(cal->base.nspots * sizeof(double));
    if (!cx || !cy) { free(cx); free(cy); return -1; }
    int ret = rippa_compute_centroids(frame, width, height, &cal->base, &cal->cfg, cx, cy);
    if (ret == 0)
        rippa_compute_deltas(cx, cy, &cal->base, (int)cal->base.nspots, out_dx, out_dy);
    free(cx); free(cy);
    return ret;
}

/* ---- Zonal ------------------------------------------------------------- */
RIPRA_API int rippra_reconstruct_zonal(void *cal_ptr,
                                        const double *dx, const double *dy,
                                        const rippra_api_config *cfg,
                                        double *out_phase)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->zmesh_ready) {
        int ret = rippra_zonal_setup(&cal->base, &cal->cfg, &cal->zmesh);
        if (ret != 0) return ret;
        cal->zmesh_ready = 1;
    }
    return rippra_zonal_reconstruct(&cal->zmesh, dx, dy, &cal->cfg, out_phase);
}

/* ---- Modal ------------------------------------------------------------- */
RIPRA_API int rippra_reconstruct_modal(void *cal_ptr,
                                        const double *dx, const double *dy,
                                        const rippra_api_config *cfg,
                                        double *out_coeffs)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->mmodel_ready) {
        int ret = rippra_modal_setup(&cal->base, &cal->cfg, &cal->mmodel);
        if (ret != 0) return ret;
        cal->mmodel_ready = 1;
    }
    return rippra_modal_reconstruct(&cal->mmodel, dx, dy, &cal->cfg, out_coeffs);
}

/* ---- Full Pipeline ----------------------------------------------------- */
RIPRA_API int rippra_process_frame(void *cal_ptr,
                                    const double *frame,
                                    int width, int height,
                                    const rippra_api_config *cfg,
                                    double *out_dx, double *out_dy,
                                    double *out_coeffs)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    int nspots = (int)cal->base.nspots;
    double *cx = (double*)malloc(nspots * sizeof(double));
    double *cy = (double*)malloc(nspots * sizeof(double));
    if (!cx || !cy) { free(cx); free(cy); return -1; }
    int ret = rippa_compute_centroids(frame, width, height, &cal->base, &cal->cfg, cx, cy);
    if (ret != 0) { free(cx); free(cy); return ret; }
    rippa_compute_deltas(cx, cy, &cal->base, nspots, out_dx, out_dy);
    free(cx); free(cy);
    if (!cal->mmodel_ready) {
        ret = rippra_modal_setup(&cal->base, &cal->cfg, &cal->mmodel);
        if (ret != 0) return ret;
        cal->mmodel_ready = 1;
    }
    return rippra_modal_reconstruct(&cal->mmodel, out_dx, out_dy, &cal->cfg, out_coeffs);
}

/* ---- Turbulence -------------------------------------------------------- */
RIPRA_API double rippra_compute_r0(const double *dx_series,
                                    const double *dy_series,
                                    int nframes, int nspots,
                                    const rippra_api_config *cfg)
{
    rippa_config internal;
    api_cfg_to_internal(cfg, &internal);
    return rippra_compute_r0_impl(dx_series, dy_series, nframes, nspots, &internal);
}

RIPRA_API double rippra_compute_tau0(const double *dx_series,
                                      const double *dy_series,
                                      int nframes, int nspots,
                                      double frame_rate)
{
    return rippra_compute_tau0_impl(dx_series, dy_series, nframes, nspots, frame_rate);
}

/* ---- DM Mapping -------------------------------------------------------- */
RIPRA_API int rippra_dm_map(const double *target_phase, int nnodes,
                              void *cal_ptr,
                              const rippra_api_config *cfg,
                              double *out_commands)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->zmesh_ready) {
        int ret = rippra_zonal_setup(&cal->base, &cal->cfg, &cal->zmesh);
        if (ret != 0) return ret;
        cal->zmesh_ready = 1;
    }
    return rippra_dm_map_impl(target_phase, nnodes, &cal->zmesh, &cal->cfg, out_commands);
}

/* ---- Closed-Loop AO Control -------------------------------------------- */
RIPRA_API int rippra_dm_apply(const double *dm_commands, int nnodes,
                               void *cal_ptr,
                               const rippra_api_config *cfg,
                               const double *input_phase,
                               double *out_residual)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->zmesh_ready) {
        int ret = rippra_zonal_setup(&cal->base, &cal->cfg, &cal->zmesh);
        if (ret != 0) return ret;
        cal->zmesh_ready = 1;
    }
    return rippra_dm_apply_impl(dm_commands, nnodes, &cal->zmesh, &cal->cfg, input_phase, out_residual);
}

RIPRA_API int rippra_closed_loop_step(void *cal_ptr,
                                       const double *measured_phase,
                                       int nnodes,
                                       const rippra_api_config *cfg,
                                       double *dm_commands,
                                       double gain)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->zmesh_ready) {
        int ret = rippra_zonal_setup(&cal->base, &cal->cfg, &cal->zmesh);
        if (ret != 0) return ret;
        cal->zmesh_ready = 1;
    }
    return rippra_closed_loop_step_impl(measured_phase, nnodes, &cal->zmesh, &cal->cfg, dm_commands, gain);
}

RIPRA_API int rippra_closed_loop_run(void *cal_ptr,
                                      const double *initial_phase,
                                      int nnodes,
                                      const rippra_api_config *cfg,
                                      double *dm_commands,
                                      double gain,
                                      int max_iter,
                                      double target_rms,
                                      int *out_iters,
                                      double *out_residual_rms)
{
    api_calibration *cal = (api_calibration*)cal_ptr;
    (void)cfg;
    if (!cal->zmesh_ready) {
        int ret = rippra_zonal_setup(&cal->base, &cal->cfg, &cal->zmesh);
        if (ret != 0) return ret;
        cal->zmesh_ready = 1;
    }
    return rippra_closed_loop_run_impl(initial_phase, nnodes, &cal->zmesh, &cal->cfg,
                                        dm_commands, gain, max_iter, target_rms,
                                        out_iters, out_residual_rms);
}
