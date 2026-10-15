/*
 * rippra/src/stream.c - Real-time streaming pipeline implementation
 */
#include "rippra/stream.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RIPPRA_QUALITY_MAGIC 0x51554c31u

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

/* ---- Internal ring buffer entry ----------------------------------------- */
typedef struct {
    int64_t frame_id;
    double  timestamp_ms;
    double *frame;           /* raw frame data (width * height) */
    int     width, height;
    int     occupied;        /* 1 = slot filled, waiting for processing */
    int     processed;       /* 1 = processing done, result ready for dequeue */
    rippra_stream_result result;
} ring_entry;

/* ---- Pipeline handle (opaque) -------------------------------------------- */
struct rippra_stream {
    /* Configuration copies */
    rippa_config         cfg;
    rippra_calibration   cal;
    rippra_zonal_mesh    mesh;
    rippra_modal_model   model;

    /* Frame dimensions */
    int width, height;

    /* Ping-pong frame buffers for DMA-style double buffering */
    double *ping_buffer;
    double *pong_buffer;
    int     ping_in_use;       /* 0 = ping free, 1 = being captured */

    /* Ring buffer */
    ring_entry *ring;
    int         ring_capacity;
    int         ring_head;     /* read index for processing */
    int         ring_tail;     /* write index for new frames */

    /* Result queue (a simple ptr ring for processed results) */
    rippra_stream_result *results;
    int                   results_capacity;
    int                   results_head;
    int                   results_tail;

    /* Frame counter */
    int64_t next_frame_id;

    /* Turbulence window */
    int turb_nframes;
    double *dx_history;        /* rolling history for r0/tau0 */
    double *dy_history;
    int     history_filled;

    /* Processing-thread scratch; avoids per-frame allocation in the quality
     * gate and keeps the normal path deterministic. */
    double *work_dx;
    double *work_dy;
    rippra_spot_metrics *spot_metrics;
    rippra_quality_fallback_fn fallback_fn;
    void *fallback_ctx;

    /* Synchronisation */
#ifdef _WIN32
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
};

/* ---- Lock helpers -------------------------------------------------------- */
static inline void stream_lock(rippra_stream *s) {
#ifdef _WIN32
    EnterCriticalSection(&s->lock);
#else
    pthread_mutex_lock(&s->lock);
#endif
}

static inline void stream_unlock(rippra_stream *s) {
#ifdef _WIN32
    LeaveCriticalSection(&s->lock);
#else
    pthread_mutex_unlock(&s->lock);
#endif
}

/* ---- Init --------------------------------------------------------------- */
rippra_stream *rippra_stream_init(const rippa_config *cfg,
                                   const rippra_calibration *cal,
                                   const rippra_zonal_mesh *mesh,
                                   const rippra_modal_model *model,
                                   int frame_width, int frame_height)
{
    if (!cfg || !cal || !mesh || !model || frame_width <= 0 ||
        frame_height <= 0 || rippa_config_validate(cfg) != 0 ||
        cal->nspots <= 0 || mesh->nspots != cal->nspots ||
        model->nspots != cal->nspots || cal->width != frame_width ||
        cal->height != frame_height || !cal->subaps || !mesh->node_u ||
        !mesh->node_v || !mesh->G || !mesh->Gpinv || !model->mode_j ||
        !model->mode_n || !model->mode_m || !model->Zprime ||
        !model->Zprime_pinv || mesh->nnodes <= 0 || model->nmodes <= 0) {
        return NULL;
    }

    rippra_stream *s = (rippra_stream *)calloc(1, sizeof(rippra_stream));
    if (!s) return NULL;

#ifdef _WIN32
    InitializeCriticalSection(&s->lock);
#else
    pthread_mutex_init(&s->lock, NULL);
#endif

    /* Copy configuration */
    memcpy(&s->cfg, cfg, sizeof(rippa_config));
    s->width  = frame_width;
    s->height = frame_height;

    /* Deep-copy calibration */
    memcpy(&s->cal, cal, sizeof(rippra_calibration));
    s->cal.quality_magic =
        cal->quality_magic == RIPPRA_QUALITY_MAGIC ? RIPPRA_QUALITY_MAGIC : 0;
    s->cal.subaps = (rippra_subap *)malloc(cal->nspots * sizeof(rippra_subap));
    s->cal.ref_mass = NULL;
    s->cal.ref_background_mean = NULL;
    s->cal.ref_background_std = NULL;
    s->cal.ref_snr = NULL;
    s->cal.ref_second_moment = NULL;
    s->cal.ref_second_moment_x = NULL;
    s->cal.ref_second_moment_y = NULL;
    s->cal.quality_neighbor_count = NULL;
    s->cal.quality_neighbors = NULL;
    if (!s->cal.subaps) goto init_fail;
    memcpy(s->cal.subaps, cal->subaps, cal->nspots * sizeof(rippra_subap));
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_mass) {
        s->cal.ref_mass = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_mass) goto init_fail;
        memcpy(s->cal.ref_mass, cal->ref_mass, (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_background_mean) {
        s->cal.ref_background_mean = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_background_mean) goto init_fail;
        memcpy(s->cal.ref_background_mean, cal->ref_background_mean,
               (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_background_std) {
        s->cal.ref_background_std = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_background_std) goto init_fail;
        memcpy(s->cal.ref_background_std, cal->ref_background_std,
               (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_snr) {
        s->cal.ref_snr = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_snr) goto init_fail;
        memcpy(s->cal.ref_snr, cal->ref_snr, (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_second_moment) {
        s->cal.ref_second_moment = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_second_moment) goto init_fail;
        memcpy(s->cal.ref_second_moment, cal->ref_second_moment,
               (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_second_moment_x) {
        s->cal.ref_second_moment_x = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_second_moment_x) goto init_fail;
        memcpy(s->cal.ref_second_moment_x, cal->ref_second_moment_x,
               (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC && cal->ref_second_moment_y) {
        s->cal.ref_second_moment_y = (double *)malloc((size_t)cal->nspots * sizeof(double));
        if (!s->cal.ref_second_moment_y) goto init_fail;
        memcpy(s->cal.ref_second_moment_y, cal->ref_second_moment_y,
               (size_t)cal->nspots * sizeof(double));
    }
    if (cal->quality_magic == RIPPRA_QUALITY_MAGIC &&
        cal->quality_neighbor_count && cal->quality_neighbors) {
        s->cal.quality_neighbor_count =
            (int *)malloc((size_t)cal->nspots * sizeof(int));
        s->cal.quality_neighbors =
            (int *)malloc((size_t)cal->nspots * 8 * sizeof(int));
        if (!s->cal.quality_neighbor_count || !s->cal.quality_neighbors)
            goto init_fail;
        memcpy(s->cal.quality_neighbor_count, cal->quality_neighbor_count,
               (size_t)cal->nspots * sizeof(int));
        memcpy(s->cal.quality_neighbors, cal->quality_neighbors,
               (size_t)cal->nspots * 8 * sizeof(int));
    }

    /* Deep-copy zonal mesh */
    memcpy(&s->mesh, mesh, sizeof(rippra_zonal_mesh));
    s->mesh.node_u = NULL; s->mesh.node_v = NULL;
    s->mesh.G = NULL; s->mesh.Gpinv = NULL;
    s->mesh.node_u   = (int *)malloc(mesh->nnodes * sizeof(int));
    s->mesh.node_v   = (int *)malloc(mesh->nnodes * sizeof(int));
    s->mesh.G        = (double *)malloc((size_t)2 * cal->nspots * mesh->nnodes * sizeof(double));
    s->mesh.Gpinv    = (double *)malloc((size_t)mesh->nnodes * 2 * cal->nspots * sizeof(double));
    if (!s->mesh.node_u || !s->mesh.node_v || !s->mesh.G || !s->mesh.Gpinv) goto init_fail;
    memcpy(s->mesh.node_u, mesh->node_u, mesh->nnodes * sizeof(int));
    memcpy(s->mesh.node_v, mesh->node_v, mesh->nnodes * sizeof(int));
    memcpy(s->mesh.G,     mesh->G,     (size_t)2 * cal->nspots * mesh->nnodes * sizeof(double));
    memcpy(s->mesh.Gpinv, mesh->Gpinv, (size_t)mesh->nnodes * 2 * cal->nspots * sizeof(double));

    /* Deep-copy modal model */
    memcpy(&s->model, model, sizeof(rippra_modal_model));
    s->model.mode_j = NULL; s->model.mode_n = NULL; s->model.mode_m = NULL;
    s->model.Zprime = NULL; s->model.Zprime_pinv = NULL;
    s->model.mode_j = (int *)malloc(model->nmodes * sizeof(int));
    s->model.mode_n = (int *)malloc(model->nmodes * sizeof(int));
    s->model.mode_m = (int *)malloc(model->nmodes * sizeof(int));
    s->model.Zprime     = (double *)malloc((size_t)2 * cal->nspots * model->nmodes * sizeof(double));
    s->model.Zprime_pinv= (double *)malloc((size_t)model->nmodes * 2 * cal->nspots * sizeof(double));
    if (!s->model.mode_j || !s->model.mode_n || !s->model.mode_m ||
        !s->model.Zprime || !s->model.Zprime_pinv) goto init_fail;
    memcpy(s->model.mode_j, model->mode_j, model->nmodes * sizeof(int));
    memcpy(s->model.mode_n, model->mode_n, model->nmodes * sizeof(int));
    memcpy(s->model.mode_m, model->mode_m, model->nmodes * sizeof(int));
    memcpy(s->model.Zprime,      model->Zprime,      (size_t)2 * cal->nspots * model->nmodes * sizeof(double));
    memcpy(s->model.Zprime_pinv, model->Zprime_pinv, (size_t)model->nmodes * 2 * cal->nspots * sizeof(double));

    /* Double buffers */
    size_t frame_sz = (size_t)frame_width * frame_height;
    s->ping_buffer = (double *)calloc(frame_sz, sizeof(double));
    s->pong_buffer = (double *)calloc(frame_sz, sizeof(double));
    if (!s->ping_buffer || !s->pong_buffer) goto init_fail;

    /* Ring buffer */
    s->ring_capacity = RIPPRA_STREAM_MAX_FRAMES;
    s->ring = (ring_entry *)calloc(s->ring_capacity, sizeof(ring_entry));
    if (!s->ring) goto init_fail;
    s->ring_head = 0;
    s->ring_tail = 0;

    /* Result queue */
    s->results_capacity = RIPPRA_STREAM_MAX_FRAMES;
    s->results = (rippra_stream_result *)calloc(s->results_capacity,
                                                  sizeof(rippra_stream_result));
    if (!s->results) goto init_fail;
    s->results_head = 0;
    s->results_tail = 0;

    /* Pre-allocate result buffers */
    int nspots = cal->nspots;
    int nnodes = mesh->nnodes;
    int nmodes = model->nmodes;
    for (int i = 0; i < s->results_capacity; ++i) {
        s->results[i].cx = (double *)calloc(nspots, sizeof(double));
        s->results[i].cy = (double *)calloc(nspots, sizeof(double));
        s->results[i].dx = (double *)calloc(nspots, sizeof(double));
        s->results[i].dy = (double *)calloc(nspots, sizeof(double));
        s->results[i].spot_mass = (double *)calloc(nspots, sizeof(double));
        s->results[i].spot_mask = (int *)calloc(nspots, sizeof(int));
        s->results[i].mask = s->results[i].spot_mask;
        s->results[i].W  = (double *)calloc(nnodes, sizeof(double));
        s->results[i].zernike_coeffs = (double *)calloc(nmodes, sizeof(double));
        s->results[i].dm_commands     = (double *)calloc(nnodes, sizeof(double));
        s->results[i].nspots = nspots;
        s->results[i].nnodes = nnodes;
        s->results[i].nmodes = nmodes;
        if (!s->results[i].cx || !s->results[i].cy || !s->results[i].dx ||
            !s->results[i].dy || !s->results[i].spot_mass ||
            !s->results[i].spot_mask || !s->results[i].W ||
            !s->results[i].zernike_coeffs || !s->results[i].dm_commands)
            goto init_fail;
    }

    /* Turbulence history */
    s->turb_nframes = 100;
    s->dx_history = (double *)calloc((size_t)s->turb_nframes * nspots, sizeof(double));
    s->dy_history = (double *)calloc((size_t)s->turb_nframes * nspots, sizeof(double));
    s->work_dx = (double *)malloc((size_t)nspots * sizeof(double));
    s->work_dy = (double *)malloc((size_t)nspots * sizeof(double));
    s->spot_metrics = (rippra_spot_metrics *)calloc((size_t)nspots,
                                                      sizeof(rippra_spot_metrics));
    if (!s->dx_history || !s->dy_history || !s->work_dx || !s->work_dy ||
        !s->spot_metrics) goto init_fail;
    s->history_filled = 0;

    s->next_frame_id = 1;

    return s;

init_fail:
    rippra_stream_shutdown(s);
    return NULL;
}

/* ---- Shutdown ------------------------------------------------------------ */
void rippra_stream_shutdown(rippra_stream *s)
{
    if (!s) return;

    /* Free ring buffers */
    for (int i = 0; i < s->ring_capacity; ++i) {
        if (s->ring[i].frame) free(s->ring[i].frame);
    }

    /* Free result buffers */
    for (int i = 0; i < s->results_capacity; ++i) {
        free(s->results[i].cx);
        free(s->results[i].cy);
        free(s->results[i].dx);
        free(s->results[i].dy);
        free(s->results[i].spot_mass);
        free(s->results[i].spot_mask);
        free(s->results[i].W);
        free(s->results[i].zernike_coeffs);
        free(s->results[i].dm_commands);
    }

    free(s->ping_buffer);
    free(s->pong_buffer);
    free(s->ring);
    free(s->results);
    free(s->dx_history);
    free(s->dy_history);
    free(s->work_dx);
    free(s->work_dy);
    free(s->spot_metrics);

    rippa_calibration_free(&s->cal);
    rippra_zonal_free(&s->mesh);
    rippra_modal_free(&s->model);

#ifdef _WIN32
    DeleteCriticalSection(&s->lock);
#else
    pthread_mutex_destroy(&s->lock);
#endif

    free(s);
}

/* ---- Enqueue (producer) -------------------------------------------------- */
int64_t rippra_stream_enqueue(rippra_stream *s,
                               const double *frame, int w, int h)
{
    if (!s || !frame || w != s->width || h != s->height) return -1;
    stream_lock(s);

    int next_tail = (s->ring_tail + 1) % s->ring_capacity;
    if (next_tail == s->ring_head) {
        stream_unlock(s);
        return -1; /* full */
    }

    ring_entry *e = &s->ring[s->ring_tail];
    if (!e->frame) {
        e->frame = (double *)malloc((size_t)s->width * s->height * sizeof(double));
        if (!e->frame) {
            stream_unlock(s);
            return -1;
        }
        e->width = w;
        e->height = h;
    }
    memcpy(e->frame, frame, (size_t)s->width * s->height * sizeof(double));
    e->frame_id    = s->next_frame_id++;
    e->timestamp_ms = 0.0; /* caller can set via clock */
    e->occupied    = 1;
    e->processed   = 0;

    s->ring_tail = next_tail;
    stream_unlock(s);
    return e->frame_id;
}

/* ---- Pending / ready counts ---------------------------------------------- */
int rippra_stream_pending(rippra_stream *s)
{
    stream_lock(s);
    int n = (s->ring_tail - s->ring_head + s->ring_capacity) % s->ring_capacity;
    stream_unlock(s);
    return n;
}

int rippra_stream_ready(rippra_stream *s)
{
    stream_lock(s);
    int n = (s->results_tail - s->results_head + s->results_capacity) % s->results_capacity;
    stream_unlock(s);
    return n;
}

/* Helper for nearest-neighbor spatial interpolation of occluded/lost spot displacements in stream */
static void stream_interpolate_lost_spots(const double *dx, const double *dy,
                                          const int *mask, int nspots,
                                          const rippra_calibration *cal,
                                          double *out_dx, double *out_dy)
{
    int i, j;
    int has_invalid = 0;
    for (i = 0; i < nspots; ++i) {
        if (mask[i] == 0) { has_invalid = 1; break; }
    }
    if (!has_invalid) {
        memcpy(out_dx, dx, nspots * sizeof(double));
        memcpy(out_dy, dy, nspots * sizeof(double));
        return;
    }
    for (i = 0; i < nspots; ++i) {
        if (mask[i]) {
            out_dx[i] = dx[i];
            out_dy[i] = dy[i];
        } else {
            double min_d2 = 1e20;
            int best_j = -1;
            for (j = 0; j < nspots; ++j) {
                if (mask[j]) {
                    double dist2 = pow(cal->subaps[i].ref_cx - cal->subaps[j].ref_cx, 2) +
                                   pow(cal->subaps[i].ref_cy - cal->subaps[j].ref_cy, 2);
                    if (dist2 < min_d2) {
                        min_d2 = dist2;
                        best_j = j;
                    }
                }
            }
            if (best_j != -1) {
                out_dx[i] = dx[best_j];
                out_dy[i] = dy[best_j];
            } else {
                out_dx[i] = 0.0;
                out_dy[i] = 0.0;
            }
        }
    }
}

static void stream_zero_result(rippra_stream_result *r)
{
    memset(r->W, 0, (size_t)r->nnodes * sizeof(double));
    memset(r->zernike_coeffs, 0, (size_t)r->nmodes * sizeof(double));
    memset(r->dm_commands, 0, (size_t)r->nnodes * sizeof(double));
    r->r0 = 0.0;
    r->tau0 = 0.0;
}

/* ---- Process (consumer) ------------------------------------------------- */
int rippra_stream_process(rippra_stream *s)
{
    stream_lock(s);

    if (s->ring_head == s->ring_tail) {
        stream_unlock(s);
        return -1; /* nothing pending */
    }

    ring_entry *e = &s->ring[s->ring_head];
    if (!e->occupied) {
        stream_unlock(s);
        return -1;
    }

    int nspots = s->cal.nspots;

    if ((s->results_tail + 1) % s->results_capacity == s->results_head) {
        stream_unlock(s);
        return -2; /* result queue full; leave input pending */
    }

    /* Prepare output slot */
    int rslot = s->results_tail;
    rippra_stream_result *r = &s->results[rslot];
    r->frame_id    = e->frame_id;
    r->timestamp_ms = e->timestamp_ms;
    r->nspots = nspots;
    r->nnodes = s->mesh.nnodes;
    r->nmodes = s->model.nmodes;
    r->quality = 0.0;
    r->status = RIPPRA_FRAME_LOST;
    memset(r->cx, 0, (size_t)nspots * sizeof(double));
    memset(r->cy, 0, (size_t)nspots * sizeof(double));
    memset(r->dx, 0, (size_t)nspots * sizeof(double));
    memset(r->dy, 0, (size_t)nspots * sizeof(double));
    memset(r->spot_mass, 0, (size_t)nspots * sizeof(double));
    memset(r->spot_mask, 0, (size_t)nspots * sizeof(int));
    stream_zero_result(r);

    /* Step 1: centroiding */
    int centroid_rc = rippa_compute_centroids_with_metrics(
        e->frame, e->width, e->height, &s->cal, &s->cfg,
        r->cx, r->cy, s->spot_metrics);
    if (centroid_rc != 0) {
        r->quality = 0.0;
        r->status = RIPPRA_STATUS_ERROR;
        stream_zero_result(r);
        e->occupied = 0;
        s->ring_head = (s->ring_head + 1) % s->ring_capacity;
        s->results_tail = (s->results_tail + 1) % s->results_capacity;
        stream_unlock(s);
        return 0;
    }

    /* Step 2: retain raw deltas/mask for diagnostics; interpolation is only
     * applied to the private reconstruction work vectors. */
    rippa_compute_deltas(r->cx, r->cy, &s->cal, nspots, r->dx, r->dy,
                         r->spot_mask);
    for (int i = 0; i < nspots; ++i) {
        r->spot_mass[i] = s->spot_metrics[i].mass;
        if (!isfinite(s->spot_metrics[i].mass) || s->spot_metrics[i].mass <= 1e-9)
            r->spot_mask[i] = 0;
    }
    r->quality = rippra_frame_quality_with_metrics(
        r->dx, r->dy, r->spot_mass, r->spot_mask, s->spot_metrics,
        nspots, &s->cfg, &s->cal, NULL);
    if (r->quality >= s->cfg.quality_soft)
        r->status = RIPPRA_FRAME_NOMINAL;
    else if (r->quality >= s->cfg.quality_hard)
        r->status = RIPPRA_FRAME_DEGRADED;
    else
        r->status = RIPPRA_FRAME_LOST;

    if (r->status == RIPPRA_FRAME_LOST) {
        stream_zero_result(r);
        e->occupied = 0;
        s->ring_head = (s->ring_head + 1) % s->ring_capacity;
        s->results_tail = (s->results_tail + 1) % s->results_capacity;
        stream_unlock(s);
        return 0;
    }

    if (r->status == RIPPRA_FRAME_DEGRADED) {
        int fallback_ok = 0;
        if (s->fallback_fn) {
            int callback_rc = s->fallback_fn(
                r->dx, r->dy, r->spot_mask, nspots, s->dx_history,
                s->fallback_ctx, s->work_dx, s->work_dy);
            if (callback_rc == 0) {
                fallback_ok = 1;
                for (int i = 0; i < nspots; ++i) {
                    if (!isfinite(s->work_dx[i]) || !isfinite(s->work_dy[i])) {
                        fallback_ok = 0;
                        break;
                    }
                }
            }
        }
        if (!fallback_ok)
            stream_interpolate_lost_spots(r->dx, r->dy, r->spot_mask, nspots,
                                          &s->cal, s->work_dx, s->work_dy);
    } else {
        memcpy(s->work_dx, r->dx, (size_t)nspots * sizeof(double));
        memcpy(s->work_dy, r->dy, (size_t)nspots * sizeof(double));
    }

    /* Step 3: zonal reconstruction */
    int recon_rc = rippra_zonal_reconstruct(&s->mesh, s->work_dx, s->work_dy,
                                            &s->cfg, r->W);

    /* Step 4: modal reconstruction */
    if (recon_rc == 0)
        recon_rc = rippra_modal_reconstruct(&s->model, s->work_dx, s->work_dy,
                                            &s->cfg, r->zernike_coeffs);
    if (recon_rc != 0) {
        r->quality = 0.0;
        r->status = RIPPRA_STATUS_ERROR;
        stream_zero_result(r);
    }

    /* Step 5: turbulence (rolling window) */
    if (r->status != RIPPRA_FRAME_LOST) {
        int idx = (int)((e->frame_id - 1) % s->turb_nframes);
        memcpy(&s->dx_history[(size_t)idx * nspots], s->work_dx,
               nspots * sizeof(double));
        memcpy(&s->dy_history[(size_t)idx * nspots], s->work_dy,
               nspots * sizeof(double));
        if (idx == s->turb_nframes - 1) s->history_filled = 1;

        if (s->history_filled) {
            r->r0 = rippra_compute_r0_impl(s->dx_history, s->dy_history,
                                       s->turb_nframes, nspots, &s->cfg);
            r->tau0 = rippra_compute_tau0_impl(s->dx_history, s->dy_history,
                                           s->turb_nframes, nspots, 1000.0);
        } else {
            r->r0 = 0.0;
            r->tau0 = 0.0;
        }
    }

    /* Step 6: DM command map.  A lost result is always explicitly zeroed. */
    if (r->status != RIPPRA_FRAME_LOST && r->status != RIPPRA_STATUS_ERROR &&
        rippra_dm_map_impl(r->W, s->mesh.nnodes, &s->mesh, &s->cfg,
                           r->dm_commands) != 0) {
        r->quality = 0.0;
        r->status = RIPPRA_STATUS_ERROR;
        stream_zero_result(r);
    }

    /* Mark ring entry as processed */
    e->occupied = 0;
    s->ring_head = (s->ring_head + 1) % s->ring_capacity;

    /* Advance result tail */
    s->results_tail = (s->results_tail + 1) % s->results_capacity;

    stream_unlock(s);
    return 0;
}

/* ---- Dequeue (output consumer) ------------------------------------------- */
const rippra_stream_result *rippra_stream_dequeue(rippra_stream *s)
{
    stream_lock(s);

    if (s->results_head == s->results_tail) {
        stream_unlock(s);
        return NULL;
    }

    const rippra_stream_result *r = &s->results[s->results_head];
    s->results_head = (s->results_head + 1) % s->results_capacity;

    stream_unlock(s);
    return r;
}

/* ---- Convenience: process one frame synchronously ------------------------ */
const rippra_stream_result *rippra_stream_process_one(rippra_stream *s,
                                                       const double *frame,
                                                       int w, int h)
{
    int64_t fid = rippra_stream_enqueue(s, frame, w, h);
    if (fid < 0) return NULL;

    /* Wait until result is ready (busy-poll) */
    while (rippra_stream_process(s) != 0) { /* spin */ }

    const rippra_stream_result *r = NULL;
    do {
        r = rippra_stream_dequeue(s);
    } while (!r || r->frame_id != fid);

    return r;
}

/* ---- Turbulence window --------------------------------------------------- */
void rippra_stream_set_turbulence_window(rippra_stream *s, int nframes)
{
    if (nframes < 2) nframes = 2;
    stream_lock(s);
    int nspots = s->cal.nspots;
    double *new_dx = (double *)calloc((size_t)nframes * nspots, sizeof(double));
    double *new_dy = (double *)calloc((size_t)nframes * nspots, sizeof(double));
    if (new_dx && new_dy) {
        free(s->dx_history);
        free(s->dy_history);
        s->dx_history = new_dx;
        s->dy_history = new_dy;
        s->turb_nframes = nframes;
        s->history_filled = 0;
    } else {
        free(new_dx);
        free(new_dy);
    }
    stream_unlock(s);
}

int rippra_stream_set_fallback(rippra_stream *s,
                               rippra_quality_fallback_fn fn, void *ctx)
{
    if (!s) return -1;
    stream_lock(s);
    s->fallback_fn = fn;
    s->fallback_ctx = ctx;
    stream_unlock(s);
    return 0;
}
