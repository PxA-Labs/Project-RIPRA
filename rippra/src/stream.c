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
    rippra_stream *s = (rippra_stream *)calloc(1, sizeof(rippra_stream));
    if (!s) return NULL;

    /* Copy configuration */
    memcpy(&s->cfg, cfg, sizeof(rippa_config));
    s->width  = frame_width;
    s->height = frame_height;

    /* Deep-copy calibration */
    memcpy(&s->cal, cal, sizeof(rippra_calibration));
    s->cal.subaps = (rippra_subap *)malloc(cal->nspots * sizeof(rippra_subap));
    memcpy(s->cal.subaps, cal->subaps, cal->nspots * sizeof(rippra_subap));

    /* Deep-copy zonal mesh */
    memcpy(&s->mesh, mesh, sizeof(rippra_zonal_mesh));
    s->mesh.node_u   = (int *)malloc(mesh->nnodes * sizeof(int));
    s->mesh.node_v   = (int *)malloc(mesh->nnodes * sizeof(int));
    s->mesh.G        = (double *)malloc((size_t)2 * cal->nspots * mesh->nnodes * sizeof(double));
    s->mesh.Gpinv    = (double *)malloc((size_t)mesh->nnodes * 2 * cal->nspots * sizeof(double));
    memcpy(s->mesh.node_u, mesh->node_u, mesh->nnodes * sizeof(int));
    memcpy(s->mesh.node_v, mesh->node_v, mesh->nnodes * sizeof(int));
    memcpy(s->mesh.G,     mesh->G,     (size_t)2 * cal->nspots * mesh->nnodes * sizeof(double));
    memcpy(s->mesh.Gpinv, mesh->Gpinv, (size_t)mesh->nnodes * 2 * cal->nspots * sizeof(double));

    /* Deep-copy modal model */
    memcpy(&s->model, model, sizeof(rippra_modal_model));
    s->model.mode_j = (int *)malloc(model->nmodes * sizeof(int));
    s->model.mode_n = (int *)malloc(model->nmodes * sizeof(int));
    s->model.mode_m = (int *)malloc(model->nmodes * sizeof(int));
    s->model.Zprime     = (double *)malloc((size_t)2 * cal->nspots * model->nmodes * sizeof(double));
    s->model.Zprime_pinv= (double *)malloc((size_t)model->nmodes * 2 * cal->nspots * sizeof(double));
    memcpy(s->model.mode_j, model->mode_j, model->nmodes * sizeof(int));
    memcpy(s->model.mode_n, model->mode_n, model->nmodes * sizeof(int));
    memcpy(s->model.mode_m, model->mode_m, model->nmodes * sizeof(int));
    memcpy(s->model.Zprime,      model->Zprime,      (size_t)2 * cal->nspots * model->nmodes * sizeof(double));
    memcpy(s->model.Zprime_pinv, model->Zprime_pinv, (size_t)model->nmodes * 2 * cal->nspots * sizeof(double));

    /* Double buffers */
    size_t frame_sz = (size_t)frame_width * frame_height;
    s->ping_buffer = (double *)calloc(frame_sz, sizeof(double));
    s->pong_buffer = (double *)calloc(frame_sz, sizeof(double));

    /* Ring buffer */
    s->ring_capacity = RIPPRA_STREAM_MAX_FRAMES;
    s->ring = (ring_entry *)calloc(s->ring_capacity, sizeof(ring_entry));
    s->ring_head = 0;
    s->ring_tail = 0;

    /* Result queue */
    s->results_capacity = RIPPRA_STREAM_MAX_FRAMES;
    s->results = (rippra_stream_result *)calloc(s->results_capacity,
                                                  sizeof(rippra_stream_result));
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
        s->results[i].W  = (double *)calloc(nnodes, sizeof(double));
        s->results[i].zernike_coeffs = (double *)calloc(nmodes, sizeof(double));
        s->results[i].dm_commands     = (double *)calloc(nnodes, sizeof(double));
        s->results[i].nspots = nspots;
        s->results[i].nnodes = nnodes;
        s->results[i].nmodes = nmodes;
    }

    /* Turbulence history */
    s->turb_nframes = 100;
    s->dx_history = (double *)calloc((size_t)s->turb_nframes * nspots, sizeof(double));
    s->dy_history = (double *)calloc((size_t)s->turb_nframes * nspots, sizeof(double));
    s->history_filled = 0;

    s->next_frame_id = 1;

#ifdef _WIN32
    InitializeCriticalSection(&s->lock);
#else
    pthread_mutex_init(&s->lock, NULL);
#endif

    return s;
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
    stream_lock(s);

    int next_tail = (s->ring_tail + 1) % s->ring_capacity;
    if (next_tail == s->ring_head) {
        stream_unlock(s);
        return -1; /* full */
    }

    ring_entry *e = &s->ring[s->ring_tail];
    if (!e->frame) {
        e->frame = (double *)malloc((size_t)w * h * sizeof(double));
        e->width = w;
        e->height = h;
    }
    memcpy(e->frame, frame, (size_t)w * h * sizeof(double));
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

    /* Prepare output slot */
    int rslot = s->results_tail;
    rippra_stream_result *r = &s->results[rslot];
    r->frame_id    = e->frame_id;
    r->timestamp_ms = e->timestamp_ms;
    r->nspots = nspots;
    r->nnodes = s->mesh.nnodes;
    r->nmodes = s->model.nmodes;

    /* Step 1: centroiding */
    rippa_compute_centroids(e->frame, e->width, e->height,
                            &s->cal, &s->cfg,
                            r->cx, r->cy);

    /* Step 2: deltas */
    rippa_compute_deltas(r->cx, r->cy, &s->cal, nspots, r->dx, r->dy);

    /* Step 3: zonal reconstruction */
    rippra_zonal_reconstruct(&s->mesh, r->dx, r->dy, &s->cfg, r->W);

    /* Step 4: modal reconstruction */
    rippra_modal_reconstruct(&s->model, r->dx, r->dy, &s->cfg,
                             r->zernike_coeffs);

    /* Step 5: turbulence (rolling window) */
    {
        int idx = (int)((e->frame_id - 1) % s->turb_nframes);
        memcpy(&s->dx_history[(size_t)idx * nspots], r->dx,
               nspots * sizeof(double));
        memcpy(&s->dy_history[(size_t)idx * nspots], r->dy,
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

    /* Step 6: DM command map */
    rippra_dm_map(r->W, s->mesh.nnodes, &s->mesh, &s->cfg, r->dm_commands);

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
    free(s->dx_history);
    free(s->dy_history);
    s->turb_nframes = nframes;
    s->dx_history = (double *)calloc((size_t)nframes * nspots, sizeof(double));
    s->dy_history = (double *)calloc((size_t)nframes * nspots, sizeof(double));
    s->history_filled = 0;
    stream_unlock(s);
}
