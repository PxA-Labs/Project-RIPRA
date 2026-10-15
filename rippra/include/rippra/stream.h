/*
 * rippra/include/rippra/stream.h - Real-time streaming pipeline for RIPPA
 *
 * Double-buffered (ping-pong) frame acquisition + processing pipeline.
 * Designed for continuous SH-WFS frame capture and wavefront reconstruction.
 *
 * Usage:
 *   1. Call rippra_stream_init() with calibration + system config
 *   2. Launch separate threads:
 *      - Main loop: call rippra_stream_enqueue() for each incoming frame
 *      - Processing thread: call rippra_stream_process() in a loop
 *      - Output thread: call rippra_stream_dequeue() for results
 *   3. rippra_stream_shutdown() to clean up
 *
 * Thread-safety: internally mutex-locked circular buffer (SPSC ring buffer).
 */
#ifndef RIPPRA_STREAM_H
#define RIPPRA_STREAM_H

#include <stdint.h>
#include "rippra/centroid.h"
#include "rippra/recon.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Constants ----------------------------------------------------------- */
#define RIPPRA_STREAM_MAX_FRAMES 256   /* circular buffer capacity */
#define RIPPRA_STREAM_NUM_BUFS     2   /* ping-pong frame buffers   */
/* ---- Frame quality status ----------------------------------------------- */
typedef enum {
    RIPPRA_STATUS_NOMINAL = 0,
    RIPPRA_STATUS_DEGRADED = 1,
    RIPPRA_STATUS_LOST = 2,
    RIPPRA_STATUS_ERROR = 3,
    /* Compatibility names retained for the initial Issue #93 implementation. */
    RIPPRA_FRAME_NOMINAL = RIPPRA_STATUS_NOMINAL,
    RIPPRA_FRAME_DEGRADED = RIPPRA_STATUS_DEGRADED,
    RIPPRA_FRAME_LOST = RIPPRA_STATUS_LOST
} rippra_frame_status;

/* Synchronous degraded-frame hook.  The callback does not own any argument;
 * observed arrays and history are valid only for the duration of the call.
 * ctx is caller-owned and is never freed by the stream.  It runs on the
 * processing thread while stream processing is serialized; it must not call
 * stream APIs or retain the array pointers.  Return 0 and finite output
 * arrays to accept the completed slopes; any other result selects the
 * built-in spatial interpolation fallback. */
typedef int (*rippra_quality_fallback_fn)(
    const double *observed_dx, const double *observed_dy,
    const int *mask, int nspots, const double *history, void *ctx,
    double *out_dx, double *out_dy);

/* ---- Per-frame result ---------------------------------------------------- */
typedef struct {
    int64_t  frame_id;
    double   timestamp_ms;       /* wall clock at capture             */

    /* centroid data (host, heap) */
    double  *cx, *cy;
    double  *dx, *dy;
    int      nspots;
    double *spot_mass;       /* integrated intensity per spot */
    int *spot_mask;          /* 1 = valid centroid/mass, 0 = invalid */
    int *mask;               /* compatibility alias for spot_mask */
    double quality;          /* frame quality score in [0, 1] */
    rippra_frame_status status;
    /* zonal reconstruction */
    double  *W;                  /* phase heights (nnodes)           */
    int      nnodes;

    /* modal reconstruction */
    double  *zernike_coeffs;     /* Zernike coefficients (nmodes)    */
    int      nmodes;

    /* turbulence metrics (estimated over a window) */
    double   r0;                 /* Fried parameter (m)              */
    double   tau0;               /* coherence time (s)               */

    /* DM command map */
    double  *dm_commands;        /* actuator strokes (nnodes)        */
} rippra_stream_result;

/* ---- Pipeline handle ----------------------------------------------------- */
typedef struct rippra_stream rippra_stream;

/* ---- API ----------------------------------------------------------------- */

/* Initialize the streaming pipeline.
 *   cfg  - system configuration (copied)
 *   cal  - calibration data (copied)
 *   mesh - zonal mesh (copied; must have been set up with rippra_zonal_setup)
 *   model- modal model (copied; must have been set up with rippra_modal_setup)
 *   frame_width, frame_height - frame dimensions in pixels
 * Returns NULL on allocation failure.
 */
rippra_stream *rippra_stream_init(const rippa_config *cfg,
                                   const rippra_calibration *cal,
                                   const rippra_zonal_mesh *mesh,
                                   const rippra_modal_model *model,
                                   int frame_width, int frame_height);

/* Shut down the pipeline, free all resources. Blocks until processing done. */
void rippra_stream_shutdown(rippra_stream *s);

/* ---- Frame producers (call from acquisition thread) ---------------------- */

/* Enqueue a raw frame for processing.
 *  frame  - raw frame pixels (double [width * height], normalised [0,1])
 *  w, h   - frame dimensions
 *  Returns a frame_id (>0) on success, -1 if ring buffer is full.
 *  The caller retains ownership of frame[] (it is copied internally).
 */
int64_t rippra_stream_enqueue(rippra_stream *s,
                               const double *frame, int w, int h);

/* Return the number of frames waiting to be processed. */
int rippra_stream_pending(rippra_stream *s);

/* Return the number of results ready for dequeue. */
int rippra_stream_ready(rippra_stream *s);

/* ---- Frame consumers (call from processing & output threads) ------------- */

/* Process the next frame from the ring buffer.
 *  Calls centroiding and routes the frame through the quality gate before
 *  zonal/modal reconstruction and DM mapping.  A degraded frame uses the
 *  built-in nearest-neighbour interpolation fallback.  A lost frame skips
 *  reconstruction and returns zero phase, coefficients, and DM commands.
 *  Returns 0 on success, -1 if no frame is pending, or -2 if the result
 *  queue is full (the input remains pending in that case).
 *  The result is stored internally and can be retrieved with dequeue.
 */
int rippra_stream_process(rippra_stream *s);

/* Dequeue the next processed result.
 *  Returns a pointer to the result (valid until the next dequeue call),
 *  or NULL if no results are ready.
 *  The result memory is managed internally.
 */
const rippra_stream_result *rippra_stream_dequeue(rippra_stream *s);

/* ---- Convenience: process + dequeue in one call ------------------------- */

/* Process one frame and get result. Blocking version (busy-waits). */
const rippra_stream_result *rippra_stream_process_one(rippra_stream *s,
                                                       const double *frame,
                                                       int w, int h);

/* ---- Turbulence window --------------------------------------------------- */

/* Set the number of frames used for r0 / tau0 estimation (default: 100).
 *  Must be called before any processing.
 */
void rippra_stream_set_turbulence_window(rippra_stream *s, int nframes);

/* Install or clear the degraded-frame fallback.  The stream borrows fn and
 * ctx; the caller guarantees their lifetime until this function replaces the
 * callback or the stream is shut down.  Thread-safe; takes effect for the
 * next processed frame.  Returns 0, or -1 for a NULL stream. */
int rippra_stream_set_fallback(rippra_stream *s,
                               rippra_quality_fallback_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* RIPPRA_STREAM_H */
