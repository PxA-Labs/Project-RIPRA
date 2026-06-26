/*
 * tests/test_stream.c - Test real-time streaming pipeline (double-buffering)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/la.h"
#include "rippra/recon.h"
#include "rippra/stream.h"

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    int rc;

    printf("=== RIPRA Streaming Pipeline Test ===\n\n");

    /* Load config & data */
    rippa_config_load(&cfg, "config/system.conf");
    rippa_load_raw("data_raw/sh_flat.raw", w, h, &sh_flat);
    rippa_load_raw("data_raw/img.raw", w, h, &img);

    /* Calibration (one-time) */
    rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    printf("Calibration: %d spots, pitch=%.1f px\n", cal.nspots, cal.pitch_px);

    /* Setup zonal mesh */
    rippra_zonal_mesh mesh;
    rippra_zonal_setup(&cal, &cfg, &mesh);
    printf("Zonal mesh: %d nodes\n", mesh.nnodes);

    /* Setup modal model */
    rippra_modal_model model;
    rippra_modal_setup(&cal, &cfg, &model);
    printf("Modal model: %d Zernike modes\n", model.nmodes);

    /* Initialize streaming pipeline */
    rippra_stream *s = rippra_stream_init(&cfg, &cal, &mesh, &model, w, h);
    if (!s) { printf("ERROR: stream init failed\n"); return 1; }

    printf("\n-- Enqueuing 10 frames (same img.raw repeated) --\n");
    for (int i = 0; i < 10; ++i) {
        int64_t fid = rippra_stream_enqueue(s, img, w, h);
        printf("  Enqueued frame %lld\n", (long long)fid);
    }

    printf("\n-- Processing frames --\n");
    int processed = 0;
    while (rippra_stream_pending(s) > 0) {
        rc = rippra_stream_process(s);
        if (rc == 0) processed++;
    }
    printf("  Processed: %d frames\n", processed);

    printf("\n-- Dequeuing results --\n");
    int dequeued = 0;
    const rippra_stream_result *r;
    while ((r = rippra_stream_dequeue(s)) != NULL) {
        dequeued++;
        printf("  Frame %lld: r0=%.4e m, tau0=%.4f ms, "
               "first Zernike coeffs: %+.4f %+.4f %+.4f rad\n",
               (long long)r->frame_id,
               r->r0, r->tau0 * 1000.0,
               r->zernike_coeffs[0],
               r->zernike_coeffs[1],
               r->zernike_coeffs[2]);
    }
    printf("  Dequeued: %d results\n", dequeued);

    /* Test convenience function */
    printf("\n-- process_one (convenience) --\n");
    r = rippra_stream_process_one(s, img, w, h);
    if (r) {
        printf("  Frame %lld: phase RMS=%.4e rad\n",
               (long long)r->frame_id, r->r0);
    }

    /* Cleanup */
    rippra_stream_shutdown(s);
    rippra_zonal_free(&mesh);
    rippra_modal_free(&model);
    rippa_calibration_free(&cal);
    free(sh_flat);
    free(img);

    printf("\n=== Streaming Pipeline Test Complete ===\n");
    return 0;
}
