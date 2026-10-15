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

static const char *pick_path(const char *from_rippra, const char *from_root,
                             char *selected, size_t selected_size)
{
    FILE *fp = fopen(from_rippra, "rb");
    if (fp) {
        fclose(fp);
        snprintf(selected, selected_size, "%s", from_rippra);
        return selected;
    }
    fp = fopen(from_root, "rb");
    if (fp) {
        fclose(fp);
        snprintf(selected, selected_size, "%s", from_root);
        return selected;
    }
    return NULL;
}

int main(void) {
    rippa_config cfg;
    double *sh_flat = NULL, *img = NULL;
    int w = 648, h = 492;
    rippra_calibration cal;
    int rc;
    char config_path[256], flat_path[256], frame_path[256];

    printf("=== RIPRA Streaming Pipeline Test ===\n\n");

    /* Load config & data */
    if (!pick_path("config/system.conf", "rippra/config/system.conf",
                   config_path, sizeof(config_path)) ||
        !pick_path("data_raw/sh_flat.raw", "rippra/data_raw/sh_flat.raw",
                   flat_path, sizeof(flat_path)) ||
        !pick_path("data_raw/img.raw", "rippra/data_raw/img.raw",
                   frame_path, sizeof(frame_path))) {
        fprintf(stderr, "ERROR: stream fixtures are unavailable\n");
        return 1;
    }
    if (rippa_config_load(&cfg, config_path) != 0 ||
        rippa_load_raw(flat_path, w, h, &sh_flat) != 0 ||
        rippa_load_raw(frame_path, w, h, &img) != 0) {
        fprintf(stderr, "ERROR: stream fixture loading failed\n");
        free(sh_flat); free(img);
        return 1;
    }

    /* Calibration (one-time) */
    rc = rippa_calibrate_grid(sh_flat, w, h, &cfg, &cal);
    if (rc != 0) { fprintf(stderr, "ERROR: calibration failed\n"); return 1; }
    printf("Calibration: %d spots, pitch=%.1f px\n", cal.nspots, cal.pitch_px);

    /* Setup zonal mesh */
    rippra_zonal_mesh mesh;
    if (rippra_zonal_setup(&cal, &cfg, &mesh) != 0) return 1;
    printf("Zonal mesh: %d nodes\n", mesh.nnodes);

    /* Setup modal model */
    rippra_modal_model model;
    if (rippra_modal_setup(&cal, &cfg, &model) != 0) return 1;
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
        if (!isfinite(r->quality) || r->quality < 0.0 || r->quality > 1.0 ||
            (r->status != RIPPRA_FRAME_NOMINAL &&
             r->status != RIPPRA_FRAME_DEGRADED &&
             r->status != RIPPRA_FRAME_LOST &&
             r->status != RIPPRA_STATUS_ERROR)) {
            fprintf(stderr, "ERROR: invalid frame-quality result\n");
            return 1;
        }
        printf("  Frame %lld: r0=%.4e m, tau0=%.4f ms, "
               "first Zernike coeffs: %+.4f %+.4f %+.4f rad\n",
               (long long)r->frame_id,
               r->r0, r->tau0 * 1000.0,
               r->zernike_coeffs[0],
               r->zernike_coeffs[1],
               r->zernike_coeffs[2]);
    }
    printf("  Dequeued: %d results\n", dequeued);
    if (processed != 10 || dequeued != 10) {
        fprintf(stderr, "ERROR: expected 10 processed/dequeued results\n");
        return 1;
    }

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
