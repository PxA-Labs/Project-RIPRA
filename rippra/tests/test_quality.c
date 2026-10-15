/*
 * test_quality.c - frame-quality gate and routing tests.
 *
 * This test intentionally uses the generated SH-WFS flat frame so that the
 * calibration and streaming paths are exercised together.  The test prints
 * the measured processing latency for visibility, but does not impose a
 * machine-dependent timing assertion.
 */
#include "rippra/io.h"
#include "rippra/centroid.h"
#include "rippra/recon.h"
#include "rippra/stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CHECK(cond, message) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        return 1; \
    } \
    printf("PASS: %s\n", message); \
} while (0)

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

static void occlude_spots(double *frame, int width, int height,
                          const rippra_calibration *cal,
                          int first, int count)
{
    int end = first + count;
    if (end > cal->nspots) end = cal->nspots;
    for (int k = first; k < end; ++k) {
        const rippra_subap *spot = &cal->subaps[k];
        for (int row = spot->row_min; row <= spot->row_max; ++row) {
            for (int col = spot->col_min; col <= spot->col_max; ++col) {
                if (row >= 0 && row < height && col >= 0 && col < width)
                    frame[row * width + col] = 0.0;
            }
        }
    }
}

static int all_zero(const double *values, int count)
{
    for (int i = 0; i < count; ++i)
        if (values[i] != 0.0) return 0;
    return 1;
}

int main(void)
{
    const int width = 648, height = 492;
    const size_t npix = (size_t)width * height;
    rippa_config cfg;
    rippra_calibration cal = {0};
    rippra_zonal_mesh mesh = {0};
    rippra_modal_model model = {0};
    rippra_stream *stream = NULL;
    double *flat = NULL, *attenuated = NULL, *partial = NULL, *severe = NULL;
    int rc;
    char config_path[256], flat_path[256];

    CHECK(pick_path("config/system.conf", "rippra/config/system.conf",
                    config_path, sizeof(config_path)) != NULL,
          "quality configuration fixture is available");
    CHECK(pick_path("data_raw/sh_flat.raw", "rippra/data_raw/sh_flat.raw",
                    flat_path, sizeof(flat_path)) != NULL,
          "reference frame fixture is available");
    CHECK(rippa_config_load(&cfg, config_path) == 0,
          "quality configuration loads");
    CHECK(cfg.quality_hard >= 0.0 && cfg.quality_hard < cfg.quality_soft &&
          cfg.quality_soft <= 1.0, "quality thresholds are ordered and bounded");
    {
        const char *invalid_path = "/tmp/rippra_quality_invalid.conf";
        FILE *invalid = fopen(invalid_path, "w");
        CHECK(invalid != NULL, "invalid configuration fixture opens");
        fprintf(invalid, "quality_soft = 0.30\nquality_hard = 0.60\n");
        fclose(invalid);
        CHECK(rippa_config_load(&cfg, invalid_path) != 0,
              "invalid quality threshold ordering is rejected");
        remove(invalid_path);
        CHECK(rippa_config_load(&cfg, config_path) == 0,
              "valid quality configuration can be reloaded");
    }
    CHECK(rippa_load_raw(flat_path, width, height, &flat) == 0,
          "reference frame loads");

    rc = rippa_calibrate_grid(flat, width, height, &cfg, &cal);
    CHECK(rc == 0 && cal.nspots > 0, "quality calibration succeeds");
    CHECK(cal.ref_mass != NULL && cal.ref_second_moment != NULL,
          "calibration retains reference quality statistics");
    CHECK(rippra_zonal_setup(&cal, &cfg, &mesh) == 0,
          "quality zonal setup succeeds");
    CHECK(rippra_modal_setup(&cal, &cfg, &model) == 0,
          "quality modal setup succeeds");

    stream = rippra_stream_init(&cfg, &cal, &mesh, &model, width, height);
    CHECK(stream != NULL, "quality stream initializes");

    attenuated = (double *)malloc(npix * sizeof(double));
    partial = (double *)malloc(npix * sizeof(double));
    severe = (double *)malloc(npix * sizeof(double));
    CHECK(attenuated && partial && severe, "quality test frames allocate");
    for (size_t i = 0; i < npix; ++i) attenuated[i] = 0.20 * flat[i];
    memcpy(partial, flat, npix * sizeof(double));
    memcpy(severe, flat, npix * sizeof(double));
    occlude_spots(partial, width, height, &cal, 0, cal.nspots / 2);
    occlude_spots(severe, width, height, &cal, 0, (cal.nspots * 9) / 10);

    const rippra_stream_result *clean =
        rippra_stream_process_one(stream, flat, width, height);
    CHECK(clean != NULL && clean->status == RIPPRA_FRAME_NOMINAL,
          "clean frame is nominal");
    CHECK(isfinite(clean->quality) && clean->quality >= 0.0 &&
          clean->quality <= 1.0 && clean->quality >= 0.90,
          "clean quality is high and bounded");
    CHECK(clean->spot_mass[0] > 0.0 && clean->spot_mask[0] == 1,
          "clean result exposes calibrated spot mass and validity");
    double clean_quality = clean->quality;

    const rippra_stream_result *attenuated_result =
        rippra_stream_process_one(stream, attenuated, width, height);
    CHECK(attenuated_result != NULL && isfinite(attenuated_result->quality) &&
          attenuated_result->quality >= 0.0 && attenuated_result->quality <= 1.0,
          "attenuated quality is bounded");
    printf("INFO: clean quality=%.4f attenuated quality=%.4f status=%d\n",
           clean_quality, attenuated_result->quality, attenuated_result->status);
    CHECK(attenuated_result->quality < clean_quality &&
          attenuated_result->status == RIPPRA_FRAME_DEGRADED,
          "attenuation lowers quality and routes to degraded fallback");

    const rippra_stream_result *partial_result =
        rippra_stream_process_one(stream, partial, width, height);
    CHECK(partial_result != NULL && partial_result->status == RIPPRA_FRAME_DEGRADED,
          "partial occlusion routes to degraded fallback");
    CHECK(partial_result->quality < clean_quality,
          "occlusion decreases quality");
    int invalid = 0;
    for (int i = 0; i < partial_result->nspots; ++i)
        if (!partial_result->spot_mask[i]) invalid++;
    CHECK(invalid > 0, "partial occlusion exposes invalid spots");

    const rippra_stream_result *severe_result =
        rippra_stream_process_one(stream, severe, width, height);
    CHECK(severe_result != NULL && severe_result->status == RIPPRA_FRAME_LOST,
          "severe occlusion routes to lost");
    CHECK(severe_result->quality >= 0.0 && severe_result->quality <= 1.0 &&
          severe_result->quality < partial_result->quality &&
          all_zero(severe_result->dm_commands, severe_result->nnodes) &&
          all_zero(severe_result->W, severe_result->nnodes) &&
          all_zero(severe_result->zernike_coeffs, severe_result->nmodes),
          "lost frame has bounded quality and zero outputs");

    /* A missing/blank frame must be safe and must not inherit the previous DM
     * command. */
    memset(severe, 0, npix * sizeof(double));
    const rippra_stream_result *blank_result =
        rippra_stream_process_one(stream, severe, width, height);
    CHECK(blank_result != NULL && blank_result->status == RIPPRA_FRAME_LOST &&
          blank_result->quality == 0.0 &&
          all_zero(blank_result->dm_commands, blank_result->nnodes),
          "blank frame is safely lost with no stale DM commands");

    /* The gate is designed for a short deterministic processing path, but
     * timing is reported rather than asserted because CI hosts vary. */
    {
        const int samples = 5;
        double total_us = 0.0;
        for (int i = 0; i < samples; ++i) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            const rippra_stream_result *result =
                rippra_stream_process_one(stream, flat, width, height);
            clock_gettime(CLOCK_MONOTONIC, &end);
            CHECK(result != NULL, "latency sample produces a result");
            total_us += (end.tv_sec - start.tv_sec) * 1.0e6 +
                        (end.tv_nsec - start.tv_nsec) / 1.0e3;
        }
        printf("INFO: quality-gated process_one mean latency %.2f us (target < 20 us; informational)\n",
               total_us / samples);
    }

    rippra_stream_shutdown(stream);
    rippra_modal_free(&model);
    rippra_zonal_free(&mesh);
    rippa_calibration_free(&cal);
    free(flat); free(attenuated); free(partial); free(severe);
    printf("Frame-quality gate tests complete.\n");
    return 0;
}
