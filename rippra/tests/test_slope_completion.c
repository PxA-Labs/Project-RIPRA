/*
 * tests/test_slope_completion.c — Unit test for slope-domain ring buffer
 * and predictive_ao_complete_slopes() fallback behavior.
 *
 * Part of Issue #90 (AI fallback) validation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rippra/predictive_ao.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(name, cond) do { \
    if (cond) { tests_passed++; printf("  PASS: %s\n", name); } \
    else      { tests_failed++; printf("  FAIL: %s\n", name); } \
} while (0)

static void test_slope_state_init(void) {
    printf("\n--- Slope State Init ---\n");
    PredictiveAOSlopeState state;

    predictive_ao_slope_init(&state, 137, 10);
    CHECK("nspots set", state.nspots == 137);
    CHECK("lookback set", state.lookback == 10);
    CHECK("frame_count zero", state.frame_count == 0);

    /* Default values when args are 0 */
    PredictiveAOSlopeState state2;
    predictive_ao_slope_init(&state2, 0, 0);
    CHECK("nspots defaults to MAX", state2.nspots == PREDICTIVE_AO_MAX_NSPOTS);
    CHECK("lookback defaults to LSTM_LOOKBACK", state2.lookback == LSTM_LOOKBACK);

    predictive_ao_slope_free(&state);
    CHECK("free resets frame_count", state.frame_count == 0);
}

static void test_slope_push(void) {
    printf("\n--- Slope Push ---\n");
    PredictiveAOSlopeState state;
    int nspots = 4;  /* small for testing */
    int lookback = 3;

    predictive_ao_slope_init(&state, nspots, lookback);

    float dx[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dy[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int mask[] = {1, 1, 0, 1};

    /* Push first frame */
    predictive_ao_slope_push(&state, dx, dy, mask, nspots);
    CHECK("frame_count incremented", state.frame_count == 1);

    /* Check buffer layout: [dx, dy, mask, inv_mask] */
    CHECK("dx[0] stored", state.buffer[0] == 1.0f);
    CHECK("dx[3] stored", state.buffer[3] == 4.0f);
    CHECK("dy[0] stored", state.buffer[nspots + 0] == 5.0f);
    CHECK("mask[0] = 1", state.buffer[2 * nspots + 0] == 1.0f);
    CHECK("mask[2] = 0", state.buffer[2 * nspots + 2] == 0.0f);
    CHECK("inv_mask[2] = 1", state.buffer[3 * nspots + 2] == 1.0f);
    CHECK("inv_mask[0] = 0", state.buffer[3 * nspots + 0] == 0.0f);

    /* Push frames to fill the buffer */
    float dx2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float dy2[] = {50.0f, 60.0f, 70.0f, 80.0f};
    int mask2[] = {1, 1, 1, 1};
    predictive_ao_slope_push(&state, dx2, dy2, mask2, nspots);
    CHECK("frame_count = 2", state.frame_count == 2);

    float dx3[] = {100.0f, 200.0f, 300.0f, 400.0f};
    float dy3[] = {500.0f, 600.0f, 700.0f, 800.0f};
    predictive_ao_slope_push(&state, dx3, dy3, NULL, nspots);
    CHECK("frame_count = 3 (buffer full)", state.frame_count == 3);

    /* Verify NULL mask defaults to all-valid */
    int feat = 4 * nspots;
    CHECK("NULL mask -> mask[0] = 1",
          state.buffer[2 * feat + 2 * nspots + 0] == 1.0f);

    /* Push one more (should shift buffer) */
    float dx4[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float dy4[] = {-5.0f, -6.0f, -7.0f, -8.0f};
    int mask4[] = {0, 0, 0, 0};
    predictive_ao_slope_push(&state, dx4, dy4, mask4, nspots);
    CHECK("frame_count = 4 (after shift)", state.frame_count == 4);

    /* After shift, the last row should contain dx4 */
    float *last_row = &state.buffer[(lookback - 1) * feat];
    CHECK("last row dx[0] after shift", last_row[0] == -1.0f);
    CHECK("last row dy[0] after shift", last_row[nspots] == -5.0f);
    CHECK("last row mask[0] = 0", last_row[2 * nspots] == 0.0f);

    predictive_ao_slope_free(&state);
}

static void test_complete_slopes_null_model(void) {
    printf("\n--- Complete Slopes (NULL model fallback) ---\n");
    PredictiveAOSlopeState state;
    int nspots = 4;

    predictive_ao_slope_init(&state, nspots, 3);

    float dx[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dy[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int mask[] = {1, 0, 1, 0};
    float out_dx[4], out_dy[4];

    /* With NULL model, should return 1 (graceful fallback) */
    int rc = predictive_ao_complete_slopes(NULL, &state, dx, dy, mask, nspots,
                                            out_dx, out_dy);
    CHECK("NULL model returns 1 (fallback)", rc == 1);
    CHECK("frame_count incremented by complete_slopes", state.frame_count == 1);

    predictive_ao_slope_free(&state);
}

static void test_complete_slopes_edge_cases(void) {
    printf("\n--- Complete Slopes (Edge Cases) ---\n");
    PredictiveAOSlopeState state;
    float out_dx[4], out_dy[4];

    /* NULL state */
    int rc = predictive_ao_complete_slopes(NULL, NULL, NULL, NULL, NULL, 0,
                                            out_dx, out_dy);
    CHECK("NULL state returns -1", rc == -1);

    /* Zero nspots */
    predictive_ao_slope_init(&state, 4, 3);
    rc = predictive_ao_complete_slopes(NULL, &state, NULL, NULL, NULL, 0,
                                        out_dx, out_dy);
    CHECK("nspots=0 returns -1", rc == -1);

    /* NULL output buffers */
    float dx[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dy[] = {5.0f, 6.0f, 7.0f, 8.0f};
    int mask[] = {1, 1, 1, 1};
    rc = predictive_ao_complete_slopes(NULL, &state, dx, dy, mask, 4,
                                        NULL, out_dy);
    CHECK("NULL out_dx returns -1", rc == -1);
    rc = predictive_ao_complete_slopes(NULL, &state, dx, dy, mask, 4,
                                        out_dx, NULL);
    CHECK("NULL out_dy returns -1", rc == -1);

    predictive_ao_slope_free(&state);
}

static void test_predictive_ao_model_load_unload(void) {
    printf("\n--- Model Load/Unload (stub) ---\n");

    /* Without RIPRA_ONNXRT defined, load_model returns NULL */
    LSTMInference *ctx = predictive_ao_load_model("nonexistent.onnx");
    CHECK("load_model returns NULL (no ONNXRT)", ctx == NULL);

    /* Unload NULL should not crash */
    predictive_ao_unload(NULL);
    CHECK("unload(NULL) does not crash", 1);

    /* Infer with NULL context */
    float dummy_in[4] = {0};
    float dummy_out[2] = {0};
    int64_t shape[2] = {1, 4};
    int rc = predictive_ao_infer(NULL, dummy_in, 2, shape, dummy_out, 2);
    CHECK("infer(NULL) returns -1", rc == -1);
}

static void test_model_set_geometry(void) {
    printf("\n--- Model Set Geometry ---\n");

    /* NULL context should not crash */
    predictive_ao_model_set_geometry(NULL, 137, 10);
    CHECK("set_geometry(NULL) does not crash", 1);
}

int main(void) {
    printf("=== RIPRA Slope Completion Unit Test ===\n");

    test_slope_state_init();
    test_slope_push();
    test_complete_slopes_null_model();
    test_complete_slopes_edge_cases();
    test_predictive_ao_model_load_unload();
    test_model_set_geometry();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
