#include "rippra/predictive_ao.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(_WIN32) && defined(RIPPR_DYNAMIC)
__declspec(dllexport)
#endif

/* ---- ONNX Runtime inference ---- */

struct LSTMInference {
    void* ort_env;         /* OrtEnv* */
    void* ort_session;     /* OrtSession* */
    int   nspots;          /* expected number of sub-apertures (for slope completion) */
    int   lookback;        /* history length in frames */
};

/*
 * Use the ONNX Runtime C API when the library is available.  If the SDK is
 * not linked, the functions below still compile but return failure, so the
 * runtime falls back to spatial interpolation / persistence prediction.
 *
 * To enable, link with -lonnxruntime and define RIPRA_ONNXRT.  The CI static
 * build deliberately leaves it undefined so the library has no external ML
 * dependency; ONNX models are validated in the Python test suite instead.
 */
#ifdef RIPRA_ONNXRT

#include <onnxruntime/core/session/onnxruntime_c_api.h>

static const OrtApi* predictive_ao_api(void) {
    const OrtApiBase* base = OrtGetApiBase();
    return base ? base->GetApi(ORT_API_VERSION) : NULL;
}

LSTMInference* predictive_ao_load_model(const char* onnx_path) {
    const OrtApi* api = predictive_ao_api();
    if (!api) return NULL;

    LSTMInference* ctx = (LSTMInference*)calloc(1, sizeof(LSTMInference));
    if (!ctx) return NULL;

    OrtEnv* env = NULL;
    OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "predictive_ao", &env);
    if (status) { api->ReleaseStatus(status); free(ctx); return NULL; }
    ctx->ort_env = env;

    OrtSessionOptions* opts = NULL;
    status = api->CreateSessionOptions(&opts);
    if (status) { api->ReleaseStatus(status); api->ReleaseEnv(env); free(ctx); return NULL; }

    api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_EXTENDED);

    OrtSession* session = NULL;
    status = api->CreateSession(env, onnx_path, opts, &session);
    api->ReleaseSessionOptions(opts);
    if (status) { api->ReleaseStatus(status); api->ReleaseEnv(env); free(ctx); return NULL; }
    ctx->ort_session = session;

    /* Defaults for the existing Zernike predictor.  Slope-completion callers
       update these via predictive_ao_model_set_geometry(). */
    ctx->nspots = 0;
    ctx->lookback = LSTM_LOOKBACK;

    return ctx;
}

int predictive_ao_infer(LSTMInference* ctx, const float* input, int input_rank,
                          const int64_t* shape, float* output, int output_len) {
    const OrtApi* api = predictive_ao_api();
    if (!api || !ctx || !ctx->ort_session) return -1;

    OrtMemoryInfo* mem = NULL;
    OrtStatus* status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);
    if (status) { api->ReleaseStatus(status); return -1; }

    size_t nfloats = 1;
    for (int i = 0; i < input_rank; ++i) nfloats *= (size_t)shape[i];

    OrtValue* input_tensor = NULL;
    status = api->CreateTensorWithDataAsOrtValue(mem, (float*)input,
                                                 nfloats * sizeof(float),
                                                 shape, input_rank,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                 &input_tensor);
    if (status) { api->ReleaseStatus(status); api->ReleaseMemoryInfo(mem); return -1; }

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;

    status = api->Run(ctx->ort_session, NULL, input_names,
                      (const OrtValue* const*)&input_tensor, 1,
                      output_names, 1, &output_tensor);

    api->ReleaseMemoryInfo(mem);
    api->ReleaseValue(input_tensor);

    if (status) { api->ReleaseStatus(status); return -1; }

    float* out_data = NULL;
    api->GetTensorMutableData(output_tensor, (void**)&out_data);
    if (out_data) memcpy(output, out_data, output_len * sizeof(float));
    api->ReleaseValue(output_tensor);

    return out_data ? 0 : -1;
}

void predictive_ao_unload(LSTMInference* ctx) {
    if (!ctx) return;
    const OrtApi* api = predictive_ao_api();
    if (api) {
        if (ctx->ort_session) api->ReleaseSession((OrtSession*)ctx->ort_session);
        if (ctx->ort_env) api->ReleaseEnv((OrtEnv*)ctx->ort_env);
    }
    free(ctx);
}

#else
/* Stub implementation (when ONNX Runtime is not linked) */

LSTMInference* predictive_ao_load_model(const char* onnx_path) {
    (void)onnx_path;
    return NULL;
}

int predictive_ao_infer(LSTMInference* ctx, const float* input, int input_rank,
                          const int64_t* shape, float* output, int output_len) {
    (void)ctx; (void)input; (void)input_rank; (void)shape; (void)output; (void)output_len;
    return -1;
}

void predictive_ao_unload(LSTMInference* ctx) {
    (void)ctx;
}
#endif

void predictive_ao_model_set_geometry(LSTMInference* ctx, int nspots, int lookback) {
    if (!ctx) return;
    ctx->nspots = nspots;
    ctx->lookback = lookback;
}

/* ---- State management ---- */

void predictive_ao_state_init(PredictiveAOState* state) {
    memset(state, 0, sizeof(PredictiveAOState));
}

void predictive_ao_push(PredictiveAOState* state, const float coeffs[LSTM_NMODES]) {
    if (state->frame_count < LSTM_LOOKBACK) {
        /* Still filling buffer: append */
        memcpy(state->history[state->frame_count], coeffs, LSTM_NMODES * sizeof(float));
    } else {
        /* Shift and insert */
        memmove(&state->history[0], &state->history[1],
                (LSTM_LOOKBACK - 1) * LSTM_NMODES * sizeof(float));
        memcpy(state->history[LSTM_LOOKBACK - 1], coeffs, LSTM_NMODES * sizeof(float));
    }
    state->frame_count++;
}

const float* predictive_ao_history(const PredictiveAOState* state) {
    return (const float*)state->history;
}

/* ---- Slope completion state (parallel ring buffer) ---- */

void predictive_ao_slope_init(PredictiveAOSlopeState* state, int nspots, int lookback) {
    memset(state, 0, sizeof(PredictiveAOSlopeState));
    state->nspots = nspots > 0 ? nspots : PREDICTIVE_AO_MAX_NSPOTS;
    state->lookback = lookback > 0 ? lookback : LSTM_LOOKBACK;
}

void predictive_ao_slope_free(PredictiveAOSlopeState* state) {
    if (!state) return;
    memset(state, 0, sizeof(PredictiveAOSlopeState));
}

void predictive_ao_slope_push(PredictiveAOSlopeState* state,
                              const float* dx, const float* dy,
                              const int* mask, int nspots) {
    if (nspots <= 0 || nspots > PREDICTIVE_AO_MAX_NSPOTS) return;
    if (state->nspots == 0) state->nspots = nspots;
    else if (nspots != state->nspots) return;
    int L = state->lookback;
    if (L <= 0) L = LSTM_LOOKBACK;
    if (state->frame_count < (size_t)L) {
        float* slot = &state->buffer[state->frame_count * 4 * state->nspots];
        for (int i = 0; i < nspots; ++i) {
            slot[i]              = dx[i];
            slot[i + state->nspots]   = dy[i];
            slot[i + 2*state->nspots] = mask ? (float)mask[i] : 1.0f;
            slot[i + 3*state->nspots] = mask ? (1.0f - (float)mask[i]) : 0.0f;
        }
    } else {
        size_t row_bytes = 4 * state->nspots * sizeof(float);
        memmove(state->buffer, state->buffer + 4 * state->nspots, ((size_t)L - 1) * row_bytes);
        float* slot = &state->buffer[((size_t)L - 1) * 4 * state->nspots];
        for (int i = 0; i < nspots; ++i) {
            slot[i]              = dx[i];
            slot[i + state->nspots]   = dy[i];
            slot[i + 2*state->nspots] = mask ? (float)mask[i] : 1.0f;
            slot[i + 3*state->nspots] = mask ? (1.0f - (float)mask[i]) : 0.0f;
        }
    }
    state->frame_count++;
}

static int predictive_ao_ready(const PredictiveAOSlopeState* state) {
    return state->frame_count >= (size_t)state->lookback;
}

/* ---- Control loop ---- */

float predictive_ao_step(PredictiveAOState* state, const float coeffs[LSTM_NMODES],
                         LSTMInference* lstm, float gain, float dm_out[LSTM_NMODES]) {
    float residual_error[LSTM_NMODES];
    float target_correction[LSTM_NMODES];

    /* 1. Current residual = measured coeffs + accumulated DM correction */
    for (int i = 0; i < LSTM_NMODES; i++) {
        residual_error[i] = coeffs[i] + state->dm_correction[i];
    }
    float rms = 0.0f;
    for (int i = 0; i < LSTM_NMODES; i++) {
        rms += residual_error[i] * residual_error[i];
    }
    rms = sqrtf(rms / LSTM_NMODES);

    /* 2. Push current residual into history buffer */
    predictive_ao_push(state, residual_error);

    /* 3. Determine target correction */
    if (lstm && state->frame_count >= LSTM_LOOKBACK) {
        /* Predictive: use LSTM to predict future residual, correct that */
        float predicted_residual[LSTM_NMODES];
        int64_t shape[3] = {1, LSTM_LOOKBACK, LSTM_NMODES};
        if (predictive_ao_infer(lstm, (const float*)state->history, 3, shape,
                                predicted_residual, LSTM_NMODES) == 0) {
            for (int i = 0; i < LSTM_NMODES; i++) {
                target_correction[i] = -gain * predicted_residual[i];
            }
        } else {
            /* Fallback to persistence prediction */
            for (int i = 0; i < LSTM_NMODES; i++) {
                target_correction[i] = -gain * residual_error[i];
            }
        }
    } else {
        /* Standard closed-loop: correct current residual */
        for (int i = 0; i < LSTM_NMODES; i++) {
            target_correction[i] = -gain * residual_error[i];
        }
    }

    /* 4. Update DM state */
    for (int i = 0; i < LSTM_NMODES; i++) {
        state->dm_correction[i] += target_correction[i];
        dm_out[i] = target_correction[i];
    }

    return rms;
}

/*
 * Complete missing or corrupted sub-aperture slopes using the ONNX Runtime
 * slope-completion model.  The state buffer holds the last L frames; each
 * frame stores [dx, dy, mask, inv_mask] (4*nspots floats).  The current frame
 * is the last row and may already have valid entries; the model returns a
 * full slope vector.
 *
 * Returns 0 on success, 1 if no model is loaded (caller should fall back to
 * spatial interpolation), negative on runtime error.
 */
int predictive_ao_complete_slopes(LSTMInference* model,
                                    PredictiveAOSlopeState* state,
                                    const float* observed_dx,
                                    const float* observed_dy,
                                    const int* mask, int nspots,
                                    float* out_dx, float* out_dy) {
    if (!state || nspots <= 0 || nspots > PREDICTIVE_AO_MAX_NSPOTS || !out_dx || !out_dy) return -1;
    if (state->nspots > 0 && nspots != state->nspots) return -1;

    /* Update ring buffer with current partial observation */
    predictive_ao_slope_push(state, observed_dx, observed_dy, mask, nspots);

    if (!model) return 1;  /* graceful fallback to spatial interpolation */
    if (!predictive_ao_ready(state)) return 1;  /* not enough history yet */

    /* Ensure model geometry matches runtime */
    if (model->nspots == 0) predictive_ao_model_set_geometry(model, nspots, state->lookback);

    int L = state->lookback;
    int feat = 4 * state->nspots;
    int64_t shape[3] = {1, L, feat};
    float* output = (float*)malloc(2 * state->nspots * sizeof(float));
    if (!output) return -2;

    int rc = predictive_ao_infer(model, state->buffer, 3, shape, output, 2 * state->nspots);
    if (rc != 0) { free(output); return rc; }

    for (int i = 0; i < nspots; ++i) {
        out_dx[i] = output[i];
        out_dy[i] = output[i + state->nspots];
    }

    free(output);
    return 0;
}
