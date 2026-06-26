#include "rippra/predictive_ao.h"
#include <string.h>
#include <math.h>

#if defined(_WIN32) && defined(RIPPR_DYNAMIC)
__declspec(dllexport)
#endif

/* ---- LSTM ONNX Runtime inference ---- */

struct LSTMInference {
    void* ort_env;         /* OrtEnv* */
    void* ort_session;     /* OrtSession* */
};

#if 0
/* Full ONNX Runtime integration requires linking against onnxruntime.lib.
   Below is the complete implementation using the ONNX Runtime C API.
   Uncomment and link with -lonnxruntime when ONNX Runtime SDK is available. */

#include <onnxruntime/core/session/onnxruntime_c_api.h>

LSTMInference* predictive_ao_load_model(const char* onnx_path) {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) return NULL;

    LSTMInference* ctx = (LSTMInference*)calloc(1, sizeof(LSTMInference));
    if (!ctx) return NULL;

    OrtEnv* env = NULL;
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "predictive_ao", &env);
    if (!env) { free(ctx); return NULL; }
    ctx->ort_env = env;

    OrtSessionOptions* opts = NULL;
    api->CreateSessionOptions(&opts);
    api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_EXTENDED);

    OrtSession* session = NULL;
    api->CreateSession(env, onnx_path, opts, &session);
    api->ReleaseSessionOptions(opts);
    if (!session) { api->ReleaseEnv(env); free(ctx); return NULL; }
    ctx->ort_session = session;

    return ctx;
}

int predictive_ao_infer(LSTMInference* ctx, const float input[LSTM_LOOKBACK][LSTM_NMODES],
                        float output[LSTM_NMODES]) {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api || !ctx || !ctx->ort_session) return -1;

    OrtMemoryInfo* mem = NULL;
    api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem);

    const int64_t shape[3] = {1, LSTM_LOOKBACK, LSTM_NMODES};
    OrtValue* input_tensor = NULL;
    api->CreateTensorWithDataAsOrtValue(mem, (float*)input, 3 * sizeof(shape), shape, 3,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor = NULL;

    OrtStatus* status = api->Run(ctx->ort_session, NULL, input_names, (const OrtValue* const*)&input_tensor,
                                 1, output_names, 1, &output_tensor);

    api->ReleaseMemoryInfo(mem);
    api->ReleaseValue(input_tensor);

    if (status) {
        api->ReleaseStatus(status);
        return -1;
    }

    float* out_data = NULL;
    api->GetTensorMutableData(output_tensor, (void**)&out_data);
    memcpy(output, out_data, LSTM_NMODES * sizeof(float));
    api->ReleaseValue(output_tensor);

    return 0;
}

void predictive_ao_unload(LSTMInference* ctx) {
    if (!ctx) return;
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
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

int predictive_ao_infer(LSTMInference* ctx, const float input[LSTM_LOOKBACK][LSTM_NMODES],
                        float output[LSTM_NMODES]) {
    (void)ctx; (void)input; (void)output;
    return -1;
}

void predictive_ao_unload(LSTMInference* ctx) {
    (void)ctx;
}
#endif

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
        if (predictive_ao_infer(lstm, state->history, predicted_residual) == 0) {
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
