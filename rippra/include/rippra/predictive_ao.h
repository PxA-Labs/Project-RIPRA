#ifndef RIPPR_PREDICTIVE_AO_H
#define RIPPR_PREDICTIVE_AO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define PREDICTIVE_AO_VERSION "11.2.0"
#define LSTM_LOOKBACK     10
#define LSTM_NMODES       20
#define LSTM_HIDDEN       128
#define LSTM_NLAYERS      2

/* ONNX Runtime LSTM session handle (opaque) */
typedef struct LSTMInference LSTMInference;

/* Predictive AO state */
typedef struct {
    float history[LSTM_LOOKBACK][LSTM_NMODES];  /* sliding window */
    int   frame_count;                           /* frames seen so far */
    float dm_correction[LSTM_NMODES];            /* accumulated DM state in Zernike space */
} PredictiveAOState;

/* ---------- LSTM ONNX Runtime inference ---------- */

/* Load LSTM ONNX model. Returns NULL on failure. */
LSTMInference* predictive_ao_load_model(const char* onnx_path);

/* Run ONNX inference. Input: float[LSTM_LOOKBACK][LSTM_NMODES], Output: float[LSTM_NMODES]. Returns 0 on success. */
int predictive_ao_infer(LSTMInference* ctx, const float input[LSTM_LOOKBACK][LSTM_NMODES],
                        float output[LSTM_NMODES]);

/* Unload model */
void predictive_ao_unload(LSTMInference* ctx);

/* ---------- State management ---------- */

/* Initialize predictive AO state (zero history, zero DM correction) */
void predictive_ao_state_init(PredictiveAOState* state);

/* Shift buffer and insert latest coefficients */
void predictive_ao_push(PredictiveAOState* state, const float coeffs[LSTM_NMODES]);

/* Get pointer to current history buffer (LSTM_LOOKBACK × LSTM_NMODES) */
const float* predictive_ao_history(const PredictiveAOState* state);

/* ---------- Control loop ---------- */

/*
 * Single predictive AO step.
 *   state   - predictive AO state (history + DM correction)
 *   coeffs  - measured Zernike coefficients from current frame [NMODES]
 *   lstm    - LSTM model (NULL to fall back to persistence / standard CL)
 *   gain    - closed-loop gain (0, 1]
 *   dm_out  - output: delta DM commands [NMODES] (feed to rippra_dm_map)
 *
 * Returns residual RMS (rad) after correction.
 */
float predictive_ao_step(PredictiveAOState* state, const float coeffs[LSTM_NMODES],
                         LSTMInference* lstm, float gain, float dm_out[LSTM_NMODES]);

#ifdef __cplusplus
}
#endif

#endif /* RIPPR_PREDICTIVE_AO_H */
