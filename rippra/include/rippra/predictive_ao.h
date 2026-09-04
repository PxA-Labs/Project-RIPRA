#ifndef RIPPR_PREDICTIVE_AO_H
#define RIPPR_PREDICTIVE_AO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define PREDICTIVE_AO_VERSION "11.3.0"
#define LSTM_LOOKBACK     10
#define LSTM_NMODES       20
#define LSTM_HIDDEN       128
#define LSTM_NLAYERS      2
#define PREDICTIVE_AO_MAX_NSPOTS 512

/* ONNX Runtime session handle (opaque) */
typedef struct LSTMInference LSTMInference;

/* Predictive AO state (Zernike-coefficient history for future-mode prediction) */
typedef struct {
    float history[LSTM_LOOKBACK][LSTM_NMODES];  /* sliding window */
    int   frame_count;                           /* frames seen so far */
    float dm_correction[LSTM_NMODES];            /* accumulated DM state in Zernike space */
} PredictiveAOState;

/* Slope-domain ring-buffer state for slope completion (#90) */
typedef struct {
    float  buffer[LSTM_LOOKBACK * 4 * PREDICTIVE_AO_MAX_NSPOTS]; /* [L, 4*nspots] flattened */
    size_t frame_count;                       /* frames pushed so far */
    int    nspots;                            /* active sub-apertures */
    int    lookback;                          /* history length L */
} PredictiveAOSlopeState;

/* ---------- ONNX Runtime inference ---------- */

/* Load ONNX model. Returns NULL on failure. */
LSTMInference* predictive_ao_load_model(const char* onnx_path);

/*
 * Run ONNX inference with arbitrary input shape.
 *   input       - flat float buffer
 *   input_rank  - rank of the input tensor (usually 3)
 *   shape       - int64_t[input_rank] tensor dimensions
 *   output      - pre-allocated float buffer
 *   output_len  - number of floats expected
 */
int predictive_ao_infer(LSTMInference* ctx, const float* input, int input_rank,
                        const int64_t* shape, float* output, int output_len);

/* Unload model */
void predictive_ao_unload(LSTMInference* ctx);

/* Update the expected input geometry for a loaded slope-completion model. */
void predictive_ao_model_set_geometry(LSTMInference* ctx, int nspots, int lookback);

/* ---------- State management (Zernike history) ---------- */

/* Initialize predictive AO state (zero history, zero DM correction) */
void predictive_ao_state_init(PredictiveAOState* state);

/* Shift buffer and insert latest coefficients */
void predictive_ao_push(PredictiveAOState* state, const float coeffs[LSTM_NMODES]);

/* Get pointer to current history buffer (LSTM_LOOKBACK × LSTM_NMODES) */
const float* predictive_ao_history(const PredictiveAOState* state);

/* ---------- Slope completion state ---------- */

/* Initialize a slope-domain ring buffer. */
void predictive_ao_slope_init(PredictiveAOSlopeState* state, int nspots, int lookback);

/* Free a slope-domain ring buffer (resets to zero). */
void predictive_ao_slope_free(PredictiveAOSlopeState* state);

/* Push one partial observation [dx, dy, mask] into the slope ring buffer. */
void predictive_ao_slope_push(PredictiveAOSlopeState* state,
                              const float* dx, const float* dy,
                              const int* mask, int nspots);

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

/*
 * Complete missing or corrupted sub-aperture slopes using the ONNX Runtime
 * slope-completion model.  The state buffer holds the last L frames; the
 * current frame is the last row.  The model returns a full slope vector.
 *
 * Returns 0 on success, 1 if no model is loaded / not enough history
 * (graceful fallback to spatial interpolation), negative on error.
 */
int predictive_ao_complete_slopes(LSTMInference* model,
                                    PredictiveAOSlopeState* state,
                                    const float* observed_dx,
                                    const float* observed_dy,
                                    const int* mask, int nspots,
                                    float* out_dx, float* out_dy);

#ifdef __cplusplus
}
#endif

#endif /* RIPPR_PREDICTIVE_AO_H */
