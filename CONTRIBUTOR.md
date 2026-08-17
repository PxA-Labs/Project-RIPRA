# My Contributions — Shlok Parekh

GitHub: [@Shlok-Parekh09](https://github.com/Shlok-Parekh09)

---

## Feature: Slope-Domain Sequence Completion Model (AI Fallback)

**Umbrella Issue**: [#90 — AI and Radio fallback modes](https://github.com/PxA-Labs/Project-RIPRA/issues/90)  
**Status**: ✅ Completed

### Overview

Designing and implementing a slope-domain sequence model that reconstructs missing or corrupted wavefront slopes (Δx, Δy) per sub-aperture from the temporal consistency of past frames. This is triggered by the frame-quality gate ([#93](https://github.com/PxA-Labs/Project-RIPRA/issues/93)) when the detector reports partial signal loss (mild fog, scattering, or partial occlusion). The completed slopes are fed to the existing zonal/modal reconstructors so the AO loop continues without forcing the DM into invalid geometries.

### Problem Formulation

- **Data model**: Per-spot slope vector s_t ∈ ℝ^{2N} with observation mask m_t ∈ {0,1}^{2N}
- **Objective**: Recover the full current slope vector from L frames of history + the partial current observation (amortized posterior via LSTM)
- **Architecture**: Mask-conditioned LSTM with persistence-prior residual head — input `[batch, L+1, 4N]`, output `[batch, 2N]`

### Components Contributed

#### 1. Model Design & Specification
- Formulated the latent state-space model (Kalman analogy for Taylor frozen-flow turbulence)
- Specified the mask-conditioned LSTM architecture (`SlopeCompletionLSTM`) with persistence-prior residual
- Designed the three-component loss function:
  - **Masked reconstruction loss** — gradients only from missing entries
  - **Zernike-consistency regularizer** — projects slopes onto the modal derivative basis to enforce physical plausibility
  - **Temporal smoothness regularizer** — exploits Taylor frozen-flow assumption

#### 2. Physical Constraint Enforcement (Safety)
- Specified the Zernike projection constraint (truncates high-order modes via rcond cutoff)
- Specified the per-spot stroke clamp (±pitch_px/2) to prevent actuator cross-talk
- These constraints guarantee the model output maps into valid actuator space

#### 3. Stream Integration Design
- Designed the frame-quality gate dispatch logic:
  - **OK** (≥80% valid): standard path, push into temporal buffer
  - **DEGRADED** (20–80% valid): AI slope completion → spatial fallback → physical constraints
  - **LOST** (<20% valid): zero slopes, park DM
- Specified the graceful fallback chain: ONNX model → spatial interpolation → zero-out

#### 4. C Integration Architecture
- Designed the `predictive_ao_complete_slopes()` API with return codes:
  - `0` = success
  - `1` = model unavailable (graceful fallback)
  - negative = error
- Specified the slope ring buffer (`PredictiveAOSlopeState`) that packs `[dx, dy, mask, inv_mask]` per frame
- Designed the ONNX Runtime integration path behind `#ifdef RIPRA_ONNXRT` for zero-dependency static builds

#### 5. Dataset Design
- Specified the observation masking schedule for training data:
  - Clean (15%), mild Bernoulli dropout (15%), structured angular wedge (25%), mixed (45%), heavy (75%), boundary (95%)
- Specified the AR(1) temporal correlation model matching Kolmogorov + Taylor frozen-flow physics

#### 6. Validation & Acceptance Criteria
- Defined the acceptance criteria matrix:
  - Masked-completion RMSE ≤ 30% of spatial-interpolation baseline
  - Sequence-level split with `check_split_leakage()` assertion
  - Inference latency < 1 ms (CPU EP; CUDA EP where available)
  - Adversarial mask testing for physical constraint verification
  - Existing 38/38 C tests must continue passing

### What's Done

| Component | Status |
|---|---|
| `SlopeCompletionLSTM` model architecture | ✅ Implemented |
| Masked slope loss (rec + Zernike + temporal) | ✅ Implemented |
| `ZernikeConsistencyLoss` differentiable module | ✅ Implemented |
| Dataset generator with structured masking | ✅ Implemented |
| `train_sequence.py` — `complete_slopes` task | ✅ Implemented |
| ONNX export for slope completion model | ✅ Implemented |
| C `predictive_ao_complete_slopes()` API | ✅ Implemented |
| C slope ring buffer (`PredictiveAOSlopeState`) | ✅ Implemented |
| Stream integration (quality gate + AI fallback) | ✅ Implemented |
| Physical constraints (Zernike projection + stroke clamp) | ✅ Implemented |

### What Remains (In Progress)

| Component | Status |
|---|---|
| End-to-end Python validation test (`test_slope_completion.py`) | ✅ Implemented |
| C unit test for ring buffer + fallback (`test_slope_completion.c`) | ✅ Implemented |
| `docs/algorithms.md` — slope completion section | ✅ Implemented |
| CI integration (CMakeLists + workflow update) | ✅ Implemented |
| AGENTS.md update with completion status | ✅ Implemented |

### Key Files

| File | Role |
|---|---|
| `rippra/ml/sequence_models.py` | `SlopeCompletionLSTM` class (lines 93–132) |
| `rippra/ml/train_sequence.py` | Training loop + `masked_slope_loss` + `ZernikeConsistencyLoss` |
| `rippra/tools/generate_dataset.py` | Synthetic data with structured masking schedules |
| `rippra/ml/export_onnx.py` | ONNX export (slope completion section, lines 106–117) |
| `rippra/src/predictive_ao.c` | C ONNX RT integration + slope ring buffer + `complete_slopes` |
| `rippra/include/rippra/predictive_ao.h` | API declarations + `PredictiveAOSlopeState` struct |
| `rippra/src/stream.c` | Frame-quality gate dispatch + AI fallback wiring |
| `rippra/include/rippra/stream.h` | `rippra_frame_quality` enum + gate API |

---

## Related Contributions

- Reviewed and provided feedback on the existing LSTM predictive AO pipeline
- Contributed to the mathematical formulation in the project documentation
- Participated in architecture discussions for the umbrella AI/Radio fallback feature (#90)
