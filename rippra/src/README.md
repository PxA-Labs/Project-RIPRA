# C Library — `rippra/src/`

Real-time SH-WFS processing pipeline implemented in C99 (MinGW GCC).

## Files

| File | Purpose | Key Functions |
|---|---|---|
| `centroid.c` | Spot centroid detection via TCoG | `rippa_calibrate_grid()`, `rippa_compute_centroids()`, `rippa_compute_deltas()`, `tcog_window_fast()`, `rippa_compute_centroids_refined()` |
| `io.c` | RAW frame I/O and config parsing | `rippa_load_raw()`, `rippa_save_raw()`, `rippa_config_load()` |
| `la.c` | Linear algebra utilities | `matrix_mul()`, `vector_mul()`, `svd()`, `pseudoinverse()` |
| `recon.c` | Zonal/modal reconstruction, turbulence, DM mapping | `rippra_zonal_setup()`, `rippra_zonal_reconstruct()`, `rippra_modal_setup()`, `rippra_modal_reconstruct()`, `rippra_compute_r0()`, `rippra_compute_tau0()`, `rippra_dm_map()`, `rippra_wavefront_rms_lambda()` |
| `stream.c` | Real-time streaming pipeline | `rippra_stream_create()`, `rippra_stream_process()`, `rippra_stream_get_result()` |
| `rippra_api.c` | Public C API wrapper | `rippa_config_load()`, `rippa_calibrate()`, `rippa_centroid()`, `rippa_reconstruct_zonal()`, `rippa_reconstruct_modal()`, `rippa_compute_r0()`, `rippa_compute_tau0()`, `rippa_dm_map()`, `rippa_process_frame()`, `rippa_wavefront_rms_lambda()` |

## Pipeline Stages

1. **Config** → `rippa_config_load()` — parse `system.conf`
2. **Load** → `rippa_load_raw()` — read binary frame
3. **Calibrate** → `rippa_calibrate_grid()` — detect reference spot grid from flat
4. **Centroid** → `rippa_compute_centroids()` — TCoG on each sub-aperture
5. **Deltas** → `rippa_compute_deltas()` — subtract reference positions
6. **Zonal recon** → `rippra_zonal_reconstruct()` — Fried geometry phase
7. **Modal recon** → `rippra_modal_reconstruct()` — Zernike coefficients
8. **Turbulence** → `rippra_compute_r0()` / `rippra_compute_tau0()`
9. **DM map** → `rippra_dm_map()` — actuator commands

## Latency

Hot-path (per-frame compute, excludes one-time I/O):

| Stage | Time |
|---|---|
| Fast centroid (merged TCoG) | $482\,\mu\text{s}$ |
| Reconstruction (zonal + modal) | $194\,\mu\text{s}$ |
| DM mapping | $85\,\mu\text{s}$ |
| **Hot-path total** | **$761\,\mu\text{s}$ (~0.76 ms)** |

End-to-end (first frame, includes I/O): **$2.26$ ms** (steady-state: $761\,\mu\text{s}$).

## Dependencies

- C99 compiler (MinGW GCC, GCC, Clang)
- OpenMP (optional, for parallel loops)
- No external libraries (hand-rolled SVD, matrix ops)
