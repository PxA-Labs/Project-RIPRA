# API Reference

## C API — `rippra/include/rippra/rippra_api.h`

The shared library exposes the complete wavefront reconstruction pipeline. All
functions return `int` (0 = success, nonzero = error) except where noted. The
library is built with `BUILD_RIPRA_DLL` defined for shared-library builds.

### Version

```c
const char* rippra_version(void);
```

Returns the library version string.

### Configuration

#### `rippra_api_config`

```c
typedef struct rippra_api_config {
    double camera_pixsize;  // Camera pixel size (m)
    int    frame_width;     // Frame width (px)
    int    frame_height;    // Frame height (px)
    int    totlenses;       // Max lenslet count
    double flength;         // Lenslet focal length (m)
    double pitch;           // Lenslet pitch (m)
    double sa_radius;       // Sub-aperture radius (px)
    double pupil_radius;    // Pupil radius (m)
    double wavelength;      // Observation wavelength (m)
    double thresh_binary;   // Binary threshold
    double centroid_percent;// Centroid window fraction
    int    coarse_grid_radius; // Coarse grid search radius (px)
    int    zernike_nmax;    // Max Zernike radial order
    int    dm_nact_x;       // DM actuators in X
    int    dm_nact_y;       // DM actuators in Y
    double coupling;        // DM influence coupling
} rippra_api_config;
```

#### `rippra_default_config`

```c
rippra_api_config rippra_default_config(void);
```

Returns a config struct initialized with defaults matching
`rippra/config/system.conf`.

#### `rippra_config_load`

```c
int rippra_config_load(rippra_api_config *cfg, const char *path);
```

Parse system configuration from a `system.conf` file.

| Parameter | Description |
|-----------|-------------|
| `cfg`     | Output config struct |
| `path`    | Path to config file (e.g. `"config/system.conf"`) |

### Calibration

#### `rippra_calibrate`

```c
void* rippra_calibrate(const double *flat_frame,
                        int width, int height,
                        const rippra_api_config *cfg);
```

Detect reference spot grid from a flat-field (unabberated) frame.

| Parameter    | Description |
|--------------|-------------|
| `flat_frame` | Flat-field frame, width × height doubles in row-major order |
| `width`      | Frame width (px) |
| `height`     | Frame height (px) |
| `cfg`        | System config |

**Returns:** Opaque calibration handle (`void*`), or `NULL` on failure.

#### `rippra_calibration_free`

```c
void rippra_calibration_free(void *cal);
```

Release calibration resources.

#### `rippra_calibration_nspots`

```c
int rippra_calibration_nspots(void *cal);
```

Returns the number of detected sub-apertures (spots).

#### `rippra_calibration_nnodes`

```c
int rippra_calibration_nnodes(void *cal);
```

Returns the number of phase grid nodes computed during zonal setup. Call this
to determine the buffer size needed for `rippra_reconstruct_zonal`,
`rippra_dm_map`, `rippra_dm_apply`, `rippra_closed_loop_step`, and
`rippra_closed_loop_run`.

#### `rippra_calibration_ref_centroids`

```c
void rippra_calibration_ref_centroids(void *cal,
                                      double *out_cx,
                                      double *out_cy);
```

Write reference centroid coordinates into `out_cx` and `out_cy` (nspots doubles
each).

### Centroiding

#### `rippra_centroid`

```c
int rippra_centroid(void *cal,
                     const double *frame,
                     int width, int height,
                     double *out_dx, double *out_dy,
                     int *out_mask);
```

Compute spot centroid displacements from an aberrated frame.

| Parameter  | Description |
|------------|-------------|
| `cal`      | Calibration handle (from `rippra_calibrate`) |
| `frame`    | Aberrated frame, width × height doubles |
| `width`    | Frame width (px) |
| `height`   | Frame height (px) |
| `out_dx`   | Output X displacements (nspots doubles) |
| `out_dy`   | Output Y displacements (nspots doubles) |
| `out_mask` | Output validity mask (nspots ints; non-zero = valid) |

### Zonal Reconstruction

#### `rippra_reconstruct_zonal`

```c
int rippra_reconstruct_zonal(void *cal,
                              const double *dx,
                              const double *dy,
                              const int *mask,
                              const rippra_api_config *cfg,
                              double *out_phase);
```

Fried-geometry zonal wavefront reconstruction from spot displacements.

| Parameter  | Description |
|------------|-------------|
| `cal`      | Calibration handle |
| `dx, dy`   | Spot displacements (nspots each) |
| `mask`     | Spot validity mask (nspots) |
| `cfg`      | System config |
| `out_phase`| Output phase heights at grid nodes (nnodes doubles) |

### Modal Reconstruction

#### `rippra_reconstruct_modal`

```c
int rippra_reconstruct_modal(void *cal,
                              const double *dx,
                              const double *dy,
                              const int *mask,
                              const rippra_api_config *cfg,
                              double *out_coeffs);
```

Zernike modal reconstruction from spot displacements.

| Parameter   | Description |
|-------------|-------------|
| `cal`       | Calibration handle |
| `dx, dy`    | Spot displacements (nspots each) |
| `mask`      | Spot validity mask (nspots) |
| `cfg`       | System config |
| `out_coeffs`| Output Zernike coefficients (nmodes doubles, radians) |

### Full Pipeline (Centroid + Modal)

#### `rippra_process_frame`

```c
int rippra_process_frame(void *cal,
                          const double *frame,
                          int width, int height,
                          const rippra_api_config *cfg,
                          double *out_dx, double *out_dy,
                          int *out_mask,
                          double *out_coeffs);
```

End-to-end per-frame pipeline: centroid + modal reconstruction.

| Parameter    | Description |
|--------------|-------------|
| `cal`        | Calibration handle |
| `frame`      | Aberrated frame (width × height doubles) |
| `width,height`| Frame dimensions |
| `cfg`        | System config |
| `out_dx,out_dy`| Output spot displacements (nspots each) |
| `out_mask`   | Output validity mask (nspots) |
| `out_coeffs` | Output Zernike coefficients (nmodes doubles) |

### Turbulence Characterization

#### `rippra_compute_r0`

```c
double rippra_compute_r0(const double *dx_series,
                          const double *dy_series,
                          int nframes, int nspots,
                          const rippra_api_config *cfg);
```

Fried parameter r₀ from time-series displacement variance.

| Parameter   | Description |
|-------------|-------------|
| `dx_series` | X displacements, concatenated: nframes × nspots doubles |
| `dy_series` | Y displacements, concatenated: nframes × nspots doubles |
| `nframes`   | Number of frames in series |
| `nspots`    | Number of sub-apertures |
| `cfg`       | System config |

**Returns:** r₀ in meters.

#### `rippra_compute_tau0`

```c
double rippra_compute_tau0(const double *dx_series,
                            const double *dy_series,
                            int nframes, int nspots,
                            double frame_rate);
```

Coherence time τ₀ from temporal auto-covariance 1/e decay.

| Parameter   | Description |
|-------------|-------------|
| `dx_series` | X displacements (nframes × nspots doubles) |
| `dy_series` | Y displacements (nframes × nspots doubles) |
| `nframes`   | Number of frames |
| `nspots`    | Number of sub-apertures |
| `frame_rate`| Frame rate (Hz) |

**Returns:** τ₀ in seconds.

### DM Mapping

#### `rippra_dm_map`

```c
int rippra_dm_map(const double *target_phase,
                   int nnodes,
                   void *cal,
                   const rippra_api_config *cfg,
                   double *out_commands);
```

Compute DM actuator commands from a target phase.

| Parameter      | Description |
|----------------|-------------|
| `target_phase` | Target phase heights (nnodes doubles) |
| `nnodes`       | Number of phase nodes |
| `cal`          | Calibration handle |
| `cfg`          | System config |
| `out_commands` | Output DM actuator commands (nnodes doubles) |

### Closed-Loop AO Control

#### `rippra_dm_apply`

```c
int rippra_dm_apply(const double *dm_commands,
                     int nnodes,
                     void *cal,
                     const rippra_api_config *cfg,
                     const double *input_phase,
                     double *out_residual);
```

Apply DM commands to an input phase: residual = input + C · commands.

#### `rippra_closed_loop_step`

```c
int rippra_closed_loop_step(void *cal,
                             const double *measured_phase,
                             int nnodes,
                             const rippra_api_config *cfg,
                             double *dm_commands,
                             double gain);
```

Single closed-loop iteration: update DM commands with integrator gain.

#### `rippra_closed_loop_run`

```c
int rippra_closed_loop_run(void *cal,
                            const double *initial_phase,
                            int nnodes,
                            const rippra_api_config *cfg,
                            double *dm_commands,
                            double gain,
                            int max_iter,
                            double target_rms,
                            int *out_iters,
                            double *out_residual_rms);
```

Run closed-loop control to convergence.

| Parameter        | Description |
|------------------|-------------|
| `cal`            | Calibration handle |
| `initial_phase`  | Initial measured phase (nnodes doubles) |
| `nnodes`         | Number of phase nodes |
| `cfg`            | System config |
| `dm_commands`    | Input/output DM commands (nnodes doubles) |
| `gain`           | Integrator gain (0 < gain ≤ 1) |
| `max_iter`       | Maximum iterations |
| `target_rms`     | Target residual RMS (rad) to stop early |
| `out_iters`      | Output iteration count |
| `out_residual_rms`| Output final residual RMS |

#### `rippra_strehl_ratio`

```c
double rippra_strehl_ratio(const double *phase, int nnodes);
```

Marechal-approximation Strehl ratio: S = exp(-σ²) where σ² is the phase
variance.

---

## Worked Example

Below is a complete worked example using only the public C API. It loads a
flat-frame and an aberrated frame from disk (generated by
`python ml/synthetic_shwfs.py`), calibrates the spot grid, computes spot
displacements, reconstructs the wavefront (zonal + modal), characterizes
turbulence, maps to DM commands, and runs one closed-loop step.

```c
#include "rippra/rippra_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double* load_raw(const char *path, int n) {
    double *buf = malloc(n * sizeof(double));
    if (!buf) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) { free(buf); return NULL; }
    size_t got = fread(buf, sizeof(double), n, fp);
    fclose(fp);
    return (got == (size_t)n) ? buf : (free(buf), NULL);
}

int main(void) {
    rippra_api_config cfg = rippra_default_config();
    int npix = cfg.frame_width * cfg.frame_height;

    /* 1. Load flat frame and calibrate */
    double *flat = load_raw("data_raw/sh_flat.raw", npix);
    if (!flat) { fprintf(stderr, "Load flat failed\n"); return 1; }

    void *cal = rippra_calibrate(flat, cfg.frame_width,
                                  cfg.frame_height, &cfg);
    free(flat);
    if (!cal) { fprintf(stderr, "Calibration failed\n"); return 1; }

    int nspots = rippra_calibration_nspots(cal);
    printf("Calibrated %d spots\n", nspots);

    /* 2. Load aberrated frame and compute centroids */
    double *frame = load_raw("data_raw/img.raw", npix);
    if (!frame) { rippra_calibration_free(cal); return 1; }

    double *dx = malloc(nspots * sizeof(double));
    double *dy = malloc(nspots * sizeof(double));
    int    *mask = malloc(nspots * sizeof(int));

    int ret = rippra_centroid(cal, frame, cfg.frame_width,
                               cfg.frame_height, dx, dy, mask);
    free(frame);
    if (ret != 0) { /* handle error */ }

    printf("Centroids computed\n");

    /* 3. Query phase grid nodes and do zonal reconstruction */
    int nnodes = rippra_calibration_nnodes(cal);
    double *phase = malloc(nnodes * sizeof(double));
    ret = rippra_reconstruct_zonal(cal, dx, dy, mask, &cfg, phase);

    /* 4. Modal reconstruction */
    int nmodes = (cfg.zernike_nmax + 1) * (cfg.zernike_nmax + 2) / 2;
    double *coeffs = malloc(nmodes * sizeof(double));
    ret = rippra_reconstruct_modal(cal, dx, dy, mask, &cfg, coeffs);

    printf("Reconstructed %d Zernike modes\n", nmodes);

    /* 5. Turbulence characterization */
    double r0  = rippra_compute_r0(dx, dy, 1, nspots, &cfg);
    double tau0 = rippra_compute_tau0(dx, dy, 1, nspots, 1000.0);
    printf("r0 = %.6f m, tau0 = %.6f s\n", r0, tau0);

    /* 6. DM mapping */
    double *dm_cmds = malloc(nnodes * sizeof(double));
    ret = rippra_dm_map(phase, nnodes, cal, &cfg, dm_cmds);

    /* 7. Closed-loop step */
    ret = rippra_closed_loop_step(cal, phase, nnodes,
                                   &cfg, dm_cmds, 0.5);

    /* 8. Strehl ratio */
    double strehl = rippra_strehl_ratio(phase, nnodes);
    printf("Strehl ratio = %.4f\n", strehl);

    /* Cleanup */
    free(dx); free(dy); free(mask);
    free(phase); free(coeffs); free(dm_cmds);
    rippra_calibration_free(cal);
    return 0;
}
```

See `rippra/tests/doc_example.c` for the full compilable version run in CI.

---

## Error Codes

| Code | Meaning |
|------|---------|
| 0    | Success |
| -1   | General error |
| -2   | File not found |
| -3   | Invalid config |
| -4   | Calibration failed |
| -5   | Reconstruction failed |
