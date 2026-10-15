# Frame-quality gate

The streaming pipeline evaluates each camera frame after centroiding and
before wavefront reconstruction.  The result is exposed through
`rippra_stream_result.quality`, a finite score clamped to `[0, 1]`, and
`rippra_stream_result.status`:

| Score | Status | Processing |
|---|---|---|
| `Q >= quality_soft` | `RIPPRA_STATUS_NOMINAL` | Reconstruct and map the measured centroids normally. |
| `quality_hard <= Q < quality_soft` | `RIPPRA_STATUS_DEGRADED` | Reconstruct after fallback completion of invalid spots. |
| `Q < quality_hard` | `RIPPRA_STATUS_LOST` | Skip reconstruction and DM mapping; phase, modal, and DM outputs are zero. |

The default thresholds are:

```text
quality_soft = 0.60
quality_hard = 0.30
```

They are configured in `rippra/config/system.conf` and validated when the
configuration is loaded and when a stream is initialized.  Both values must be
in `[0, 1]`, with `quality_hard < quality_soft`.

## Calibration requirements

`rippa_calibrate_grid()` captures per-spot reference values from the supplied
flat frame using the same thresholded CoG window used during streaming:

* integrated spot mass;
* background mean and standard deviation;
* signal-to-background statistic;
* intensity-weighted centroid second moment.

`rippra_quality_calibrate()` publishes those retained values as a non-owning
`rippra_quality_ref` view.  The calibration frame must contain the complete usable spot grid at the same
frame dimensions as the stream.  Spots with an invalid or effectively zero
reference mass are excluded from calibration.  The stream copies these
reference arrays along with the existing calibration state; callers still
own and may free their original calibration after `rippra_stream_init()`.

## Score components and limitations

The implemented score combines:

1. valid-spot fraction;
2. robust per-spot mass agreement with the calibrated mass;
3. current signal/background agreement with the calibrated statistic;
4. second-moment agreement with the calibrated spot shape; and
5. a weak nearest-neighbour displacement-coherence term.

The fixed weights are `0.85`, `0.05`, `0.05`, and `0.05` for validity, mass/SNR,
shape, and coherence respectively.  The high validity weight is intentional:
partial spot loss must cross the soft gate even when the median of the
remaining spots still looks nominal.  All terms are bounded before the
weighted combination is clamped.  No independent camera saturation model or
advanced shape classifier is claimed here.

The degraded path supports `rippra_stream_set_fallback()`.  The stream borrows
the callback and context, invokes it synchronously on the processing thread,
and requires finite output slopes.  A callback failure or invalid output
selects the built-in nearest-neighbour interpolation.  Invalid measured
`dx`/`dy` values remain visible in the result arrays and are only replaced in
private reconstruction scratch arrays.

Lost frames do not update the turbulence history, and result slots are
explicitly zeroed on every lost/error route so a previous DM command cannot
leak into the output.

## Build and tests

From the repository root:

```bash
cmake -S . -B build-quality -DRIPRA_BUILD_TESTS=ON
cmake --build build-quality --parallel
(cd rippra && python ml/synthetic_shwfs.py)
ctest --test-dir build-quality -R '^(test_quality|test_stream|test_simd)$' \
      --output-on-failure
```

`test_quality` checks clean, attenuated, partially occluded, severely
occluded, and blank frames, including bounded scores and zero DM output for a
lost frame.  It prints measured latency for visibility but deliberately does
not use a hard-coded microsecond assertion; full centroiding, reconstruction,
and DM mapping are workload- and host-dependent.

`benchmark_quality` uses 10 warm-up iterations and 200 timed iterations on
the generated 137-spot frame.  It reports both the quality reduction over
already extracted spot arrays and the complete centroid-plus-quality path,
including p99; results are informational rather than a CI gate.
