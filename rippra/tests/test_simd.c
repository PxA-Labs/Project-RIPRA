#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "rippra/simd.h"

static int test_simd_detect(void)
{
    rippra_simd_level level = rippra_simd_detect();
    printf("SIMD level: %s\n", rippra_simd_level_name(level));
    return (level >= RIPPRA_SIMD_NONE) ? 0 : -1;
}

static double frand(void)
{
    return (double)rand() / (double)RAND_MAX;
}

static int test_tcog_accuracy(void)
{
    int w = 40, h = 40;
    double frame[40 * 40];
    for (int i = 0; i < w * h; i++)
        frame[i] = frand() * 5.0;

    /* Place a bright spot at roughly (20,20) */
    for (int j = 15; j <= 25; j++)
        for (int i = 15; i <= 25; i++)
            frame[j * w + i] += 100.0 * exp(-((i-20)*(i-20) + (j-20)*(j-20)) / 8.0);

    double cx_scalar, cy_scalar, mass_scalar;
    double cx_simd, cy_simd, mass_simd;

    /* Force scalar */
    rippra_simd_force_level(RIPPRA_SIMD_NONE);
    rippra_simd_tcog_window_fast(frame, w, 15, 25, 15, 25, 0.5, &cx_scalar, &cy_scalar, &mass_scalar);

    /* Force AVX2 */
    rippra_simd_force_level(RIPPRA_SIMD_AVX2);
    rippra_simd_tcog_window_fast(frame, w, 15, 25, 15, 25, 0.5, &cx_simd, &cy_simd, &mass_simd);

    /* Reset to auto-detect */
    rippra_simd_force_level(-1);

    double err = fabs(cx_scalar - cx_simd) + fabs(cy_scalar - cy_simd) + fabs(mass_scalar - mass_simd);
    printf("Scalar: cx=%.6f cy=%.6f mass=%.6f\n", cx_scalar, cy_scalar, mass_scalar);
    printf("AVX2:   cx=%.6f cy=%.6f mass=%.6f\n", cx_simd, cy_simd, mass_simd);
    printf("Total error: %.2e  %s\n", err, err < 1e-12 ? "PASS" : "FAIL");

    return (err < 1e-12) ? 0 : -1;
}

int main(void)
{
    int rc = 0;
    printf("=== SIMD Tests ===\n\n");

    if (test_simd_detect() != 0) {
        printf("FAIL: simd_detect\n");
        rc = 1;
    }

    /* Test works on whatever level is available, including scalar fallback */
    printf("\n--- TCoG accuracy test ---\n");
    if (test_tcog_accuracy() != 0) {
        printf("FAIL: tcog accuracy\n");
        rc = 1;
    }

    printf("\n%s\n", rc == 0 ? "All tests PASSED" : "Some tests FAILED");
    return rc;
}
