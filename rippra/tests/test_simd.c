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

/* Compare scalar vs AVX2 on a fresh synthetic window.
   Returns 0 if bit-exact, -1 otherwise. */
static int compare_paths(int col_min, int col_max, int row_min, int row_max,
                         double percent)
{
    int w = 40, h = 40;
    double frame[40 * 40];
    for (int i = 0; i < w * h; i++)
        frame[i] = frand() * 5.0;
    for (int j = row_min; j <= row_max; j++)
        for (int i = col_min; i <= col_max; i++)
            frame[j * w + i] += 100.0 * exp(-((i-20)*(i-20)+(j-20)*(j-20)) / 8.0);

    double cx_s, cy_s, m_s, cx_a, cy_a, m_a;

    rippra_simd_level max_level = rippra_simd_detect();

    rippra_simd_force_level(RIPPRA_SIMD_NONE);
    rippra_simd_tcog_window_fast(frame, w, col_min, col_max, row_min, row_max,
                                 percent, &cx_s, &cy_s, &m_s);

    if (max_level < RIPPRA_SIMD_AVX2) {
        /* AVX2 not supported on this platform — skip comparison */
        return 0;
    }

    rippra_simd_force_level(RIPPRA_SIMD_AVX2);
    rippra_simd_tcog_window_fast(frame, w, col_min, col_max, row_min, row_max,
                                 percent, &cx_a, &cy_a, &m_a);

    double err = fabs(cx_s - cx_a) + fabs(cy_s - cy_a) + fabs(m_s - m_a);
    if (err >= 1e-12) {
        printf("  FAIL [%d..%d]x[%d..%d]: err=%.2e\n", col_min, col_max, row_min, row_max, err);
        return -1;
    }
    return 0;
}

static int test_tcog_accuracy(void)
{
    int rc = 0;
    printf("  Small window (7x7) ....  "); rc |= compare_paths(16, 23, 16, 23, 0.5);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Wide window (11x7) ....  "); rc |= compare_paths(14, 25, 16, 23, 0.5);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Tall window (7x11) ....  "); rc |= compare_paths(16, 23, 14, 25, 0.5);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Single row (1x7) ......  "); rc |= compare_paths(16, 23, 20, 20, 0.5);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Single column (7x1) ...  "); rc |= compare_paths(20, 20, 16, 23, 0.5);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Threshold 0% .........  "); rc |= compare_paths(16, 23, 16, 23, 0.0);
    printf("%s\n", rc ? "" : "PASS");

    printf("  Threshold 100% .......  "); rc |= compare_paths(16, 23, 16, 23, 1.0);
    printf("%s\n", rc ? "" : "PASS");

    return rc;
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
