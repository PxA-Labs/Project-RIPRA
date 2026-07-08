/*
 * simd_avx2.c — AVX2-accelerated TCoG window kernel
 *
 * Compile with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC).
 * Runtime dispatch in simd.c ensures this code is only called on
 * CPUs that support AVX2.
 */
#include "rippra/simd.h"

#include <immintrin.h>
#include <math.h>

/* ---- Horizontal reductions for __m256d ---- */

static inline double hmin256_pd(__m256d x)
{
    __m128d lo = _mm256_castpd256_pd128(x);
    __m128d hi = _mm256_extractf128_pd(x, 1);
    lo = _mm_min_pd(lo, hi);
    __m128d hi_lo = _mm_unpackhi_pd(lo, lo);
    lo = _mm_min_sd(lo, hi_lo);
    return _mm_cvtsd_f64(lo);
}

static inline double hmax256_pd(__m256d x)
{
    __m128d lo = _mm256_castpd256_pd128(x);
    __m128d hi = _mm256_extractf128_pd(x, 1);
    lo = _mm_max_pd(lo, hi);
    __m128d hi_lo = _mm_unpackhi_pd(lo, lo);
    lo = _mm_max_sd(lo, hi_lo);
    return _mm_cvtsd_f64(lo);
}

static inline double hsum256_pd(__m256d x)
{
    __m128d lo = _mm256_castpd256_pd128(x);
    __m128d hi = _mm256_extractf128_pd(x, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d hi_lo = _mm_unpackhi_pd(lo, lo);
    lo = _mm_add_sd(lo, hi_lo);
    return _mm_cvtsd_f64(lo);
}

/* ---- AVX2 TCoG window kernel ---- */

void tcog_window_fast_avx2(const double *frame, int w,
                            int col_min, int col_max,
                            int row_min, int row_max,
                            double centroid_percent,
                            double *out_cx, double *out_cy,
                            double *out_mass)
{
    int i, j;
    double sx = 0.0, sy = 0.0, m = 0.0;
    double mn = 1e18, mx = -1e18;

    /* Pass 1: min/max over window */
    for (j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * w;
        __m256d v_mn = _mm256_set1_pd(1e18);
        __m256d v_mx = _mm256_set1_pd(-1e18);
        i = col_min;
        for (; i + 3 <= col_max; i += 4) {
            __m256d v = _mm256_loadu_pd(&row[i]);
            v_mn = _mm256_min_pd(v_mn, v);
            v_mx = _mm256_max_pd(v_mx, v);
        }
        double row_mn = hmin256_pd(v_mn);
        double row_mx = hmax256_pd(v_mx);
        for (; i <= col_max; ++i) {
            double v = row[i];
            if (v < row_mn) row_mn = v;
            if (v > row_mx) row_mx = v;
        }
        if (row_mn < mn) mn = row_mn;
        if (row_mx > mx) mx = row_mx;
    }

    double level = mn + centroid_percent * (mx - mn);

    /* Pass 2: thresholded CoG with AVX2 */
    const __m256d v_level = _mm256_set1_pd(level);
    const __m256d v_zero = _mm256_setzero_pd();
    for (j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * w;
        const double j_dbl = (double)j;
        __m256d v_sx = v_zero;
        __m256d v_m  = v_zero;
        i = col_min;
        for (; i + 3 <= col_max; i += 4) {
            __m256d v = _mm256_loadu_pd(&row[i]);
            __m256d mask = _mm256_cmp_pd(v, v_level, _CMP_GE_OQ);
            v = _mm256_blendv_pd(v_zero, v, mask);
            __m256d idx = _mm256_set_pd((double)(i+3), (double)(i+2),
                                        (double)(i+1), (double)(i+0));
            v_sx = _mm256_fmadd_pd(idx, v, v_sx);
            v_m  = _mm256_add_pd(v_m, v);
        }
        double row_sx = hsum256_pd(v_sx);
        double row_m  = hsum256_pd(v_m);
        for (; i <= col_max; ++i) {
            double v = row[i];
            if (v < level) continue;
            row_sx += (double)i * v;
            row_m  += v;
        }
        sx += row_sx;
        sy += j_dbl * row_m;
        m  += row_m;
    }

    if (m > 1e-9) {
        *out_cx = sx / m;
        *out_cy = sy / m;
    } else {
        *out_cx = NAN;
        *out_cy = NAN;
    }
    *out_mass = m;
}
