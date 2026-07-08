#ifndef RIPPRA_SIMD_H
#define RIPPRA_SIMD_H

typedef enum {
    RIPPRA_SIMD_NONE   = 0,
    RIPPRA_SIMD_SSE    = 1,
    RIPPRA_SIMD_AVX    = 2,
    RIPPRA_SIMD_AVX2   = 3,
    RIPPRA_SIMD_AVX512 = 4
} rippra_simd_level;

rippra_simd_level rippra_simd_detect(void);
const char *rippra_simd_level_name(rippra_simd_level level);

void rippra_simd_tcog_window_fast(const double *frame, int w,
                                   int col_min, int col_max,
                                   int row_min, int row_max,
                                   double centroid_percent,
                                   double *out_cx, double *out_cy,
                                   double *out_mass);

void rippra_simd_force_level(int level);

#endif
