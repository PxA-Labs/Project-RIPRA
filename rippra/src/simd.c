#include "rippra/simd.h"

#include <math.h>
#include <stdlib.h>

/* ---- CPU feature detection (x86) ---- */

#if defined(_MSC_VER)
#include <intrin.h>
static int cpuid_has_avx2(void)
{
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7) return 0;
    __cpuidex(cpuInfo, 7, 0);
    int ebx = cpuInfo[1];
    return (ebx >> 5) & 1; /* EBX bit 5 = AVX2 */
}
static int cpuid_has_avx(void)
{
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 1);
    int ecx = cpuInfo[2];
    return (ecx >> 28) & 1; /* ECX bit 28 = AVX */
}
static int cpuid_osxsave(void)
{
    int cpuInfo[4] = {0};
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] >> 27) & 1;
}
static int xgetbv_ymm(void)
{
    unsigned long long xcr0 = _xgetbv(0);
    return (xcr0 & 6) == 6; /* XMM + YMM state */
}
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static int cpuid_has_avx2(void)
{
    unsigned int eax, ebx = 0, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return (ebx >> 5) & 1;
    return 0;
}
static int cpuid_has_avx(void)
{
    unsigned int eax, ebx, ecx = 0, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return (ecx >> 28) & 1;
    return 0;
}
static int cpuid_osxsave(void)
{
    unsigned int eax, ebx, ecx = 0, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return (ecx >> 27) & 1;
    return 0;
}
static int xgetbv_ymm(void)
{
    unsigned int eax = 0, edx = 0;
    __asm__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (eax & 6) == 6;
}
#else
static int cpuid_has_avx2(void) { return 0; }
static int cpuid_has_avx(void)  { return 0; }
static int cpuid_osxsave(void)    { return 0; }
static int xgetbv_ymm(void)       { return 0; }
#endif

rippra_simd_level rippra_simd_detect(void)
{
#ifdef _MSC_VER
    /* MSVC codegen for AVX2 intrinsics (cmp_pd, fmadd_pd) produces illegal
       instructions or segfaults in CI (windows-2025-vs2026).  Disable AVX2
       path on MSVC until the kernel can be debugged; scalar fallback is used. */
    return RIPPRA_SIMD_SSE;
#else
    if (!cpuid_has_avx())     return RIPPRA_SIMD_SSE;
    if (!cpuid_osxsave())     return RIPPRA_SIMD_SSE;
    if (!xgetbv_ymm())        return RIPPRA_SIMD_SSE;
    if (cpuid_has_avx2())     return RIPPRA_SIMD_AVX2;
    return RIPPRA_SIMD_AVX;
#endif
}

const char *rippra_simd_level_name(rippra_simd_level level)
{
    switch (level) {
        case RIPPRA_SIMD_NONE:   return "none (scalar)";
        case RIPPRA_SIMD_SSE:    return "SSE";
        case RIPPRA_SIMD_AVX:    return "AVX";
        case RIPPRA_SIMD_AVX2:   return "AVX2";
        case RIPPRA_SIMD_AVX512: return "AVX-512";
        default:                 return "unknown";
    }
}

/* ---- Override for testing ---- */
static int force_level = -1;

void rippra_simd_force_level(int level)
{
    force_level = level;
}

/* ---- Scalar fallback (identical algorithm to centroid.c) ---- */

static void tcog_window_fast_scalar(const double *frame, int w,
                                     int col_min, int col_max,
                                     int row_min, int row_max,
                                     double centroid_percent,
                                     double *out_cx, double *out_cy,
                                     double *out_mass)
{
    double sx = 0.0, sy = 0.0, m = 0.0;
    double mn = 1e18, mx = -1e18;
    for (int j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * w;
        for (int i = col_min; i <= col_max; ++i) {
            double v = row[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    double level = mn + centroid_percent * (mx - mn);
    for (int j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * w;
        for (int i = col_min; i <= col_max; ++i) {
            double v = row[i];
            if (v < level) continue;
            sx += (double)i * v;
            sy += (double)j * v;
            m += v;
        }
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

/* ---- AVX2 implementation (defined in simd_avx2.c; not available on MSVC) ---- */
#ifndef _MSC_VER
extern void tcog_window_fast_avx2(const double *frame, int w,
                                   int col_min, int col_max,
                                   int row_min, int row_max,
                                   double centroid_percent,
                                   double *out_cx, double *out_cy,
                                   double *out_mass);
#endif

/* ---- Dispatch (thread-safe, respects force_level) ---- */
/* cached_level has a benign race: all threads compute the same value */
static volatile int cached_level = -1;

void rippra_simd_tcog_window_fast(const double *frame, int w,
                                   int col_min, int col_max,
                                   int row_min, int row_max,
                                   double centroid_percent,
                                   double *out_cx, double *out_cy,
                                   double *out_mass)
{
    rippra_simd_level level;
    if (force_level >= 0) {
        level = (rippra_simd_level)force_level;
    } else {
        if (cached_level < 0)
            cached_level = (int)rippra_simd_detect();
        level = (rippra_simd_level)cached_level;
    }
#ifndef _MSC_VER
    if (level >= RIPPRA_SIMD_AVX2) {
        tcog_window_fast_avx2(frame, w, col_min, col_max, row_min, row_max,
                              centroid_percent, out_cx, out_cy, out_mass);
        return;
    }
#endif
    tcog_window_fast_scalar(frame, w, col_min, col_max, row_min, row_max,
                            centroid_percent, out_cx, out_cy, out_mass);
}
