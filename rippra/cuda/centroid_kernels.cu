/*
 * rippra/cuda/centroid_kernels.cu - CUDA kernels for centroid computation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rippra/centroid.h"
#include "rippra_cuda.h"

/* ------------------------------------------------------------------------- */
/* Device helpers                                                            */
/* ------------------------------------------------------------------------- */

__device__ static double d_tcog_window(const double *frame, int width,
                                       int col_min, int col_max,
                                       int row_min, int row_max,
                                       double level,
                                       double *out_cx, double *out_cy,
                                       double *out_mass)
{
    double sx = 0.0, sy = 0.0, m = 0.0;
    for (int j = row_min; j <= row_max; ++j) {
        const double *row = frame + (size_t)j * width;
        for (int i = col_min; i <= col_max; ++i) {
            double v = row[i];
            if (v < level) continue;
            sx += (double)i * v;
            sy += (double)j * v;
            m += v;
        }
    }
    if (m > 0.0) {
        *out_cx = sx / m;
        *out_cy = sy / m;
    } else {
        *out_cx = 0.5 * (col_min + col_max);
        *out_cy = 0.5 * (row_min + row_max);
    }
    *out_mass = m;
    return m;
}

/* ------------------------------------------------------------------------- */
/* Kernel: compute centroids for all sub-apertures                           */
/* One thread per sub-aperture.                                              */
/* ------------------------------------------------------------------------- */
__global__ void centroid_kernel(const double *frame, int width, int height,
                                const double *ref_cx, const double *ref_cy,
                                const int *col_min, const int *col_max,
                                const int *row_min, const int *row_max,
                                int nspots, double centroid_percent,
                                double *cx, double *cy)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nspots) return;

    int cmin = col_min[k];
    int cmax = col_max[k];
    int rmin = row_min[k];
    int rmax = row_max[k];

    double wmin = 1e18, wmax = -1e18;
    for (int b = rmin; b <= rmax; ++b) {
        for (int a = cmin; a <= cmax; ++a) {
            double v = frame[(size_t)b * width + a];
            if (v < wmin) wmin = v;
            if (v > wmax) wmax = v;
        }
    }
    double wlevel = wmin + centroid_percent * (wmax - wmin);
    double mass;
    d_tcog_window(frame, width, cmin, cmax, rmin, rmax, wlevel,
                  &cx[k], &cy[k], &mass);
}

/* ------------------------------------------------------------------------- */
/* Kernel: compute deltas (deviation from reference)                          */
/* ------------------------------------------------------------------------- */
__global__ void delta_kernel(const double *cx, const double *cy,
                             const double *ref_cx, const double *ref_cy,
                             int nspots, double *dx, double *dy)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nspots) return;
    dx[k] = cx[k] - ref_cx[k];
    dy[k] = cy[k] - ref_cy[k];
}

/* ------------------------------------------------------------------------- */
/* Host API                                                                  */
/* ------------------------------------------------------------------------- */

struct CentroidGPUData {
    double *d_ref_cx;
    double *d_ref_cy;
    int *d_col_min;
    int *d_col_max;
    int *d_row_min;
    int *d_row_max;
    int nspots;
    int initialized;
};

static struct CentroidGPUData g_centroid = {0};

int rippra_cuda_centroid_init(const rippra_calibration *cal, int width, int height)
{
    int nspots = cal->nspots;

    double *h_ref_cx = (double *)malloc(nspots * sizeof(double));
    double *h_ref_cy = (double *)malloc(nspots * sizeof(double));
    int *h_col_min = (int *)malloc(nspots * sizeof(int));
    int *h_col_max = (int *)malloc(nspots * sizeof(int));
    int *h_row_min = (int *)malloc(nspots * sizeof(int));
    int *h_row_max = (int *)malloc(nspots * sizeof(int));
    if (!h_ref_cx || !h_ref_cy || !h_col_min || !h_col_max || !h_row_min || !h_row_max)
        goto fail;

    for (int k = 0; k < nspots; ++k) {
        h_ref_cx[k]  = cal->subaps[k].ref_cx;
        h_ref_cy[k]  = cal->subaps[k].ref_cy;
        h_col_min[k] = cal->subaps[k].col_min;
        h_col_max[k] = cal->subaps[k].col_max;
        h_row_min[k] = cal->subaps[k].row_min;
        h_row_max[k] = cal->subaps[k].row_max;
    }

    cudaError_t err;
    err = cudaMalloc(&g_centroid.d_ref_cx,  nspots * sizeof(double)); if (err) goto fail;
    err = cudaMalloc(&g_centroid.d_ref_cy,  nspots * sizeof(double)); if (err) goto fail;
    err = cudaMalloc(&g_centroid.d_col_min, nspots * sizeof(int));    if (err) goto fail;
    err = cudaMalloc(&g_centroid.d_col_max, nspots * sizeof(int));    if (err) goto fail;
    err = cudaMalloc(&g_centroid.d_row_min, nspots * sizeof(int));    if (err) goto fail;
    err = cudaMalloc(&g_centroid.d_row_max, nspots * sizeof(int));    if (err) goto fail;

    cudaMemcpy(g_centroid.d_ref_cx,  h_ref_cx,  nspots * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_centroid.d_ref_cy,  h_ref_cy,  nspots * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_centroid.d_col_min, h_col_min, nspots * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(g_centroid.d_col_max, h_col_max, nspots * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(g_centroid.d_row_min, h_row_min, nspots * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(g_centroid.d_row_max, h_row_max, nspots * sizeof(int),    cudaMemcpyHostToDevice);

    g_centroid.nspots = nspots;
    g_centroid.initialized = 1;

    free(h_ref_cx); free(h_ref_cy);
    free(h_col_min); free(h_col_max);
    free(h_row_min); free(h_row_max);
    return 0;

fail:
    free(h_ref_cx); free(h_ref_cy);
    free(h_col_min); free(h_col_max);
    free(h_row_min); free(h_row_max);
    return -1;
}

int rippra_cuda_compute_centroids(const double *d_frame, int width, int height,
                                  const rippra_calibration *cal,
                                  const rippa_config *cfg,
                                  double *d_cx, double *d_cy)
{
    if (!g_centroid.initialized) return -1;
    int nspots = g_centroid.nspots;
    int threads = 128;
    int blocks = (nspots + threads - 1) / threads;

    centroid_kernel<<<blocks, threads>>>(
        d_frame, width, height,
        g_centroid.d_ref_cx, g_centroid.d_ref_cy,
        g_centroid.d_col_min, g_centroid.d_col_max,
        g_centroid.d_row_min, g_centroid.d_row_max,
        nspots, cfg->centroid_percent,
        d_cx, d_cy
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

int rippra_cuda_compute_deltas(double *d_cx, double *d_cy, int nspots,
                                double *d_dx, double *d_dy)
{
    if (!g_centroid.initialized) return -1;
    int threads = 128;
    int blocks = (nspots + threads - 1) / threads;

    delta_kernel<<<blocks, threads>>>(
        d_cx, d_cy,
        g_centroid.d_ref_cx, g_centroid.d_ref_cy,
        nspots, d_dx, d_dy
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

void rippra_cuda_centroid_free(void)
{
    if (g_centroid.d_ref_cx)  cudaFree(g_centroid.d_ref_cx);
    if (g_centroid.d_ref_cy)  cudaFree(g_centroid.d_ref_cy);
    if (g_centroid.d_col_min)  cudaFree(g_centroid.d_col_min);
    if (g_centroid.d_col_max)  cudaFree(g_centroid.d_col_max);
    if (g_centroid.d_row_min)  cudaFree(g_centroid.d_row_min);
    if (g_centroid.d_row_max)  cudaFree(g_centroid.d_row_max);
    memset(&g_centroid, 0, sizeof(g_centroid));
}
