/*
 * rippra/cuda/matrix_kernels.cu - CUDA kernels for matrix operations
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/centroid.h"
#include "rippra/recon.h"
#include "rippra_cuda.h"

/* ----------------------------------------------------------------------- */
/* Kernel: matrix-vector multiply y = A * x                                */
/* A: (m x n) row-major, x: (n), y: (m)                                   */
/* One block per output row, one thread per column element.                */
/* ----------------------------------------------------------------------- */
__global__ void matvec_kernel(const double *A, const double *x, double *y,
                              size_t m, size_t n)
{
    size_t i = blockIdx.x;
    if (i >= m) return;

    double sum = 0.0;
    const double *arow = A + i * n;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        sum += arow[j] * x[j];
    }

    __shared__ double smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) y[i] = smem[0];
}

/* ----------------------------------------------------------------------- */
/* Kernel: matrix-matrix multiply C = A * B (both row-major)               */
/* A: (m x k), B: (k x n), C: (m x n)                                     */
/* ----------------------------------------------------------------------- */
__global__ void matmul_kernel(const double *A, const double *B, double *C,
                              size_t m, size_t k, size_t n)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;

    double sum = 0.0;
    for (size_t p = 0; p < k; ++p)
        sum += A[i * k + p] * B[p * n + j];
    C[i * n + j] = sum;
}

/* ----------------------------------------------------------------------- */
/* Kernel: assemble slope vector s = [dx * (p/f); dy * (p/f)] and compute  */
/* zonal reconstruction W = Gpinv * s                                      */
/* ----------------------------------------------------------------------- */
__global__ void zonal_recon_kernel(const double *Gpinv,
                                   const double *dx, const double *dy,
                                   int nspots, double p, double f,
                                   int nnodes, double *W,
                                   double *s_shared)
{
    /* Thread 0 assembles s */
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < nspots; ++i) {
            s_shared[i] = dx[i] * (p / f);
            s_shared[i + nspots] = dy[i] * (p / f);
        }
    }
    __syncthreads();

    /* Each block handles one output element of W */
    size_t i = blockIdx.x;
    if (i >= (size_t)nnodes) return;

    double sum = 0.0;
    const double *grow = Gpinv + i * (size_t)(2 * nspots);
    for (size_t j = threadIdx.x; j < (size_t)(2 * nspots); j += blockDim.x) {
        sum += grow[j] * s_shared[j];
    }

    __shared__ double smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) W[i] = smem[0];
}

/* ----------------------------------------------------------------------- */
/* Kernel: modal reconstruction coeffs = Zprime_pinv * s                   */
/* ----------------------------------------------------------------------- */
__global__ void modal_recon_kernel(const double *Zprime_pinv,
                                   const double *dx, const double *dy,
                                   int nspots, double p, double f,
                                   double lambda, int nmodes,
                                   double *coeffs,
                                   double *s_shared)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < nspots; ++i) {
            s_shared[i] = dx[i] * p;
            s_shared[i + nspots] = dy[i] * p;
        }
    }
    __syncthreads();

    size_t i = blockIdx.x;
    if (i >= (size_t)nmodes) return;

    double sum = 0.0;
    const double *prow = Zprime_pinv + i * (size_t)(2 * nspots);
    for (size_t j = threadIdx.x; j < (size_t)(2 * nspots); j += blockDim.x) {
        sum += prow[j] * s_shared[j];
    }

    __shared__ double smem[256];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        coeffs[i] = (1.0 / f) * (smem[0]) * (2.0 * M_PI / lambda);
}

/* ----------------------------------------------------------------------- */
/* Host API wrappers                                                       */
/* ----------------------------------------------------------------------- */

int rippra_cuda_matvec(const double *d_A, const double *d_x, double *d_dst,
                       size_t m, size_t n)
{
    int threads = 256;
    dim3 grid((unsigned int)m, 1, 1);
    matvec_kernel<<<grid, threads>>>(d_A, d_x, d_dst, m, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

int rippra_cuda_matmul(const double *d_A, const double *d_B, double *d_dst,
                       size_t m, size_t k, size_t n)
{
    int tile = 16;
    dim3 block(tile, tile, 1);
    dim3 grid((unsigned int)((n + tile - 1) / tile),
              (unsigned int)((m + tile - 1) / tile), 1);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_dst, m, k, n);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

int rippra_cuda_zonal_reconstruct(const double *d_dx, const double *d_dy,
                                   const rippra_zonal_mesh *mesh,
                                   const rippa_config *cfg,
                                   double *d_W)
{
    int nspots = (int)cfg->totlenses;
    int nnodes = mesh->nnodes;

    double *d_s = NULL;
    cudaError_t err;
    err = cudaMalloc(&d_s, (size_t)(2 * nspots) * sizeof(double));
    if (err) return -1;

    zonal_recon_kernel<<<nnodes, 256>>>(
        mesh->Gpinv, d_dx, d_dy,
        nspots, cfg->camera_pixsize, cfg->flength,
        nnodes, d_W, d_s
    );

    cudaFree(d_s);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

int rippra_cuda_modal_reconstruct(const double *d_dx, const double *d_dy,
                                   const rippra_modal_model *model,
                                   const rippa_config *cfg,
                                   double *d_coeffs)
{
    int nspots = (int)cfg->totlenses;
    int nmodes = model->nmodes;

    double *d_s = NULL;
    cudaError_t err;
    err = cudaMalloc(&d_s, (size_t)(2 * nspots) * sizeof(double));
    if (err) return -1;

    modal_recon_kernel<<<nmodes, 256>>>(
        model->Zprime_pinv, d_dx, d_dy,
        nspots, cfg->camera_pixsize, cfg->flength,
        cfg->wavelength, nmodes,
        d_coeffs, d_s
    );

    cudaFree(d_s);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

/* ----------------------------------------------------------------------- */
/* Utility                                                                  */
/* ----------------------------------------------------------------------- */

int rippra_cuda_malloc_host_to_device(double **d_ptr, const double *h_ptr, size_t n)
{
    cudaError_t err = cudaMalloc(d_ptr, n * sizeof(double));
    if (err) return -1;
    err = cudaMemcpy(*d_ptr, h_ptr, n * sizeof(double), cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int rippra_cuda_malloc_device_to_host(double **h_ptr, const double *d_ptr, size_t n)
{
    *h_ptr = (double *)malloc(n * sizeof(double));
    if (!*h_ptr) return -1;
    cudaError_t err = cudaMemcpy(*h_ptr, d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

void rippra_cuda_free(void *d_ptr)
{
    if (d_ptr) cudaFree(d_ptr);
}

int rippra_cuda_check_error(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line,
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
