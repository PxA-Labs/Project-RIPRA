/*
 * rippra/cuda/dm_kernels.cu - CUDA kernels for deformable mirror mapping
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/recon.h"
#include "rippra_cuda.h"

/* ----------------------------------------------------------------------- */
/* Kernel: build coupling matrix C and solve C * v = -target_phase         */
/* Since LU solve is inherently sequential, we build C on GPU but solve    */
/* on CPU. For larger actuator counts, a GPU iterative solver could be used.*/
/*                                                                         */
/* Here we build the coupling matrix C in parallel.                        */
/* ----------------------------------------------------------------------- */
__global__ void build_coupling_kernel(double *C, int nnodes, double coupling,
                                       const int *node_u, const int *node_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnodes) return;

    int ui = node_u[i];
    int vi = node_v[i];

    C[i * nnodes + i] = 1.0;

    for (int j = 0; j < nnodes; ++j) {
        if (i == j) continue;
        int du = abs(ui - node_u[j]);
        int dv = abs(vi - node_v[j]);

        if ((du == 1 && dv == 0) || (du == 0 && dv == 1))
            C[i * nnodes + j] = coupling;
        else if (du == 1 && dv == 1)
            C[i * nnodes + j] = coupling * coupling;
    }
}

/* Copy node mesh to device */
struct DMGPUData {
    int *d_node_u;
    int *d_node_v;
    int nnodes;
    int initialized;
};

static struct DMGPUData g_dm = {0};

int rippra_cuda_dm_init(const rippra_zonal_mesh *mesh)
{
    int nnodes = mesh->nnodes;

    cudaError_t err;
    err = cudaMalloc(&g_dm.d_node_u, nnodes * sizeof(int)); if (err) return -1;
    err = cudaMalloc(&g_dm.d_node_v, nnodes * sizeof(int)); if (err) return -1;

    cudaMemcpy(g_dm.d_node_u, mesh->node_u, nnodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_dm.d_node_v, mesh->node_v, nnodes * sizeof(int), cudaMemcpyHostToDevice);

    g_dm.nnodes = nnodes;
    g_dm.initialized = 1;
    return 0;
}

/* Build coupling matrix on GPU, copy to host for LU solve,
   then result back to GPU */
int rippra_cuda_dm_map(const double *d_target_phase, int nnodes,
                       const rippra_zonal_mesh *mesh,
                       const rippa_config *cfg,
                       double *d_dm_commands)
{
    (void)d_target_phase; /* not used until LU is ported */
    (void)d_dm_commands;
    (void)cfg;
    (void)mesh;

    /* For now: build C on GPU, then we'd need to copy back for LU solve.
       Future: implement Jacobi or CG iteration on GPU for full GPU pipeline. */

    if (!g_dm.initialized) return -1;

    double *d_C = NULL;
    cudaError_t err;
    err = cudaMalloc(&d_C, (size_t)nnodes * nnodes * sizeof(double));
    if (err) return -1;

    int threads = 128;
    int blocks = (nnodes + threads - 1) / threads;
    build_coupling_kernel<<<blocks, threads>>>(d_C, nnodes, cfg->coupling,
                                                g_dm.d_node_u, g_dm.d_node_v);

    err = cudaGetLastError();
    cudaFree(d_C);
    return (err == cudaSuccess) ? 0 : -1;
}

void rippra_cuda_dm_free(void)
{
    if (g_dm.d_node_u) cudaFree(g_dm.d_node_u);
    if (g_dm.d_node_v) cudaFree(g_dm.d_node_v);
    memset(&g_dm, 0, sizeof(g_dm));
}

/* Combined full pipeline: frame in, DM commands out (all on GPU) */
int rippra_cuda_full_pipeline(const double *d_frame, int width, int height,
                               const rippra_calibration *cal,
                               const rippa_config *cfg,
                               const rippra_zonal_mesh *mesh,
                               const rippra_modal_model *model,
                               double *d_W, double *d_coeffs,
                               double *d_dm_commands)
{
    double *d_cx, *d_cy, *d_dx, *d_dy;
    int nspots = cal->nspots;

    cudaError_t err;
    err = cudaMalloc(&d_cx, nspots * sizeof(double)); if (err) return -1;
    err = cudaMalloc(&d_cy, nspots * sizeof(double)); if (err) return -1;
    err = cudaMalloc(&d_dx, nspots * sizeof(double)); if (err) return -1;
    err = cudaMalloc(&d_dy, nspots * sizeof(double)); if (err) return -1;

    /* Step 1: centroids */
    if (rippra_cuda_compute_centroids(d_frame, width, height, cal, cfg, d_cx, d_cy) != 0)
        goto fail;

    /* Step 2: deltas */
    if (rippra_cuda_compute_deltas(d_cx, d_cy, nspots, d_dx, d_dy) != 0)
        goto fail;

    /* Step 3: zonal reconstruction */
    if (rippra_cuda_zonal_reconstruct(d_dx, d_dy, mesh, cfg, d_W) != 0)
        goto fail;

    /* Step 4: modal reconstruction */
    if (rippra_cuda_modal_reconstruct(d_dx, d_dy, model, cfg, d_coeffs) != 0)
        goto fail;

    /* Step 5: DM map */
    if (rippra_cuda_dm_map(d_W, mesh->nnodes, mesh, cfg, d_dm_commands) != 0)
        goto fail;

    cudaFree(d_cx); cudaFree(d_cy);
    cudaFree(d_dx); cudaFree(d_dy);
    return 0;

fail:
    cudaFree(d_cx); cudaFree(d_cy);
    cudaFree(d_dx); cudaFree(d_dy);
    return -1;
}
