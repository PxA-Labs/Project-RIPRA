/*
 * rippra/cuda/dm_kernels.cu - CUDA kernels for deformable mirror mapping
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/recon.h"
#include "rippra_cuda.h"

/*
 * The DM map deliberately remains a host fallback.  The authoritative CPU
 * implementation owns coupling-matrix inversion and all safety policy
 * (disabled limits, saturation, parking, and invalid-input handling).  CUDA
 * owns only the transport around that implementation; this avoids ever
 * clamping or returning uninitialized device output.
 */
/* Retained for compatibility with callers that initialize DM GPU resources.
 * The host fallback itself does not depend on these cached device arrays. */
struct DMGPUData {
    int *d_node_u;
    int *d_node_v;
    int nnodes;
    int initialized;
};

static struct DMGPUData g_dm = {0};

int rippra_cuda_dm_init(const rippra_zonal_mesh *mesh)
{
    if (!mesh || mesh->nnodes <= 0 || !mesh->node_u || !mesh->node_v)
        return -1;

    /* Reinitialization must not leak the previous mesh allocation. */
    rippra_cuda_dm_free();

    int nnodes = mesh->nnodes;

    cudaError_t err;
    err = cudaMalloc(&g_dm.d_node_u, (size_t)nnodes * sizeof(int));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&g_dm.d_node_v, (size_t)nnodes * sizeof(int));
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(g_dm.d_node_u, mesh->node_u,
                     (size_t)nnodes * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(g_dm.d_node_v, mesh->node_v,
                     (size_t)nnodes * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    g_dm.nnodes = nnodes;
    g_dm.initialized = 1;
    return 0;

fail:
    rippra_cuda_dm_free();
    return -1;
}

int rippra_cuda_dm_map_ex(const double *d_target_phase, int nnodes,
                          const rippra_zonal_mesh *mesh,
                          const rippa_config *cfg,
                          double *d_dm_commands, int *out_status)
{
    size_t bytes;
    double *h_target = NULL;
    double *h_commands = NULL;
    int cpu_status = RIPPRA_DM_ERROR;
    int cpu_rc;
    cudaError_t err;

    if (out_status) *out_status = RIPPRA_DM_ERROR;

    /* Validate the output extent first so invalid inputs can still be parked. */
    if (!d_dm_commands || nnodes <= 0)
        return -1;

    if ((size_t)nnodes > (size_t)-1 / sizeof(double))
        return -1;
    bytes = (size_t)nnodes * sizeof(double);

    /* Clear first: all subsequent error paths leave a safe device result. */
    err = cudaMemset(d_dm_commands, 0, bytes);
    if (err != cudaSuccess)
        return -1;

    if (!d_target_phase || !mesh || !cfg || mesh->nnodes != nnodes ||
        !mesh->node_u || !mesh->node_v)
        return -1;

    h_target = (double *)malloc(bytes);
    h_commands = (double *)calloc((size_t)nnodes, sizeof(double));
    if (!h_target || !h_commands) {
        free(h_target);
        free(h_commands);
        return -1;
    }

    err = cudaMemcpy(h_target, d_target_phase, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_target);
        free(h_commands);
        return -1;
    }

    /*
     * Do not duplicate the CPU solver or safety policy here.  In particular,
     * a successful return includes RIPPRA_DM_SATURATED and RIPPRA_DM_PARKED;
     * only a negative CPU result (or invalid output) is an API error.
     */
    cpu_rc = rippra_dm_map_ex_impl(h_target, nnodes, mesh, cfg,
                                   h_commands, &cpu_status);
    if (cpu_rc != 0 || cpu_status < RIPPRA_DM_OK ||
        cpu_status > RIPPRA_DM_PARKED) {
        free(h_target);
        free(h_commands);
        return -1;
    }

    /* Never transfer NaN/Inf into the device command stream. */
    for (int i = 0; i < nnodes; ++i) {
        if (!isfinite(h_commands[i])) {
            free(h_target);
            free(h_commands);
            return -1;
        }
    }

    err = cudaMemcpy(d_dm_commands, h_commands, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        free(h_target);
        free(h_commands);
        return -1;
    }

    if (out_status) *out_status = cpu_status;
    free(h_target);
    free(h_commands);
    return 0;
}

int rippra_cuda_dm_map(const double *d_target_phase, int nnodes,
                       const rippra_zonal_mesh *mesh,
                       const rippa_config *cfg,
                       double *d_dm_commands)
{
    return rippra_cuda_dm_map_ex(d_target_phase, nnodes, mesh, cfg,
                                 d_dm_commands, NULL);
}

void rippra_cuda_dm_free(void)
{
    if (g_dm.d_node_u) cudaFree(g_dm.d_node_u);
    if (g_dm.d_node_v) cudaFree(g_dm.d_node_v);
    memset(&g_dm, 0, sizeof(g_dm));
}

/* Combined full pipeline: frame in, DM commands out (all on GPU). */
int rippra_cuda_full_pipeline_ex(const double *d_frame, int width, int height,
                                  const rippra_calibration *cal,
                                  const rippa_config *cfg,
                                  const rippra_zonal_mesh *mesh,
                                  const rippra_modal_model *model,
                                  double *d_W, double *d_coeffs,
                                  double *d_dm_commands, int *out_status)
{
    double *d_cx, *d_cy, *d_dx, *d_dy;
    int nspots = cal->nspots;

    cudaError_t err;
    if (out_status) *out_status = RIPPRA_DM_ERROR;
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
    if (rippra_cuda_dm_map_ex(d_W, mesh->nnodes, mesh, cfg, d_dm_commands,
                              out_status) != 0)
        goto fail;

    cudaFree(d_cx); cudaFree(d_cy);
    cudaFree(d_dx); cudaFree(d_dy);
    return 0;

fail:
    cudaFree(d_cx); cudaFree(d_cy);
    cudaFree(d_dx); cudaFree(d_dy);
    return -1;
}

int rippra_cuda_full_pipeline(const double *d_frame, int width, int height,
                              const rippra_calibration *cal,
                              const rippa_config *cfg,
                              const rippra_zonal_mesh *mesh,
                              const rippra_modal_model *model,
                              double *d_W, double *d_coeffs,
                              double *d_dm_commands)
{
    return rippra_cuda_full_pipeline_ex(d_frame, width, height, cal, cfg,
                                        mesh, model, d_W, d_coeffs,
                                        d_dm_commands, NULL);
}
