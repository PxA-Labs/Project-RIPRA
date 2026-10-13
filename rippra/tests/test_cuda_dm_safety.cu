/*
 * Focused CUDA DM mapping/safety test.
 *
 * This test is intentionally standalone: it exercises the device buffer
 * transport and compares the CUDA host fallback with the CPU authority for
 * each safety outcome.  Register it with CMake only when CUDA is available.
 */
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rippra/recon.h"
#include "../cuda/rippra_cuda.h"

static int g_tests;
static int g_passed;

static void check(const char *label, int condition)
{
    ++g_tests;
    if (condition) {
        ++g_passed;
        printf("  PASS %s\n", label);
    } else {
        printf("  FAIL %s\n", label);
    }
}

static int commands_match(const double *a, const double *b, int n)
{
    for (int i = 0; i < n; ++i) {
        double scale = fmax(1.0, fmax(fabs(a[i]), fabs(b[i])));
        if (!isfinite(a[i]) || !isfinite(b[i]) ||
            fabs(a[i] - b[i]) > 1e-12 * scale)
            return 0;
    }
    return 1;
}

static int all_zero(const double *values, int n)
{
    for (int i = 0; i < n; ++i) {
        if (values[i] != 0.0)
            return 0;
    }
    return 1;
}

static void init_config(rippa_config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->coupling = 0.0;
    cfg->dm_park_threshold = 0.30;
    cfg->dm_max_stroke = 0.0;
}

static int run_case(const char *name, const double *target,
                    const rippra_zonal_mesh *mesh, const rippa_config *cfg,
                    const double *expected_commands, int expected_rc,
                    int expected_status, double *d_target,
                    double *d_commands, int use_wrapper)
{
    const int nnodes = mesh->nnodes;
    double cpu_commands[4] = {0.0, 0.0, 0.0, 0.0};
    double gpu_commands[4] = {0.0, 0.0, 0.0, 0.0};
    const double stale_commands[4] = {99.0, -99.0, 99.0, -99.0};
    int cpu_status = RIPPRA_DM_ERROR;
    int gpu_status = RIPPRA_DM_ERROR;
    int cpu_rc;
    int gpu_rc;
    cudaError_t err;

    printf("\n%s:\n", name);

    cpu_rc = rippra_dm_map_ex_impl(target, nnodes, mesh, cfg,
                                   cpu_commands, &cpu_status);
    check("cpu_return_matches_case",
          (expected_rc == 0 && cpu_rc == 0) ||
          (expected_rc < 0 && cpu_rc < 0));
    check("cpu_status_matches_case", cpu_status == expected_status);

    err = cudaMemcpy(d_target, target, (size_t)nnodes * sizeof(double),
                     cudaMemcpyHostToDevice);
    check("target_copy_to_device", err == cudaSuccess);
    if (err != cudaSuccess)
        return 0;

    /* Make stale-output bugs observable on both success and failure paths. */
    err = cudaMemcpy(d_commands, stale_commands,
                     (size_t)nnodes * sizeof(double), cudaMemcpyHostToDevice);
    check("stale_output_seed", err == cudaSuccess);
    if (err != cudaSuccess)
        return 0;

    if (use_wrapper) {
        gpu_rc = rippra_cuda_dm_map(d_target, nnodes, mesh, cfg, d_commands);
    } else {
        gpu_rc = rippra_cuda_dm_map_ex(d_target, nnodes, mesh, cfg,
                                       d_commands, &gpu_status);
    }

    err = cudaMemcpy(gpu_commands, d_commands,
                     (size_t)nnodes * sizeof(double), cudaMemcpyDeviceToHost);
    check("commands_copy_to_host", err == cudaSuccess);
    if (err != cudaSuccess)
        return 0;

    check("return_sign_matches_cpu",
          (cpu_rc == 0 && gpu_rc == 0) || (cpu_rc != 0 && gpu_rc < 0));

    if (cpu_rc == 0) {
        check("status_is_successful_safety_status",
              cpu_status == RIPPRA_DM_OK ||
              cpu_status == RIPPRA_DM_SATURATED ||
              cpu_status == RIPPRA_DM_PARKED);
        if (!use_wrapper)
            check("status_matches_cpu", gpu_status == cpu_status);
        check("commands_match_cpu", commands_match(cpu_commands,
                                                    gpu_commands, nnodes));
        check("commands_match_expected", commands_match(expected_commands,
                                                         gpu_commands, nnodes));
    } else {
        check("failed_map_status_is_error", cpu_status == RIPPRA_DM_ERROR);
        if (!use_wrapper)
            check("failed_status_is_error", gpu_status == RIPPRA_DM_ERROR);
        check("failed_map_zeroes_device_output", all_zero(gpu_commands, nnodes));
        check("failed_output_matches_expected",
              commands_match(expected_commands, gpu_commands, nnodes));
    }

    return 0;
}

int main(void)
{
    const int nnodes = 4;
    const int node_u[4] = {0, 1, 0, 1};
    const int node_v[4] = {0, 0, 1, 1};
    const double disabled_target[4] = {1.0, -2.0, 0.5, -0.25};
    const double clamp_target[4] = {2.0, 0.0, 0.0, 0.0};
    const double park_target[4] = {2.0, -2.0, 2.0, -2.0};
    const double boundary_target[4] = {1.0, -1.0, 0.25, -0.25};
    const double nan_target[4] = {NAN, 0.0, 0.0, 0.0};
    const double inf_target[4] = {INFINITY, 0.0, 0.0, 0.0};
    const double disabled_expected[4] = {-1.0, 2.0, -0.5, 0.25};
    const double clamp_expected[4] = {-1.0, 0.0, 0.0, 0.0};
    const double zero_expected[4] = {0.0, 0.0, 0.0, 0.0};
    const double boundary_expected[4] = {-1.0, 1.0, -0.25, 0.25};
    rippra_zonal_mesh mesh;
    rippa_config cfg;
    double *d_target = NULL;
    double *d_commands = NULL;
    int device_count = 0;
    cudaError_t err;

    printf("=== CUDA DM Mapping/Safety Test ===\n");

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    memset(&mesh, 0, sizeof(mesh));
    mesh.nnodes = nnodes;
    mesh.node_u = (int *)node_u;
    mesh.node_v = (int *)node_v;

    err = cudaMalloc((void **)&d_target, (size_t)nnodes * sizeof(double));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc((void **)&d_commands, (size_t)nnodes * sizeof(double));
    if (err != cudaSuccess)
        goto fail;

    init_config(&cfg);
    run_case("disabled_limits", disabled_target, &mesh, &cfg,
             disabled_expected, 0, RIPPRA_DM_OK, d_target, d_commands, 0);

    init_config(&cfg);
    cfg.dm_max_stroke = 1.0;
    cfg.dm_park_threshold = 0.50;
    run_case("single_actuator_clamp", clamp_target, &mesh, &cfg,
             clamp_expected, 0, RIPPRA_DM_SATURATED, d_target, d_commands, 0);

    init_config(&cfg);
    cfg.dm_max_stroke = 1.0;
    cfg.dm_park_threshold = 0.30;
    run_case("parking", park_target, &mesh, &cfg,
             zero_expected, 0, RIPPRA_DM_PARKED, d_target, d_commands, 0);

    init_config(&cfg);
    cfg.dm_max_stroke = 1.0;
    cfg.dm_park_threshold = 0.50;
    run_case("exact_boundaries_and_wrapper", boundary_target, &mesh, &cfg,
             boundary_expected, 0, RIPPRA_DM_OK, d_target, d_commands, 1);

    init_config(&cfg);
    cfg.dm_max_stroke = 1.0;
    run_case("nan_input", nan_target, &mesh, &cfg,
             zero_expected, -1, RIPPRA_DM_ERROR, d_target, d_commands, 0);
    run_case("infinite_input", inf_target, &mesh, &cfg,
             zero_expected, -1, RIPPRA_DM_ERROR, d_target, d_commands, 0);

    cudaFree(d_target);
    cudaFree(d_commands);
    printf("\n=== %d/%d checks passed ===\n", g_passed, g_tests);
    return (g_passed == g_tests) ? 0 : 1;

fail:
    cudaFree(d_target);
    cudaFree(d_commands);
    fprintf(stderr, "ERROR: CUDA allocation failed: %s\n", cudaGetErrorString(err));
    return 1;
}
