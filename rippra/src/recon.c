/*
 * rippra/recon.c - Wavefront Reconstruction, Turbulence Characterization, and DM Mapping
 */
#include "rippra/recon.h"
#include "rippra/la.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Factorial helper */
static double factorial(int k) {
    double res = 1.0;
    for (int i = 2; i <= k; i++) {
        res *= i;
    }
    return res;
}

/* Zernike radial coefficient C_s */
static double zernike_coeff(int n, int m, int s) {
    int abs_m = abs(m);
    double num = factorial(n - s);
    double den = factorial(s) * factorial((n + abs_m)/2 - s) * factorial((n - abs_m)/2 - s);
    double sign = (s % 2 == 0) ? 1.0 : -1.0;
    return sign * num / den;
}

/* Analytical evaluation of Zernike derivatives at (x, y) */
static void evaluate_zernike_derivatives(int n, int m, double x, double y, double *dzdx, double *dzdy) {
    double rho = sqrt(x * x + y * y);
    double theta = atan2(y, x);
    int abs_m = abs(m);
    
    double R = 0.0;
    double dR = 0.0;
    
    for (int s = 0; s <= (n - abs_m)/2; s++) {
        double c = zernike_coeff(n, m, s);
        int pow_rho = n - 2*s;
        if (pow_rho > 0) {
            R += c * pow(rho, pow_rho);
            dR += c * pow_rho * pow(rho, pow_rho - 1);
        } else if (pow_rho == 0) {
            R += c;
        }
    }
    
    double norm_factor = (m == 0) ? sqrt(n + 1) : sqrt(2 * (n + 1));
    R *= norm_factor;
    dR *= norm_factor;
    
    double cos_mt = cos(abs_m * theta);
    double sin_mt = sin(abs_m * theta);
    
    double dz_drho = 0.0;
    double dz_dtheta = 0.0;
    
    if (m >= 0) {
        dz_drho = dR * cos_mt;
        dz_dtheta = -abs_m * R * sin_mt;
    } else {
        dz_drho = dR * sin_mt;
        dz_dtheta = abs_m * R * cos_mt;
    }
    
    if (rho < 1e-9) {
        if (n == 1) {
            if (m == 1) {
                *dzdx = norm_factor;
                *dzdy = 0.0;
            } else if (m == -1) {
                *dzdx = 0.0;
                *dzdy = norm_factor;
            } else {
                *dzdx = 0.0;
                *dzdy = 0.0;
            }
        } else {
            *dzdx = 0.0;
            *dzdy = 0.0;
        }
        return;
    }
    
    *dzdx = dz_drho * cos(theta) - dz_dtheta * sin(theta) / rho;
    *dzdy = dz_drho * sin(theta) + dz_dtheta * cos(theta) / rho;
}

/* Noll index j to radial order n and azimuthal frequency m */
static void noll_to_nm(int j, int *n, int *m) {
    if (j == 1) {
        *n = 0; *m = 0; return;
    }
    int current_j = 2;
    for (int ni = 1; ; ni++) {
        for (int mi = ni % 2; mi <= ni; mi += 2) {
            if (mi == 0) {
                if (current_j == j) {
                    *n = ni; *m = 0; return;
                }
                current_j++;
            } else {
                if (current_j % 2 == 1) {
                    if (current_j == j) {
                        *n = ni; *m = -mi; return;
                    }
                    if (current_j + 1 == j) {
                        *n = ni; *m = mi; return;
                    }
                } else {
                    if (current_j == j) {
                        *n = ni; *m = mi; return;
                    }
                    if (current_j + 1 == j) {
                        *n = ni; *m = -mi; return;
                    }
                }
                current_j += 2;
            }
        }
    }
}

/* Compare function for sorting node coordinates */
typedef struct {
    int u, v;
    int index;
} ZonalNode;

static int compare_nodes(const void *a, const void *b) {
    const ZonalNode *na = (const ZonalNode *)a;
    const ZonalNode *nb = (const ZonalNode *)b;
    if (na->u != nb->u) return na->u - nb->u;
    return na->v - nb->v;
}

/* ---- Zonal Reconstruction (Fried Geometry) ------------------------------- */

int rippra_zonal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_zonal_mesh *mesh) {
    int nspots = cal->nspots;
    double d = cfg->pitch;
    
    /* Allocate temporary nodes array to collect all corners */
    int max_nodes = 4 * nspots;
    ZonalNode *temp_nodes = (ZonalNode *)malloc(max_nodes * sizeof(ZonalNode));
    if (!temp_nodes) return -1;
    
    int t_cnt = 0;
    for (int k = 0; k < nspots; ++k) {
        int u = (int)round((cal->subaps[k].ref_cx - cal->pupil_cx) / cal->pitch_px);
        int v = (int)round((cal->subaps[k].ref_cy - cal->pupil_cy) / cal->pitch_px);
        
        temp_nodes[t_cnt++] = (ZonalNode){u, v, 0};
        temp_nodes[t_cnt++] = (ZonalNode){u + 1, v, 0};
        temp_nodes[t_cnt++] = (ZonalNode){u, v + 1, 0};
        temp_nodes[t_cnt++] = (ZonalNode){u + 1, v + 1, 0};
    }
    
    /* Sort and de-duplicate nodes */
    qsort(temp_nodes, t_cnt, sizeof(ZonalNode), compare_nodes);
    
    int nnodes = 0;
    for (int i = 0; i < t_cnt; ++i) {
        if (i == 0 || temp_nodes[i].u != temp_nodes[i - 1].u || temp_nodes[i].v != temp_nodes[i - 1].v) {
            nnodes++;
        }
    }
    
    mesh->nnodes = nnodes;
    mesh->node_u = (int *)malloc(nnodes * sizeof(int));
    mesh->node_v = (int *)malloc(nnodes * sizeof(int));
    mesh->G = (double *)calloc(2 * nspots * nnodes, sizeof(double));
    mesh->Gpinv = (double *)malloc(nnodes * 2 * nspots * sizeof(double));
    
    if (!mesh->node_u || !mesh->node_v || !mesh->G || !mesh->Gpinv) {
        free(temp_nodes);
        rippra_zonal_free(mesh);
        return -2;
    }
    
    /* Copy unique nodes and assign index map */
    int unique_idx = 0;
    for (int i = 0; i < t_cnt; ++i) {
        if (i == 0 || temp_nodes[i].u != temp_nodes[i - 1].u || temp_nodes[i].v != temp_nodes[i - 1].v) {
            mesh->node_u[unique_idx] = temp_nodes[i].u;
            mesh->node_v[unique_idx] = temp_nodes[i].v;
            temp_nodes[i].index = unique_idx;
            unique_idx++;
        } else {
            temp_nodes[i].index = unique_idx - 1;
        }
    }
    
    /* Build helper to map grid (u, v) back to unique node index */
    /* Since N is small (~508), we can search in mesh->node_u/v directly */
    #define FIND_NODE_IDX(u_coord, v_coord) ({ \
        int found_idx = -1; \
        for(int _ni=0; _ni<nnodes; ++_ni) { \
            if (mesh->node_u[_ni] == (u_coord) && mesh->node_v[_ni] == (v_coord)) { \
                found_idx = _ni; \
                break; \
            } \
        } \
        found_idx; \
    })
    
    /* Construct Geometry matrix G */
    for (int k = 0; k < nspots; ++k) {
        int u = (int)round((cal->subaps[k].ref_cx - cal->pupil_cx) / cal->pitch_px);
        int v = (int)round((cal->subaps[k].ref_cy - cal->pupil_cy) / cal->pitch_px);
        
        int i0 = FIND_NODE_IDX(u, v);
        int i1 = FIND_NODE_IDX(u + 1, v);
        int i2 = FIND_NODE_IDX(u, v + 1);
        int i3 = FIND_NODE_IDX(u + 1, v + 1);
        
        /* G row k (X-slope) */
        mesh->G[k * nnodes + i0] = -1.0 / (2.0 * d);
        mesh->G[k * nnodes + i1] =  1.0 / (2.0 * d);
        mesh->G[k * nnodes + i2] = -1.0 / (2.0 * d);
        mesh->G[k * nnodes + i3] =  1.0 / (2.0 * d);
        
        /* G row k + nspots (Y-slope) */
        mesh->G[(k + nspots) * nnodes + i0] = -1.0 / (2.0 * d);
        mesh->G[(k + nspots) * nnodes + i1] = -1.0 / (2.0 * d);
        mesh->G[(k + nspots) * nnodes + i2] =  1.0 / (2.0 * d);
        mesh->G[(k + nspots) * nnodes + i3] =  1.0 / (2.0 * d);
    }
    #undef FIND_NODE_IDX
    
    free(temp_nodes);
    
    /* Compute pseudo-inverse of G (size 2*nspots x nnodes) -> Gpinv (size nnodes x 2*nspots) */
    /* Singular values below rcond * max_sig are truncated to remove piston */
    int rc = rippa_pinv(mesh->G, mesh->Gpinv, 2 * nspots, nnodes, 1e-4);
    if (rc != 0) return -3;
    
    return 0;
}

void rippra_zonal_free(rippra_zonal_mesh *mesh) {
    if (mesh) {
        free(mesh->node_u); mesh->node_u = NULL;
        free(mesh->node_v); mesh->node_v = NULL;
        free(mesh->G);      mesh->G = NULL;
        free(mesh->Gpinv);  mesh->Gpinv = NULL;
        mesh->nnodes = 0;
    }
}

int rippra_zonal_reconstruct(const rippra_zonal_mesh *mesh, const double *dx, const double *dy, const rippa_config *cfg, double *W) {
    int nspots = cfg->totlenses;
    double p = cfg->camera_pixsize;
    double f = cfg->flength;
    
    /* Assemble measured slope vector: s = [dx; dy] * (p / f) */
    double *s = (double *)malloc(2 * nspots * sizeof(double));
    if (!s) return -1;
    
    for (int i = 0; i < nspots; ++i) {
        s[i] = dx[i] * (p / f);
        s[i + nspots] = dy[i] * (p / f);
    }
    
    /* Reconstruct node phase heights W = Gpinv * s */
    rippa_matvec(mesh->Gpinv, s, W, mesh->nnodes, 2 * nspots);
    
    free(s);
    return 0;
}

/* ---- Modal Reconstruction (Zernike Polynomials) -------------------------- */

int rippra_modal_setup(const rippra_calibration *cal, const rippa_config *cfg, rippra_modal_model *model) {
    int nspots = cal->nspots;
    int max_j = (cfg->zernike_nmax + 1) * (cfg->zernike_nmax + 2) / 2;
    int nmodes = max_j - 1; /* exclude piston (j = 1) */
    
    model->nmodes = nmodes;
    model->mode_j = (int *)malloc(nmodes * sizeof(int));
    model->mode_n = (int *)malloc(nmodes * sizeof(int));
    model->mode_m = (int *)malloc(nmodes * sizeof(int));
    model->Zprime = (double *)calloc(2 * nspots * nmodes, sizeof(double));
    model->Zprime_pinv = (double *)malloc(nmodes * 2 * nspots * sizeof(double));
    
    if (!model->mode_j || !model->mode_n || !model->mode_m || !model->Zprime || !model->Zprime_pinv) {
        rippra_modal_free(model);
        return -1;
    }
    
    /* Generate modes indices mapping */
    for (int i = 0; i < nmodes; i++) {
        int j = i + 2; /* offset by 2 to skip piston */
        model->mode_j[i] = j;
        noll_to_nm(j, &model->mode_n[i], &model->mode_m[i]);
    }
    
    /* Grid integration parameters */
    int M = 15; /* 15x15 integration grid */
    double rbar = cfg->sa_radius / cfg->pupil_radius;
    double kk = cfg->pupil_radius / (M_PI * cfg->sa_radius * cfg->sa_radius);
    
    /* Integrate Zernike derivatives over each active sub-aperture area */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int k = 0; k < nspots; ++k) {
        /* Canonical center of the sub-aperture */
        double x_c = (cal->subaps[k].ref_cx - cal->pupil_cx) * cfg->camera_pixsize / cfg->pupil_radius;
        double y_c = -(cal->subaps[k].ref_cy - cal->pupil_cy) * cfg->camera_pixsize / cfg->pupil_radius;
        
        for (int m_idx = 0; m_idx < nmodes; m_idx++) {
            int n = model->mode_n[m_idx];
            int m = model->mode_m[m_idx];
            
            double sum_dzdx = 0.0;
            double sum_dzdy = 0.0;
            int count_pts = 0;
            
            for (int r_step = 0; r_step < M; r_step++) {
                double dy = -rbar + 2.0 * rbar * r_step / (M - 1);
                for (int c_step = 0; c_step < M; c_step++) {
                    double dx = -rbar + 2.0 * rbar * c_step / (M - 1);
                    if (dx*dx + dy*dy <= rbar*rbar) {
                        double dzdx_val, dzdy_val;
                        evaluate_zernike_derivatives(n, m, x_c + dx, y_c + dy, &dzdx_val, &dzdy_val);
                        sum_dzdx += dzdx_val;
                        sum_dzdy += dzdy_val;
                        count_pts++;
                    }
                }
            }
            
            double avg_dzdx = (count_pts > 0) ? sum_dzdx / count_pts : 0.0;
            double avg_dzdy = (count_pts > 0) ? sum_dzdy / count_pts : 0.0;
            
            /* Fill Zprime matrix elements */
            model->Zprime[k * nmodes + m_idx] = kk * avg_dzdx;
            model->Zprime[(k + nspots) * nmodes + m_idx] = -kk * avg_dzdy;
        }
    }
    
    /* Compute pseudo-inverse of Zprime */
    int rc = rippa_pinv(model->Zprime, model->Zprime_pinv, 2 * nspots, nmodes, 1e-4);
    if (rc != 0) return -2;
    
    return 0;
}

void rippra_modal_free(rippra_modal_model *model) {
    if (model) {
        free(model->mode_j);       model->mode_j = NULL;
        free(model->mode_n);       model->mode_n = NULL;
        free(model->mode_m);       model->mode_m = NULL;
        free(model->Zprime);       model->Zprime = NULL;
        free(model->Zprime_pinv);  model->Zprime_pinv = NULL;
        model->nmodes = 0;
    }
}

int rippra_modal_reconstruct(const rippra_modal_model *model, const double *dx, const double *dy, const rippa_config *cfg, double *coeffs) {
    int nspots = cfg->totlenses;
    double p = cfg->camera_pixsize;
    double f = cfg->flength;
    double lambda = cfg->wavelength;
    
    /* Prepare displacements in meters */
    double *s = (double *)malloc(2 * nspots * sizeof(double));
    if (!s) return -1;
    
    for (int i = 0; i < nspots; ++i) {
        s[i] = dx[i] * p;
        s[i + nspots] = dy[i] * p;
    }
    
    /* Output coefficients in meters: a_m = (1/flen) * (Zprime_pinv * s) */
    double *a_m = (double *)calloc(model->nmodes, sizeof(double));
    if (!a_m) {
        free(s);
        return -2;
    }
    
    rippa_matvec(model->Zprime_pinv, s, a_m, model->nmodes, 2 * nspots);
    
    /* Convert to radians: a = a_m * (2 * PI / lambda) / f */
    for (int i = 0; i < model->nmodes; ++i) {
        coeffs[i] = (1.0 / f) * a_m[i] * (2.0 * M_PI / lambda);
    }
    
    free(s);
    free(a_m);
    return 0;
}

/* ---- Turbulence Characterization ----------------------------------------- */

double rippra_compute_r0_impl(const double *dx_series, const double *dy_series, int nframes, int nspots, const rippa_config *cfg) {
    double p = cfg->camera_pixsize;
    double f = cfg->flength;
    double d = cfg->pitch;
    double lambda = cfg->wavelength;
    
    double total_var = 0.0;
    
#ifdef _OPENMP
#pragma omp parallel for reduction(+:total_var) schedule(static)
#endif
    for (int k = 0; k < nspots; ++k) {
        /* Compute means */
        double sum_x = 0.0, sum_y = 0.0;
        for (int t = 0; t < nframes; ++t) {
            sum_x += dx_series[t * nspots + k];
            sum_y += dy_series[t * nspots + k];
        }
        double mean_x = sum_x / nframes;
        double mean_y = sum_y / nframes;
        
        /* Compute variances */
        double var_x = 0.0, var_y = 0.0;
        for (int t = 0; t < nframes; ++t) {
            double dev_x = dx_series[t * nspots + k] - mean_x;
            double dev_y = dy_series[t * nspots + k] - mean_y;
            var_x += dev_x * dev_x;
            var_y += dev_y * dev_y;
        }
        var_x = (nframes > 1) ? var_x / (nframes - 1) : 0.0;
        var_y = (nframes > 1) ? var_y / (nframes - 1) : 0.0;
        
        /* Convert to radians squared */
        double var_slope_x = var_x * (p / f) * (p / f);
        double var_slope_y = var_y * (p / f) * (p / f);
        
        total_var += 0.5 * (var_slope_x + var_slope_y);
    }
    
    double mean_var = total_var / nspots;
    if (mean_var < 1e-15) return 0.0;
    
    /* Fried parameter formula: r0 = (0.170 * lambda^2 * d^(-1/3) / mean_var)^(3/5) */
    double r0 = pow((0.170 * lambda * lambda * pow(d, -1.0/3.0)) / mean_var, 3.0/5.0);
    return r0;
}

double rippra_compute_tau0_impl(const double *dx_series, const double *dy_series, int nframes, int nspots, double frame_rate) {
    if (nframes < 2) return 0.0;
    
    /* Auto-correlation values at different time lags delta_t */
    int max_lags = nframes / 2;
    double *C = (double *)calloc(max_lags, sizeof(double));
    if (!C) return 0.0;
    
    for (int lag = 0; lag < max_lags; ++lag) {
        double sum_corr = 0.0;
        int count = 0;
        for (int k = 0; k < nspots; ++k) {
            for (int t = 0; t < nframes - lag; ++t) {
                double sx_t = dx_series[t * nspots + k];
                double sy_t = dy_series[t * nspots + k];
                double sx_t_lag = dx_series[(t + lag) * nspots + k];
                double sy_t_lag = dy_series[(t + lag) * nspots + k];
                
                sum_corr += sx_t * sx_t_lag + sy_t * sy_t_lag;
                count++;
            }
        }
        C[lag] = (count > 0) ? sum_corr / count : 0.0;
    }
    
    double C0 = C[0];
    double target = C0 / exp(1.0);
    double tau0 = 0.0;
    
    /* Find lag where C falls below target */
    int found_lag = -1;
    for (int lag = 1; lag < max_lags; ++lag) {
        if (C[lag] <= target) {
            found_lag = lag;
            break;
        }
    }
    
    if (found_lag != -1) {
        /* Interpolate fractional lag */
        double C_prev = C[found_lag - 1];
        double C_curr = C[found_lag];
        double fraction = (target - C_prev) / (C_curr - C_prev);
        double float_lag = (found_lag - 1) + fraction;
        tau0 = float_lag / frame_rate;
    } else {
        /* If correlation doesn't decay enough, assume max lag */
        tau0 = (max_lags - 1) / frame_rate;
    }
    
    free(C);
    return tau0;
}

/* ---- DM Command Map ------------------------------------------------------ */

int rippra_dm_map_impl(const double *target_phase, int nnodes, const rippra_zonal_mesh *mesh, const rippa_config *cfg, double *dm_commands) {
    double coupling = cfg->coupling;
    
    /* Construct coupling matrix C */
    double *C = (double *)calloc(nnodes * nnodes, sizeof(double));
    if (!C) return -1;
    
    for (int i = 0; i < nnodes; ++i) {
        C[i * nnodes + i] = 1.0;
    }
    
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nnodes; ++i) {
        int ui = mesh->node_u[i];
        int vi = mesh->node_v[i];
        
        for (int j = 0; j < nnodes; ++j) {
            if (i == j) continue;
            int uj = mesh->node_u[j];
            int vj = mesh->node_v[j];
            int du = abs(ui - uj);
            int dv = abs(vi - vj);
            
            if ((du == 1 && dv == 0) || (du == 0 && dv == 1)) {
                C[i * nnodes + j] = coupling;
            } else if (du == 1 && dv == 1) {
                C[i * nnodes + j] = coupling * coupling;
            }
        }
    }
    
    /* Solve conjugate wavefront: v = -C^-1 * target_phase  => C * v = -target_phase */
    double *rhs = (double *)malloc(nnodes * sizeof(double));
    if (!rhs) {
        free(C);
        return -2;
    }
    
    for (int i = 0; i < nnodes; ++i) {
        rhs[i] = -target_phase[i];
    }
    
    int rc = rippa_lusolve(C, rhs, nnodes);
    if (rc != 0) {
        free(C);
        free(rhs);
        return -3;
    }
    
    /* rhs now holds the solution actuator strokes */
    memcpy(dm_commands, rhs, nnodes * sizeof(double));
    
    free(C);
    free(rhs);
    return 0;
}

/* ---- Closed-Loop DM Control ------------------------------------------------- */

int rippra_dm_apply_impl(const double *dm_commands, int nnodes,
                          const rippra_zonal_mesh *mesh,
                          const rippa_config *cfg,
                          const double *input_phase,
                          double *output_residual) {
    /* DM ADDS its shape to the wavefront.
       dm_commands from dm_map() solves C*v = -phase, so the DM shape
       cancels the input phase.
       output_residual = input_phase + C * dm_commands
       Since C * dm_map(phase) = -phase, output_residual = 0 at convergence. */
    double coupling = cfg->coupling;
    
    for (int i = 0; i < nnodes; ++i) {
        double dm_shape = 0.0;
        int ui = mesh->node_u[i];
        int vi = mesh->node_v[i];
        
        for (int j = 0; j < nnodes; ++j) {
            if (i == j) {
                dm_shape += dm_commands[j];
            } else {
                int uj = mesh->node_u[j];
                int vj = mesh->node_v[j];
                int du = abs(ui - uj);
                int dv = abs(vi - vj);
                if ((du == 1 && dv == 0) || (du == 0 && dv == 1)) {
                    dm_shape += coupling * dm_commands[j];
                } else if (du == 1 && dv == 1) {
                    dm_shape += coupling * coupling * dm_commands[j];
                }
            }
        }
        output_residual[i] = input_phase[i] + dm_shape;
    }
    return 0;
}

int rippra_closed_loop_step_impl(const double *measured_phase, int nnodes,
                                  const rippra_zonal_mesh *mesh,
                                  const rippa_config *cfg,
                                  double *dm_commands, double gain) {
    /* Single closed-loop iteration:
       1. Compute DM delta: delta_v = gain * dm_map(measured_phase)
          dm_map gives v s.t. C*v = -measured_phase (conjugate)
       2. Accumulate: dm_commands += delta_v
       3. DM shape = C * dm_commands
       4. Residual = measured_phase + DM_shape
       
       Returns RMS of residual in radians × 1e6. */
    double *delta_v = (double *)malloc(nnodes * sizeof(double));
    if (!delta_v) return -1;
    
    /* delta_v = -C^-1 * measured_phase */
    int ret = rippra_dm_map_impl(measured_phase, nnodes, mesh, cfg, delta_v);
    if (ret != 0) {
        free(delta_v);
        return -2;
    }
    
    /* Accumulate commands with gain */
    for (int i = 0; i < nnodes; ++i) {
        dm_commands[i] += gain * delta_v[i];
    }
    
    /* Compute residual: residual = measured_phase + C * dm_commands
       At convergence, C*dm_commands ≈ -measured_phase, so residual ≈ 0. */
    double *residual = (double *)malloc(nnodes * sizeof(double));
    if (!residual) {
        free(delta_v);
        return -3;
    }
    ret = rippra_dm_apply_impl(dm_commands, nnodes, mesh, cfg, measured_phase, residual);
    if (ret != 0) {
        free(delta_v);
        free(residual);
        return -4;
    }
    
    double sum_sq = 0.0;
    for (int i = 0; i < nnodes; ++i) {
        sum_sq += residual[i] * residual[i];
    }
    
    free(residual);
    free(delta_v);
    
    double rms = sqrt(sum_sq / nnodes);
    return (int)(rms * 1e6 + 0.5);
}

int rippra_closed_loop_run_impl(const double *initial_phase, int nnodes,
                                 const rippra_zonal_mesh *mesh,
                                 const rippa_config *cfg,
                                 double *dm_commands, double gain,
                                 int max_iter, double target_rms,
                                 int *out_iters, double *out_residual_rms) {
    /* Run closed-loop AO control until convergence.
       Each iteration:
         1. WFS measures current residual = initial_phase + C * dm_commands
         2. Controller: delta_v = -gain * C^-1 * residual
         3. dm_commands += delta_v
       dm_commands is [in/out]. Returns 0=converged, 1=max_iter, negative=error. */
    
    for (int iter = 0; iter < max_iter; ++iter) {
        /* Compute current residual (what WFS would measure) */
        double *residual = (double *)malloc(nnodes * sizeof(double));
        if (!residual) return -1;
        rippra_dm_apply_impl(dm_commands, nnodes, mesh, cfg, initial_phase, residual);
        
        /* Compute RMS of residual */
        double sum_sq = 0.0;
        for (int i = 0; i < nnodes; ++i) sum_sq += residual[i] * residual[i];
        double rms = sqrt(sum_sq / nnodes);
        
        if (out_residual_rms) *out_residual_rms = rms;
        
        if (rms <= target_rms) {
            free(residual);
            if (out_iters) *out_iters = iter;
            return 0;
        }
        
        /* Compute DM update from residual */
        double *delta_v = (double *)malloc(nnodes * sizeof(double));
        if (!delta_v) { free(residual); return -2; }
        int ret = rippra_dm_map_impl(residual, nnodes, mesh, cfg, delta_v);
        if (ret != 0) { free(residual); free(delta_v); return -3; }
        
        /* Accumulate with gain */
        for (int i = 0; i < nnodes; ++i) {
            dm_commands[i] += gain * delta_v[i];
        }
        
        free(residual);
        free(delta_v);
    }
    
    if (out_iters) *out_iters = max_iter;
    return 1; /* max_iter reached without convergence */
}

double rippra_wavefront_rms_lambda(const double *phase, int nnodes, const rippa_config *cfg)
{
    double sum_sq = 0.0;
    for (int i = 0; i < nnodes; ++i)
        sum_sq += phase[i] * phase[i];
    double rms_m = sqrt(sum_sq / nnodes);
    return rms_m / cfg->wavelength;
}
