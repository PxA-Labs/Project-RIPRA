/*
 * rippa/la.c - vendored linear algebra implementation.
 *
 * Design notes:
 *   - Row-major contiguous double storage throughout.
 *   - LU uses Doolittle with partial pivoting; classic, robust.
 *   - Pseudo-inverse uses one-sided Jacobi SVD (Kogbetliantz). It is O(n^3)
 *     per sweep and converges well for the small, well-conditioned matrices
 *     we have here (interaction matrix ~254x20, geometry ~240x196). These are
 *     computed ONCE at calibration, never per-frame, so the cost is irrelevant.
 */
#include "rippra/la.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- basic helpers ------------------------------------------------------ */

void rippa_matmul(const double *A, const double *B, double *dst,
                  size_t m, size_t k, size_t n)
{
    size_t i, j, p;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            double s = 0.0;
            const double *arow = A + i * k;
            for (p = 0; p < k; ++p)
                s += arow[p] * B[p * n + j];
            dst[i * n + j] = s;
        }
    }
}

void rippa_matvec(const double *A, const double *x, double *dst,
                  size_t m, size_t n)
{
    size_t i, j;
    for (i = 0; i < m; ++i) {
        const double *arow = A + i * n;
        double s = 0.0;
        for (j = 0; j < n; ++j)
            s += arow[j] * x[j];
        dst[i] = s;
    }
}

void rippa_transpose(const double *A, double *dst, size_t m, size_t n)
{
    size_t i, j;
    for (i = 0; i < m; ++i)
        for (j = 0; j < n; ++j)
            dst[j * m + i] = A[i * n + j];
}

void rippa_copy(double *dst, const double *src, size_t n)
{
    memcpy(dst, src, n * sizeof(double));
}

double rippa_dnrm2(const double *x, size_t n)
{
    size_t i;
    double s = 0.0;
    for (i = 0; i < n; ++i) s += x[i] * x[i];
    return sqrt(s);
}

/* ---- LU with partial pivoting ------------------------------------------- */

int rippa_lusolve(double *A, double *b, size_t n)
{
    size_t *piv;
    size_t i, j, k;
    piv = (size_t *)malloc(n * sizeof(size_t));
    if (!piv) return -2;
    for (i = 0; i < n; ++i) piv[i] = i;

    for (k = 0; k < n; ++k) {
        /* pivot: find max abs in column k, rows k..n-1 */
        size_t pivrow = k;
        double maxval = fabs(A[k * n + k]);
        for (i = k + 1; i < n; ++i) {
            double v = fabs(A[i * n + k]);
            if (v > maxval) { maxval = v; pivrow = i; }
        }
        if (maxval == 0.0) { free(piv); return 1; } /* singular */

        /* swap rows k and pivrow (index + data) */
        if (pivrow != k) {
            size_t tmp = piv[k]; piv[k] = piv[pivrow]; piv[pivrow] = tmp;
            for (j = 0; j < n; ++j) {
                double t = A[k * n + j];
                A[k * n + j] = A[pivrow * n + j];
                A[pivrow * n + j] = t;
            }
        }
        /* eliminate */
        for (i = k + 1; i < n; ++i) {
            double f = A[i * n + k] / A[k * n + k];
            A[i * n + k] = f;
            for (j = k + 1; j < n; ++j)
                A[i * n + j] -= f * A[k * n + j];
        }
    }

    /* apply pivot to b */
    {
        double *tmp = (double *)malloc(n * sizeof(double));
        if (!tmp) { free(piv); return -2; }
        for (i = 0; i < n; ++i) tmp[i] = b[piv[i]];
        memcpy(b, tmp, n * sizeof(double));
        free(tmp);
    }

    /* forward solve L y = b (unit diagonal) */
    for (i = 0; i < n; ++i) {
        double s = b[i];
        for (j = 0; j < i; ++j) s -= A[i * n + j] * b[j];
        b[i] = s;
    }
    /* back solve U x = y */
    for (k = n; k-- > 0; ) {
        double s = b[k];
        for (j = k + 1; j < n; ++j) s -= A[k * n + j] * b[j];
        b[k] = s / A[k * n + k];
    }
    free(piv);
    return 0;
}

/* ---- one-sided Jacobi SVD -> pseudo-inverse ----------------------------- */
/*
 * Compute A+ via one-sided Jacobi rotation on the columns of A.
 * Standard algorithm: repeatedly rotate pairs of columns of A to make them
 * orthogonal; the column norms become the singular values and the accumulated
 * rotations give V. U = A / sigma (column-scaled).
 *
 * We then build A+ = V * diag(1/sigma) * U^T with rcond truncation.
 */
int rippa_pinv(const double *A, double *Aplus, size_t m, size_t n,
               double rcond)
{
    const int maxsweeps = 60;
    const double tol = 1e-14;
    int sweep, p, q, i, ok;
    double *W;   /* working copy of A, m x n, mutated to hold U*sigma        */
    double *V;   /* n x n right singular vectors                              */
    double *sig; /* n singular values                                        */
    size_t mn = (size_t)m * (size_t)n;
    size_t nn = (size_t)n * (size_t)n;

    W = (double *)malloc(mn * sizeof(double));
    V = (double *)malloc(nn * sizeof(double));
    sig = (double *)malloc(n * sizeof(double));
    if (!W || !V || !sig) { free(W); free(V); free(sig); return -2; }

    memcpy(W, A, mn * sizeof(double));

    /* V = identity */
    for (i = 0; i < (int)nn; ++i) V[i] = 0.0;
    for (i = 0; i < (int)n; ++i) V[i * n + i] = 1.0;

    /* one-sided Jacobi: orthogonalize columns of W pairwise */
    for (sweep = 0; sweep < maxsweeps; ++sweep) {
        int off = 0;
        for (p = 0; p < (int)n; ++p) {
            for (q = p + 1; q < (int)n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                double zeta, t, c, s;
                for (i = 0; i < (int)m; ++i) {
                    double wp = W[i * n + p];
                    double wq = W[i * n + q];
                    alpha += wp * wp;
                    beta  += wq * wq;
                    gamma += wp * wq;
                }
                if (fabs(gamma) <= tol * sqrt(alpha * beta) ||
                    alpha == 0.0 || beta == 0.0)
                    continue; /* already orthogonal */
                off = 1;
                zeta = (beta - alpha) / (2.0 * gamma);
                t = copysign(1.0, zeta) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                c = 1.0 / sqrt(1.0 + t * t);
                s = c * t;
                /* rotate columns p,q of W */
                for (i = 0; i < (int)m; ++i) {
                    double wp = W[i * n + p];
                    double wq = W[i * n + q];
                    W[i * n + p] = c * wp - s * wq;
                    W[i * n + q] = s * wp + c * wq;
                }
                /* rotate columns p,q of V */
                for (i = 0; i < (int)n; ++i) {
                    double vp = V[i * n + p];
                    double vq = V[i * n + q];
                    V[i * n + p] = c * vp - s * vq;
                    V[i * n + q] = s * vp + c * vq;
                }
            }
        }
        if (!off) break; /* converged */
    }

    /* singular values = column norms of W; scale W down to U */
    {
        double smax = 0.0;
        for (p = 0; p < (int)n; ++p) {
            double nnrm = 0.0;
            for (i = 0; i < (int)m; ++i)
                nnrm += W[i * n + p] * W[i * n + p];
            nnrm = sqrt(nnrm);
            sig[p] = nnrm;
            if (nnrm > smax) smax = nnrm;
        }
        /* W -> U = W / sigma (safe for zero sigma) */
        for (p = 0; p < (int)n; ++p) {
            double sg = sig[p];
            if (sg > 0.0) {
                double inv = 1.0 / sg;
                for (i = 0; i < (int)m; ++i)
                    W[i * n + p] *= inv;
            }
        }
        /* zero U columns for truncated sigma */
        {
            double thresh = rcond * smax;
            for (p = 0; p < (int)n; ++p) {
                if (sig[p] < thresh) {
                    for (i = 0; i < (int)m; ++i)
                        W[i * n + p] = 0.0;
                }
            }
        }
    }

    /* A+ = V * diag(inv sigma, with truncation) * U^T
     * Build column-scaled Vt first: Vscaled[i][p] = V[i][p] * (1/sigma[p])
     * then A+ (n x m) = Vscaled (n x n) * U^T (n x m).
     * U^T[p][i] = U[i][p] = W[i*n+p].
     */
    ok = 0;
    {
        double *Vscaled = (double *)malloc(nn * sizeof(double));
        if (!Vscaled) { free(W); free(V); free(sig); return -2; }
        for (i = 0; i < (int)n; ++i)
            for (p = 0; p < (int)n; ++p) {
                double sg = sig[p];
                double inv = (sg > 0.0) ? 1.0 / sg : 0.0;
                Vscaled[i * n + p] = V[i * n + p] * inv;
            }
        /* Aplus[i*m + j] = sum_p Vscaled[i*n+p] * W[j*n+p] */
        for (i = 0; i < (int)n; ++i) {
            for (int j = 0; j < (int)m; ++j) {
                double acc = 0.0;
                for (p = 0; p < (int)n; ++p)
                    acc += Vscaled[i * n + p] * W[j * n + p];
                Aplus[i * m + j] = acc;
            }
        }
        free(Vscaled);
    }

    free(W); free(V); free(sig);
    return ok;
}
