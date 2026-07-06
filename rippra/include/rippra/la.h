/*
 * rippa/la.h - vendored linear algebra (dependency-free)
 *
 * Provides:
 *   - matvec / matmul for row-major double matrices
 *   - LU decomposition with partial pivoting + solve
 *   - matrix transpose, copy
 *   - one-sided Jacobi SVD -> Moore-Penrose pseudo-inverse (rcond truncated)
 *
 * All matrices are row-major, contiguous double arrays of size rows*cols.
 * No per-call allocation in the hot solve routines.
 *
 * This is deliberately small and self-contained so the project has zero
 * external linear-algebra dependency (avoids LAPACK-on-Windows friction).
 */
#ifndef RIPRA_LA_H
#define RIPRA_LA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- basic helpers ------------------------------------------------------ */

/* dst(m x n) = A(m x k) * B(k x n) */
void rippa_matmul(const double *A, const double *B, double *dst,
                  size_t m, size_t k, size_t n);

/* dst(m) = A(m x n) * x(n) */
void rippa_matvec(const double *A, const double *x, double *dst,
                  size_t m, size_t n);

/* dst(n x m) = transpose(A)(m x n) */
void rippa_transpose(const double *A, double *dst, size_t m, size_t n);

/* dst <- src, n elements */
void rippa_copy(double *dst, const double *src, size_t n);

/* Euclidean norm of vector length n */
double rippa_dnrm2(const double *x, size_t n);

/* ---- LU solve ----------------------------------------------------------- */

/*
 * Solve A x = b in place. A is n x n row-major; on output A holds the LU
 * factorization. b is overwritten with x. Returns 0 on success, nonzero
 * if singular.
 */
int rippa_lusolve(double *A, double *b, size_t n);

/* ---- pseudo-inverse via one-sided Jacobi SVD ---------------------------- */

/*
 * Compute the Moore-Penrose pseudo-inverse of A (m x n), storing the result
 * in Aplus (n x m). Singular values below rcond * largest_singular_value are
 * zeroed (treated as null space). This is what makes the piston mode drop
 * out automatically for the geometry/interaction matrices.
 *
 * If cond is non-NULL, the condition number (smax / smin) of the non-truncated
 * singular values is stored there. If all values are truncated, *cond = 0.
 *
 * Returns 0 on success (converged), -1 if SVD did not converge within max
 * sweeps, -2 on memory allocation failure. Aplus must hold at least m*n
 * doubles.
 */
int rippa_pinv(const double *A, double *Aplus, size_t m, size_t n,
               double rcond, double *cond);

#ifdef __cplusplus
}
#endif
#endif /* RIPRA_LA_H */
