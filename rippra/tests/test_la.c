/*
 * tests/test_la.c - unit tests for the vendored linear algebra
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rippra/la.h"

static int ntests = 0, npass = 0;

static void check(const char *label, double got, double expected, double tol)
{
    ntests++;
    if (fabs(got - expected) <= tol + 1e-15) {
        npass++;
    } else {
        printf("  FAIL %s: got %.8e expected %.8e\n", label, got, expected);
    }
}

/* Test basic matvec: A*x where A is identity -> result = x */
static void test_matvec_identity(void)
{
    double A[9] = {1,0,0, 0,1,0, 0,0,1};
    double x[3] = {7.0, -3.0, 2.5};
    double y[3];
    rippa_matvec(A, x, y, 3, 3);
    check("matvec_id_0", y[0], 7.0, 1e-12);
    check("matvec_id_1", y[1], -3.0, 1e-12);
    check("matvec_id_2", y[2], 2.5, 1e-12);
}

/* Test 2x2 LU solve: [2 1; 1 3] * [x0; x1] = [5; 10] -> x=[1, 3] */
static void test_lu_solve_2x2(void)
{
    double A[4] = {2.0, 1.0, 1.0, 3.0};
    double b[2] = {5.0, 10.0};
    int rc = rippa_lusolve(A, b, 2);
    check("lu_rc", (double)rc, 0.0, 0.0);
    check("lu_x0", b[0], 1.0, 1e-12);
    check("lu_x1", b[1], 3.0, 1e-12);
}

/* Test 3x3 LU solve with known solution */
static void test_lu_solve_3x3(void)
{
    double A[9] = {1,2,3, 4,5,6, 7,8,0};  /* known determinant = 27 */
    double b[3] = {14.0, 32.0, 23.0};      /* exact solution [1,2,3] */
    int rc = rippa_lusolve(A, b, 3);
    check("lu3_rc", (double)rc, 0.0, 0.0);
    check("lu3_x0", b[0], 1.0, 1e-10);
    check("lu3_x1", b[1], 2.0, 1e-10);
    check("lu3_x2", b[2], 3.0, 1e-10);
}

/* Test pseudo-inverse of 2x2 identity */
static void test_pinv_identity(void)
{
    double A[4] = {1,0, 0,1};
    double Ap[4];
    int rc = rippa_pinv(A, Ap, 2, 2, 1e-12);
    check("pinv_id_rc", (double)rc, 0.0, 0.0);
    check("pinv_id_00", Ap[0], 1.0, 1e-10);
    check("pinv_id_01", Ap[1], 0.0, 1e-10);
    check("pinv_id_10", Ap[2], 0.0, 1e-10);
    check("pinv_id_11", Ap[3], 1.0, 1e-10);
}

/* Test pseudo-inverse of 3x2 tall matrix, verify A * A+ * A = A */
static void test_pinv_tall(void)
{
    /* A = [1 0; 0 2; 3 1] */
    double A[6] = {1,0, 0,2, 3,1};
    double Ap[6];  /* 2x3 */
    int rc = rippa_pinv(A, Ap, 3, 2, 1e-12);
    check("pinv_tall_rc", (double)rc, 0.0, 0.0);
    /* A * A+ should be 3x3 projection; A+ * A should be 2x2 identity */
    /* Check A+ * A = I (2x2) */
    {
        double AA[4];  /* Ap(2x3) * A(3x2) */
        rippa_matmul(Ap, A, AA, 2, 3, 2);
        check("pinv_AA_00", AA[0], 1.0, 1e-8);
        check("pinv_AA_01", AA[1], 0.0, 1e-8);
        check("pinv_AA_10", AA[2], 0.0, 1e-8);
        check("pinv_AA_11", AA[3], 1.0, 1e-8);
    }
}

/* Test pseudo-inverse drops near-zero singular values */
static void test_pinv_truncated(void)
{
    /* A = [1 1; 1 1; 0 0] — rank 1, sigma = [2, 0] */
    double A[6] = {1,1, 1,1, 0,0};
    double Ap[6];
    int rc = rippa_pinv(A, Ap, 3, 2, 1e-6);
    check("pinv_trunc_rc", (double)rc, 0.0, 0.0);
    /* A * A+ * A ≈ A */
    {
        double AApA[6];
        double AAp[9];   /* 3x2 * 2x3 = 3x3 */
        rippa_matmul(A, Ap, AAp, 3, 2, 3);
        rippa_matmul(AAp, A, AApA, 3, 3, 2);
        int i;
        for (i = 0; i < 6; ++i)
            check("pinv_trunc_AApA", AApA[i], A[i], 1e-6);
    }
}

int main(void)
{
    printf("=== RIPPA linear algebra tests ===\n\n");

    printf("test_matvec_identity:\n");
    test_matvec_identity();
    printf("test_lu_solve_2x2:\n");
    test_lu_solve_2x2();
    printf("test_lu_solve_3x3:\n");
    test_lu_solve_3x3();
    printf("test_pinv_identity:\n");
    test_pinv_identity();
    printf("test_pinv_tall:\n");
    test_pinv_tall();
    printf("test_pinv_truncated:\n");
    test_pinv_truncated();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
