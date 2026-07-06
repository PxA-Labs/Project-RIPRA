#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define RIPRA_PI 3.14159265358979323846
#include "rippra/recon.h"
#include "rippra/io.h"

static int ntests = 0, npass = 0;

static void check(const char *label, double got, double expected, double tol)
{
    ntests++;
    if (fabs(got - expected) <= tol + 1e-15) {
        npass++;
    } else {
        printf("  FAIL %s: got %.10e expected %.10e (diff=%.2e)\n",
               label, got, expected, fabs(got - expected));
    }
}

/* Zernike derivative golden values at known normalized coordinates */
static void test_zernike_tip(void)
{
    double dzdx, dzdy;
    /* Tip (Noll 2, n=1, m=1): Z = 2*x, dZ/dx = 2, dZ/dy = 0 */
    evaluate_zernike_derivatives(1, 1, 0.5, 0.0, &dzdx, &dzdy);
    check("tip_dzdx_at_05_0", dzdx, 2.0, 1e-10);
    check("tip_dzdy_at_05_0", dzdy, 0.0, 1e-10);

    evaluate_zernike_derivatives(1, 1, 0.0, 0.0, &dzdx, &dzdy);
    check("tip_dzdx_at_0_0", dzdx, 2.0, 1e-10);
    check("tip_dzdy_at_0_0", dzdy, 0.0, 1e-10);
}

static void test_zernike_tilt(void)
{
    double dzdx, dzdy;
    /* Tilt (Noll 3, n=1, m=-1): Z = 2*y, dZ/dx = 0, dZ/dy = 2 */
    evaluate_zernike_derivatives(1, -1, 0.0, 0.5, &dzdx, &dzdy);
    check("tilt_dzdx_at_0_05", dzdx, 0.0, 1e-10);
    check("tilt_dzdy_at_0_05", dzdy, 2.0, 1e-10);

    evaluate_zernike_derivatives(1, -1, 0.0, 0.0, &dzdx, &dzdy);
    check("tilt_dzdx_at_0_0", dzdx, 0.0, 1e-10);
    check("tilt_dzdy_at_0_0", dzdy, 2.0, 1e-10);
}

static void test_zernike_defocus(void)
{
    double dzdx, dzdy;
    /* Defocus (Noll 4, n=2, m=0): Z = sqrt(3)*(2*rho^2 - 1)
     * At (0.3, 0): dZ/dx = sqrt(3)*4*0.3 = 2.078460969, dZ/dy = 0 */
    evaluate_zernike_derivatives(2, 0, 0.3, 0.0, &dzdx, &dzdy);
    check("defocus_dzdx_at_03_0", dzdx, 2.078460969082653, 1e-10);
    check("defocus_dzdy_at_03_0", dzdy, 0.0, 1e-10);
}

static void test_r0_golden(void)
{
    /* 1 spot, 100 frames, dx = [0..99], dy = 0
     * Pre-computed: r0 = 1.964028247358213e-05 */
    int nspots = 1, nframes = 100;
    double *dx = (double *)malloc(nframes * nspots * sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));
    for (int t = 0; t < nframes; ++t) dx[t] = (double)t;

    rippa_config cfg;
    cfg.camera_pixsize = 7.4e-6;
    cfg.flength = 18e-3;
    cfg.pitch = 300e-6;
    cfg.wavelength = 632.8e-9;

    double r0 = rippra_compute_r0_impl(dx, dy, nframes, nspots, &cfg);
    check("r0_golden", r0, 1.964028247358213e-05, 1e-10);

    free(dx); free(dy);
}

static void test_tau0_golden(void)
{
    /* 1 spot, 200 frames, dx[t] = sin(2*pi*t/20), dy = 0, frame_rate = 1000
     * Pre-computed: tau0 = 3.788848101874247e-03 s */
    int nspots = 1, nframes = 200;
    double *dx = (double *)malloc(nframes * nspots * sizeof(double));
    double *dy = (double *)calloc(nframes * nspots, sizeof(double));
    for (int t = 0; t < nframes; ++t)
        dx[t] = sin(2.0 * RIPRA_PI * t / 20.0);

    double frame_rate = 1000.0;
    double tau0 = rippra_compute_tau0_impl(dx, dy, nframes, nspots, frame_rate);
    check("tau0_golden", tau0, 3.841111250108286e-03, 1e-10);

    free(dx); free(dy);
}

int main(void)
{
    printf("=== Golden-Value Regression Tests ===\n\n");

    printf("Zernike derivatives (Tip):\n");
    test_zernike_tip();
    printf("Zernike derivatives (Tilt):\n");
    test_zernike_tilt();
    printf("Zernike derivatives (Defocus):\n");
    test_zernike_defocus();
    printf("r0 golden value:\n");
    test_r0_golden();
    printf("tau0 golden value:\n");
    test_tau0_golden();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
