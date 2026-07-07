#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rippra/io.h"

static int ntests = 0, npass = 0;

static void test_check(const char *label, int cond)
{
    ntests++;
    if (cond) { npass++; printf("  PASS %s\n", label); }
    else      { printf("  FAIL %s\n", label); }
}

/* Write a byte buffer to a temp file, return the path (caller must free) */
static char *write_temp(const unsigned char *buf, size_t len)
{
    const char *tmpdir = getenv("TEMP");
    if (!tmpdir) tmpdir = ".";
    char *path = (char *)malloc(strlen(tmpdir) + 32);
    sprintf(path, "%s\\ripra_fuzz_%p.tmp", tmpdir, (void *)path);
    FILE *fp = fopen(path, "wb");
    if (fp) { fwrite(buf, 1, len, fp); fclose(fp); }
    return path;
}

/* Build a minimal valid BMP header (54 bytes) */
static void make_bmp_header(unsigned char *hdr, int w, int h, short bpp,
                            unsigned int comp, int pixel_bytes)
{
    memset(hdr, 0, 54);
    hdr[0] = 'B'; hdr[1] = 'M';
    int file_size = 54 + pixel_bytes;
    *(int *)(hdr + 2) = file_size;
    *(int *)(hdr + 10) = 54;
    *(int *)(hdr + 14) = 40;
    *(int *)(hdr + 18) = w;
    *(int *)(hdr + 22) = h;
    *(short *)(hdr + 26) = 1;
    *(short *)(hdr + 28) = bpp;
    *(unsigned int *)(hdr + 30) = comp;
    *(int *)(hdr + 34) = pixel_bytes;
}

static void test_empty_file(void)
{
    printf("\ntest_empty_file:\n");
    /* File that exists but is empty (0 bytes) */
    unsigned char buf[1] = {0};
    char *path = write_temp(buf, 0);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("empty_file_returns_error", ret < 0);
    test_check("data_null", data == NULL);
    remove(path); free(path);
}

static void test_truncated_header(void)
{
    printf("\ntest_truncated_header:\n");
    unsigned char buf[20] = {0};
    buf[0] = 'B'; buf[1] = 'M';
    char *path = write_temp(buf, 20);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("truncated_header_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_wrong_magic(void)
{
    printf("\ntest_wrong_magic:\n");
    unsigned char hdr[54] = {0};
    hdr[0] = 'X'; hdr[1] = 'M';
    char *path = write_temp(hdr, 54);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("wrong_magic_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_zero_width(void)
{
    printf("\ntest_zero_width:\n");
    unsigned char hdr[54] = {0};
    make_bmp_header(hdr, 0, 100, 8, 0, 0);
    char *path = write_temp(hdr, 54);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("zero_width_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_huge_dimensions(void)
{
    printf("\ntest_huge_dimensions:\n");
    unsigned char hdr[54] = {0};
    make_bmp_header(hdr, 20000, 20000, 8, 0, 0);
    char *path = write_temp(hdr, 54);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("huge_dimensions_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_compressed_bmp(void)
{
    printf("\ntest_compressed_bmp:\n");
    unsigned char hdr[54] = {0};
    make_bmp_header(hdr, 10, 10, 8, 1, 0);
    char *path = write_temp(hdr, 54);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("compressed_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_unsupported_bpp(void)
{
    printf("\ntest_unsupported_bpp:\n");
    unsigned char hdr[54] = {0};
    make_bmp_header(hdr, 10, 10, 4, 0, 0);
    char *path = write_temp(hdr, 54);
    if (!path) return;
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &w, &h, &data);
    test_check("unsupported_bpp_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_truncated_pixels(void)
{
    printf("\ntest_truncated_pixels:\n");
    /* Valid 54-byte header for 8-bit 10x10, but only provide a few pixel bytes */
    int w = 10, h = 10;
    int stride = (w + 3) & ~3;
    size_t file_size = 54 + stride * h;
    unsigned char *buf = (unsigned char *)calloc(1, file_size);
    make_bmp_header(buf, w, h, 8, 0, (int)(stride * h));
    /* Write only header + 5 rows worth of pixels */
    char *path = write_temp(buf, 54 + stride * 5);
    if (!path) { free(buf); return; }
    int out_w, out_h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &out_w, &out_h, &data);
    test_check("truncated_pixels_returns_neg4", ret == -4);
    test_check("data_freed_on_truncation", data == NULL);
    free(buf);
    remove(path); free(path);
}

static void test_nonexistent_file(void)
{
    printf("\ntest_nonexistent_file:\n");
    int w, h;
    double *data = NULL;
    int ret = rippa_load_bmp("nonexistent_bmp_file_xyzzy.bmp", &w, &h, &data);
    test_check("nonexistent_returns_neg1", ret == -1);
}

static void test_raw_truncated(void)
{
    printf("\ntest_raw_truncated:\n");
    /* Write a raw file shorter than expected size */
    double vals[2] = {1.0, 2.0};
    char *path = write_temp((unsigned char *)vals, 2 * sizeof(double));
    if (!path) return;
    double *data = NULL;
    int ret = rippa_load_raw(path, 10, 10, &data);
    test_check("raw_truncated_returns_neg3", ret == -3);
    remove(path); free(path);
}

static void test_raw_nonexistent(void)
{
    printf("\ntest_raw_nonexistent:\n");
    double *data = NULL;
    int ret = rippa_load_raw("nonexistent_raw_file_xyzzy.raw", 10, 10, &data);
    test_check("raw_nonexistent_returns_neg1", ret == -1);
}

static void test_bmp_with_config_mismatch(void)
{
    printf("\ntest_bmp_with_config_mismatch:\n");
    /* Create a valid tiny BMP with 8x8, but config says frame_width=648 */
    unsigned char *buf = (unsigned char *)calloc(1, 54 + 64);
    make_bmp_header(buf, 8, 8, 8, 0, 64);
    for (int i = 0; i < 64; i++) buf[54 + i] = (unsigned char)i;
    char *path = write_temp(buf, 54 + 64);
    if (!path) { free(buf); return; }
    rippa_config cfg = {0};
    cfg.frame_width = 648;
    cfg.frame_height = 492;
    double *data = NULL;
    int ret = rippa_load_bmp_with_config(path, &cfg, &data);
    test_check("config_mismatch_returns_neg5", ret == -5);
    free(buf);
    remove(path); free(path);
}

static void test_negative_height(void)
{
    printf("\ntest_negative_height:\n");
    /* Negated height means top-down BMP — still valid if h=-10 */
    int w = 4, h = -4;
    int stride = (w * 1 + 3) & ~3;
    size_t file_size = 54 + stride * 4;
    unsigned char *buf = (unsigned char *)calloc(1, file_size);
    make_bmp_header(buf, w, h, 8, 0, (int)(stride * 4));
    char *path = write_temp(buf, file_size);
    if (!path) { free(buf); return; }
    int out_w, out_h;
    double *data = NULL;
    int ret = rippa_load_bmp(path, &out_w, &out_h, &data);
    test_check("neg_height_parsed_as_valid", ret == 0);
    test_check("neg_height_output_positive", out_h == 4);
    free(buf);
    free(data);
    remove(path); free(path);
}

int main(void)
{
    printf("=== Malformed Input Fuzz Tests ===\n");

    test_empty_file();
    test_truncated_header();
    test_wrong_magic();
    test_zero_width();
    test_huge_dimensions();
    test_compressed_bmp();
    test_unsupported_bpp();
    test_truncated_pixels();
    test_nonexistent_file();
    test_raw_truncated();
    test_raw_nonexistent();
    test_bmp_with_config_mismatch();
    test_negative_height();

    printf("\n=== %d/%d tests passed ===\n", npass, ntests);
    return (npass == ntests) ? 0 : 1;
}
