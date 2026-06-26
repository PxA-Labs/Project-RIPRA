/* Minimal test to debug the toolchain */
#include <stdio.h>
#include "rippra/io.h"

int main(int argc, char **argv)
{
    printf("test_minimal starting\n");
    fflush(stdout);

    const char *base = (argc > 1) ? argv[1] : ".";
    printf("base = %s\n", base);
    fflush(stdout);

    char path[512];
    snprintf(path, sizeof(path), "%s/config/system.conf", base);
    printf("config path = %s\n", path);
    fflush(stdout);

    rippa_config cfg;
    int ret = rippa_config_load(&cfg, path);
    printf("config load returned %d\n", ret);
    fflush(stdout);

    if (ret == 0) {
        printf("Config: %dx%d, totlenses=%d\n",
               cfg.frame_width, cfg.frame_height, cfg.totlenses);
        fflush(stdout);
    }

    printf("test_minimal done\n");
    return 0;
}
