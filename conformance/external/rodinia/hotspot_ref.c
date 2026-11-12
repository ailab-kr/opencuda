/* Serial reference of the CUDA hotspot update (clamped edges), written from
   the kernel's formula, to check the CPU-path result independently. */
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
    if (argc != 5) return 2; /* hotspot_ref <n> <iterations> <temp> <power> */
    int n = atoi(argv[1]), iters = atoi(argv[2]);
    float *t = malloc(sizeof(float) * n * n), *p = malloc(sizeof(float) * n * n),
          *o = malloc(sizeof(float) * n * n);
    FILE *ft = fopen(argv[3], "r"), *fp = fopen(argv[4], "r");
    if (ft == NULL || fp == NULL) return 1;
    for (int i = 0; i < n * n; ++i)
        if (fscanf(ft, "%f", &t[i]) != 1 || fscanf(fp, "%f", &p[i]) != 1) return 1;
    float t_chip = 0.0005f, chip_height = 0.016f, chip_width = 0.016f, amb_temp = 80.0f;
    float grid_height = chip_height / n, grid_width = chip_width / n;
    float Cap = 0.5 * 1.75e6 * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * 100 * t_chip * grid_height);
    float Ry = grid_height / (2.0 * 100 * t_chip * grid_width);
    float Rz = t_chip / (100 * grid_height * grid_width);
    float max_slope = 3.0e6 / (0.5 * t_chip * 1.75e6);
    float step = 0.001 / max_slope;
    float sdc = step / Cap, Rx_1 = 1 / Rx, Ry_1 = 1 / Ry, Rz_1 = 1 / Rz;
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                int N = r > 0 ? r - 1 : 0, S = r < n - 1 ? r + 1 : n - 1;
                int W = c > 0 ? c - 1 : 0, E = c < n - 1 ? c + 1 : n - 1;
                float x = t[r * n + c];
                o[r * n + c] = x + sdc * (p[r * n + c] + (t[S * n + c] + t[N * n + c] - 2.0 * x) * Ry_1 +
                                          (t[r * n + E] + t[r * n + W] - 2.0 * x) * Rx_1 +
                                          (amb_temp - x) * Rz_1);
            }
        float *tmp = t; t = o; o = tmp;
    }
    for (int i = 0; i < n * n; ++i) printf("%d\t%g\n", i, t[i]);
    return 0;
}
