/*
 * leukocyte_avi.c -- a small synthetic input video for Rodinia's leukocyte
 * (the suite's sample video is not part of its source tree).
 *
 *   leukocyte_avi <out.avi> <frames>
 *
 * Built together with the program's own avilib.c, so the file is written
 * the way the program reads it: 640x480 frames of 8-bit grey. Each frame
 * is a textured background with a few dark rings (cells) of radius 9 to 12
 * inside the band the program looks at (rows 110 to 328), drifting a pixel
 * per frame. Fixed seed.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "avilib.h"

#define W 640
#define H 480

static unsigned rnd_state = 12345u;
static unsigned rnd(void)
{
    rnd_state = rnd_state * 1103515245u + 12345u;
    return (rnd_state >> 16) & 0x7fffu;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <out.avi> <frames>\n", argv[0]);
        return 2;
    }
    int frames = atoi(argv[1 + 1]);
    avi_t *avi = AVI_open_output_file(argv[1]);
    if (avi == NULL) {
        AVI_print_error("AVI_open_output_file");
        return 1;
    }
    AVI_set_video(avi, W, H, 30.0, "Y800");
    enum { CELLS = 6 };
    double cx[CELLS], cy[CELLS], r[CELLS];
    for (int c = 0; c < CELLS; ++c) {
        cx[c] = 60 + c * 95 + (rnd() % 20);
        cy[c] = 480 - (150 + (rnd() % 140)); /* the program flips the frame */
        r[c] = 9 + rnd() % 4;
    }
    static unsigned char img[W * H];
    for (int f = 0; f < frames; ++f) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                double v = 150 + 20 * sin(x * 0.05) * cos(y * 0.07) + (rnd() % 9);
                for (int c = 0; c < CELLS; ++c) {
                    double d = hypot(x - (cx[c] + f), y - cy[c]);
                    if (fabs(d - r[c]) < 2.0) v -= 70;
                    else if (d < r[c]) v += 15;
                }
                img[y * W + x] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }
        if (AVI_write_frame(avi, (char *)img, W * H, 1) != 0) {
            AVI_print_error("AVI_write_frame");
            return 1;
        }
    }
    AVI_close(avi);
    return 0;
}
