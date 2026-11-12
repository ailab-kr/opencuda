// Types for cu2msl_features.cu, in an included header (as many programs are
// organised). Field sizes are mixed on purpose: the host packs these structs
// with the C layout, the generated MSL must read the same bytes.
#pragma once

typedef struct {
    const float *src;   // 0
    float *dst;         // 8
    char mode;          // 16
    short tag;          // 18
    float scale;        // 20
    float2 shift;       // 24 (8-byte aligned)
} Params;               // 32

struct Pair {
    float a;            // 0
    unsigned char b;    // 4
};                      // 8
