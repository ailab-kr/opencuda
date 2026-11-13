/*
 * ocu_texture_desc.h -- what a texture fetch reads: the descriptor the
 * runtime fills (texture objects, bound texture references) and device code
 * samples (ocu_texture_fetch.h). This project's own layout, the same in host
 * C and in device code. Not part of the CUDA API.
 *
 * A texture object is the device address of one of these, in device memory,
 * so device code on any backend that runs kernels reads it with ordinary
 * loads. A texture reference carries one inside (textureReference::ocu_bound).
 */
#ifndef OCU_TEXTURE_DESC_H
#define OCU_TEXTURE_DESC_H

/* Component kinds of a texel. */
enum {
    OCU_TEXK_SIGNED = 0,   /* signed integer */
    OCU_TEXK_UNSIGNED = 1, /* unsigned integer */
    OCU_TEXK_FLOAT = 2,    /* binary32 */
    OCU_TEXK_HALF = 3,     /* binary16, promoted to binary32 before filtering */
    OCU_TEXK_RAW = 4       /* unknown (channel kind None): read as the fetch type */
};

/* What the texels are in. */
enum {
    OCU_TEXR_NONE = 0,    /* unbound reference */
    OCU_TEXR_LINEAR = 1,  /* linear memory: tex1Dfetch, no addressing, no filtering */
    OCU_TEXR_PITCH2D = 2, /* pitched 2D linear memory */
    OCU_TEXR_ARRAY = 3    /* CUDA array */
};

struct ocu_tex_desc {
    const void *base;           /* texel (0, 0, 0); NULL when unbound */
    unsigned long long bytes;   /* bytes that may be read from base */
    unsigned long long pitch;   /* bytes from one row to the next */
    unsigned long long slice;   /* bytes from one slice (or layer) to the next */
    unsigned int width;         /* texels per row */
    unsigned int height;        /* rows per slice (1 for 1D) */
    unsigned int depth;         /* slices (3D) or layers (layered); 1 otherwise */
    unsigned int texel_bytes;   /* 0 with OCU_TEXK_RAW */
    unsigned char resource;     /* OCU_TEXR_* */
    unsigned char layered;      /* depth counts layers, not a third coordinate */
    unsigned char channels;     /* 1, 2 or 4 (0 with OCU_TEXK_RAW) */
    unsigned char channel_bytes;/* 1, 2 or 4 */
    unsigned char kind;         /* OCU_TEXK_* */
    unsigned char always_norm;  /* a ...Normalized... channel kind: integers always read as [0,1] / [-1,1] */
    /* Sampling: texture objects only (a reference uses its own members). */
    unsigned char address[3];   /* cudaTextureAddressMode */
    unsigned char filter;       /* cudaTextureFilterMode */
    unsigned char read_norm;    /* cudaReadModeNormalizedFloat */
    unsigned char normalized;   /* normalized coordinates */
    unsigned char pad_[2];
    float border[4];            /* cudaTextureDesc::borderColor */
};

#endif /* OCU_TEXTURE_DESC_H */
