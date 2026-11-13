// Smoke test for textures: CUDA arrays, texture objects and (CPU path)
// texture references. One ordinary CUDA program written by the
// implementer; the full tests are written separately.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)

// Texture references exist on the CPU path only (removed in CUDA 12).
#if defined(OCU_CPU_PATH)
#define HAVE_REFS 1
#else
#define HAVE_REFS 0
#endif

struct Pair { int a, b; };

#if HAVE_REFS

texture<float, 1, cudaReadModeElementType> t_lin;
texture<Pair> t_pair;
texture<unsigned short, 2, cudaReadModeElementType> t_img;
texture<unsigned char, 1, cudaReadModeNormalizedFloat> t_norm;

__global__ void ref_kernel(float *f, Pair *p, unsigned short *s, float *nf)
{
    int i = threadIdx.x;
    f[i] = tex1Dfetch(t_lin, i - 1);         // i = 0 reads index -1: zero
    p[i] = tex1Dfetch(t_pair, i);
    s[i] = tex2D(t_img, i - 2, 1);            // clamp at the left edge
    nf[i] = tex1Dfetch(t_norm, i);
}

#endif

__global__ void obj_kernel(cudaTextureObject_t lin2d, cudaTextureObject_t wrap,
                           cudaTextureObject_t border, cudaTextureObject_t vol, float *out)
{
    out[0] = tex2D<float>(lin2d, 1.0f, 0.5f);   // halfway between texel 0 and 1 of row 0
    out[1] = tex2D<float>(lin2d, 1.25f, 0.5f);  // alpha = 0.75
    out[2] = tex1D<float>(wrap, 1.25f);         // normalized, wrap: 0.25 -> texel 1 of 4
    out[3] = tex1D<float>(wrap, -0.25f);        // 0.75 -> texel 3
    out[4] = tex1D<float>(border, -0.5f);       // out of range: border colour
    out[5] = tex3D<float>(vol, 1.5f, 0.5f, 1.5f);
}

int main()
{
#if HAVE_REFS
    // ---- references ----
    float *dl;
    Pair *dp;
    unsigned char *dn;
    std::vector<float> hl = {1, 2, 3, 4};
    std::vector<Pair> hp = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    unsigned char hn[4] = {0, 51, 255, 128};
    OK(cudaMalloc(&dl, 16));
    OK(cudaMalloc(&dp, sizeof(Pair) * 4));
    OK(cudaMalloc(&dn, 4));
    OK(cudaMemcpy(dl, hl.data(), 16, cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dp, hp.data(), sizeof(Pair) * 4, cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dn, hn, 4, cudaMemcpyHostToDevice));
    OK(cudaBindTexture(0, t_lin, dl, 16));
    OK(cudaBindTexture(nullptr, t_pair, dp, sizeof(Pair) * 4));
    OK(cudaBindTexture(0, t_norm, dn, 4));

    unsigned short img[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    cudaArray *arr;
    OK(cudaMallocArray(&arr, &t_img.channelDesc, 4, 3));
    OK(cudaMemcpy2DToArray(arr, 0, 0, img, 8, 8, 3, cudaMemcpyHostToDevice));
    OK(cudaBindTextureToArray(t_img, arr));

    float *df, *dnf;
    Pair *dpo;
    unsigned short *ds;
    OK(cudaMalloc(&df, 16));
    OK(cudaMalloc(&dnf, 16));
    OK(cudaMalloc(&dpo, sizeof(Pair) * 4));
    OK(cudaMalloc(&ds, 8));
    ref_kernel<<<1, 4>>>(df, dpo, ds, dnf);
    OK(cudaDeviceSynchronize());
    float f[4], nf[4];
    Pair po[4];
    unsigned short s[4];
    OK(cudaMemcpy(f, df, 16, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(nf, dnf, 16, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(po, dpo, sizeof po, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(s, ds, 8, cudaMemcpyDeviceToHost));
    CHECK(f[0] == 0 && f[1] == 1 && f[3] == 3);
    CHECK(po[2].a == 5 && po[3].b == 8);
    CHECK(s[0] == 5 && s[1] == 5 && s[2] == 5 && s[3] == 6);
    CHECK(nf[0] == 0.0f && nf[1] == 0.2f && nf[2] == 1.0f);
    OK(cudaUnbindTexture(t_lin));
    unsigned short back[3][4];
    OK(cudaMemcpy2DFromArray(back, 8, arr, 0, 0, 8, 3, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(back, img, sizeof img) == 0);

#endif

    // ---- objects ----
    float *p2d;
    size_t pitch;
    float rows[2][2] = {{0, 8}, {16, 24}};
    OK(cudaMallocPitch(&p2d, &pitch, 8, 2));
    OK(cudaMemcpy2D(p2d, pitch, rows, 8, 8, 2, cudaMemcpyHostToDevice));
    cudaResourceDesc rd;
    std::memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypePitch2D;
    rd.res.pitch2D.devPtr = p2d;
    rd.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    rd.res.pitch2D.width = 2;
    rd.res.pitch2D.height = 2;
    rd.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc td;
    std::memset(&td, 0, sizeof td);
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    cudaTextureObject_t lin2d = 0, wrap = 0, border = 0, vol = 0;
    OK(cudaCreateTextureObject(&lin2d, &rd, &td, nullptr));

    cudaChannelFormatDesc fd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t a1;
    float line[4] = {10, 20, 30, 40};
    OK(cudaMallocArray(&a1, &fd, 4));
    OK(cudaMemcpyToArray(a1, 0, 0, line, 16, cudaMemcpyHostToDevice));
    std::memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = a1;
    std::memset(&td, 0, sizeof td);
    td.addressMode[0] = cudaAddressModeWrap;
    td.normalizedCoords = 1;
    OK(cudaCreateTextureObject(&wrap, &rd, &td, nullptr));
    td.addressMode[0] = cudaAddressModeBorder;
    td.normalizedCoords = 0;
    td.borderColor[0] = -7.0f;
    OK(cudaCreateTextureObject(&border, &rd, &td, nullptr));

    cudaArray_t a3;
    float cube[2][2][2] = {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};
    OK(cudaMalloc3DArray(&a3, &fd, make_cudaExtent(2, 2, 2)));
    cudaMemcpy3DParms cp;
    std::memset(&cp, 0, sizeof cp);
    cp.srcPtr = make_cudaPitchedPtr(cube, 8, 2, 2);
    cp.dstArray = a3;
    cp.extent = make_cudaExtent(2, 2, 2);
    cp.kind = cudaMemcpyHostToDevice;
    OK(cudaMemcpy3D(&cp));
    rd.res.array.array = a3;
    std::memset(&td, 0, sizeof td);
    OK(cudaCreateTextureObject(&vol, &rd, &td, nullptr));

    float *dout, out[6];
    OK(cudaMalloc(&dout, sizeof out));
    obj_kernel<<<1, 1>>>(lin2d, wrap, border, vol, dout);
    OK(cudaMemcpy(out, dout, sizeof out, cudaMemcpyDeviceToHost));
    CHECK(out[0] == 4.0f);
    CHECK(out[1] == 6.0f);
    CHECK(out[2] == 20.0f);
    CHECK(out[3] == 40.0f);
    CHECK(out[4] == -7.0f);
    CHECK(out[5] == 5.0f);

    cudaTextureDesc got;
    OK(cudaGetTextureObjectTextureDesc(&got, border));
    CHECK(got.addressMode[0] == cudaAddressModeBorder && got.borderColor[0] == -7.0f);
    cudaChannelFormatDesc gd;
    cudaExtent ge;
    unsigned gf;
    OK(cudaArrayGetInfo(&gd, &ge, &gf, a3));
    CHECK(gd.x == 32 && gd.f == cudaChannelFormatKindFloat && ge.depth == 2);
    CHECK(cudaDestroyTextureObject(12345) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    for (cudaTextureObject_t o : {lin2d, wrap, border, vol}) OK(cudaDestroyTextureObject(o));
#if HAVE_REFS
    OK(cudaFreeArray(arr));
#endif
    OK(cudaFreeArray(a1));
    OK(cudaFreeArray(a3));

    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
