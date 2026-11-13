// Independent tests for textures and CUDA arrays (card runtime-texture,
// "시험 목록"). Expectations come from the public CUDA C++ Programming Guide
// (texture memory, appendix "Texture Fetching") and the Runtime API
// reference; the host reference sampler below is written from that appendix.
// Runs on the CPU path (also texture references) and on the NVIDIA path
// (texture objects only).
//
// XCHECK marks a check that disagrees with the document or the card (or
// whose expected value the documents leave open): it is counted and printed
// as EXPECTED-FAIL and does not fail the test, so the disagreement stays
// visible without hiding other results.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail, g_checks, g_xfail;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) CHECK((call) == cudaSuccess)
#define ERR(call, e) do { cudaError_t r_ = (call); ++g_checks; if (r_ != (e)) { std::printf("FAIL %s:%d: %s -> %d (%s), expected %d\n", __FILE__, __LINE__, #call, (int)r_, cudaGetErrorString(r_), (int)(e)); ++g_fail; } } while (0)
#define XCHECK(c, why) do { ++g_checks; if (!(c)) { std::printf("EXPECTED-FAIL %s:%d: %s -- %s\n", __FILE__, __LINE__, #c, why); ++g_xfail; } } while (0)
#define CHECK_EQF(got, want) do { float g_ = (got), w_ = (want); ++g_checks; if (!(g_ == w_)) { std::printf("FAIL %s:%d: %s = %.9g, expected %.9g\n", __FILE__, __LINE__, #got, (double)g_, (double)w_); ++g_fail; } } while (0)
#define CHECK_NEAR(got, want, tol) do { float g_ = (got), w_ = (want); ++g_checks; if (!(std::fabs(g_ - w_) <= (tol))) { std::printf("FAIL %s:%d: %s = %.9g, expected %.9g (tol %g)\n", __FILE__, __LINE__, #got, (double)g_, (double)w_, (double)(tol)); ++g_fail; } } while (0)

#if defined(OCU_CPU_PATH)
#define HAVE_REFS 1
#else
#define HAVE_REFS 0
#endif

// ---------------------------------------------------------------- kernels

template <class T> __global__ void k_t1(cudaTextureObject_t t, const float *x, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex1D<T>(t, x[i]);
}
template <class T> __global__ void k_t2(cudaTextureObject_t t, const float *x, const float *y, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex2D<T>(t, x[i], y[i]);
}
template <class T> __global__ void k_t3(cudaTextureObject_t t, const float *x, const float *y, const float *z, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex3D<T>(t, x[i], y[i], z[i]);
}
template <class T> __global__ void k_t1l(cudaTextureObject_t t, const float *x, const int *l, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex1DLayered<T>(t, x[i], l[i]);
}
template <class T> __global__ void k_t2l(cudaTextureObject_t t, const float *x, const float *y, const int *l, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex2DLayered<T>(t, x[i], y[i], l[i]);
}
template <class T> __global__ void k_tf(cudaTextureObject_t t, const int *x, T *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = tex1Dfetch<T>(t, x[i]);
}

template <class T> struct Dev {
    T *p = nullptr;
    size_t n;
    explicit Dev(size_t n_) : n(n_) { OK(cudaMalloc((void **)&p, (n ? n : 1) * sizeof(T))); }
    Dev(const std::vector<T> &v) : n(v.size())
    {
        OK(cudaMalloc((void **)&p, (n ? n : 1) * sizeof(T)));
        OK(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const
    {
        std::vector<T> v(n);
        OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
static unsigned grid(size_t n) { return (unsigned)((n + 63) / 64); }

template <class T> static std::vector<T> s1(cudaTextureObject_t t, const std::vector<float> &x)
{
    Dev<float> dx(x); Dev<T> o(x.size());
    k_t1<T><<<grid(x.size()), 64>>>(t, dx.p, o.p, (int)x.size());
    OK(cudaDeviceSynchronize());
    return o.get();
}
template <class T> static std::vector<T> s2(cudaTextureObject_t t, const std::vector<float> &x, const std::vector<float> &y)
{
    Dev<float> dx(x), dy(y); Dev<T> o(x.size());
    k_t2<T><<<grid(x.size()), 64>>>(t, dx.p, dy.p, o.p, (int)x.size());
    OK(cudaDeviceSynchronize());
    return o.get();
}
template <class T> static std::vector<T> s3(cudaTextureObject_t t, const std::vector<float> &x, const std::vector<float> &y, const std::vector<float> &z)
{
    Dev<float> dx(x), dy(y), dz(z); Dev<T> o(x.size());
    k_t3<T><<<grid(x.size()), 64>>>(t, dx.p, dy.p, dz.p, o.p, (int)x.size());
    OK(cudaDeviceSynchronize());
    return o.get();
}
template <class T> static std::vector<T> sf(cudaTextureObject_t t, const std::vector<int> &x)
{
    Dev<int> dx(x); Dev<T> o(x.size());
    k_tf<T><<<grid(x.size()), 64>>>(t, dx.p, o.p, (int)x.size());
    OK(cudaDeviceSynchronize());
    return o.get();
}

// ---------------------------------------------------------------- helpers

static const float NANF = std::nanf("");

struct TD {
    cudaTextureAddressMode am[3] = {cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp};
    cudaTextureFilterMode filter = cudaFilterModePoint;
    cudaTextureReadMode read = cudaReadModeElementType;
    int norm = 0;
    float border[4] = {0, 0, 0, 0};
};
static cudaTextureDesc mkdesc(const TD &t)
{
    cudaTextureDesc d;
    std::memset(&d, 0, sizeof d);
    for (int i = 0; i < 3; ++i) d.addressMode[i] = t.am[i];
    d.filterMode = t.filter;
    d.readMode = t.read;
    d.normalizedCoords = t.norm;
    for (int i = 0; i < 4; ++i) d.borderColor[i] = t.border[i];
    return d;
}
static TD mk(cudaTextureAddressMode m, cudaTextureFilterMode f, int norm, float border = 0)
{
    TD t;
    t.am[0] = t.am[1] = t.am[2] = m;
    t.filter = f;
    t.norm = norm;
    t.border[0] = border;
    return t;
}

// An array with the data (packed, `eb` bytes per texel). h = 0: 1D; d = 0: 2D
// (unless layered: then d is the layer count).
static cudaArray_t mkarr(const cudaChannelFormatDesc &cd, size_t w, size_t h, size_t d, unsigned flags,
                         const void *data, size_t eb)
{
    cudaArray_t a = nullptr;
    if (h == 0 && d == 0) {
        OK(cudaMallocArray(&a, &cd, w, 0, flags));
        if (a) OK(cudaMemcpyToArray(a, 0, 0, data, w * eb, cudaMemcpyHostToDevice));
    } else if (d == 0 && !(flags & cudaArrayLayered)) {
        OK(cudaMallocArray(&a, &cd, w, h, flags));
        if (a) OK(cudaMemcpy2DToArray(a, 0, 0, data, w * eb, w * eb, h, cudaMemcpyHostToDevice));
    } else {
        cudaExtent e = make_cudaExtent(w, h, d);
        OK(cudaMalloc3DArray(&a, &cd, e, flags));
        if (a) {
            cudaMemcpy3DParms p;
            std::memset(&p, 0, sizeof p);
            p.srcPtr = make_cudaPitchedPtr((void *)data, w * eb, w, h ? h : 1);
            p.dstArray = a;
            p.extent = make_cudaExtent(w, h ? h : 1, d);
            p.kind = cudaMemcpyHostToDevice;
            OK(cudaMemcpy3D(&p));
        }
    }
    return a;
}
static cudaTextureObject_t mkobj(cudaArray_t a, const TD &t)
{
    cudaResourceDesc r;
    std::memset(&r, 0, sizeof r);
    r.resType = cudaResourceTypeArray;
    r.res.array.array = a;
    cudaTextureDesc d = mkdesc(t);
    cudaTextureObject_t o = 0;
    OK(cudaCreateTextureObject(&o, &r, &d, nullptr));
    return o;
}

// ------------------------------------------------- host reference sampler
// Written from the appendix "Texture Fetching": point = T[floor(x)]; linear
// uses xB = x - 0.5, i = floor(xB), alpha = frac(xB) kept with 8 fractional
// bits (round to nearest, ties up); normalized coordinates are wrapped /
// mirrored first and then scaled by the size; wrap and mirror act as clamp
// when coordinates are not normalized; border gives the border value for
// every texel outside the texture. `c` is the coordinate as the program
// passes it (normalized or not).
struct Tap { int idx; double w; };   // idx -1: border
static void axis(double c, int N, cudaTextureAddressMode mode, int norm, bool linear, Tap *taps, int &nt)
{
    double x;
    if (!(c == c)) c = 0;            // NaN: index 0 (card item 6)
    if (norm) {
        if (mode == cudaAddressModeWrap) c = c - std::floor(c);
        else if (mode == cudaAddressModeMirror) {
            double f = std::floor(c), fr = c - f;
            c = (std::fmod(f, 2.0) == 0) ? fr : 1.0 - fr;
        }
        x = c * N;
    } else {
        x = c;
        if (mode == cudaAddressModeWrap || mode == cudaAddressModeMirror) mode = cudaAddressModeClamp;
    }
    auto resolve = [&](long long i) -> int {
        switch (mode) {
        case cudaAddressModeClamp: return (int)(i < 0 ? 0 : (i > N - 1 ? N - 1 : i));
        case cudaAddressModeBorder: return (i < 0 || i >= N) ? -1 : (int)i;
        case cudaAddressModeWrap: return (int)(((i % N) + N) % N);
        default: {
            long long m = ((i % (2 * N)) + 2 * N) % (2 * N);
            return (int)(m < N ? m : 2 * N - 1 - m);
        }
        }
    };
    x = std::fmax(std::fmin(x, 1e15), -1e15);
    if (!linear) {
        taps[0] = {resolve((long long)std::floor(x)), 1.0};
        nt = 1;
        return;
    }
    double xb = x - 0.5;
    double i = std::floor(xb), a = xb - i;
    a = std::floor(a * 256.0 + 0.5) / 256.0;
    taps[0] = {resolve((long long)i), 1.0 - a};
    taps[1] = {resolve((long long)i + 1), a};
    nt = 2;
}
static float refsample(const std::vector<float> &T, int W, int H, int D, double x, double y, double z,
                       const TD &td, int dims, float border)
{
    Tap tx[2], ty[2], tz[2];
    int nx = 0, ny = 1, nz = 1;
    bool lin = td.filter == cudaFilterModeLinear;
    axis(x, W, td.am[0], td.norm, lin, tx, nx);
    ty[0] = {0, 1};
    tz[0] = {0, 1};
    if (dims >= 2) axis(y, H, td.am[1], td.norm, lin, ty, ny);
    if (dims >= 3) axis(z, D, td.am[2], td.norm, lin, tz, nz);
    double sum = 0;
    for (int c = 0; c < nz; ++c)
        for (int b = 0; b < ny; ++b)
            for (int a = 0; a < nx; ++a) {
                double v;
                if (tx[a].idx < 0 || ty[b].idx < 0 || tz[c].idx < 0) v = border;
                else v = T[((size_t)tz[c].idx * H + ty[b].idx) * W + tx[a].idx];
                sum += tx[a].w * ty[b].w * tz[c].w * v;
            }
    return (float)sum;
}

// Linear filtering in border mode with a coordinate outside [0, N): the guide
// says a fetch outside the texture returns the border colour; the card says
// outside neighbours blend in the border colour. The implementation returns
// the pure border colour. Such samples are reported as EXPECTED-FAIL.
static bool amb_axis(const TD &td, int ax, double c, int N)
{
    if (td.filter != cudaFilterModeLinear || td.am[ax] != cudaAddressModeBorder) return false;
    double x = td.norm ? c * N : c;
    return x < 0 || x >= N;
}

static uint32_t rng_s = 12345;
static uint32_t rnd() { rng_s = rng_s * 1664525u + 1013904223u; return rng_s >> 8; }
// A coordinate k/1024 that is never an exact tie of the 8-bit weight.
static float gridcoord(int q)
{
    if (((q % 4) + 4) % 4 == 2) ++q;
    return q / 1024.0f;
}

// ============================================================ 1: channel descs

static void test_channel_desc()
{
    cudaChannelFormatDesc d = cudaCreateChannelDesc(8, 16, 32, 5, cudaChannelFormatKindSigned);
    CHECK(d.x == 8 && d.y == 16 && d.z == 32 && d.w == 5 && d.f == cudaChannelFormatKindSigned);
    d = cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone);
    CHECK(d.x == 0 && d.f == cudaChannelFormatKindNone);
#define CD(T, bits, kind, nc)                                                                          \
    do {                                                                                               \
        cudaChannelFormatDesc q = cudaCreateChannelDesc<T>();                                          \
        CHECK(q.x == (bits) && q.f == (kind));                                                         \
        CHECK(q.y == ((nc) >= 2 ? (bits) : 0) && q.z == ((nc) >= 4 ? (bits) : 0) && q.w == ((nc) >= 4 ? (bits) : 0)); \
    } while (0)
    CD(signed char, 8, cudaChannelFormatKindSigned, 1);
    CD(unsigned char, 8, cudaChannelFormatKindUnsigned, 1);
    CD(char1, 8, cudaChannelFormatKindSigned, 1);
    CD(char2, 8, cudaChannelFormatKindSigned, 2);
    CD(char4, 8, cudaChannelFormatKindSigned, 4);
    CD(uchar1, 8, cudaChannelFormatKindUnsigned, 1);
    CD(uchar2, 8, cudaChannelFormatKindUnsigned, 2);
    CD(uchar4, 8, cudaChannelFormatKindUnsigned, 4);
    CD(short, 16, cudaChannelFormatKindSigned, 1);
    CD(short2, 16, cudaChannelFormatKindSigned, 2);
    CD(short4, 16, cudaChannelFormatKindSigned, 4);
    CD(unsigned short, 16, cudaChannelFormatKindUnsigned, 1);
    CD(ushort2, 16, cudaChannelFormatKindUnsigned, 2);
    CD(ushort4, 16, cudaChannelFormatKindUnsigned, 4);
    CD(int, 32, cudaChannelFormatKindSigned, 1);
    CD(int2, 32, cudaChannelFormatKindSigned, 2);
    CD(int4, 32, cudaChannelFormatKindSigned, 4);
    CD(unsigned int, 32, cudaChannelFormatKindUnsigned, 1);
    CD(uint2, 32, cudaChannelFormatKindUnsigned, 2);
    CD(uint4, 32, cudaChannelFormatKindUnsigned, 4);
    CD(float, 32, cudaChannelFormatKindFloat, 1);
    CD(float1, 32, cudaChannelFormatKindFloat, 1);
    CD(float2, 32, cudaChannelFormatKindFloat, 2);
    CD(float4, 32, cudaChannelFormatKindFloat, 4);
    CD(long, (int)sizeof(long) * 8, cudaChannelFormatKindSigned, 1);
    CD(unsigned long, (int)sizeof(long) * 8, cudaChannelFormatKindUnsigned, 1);
    {   // char: either kind, 8 bits
        cudaChannelFormatDesc q = cudaCreateChannelDesc<char>();
        CHECK(q.x == 8 && q.y == 0 && (q.f == cudaChannelFormatKindSigned || q.f == cudaChannelFormatKindUnsigned));
    }
    struct Unknown { int a; };
    d = cudaCreateChannelDesc<Unknown>();
    CHECK(d.x == 0 && d.y == 0 && d.z == 0 && d.w == 0 && d.f == cudaChannelFormatKindNone);
    d = cudaCreateChannelDescHalf();
    CHECK(d.x == 16 && d.y == 0 && d.z == 0 && d.w == 0 && d.f == cudaChannelFormatKindFloat);
    d = cudaCreateChannelDescHalf1();
    CHECK(d.x == 16 && d.y == 0 && d.f == cudaChannelFormatKindFloat);
    d = cudaCreateChannelDescHalf2();
    CHECK(d.x == 16 && d.y == 16 && d.z == 0 && d.w == 0 && d.f == cudaChannelFormatKindFloat);
    d = cudaCreateChannelDescHalf4();
    CHECK(d.x == 16 && d.y == 16 && d.z == 16 && d.w == 16 && d.f == cudaChannelFormatKindFloat);
}

// ============================================================ 2: arrays

static bool same_desc(const cudaChannelFormatDesc &a, const cudaChannelFormatDesc &b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w && a.f == b.f;
}

static void test_array_alloc()
{
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
    cudaArray_t a;
    struct Shape { size_t w, h, d; unsigned flags; bool three; } ok_shapes[] = {
        {7, 0, 0, 0, false},                     // 1D
        {7, 5, 0, 0, false},                     // 2D
        {7, 5, 3, 0, true},                      // 3D
        {7, 0, 4, cudaArrayLayered, true},       // 1D layered
        {7, 5, 4, cudaArrayLayered, true},       // 2D layered
        {6, 6, 6, cudaArrayCubemap, true},       // cubemap
        {6, 6, 12, cudaArrayCubemap | cudaArrayLayered, true},
    };
    for (auto &s : ok_shapes) {
        a = nullptr;
        cudaError_t e = s.three ? cudaMalloc3DArray(&a, &f4, make_cudaExtent(s.w, s.h, s.d), s.flags)
                                : cudaMallocArray(&a, &f4, s.w, s.h, s.flags);
        ERR(e, cudaSuccess);
        if (e != cudaSuccess) continue;
        cudaChannelFormatDesc d;
        cudaExtent ex;
        unsigned fl = 99;
        ERR(cudaArrayGetInfo(&d, &ex, &fl, a), cudaSuccess);
        CHECK(same_desc(d, f4));
        CHECK(ex.width == s.w && ex.height == s.h && ex.depth == s.d);
        CHECK(fl == s.flags);
        cudaChannelFormatDesc g;
        ERR(cudaGetChannelDesc(&g, a), cudaSuccess);
        CHECK(same_desc(g, f4));
        ERR(cudaArrayGetInfo(nullptr, &ex, nullptr, a), cudaSuccess);
        ERR(cudaArrayGetInfo(&d, nullptr, nullptr, a), cudaSuccess);
        ERR(cudaArrayGetInfo(nullptr, nullptr, &fl, a), cudaSuccess);
        ERR(cudaArrayGetInfo(nullptr, nullptr, nullptr, a), cudaSuccess);
        ERR(cudaFreeArray(a), cudaSuccess);
        ERR(cudaFreeArray(a), cudaErrorInvalidValue);   // second free
    }
    a = nullptr;
    ERR(cudaMallocArray(&a, &f1, 0, 0, 0), cudaErrorInvalidValue);
    ERR(cudaMallocArray(&a, &f1, 0, 4, 0), cudaErrorInvalidValue);
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(0, 0, 0), 0), cudaErrorInvalidValue);
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(4, 0, 3), 0), cudaErrorInvalidValue);   // depth, no height, not layered
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(4, 5, 6), cudaArrayCubemap), cudaErrorInvalidValue);   // w != h
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(4, 4, 5), cudaArrayCubemap), cudaErrorInvalidValue);   // depth != 6
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(4, 4, 7), cudaArrayCubemap | cudaArrayLayered), cudaErrorInvalidValue);
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(4, 4, 3), 0x1000u), cudaErrorInvalidValue);   // unknown flag
    ERR(cudaMallocArray(&a, &f1, 4, 4, cudaArrayLayered), cudaErrorInvalidValue);
    ERR(cudaMallocArray(&a, &f1, 4, 4, cudaArrayCubemap), cudaErrorInvalidValue);
    cudaChannelFormatDesc bad[] = {
        cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned),    // 3 components
        cudaCreateChannelDesc(8, 16, 0, 0, cudaChannelFormatKindUnsigned),   // bits differ
        cudaCreateChannelDesc(12, 0, 0, 0, cudaChannelFormatKindSigned),     // not 8/16/32
        cudaCreateChannelDesc(64, 0, 0, 0, cudaChannelFormatKindUnsigned),
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindFloat),       // float: 16/32 only
        cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone),
        cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindUnsigned),    // no bits at all
        cudaCreateChannelDesc(0, 8, 0, 0, cudaChannelFormatKindUnsigned),    // x missing
    };
    for (auto &b : bad) {
        a = nullptr;
        ERR(cudaMallocArray(&a, &b, 4, 4, 0), cudaErrorInvalidValue);
        if (a) cudaFreeArray(a);
    }
    cudaChannelFormatDesc good[] = {
        cudaCreateChannelDescHalf(), cudaCreateChannelDescHalf2(), cudaCreateChannelDescHalf4(),
        cudaCreateChannelDesc<uchar4>(), cudaCreateChannelDesc<short2>(), cudaCreateChannelDesc<uint2>(),
        cudaCreateChannelDesc<int>(), cudaCreateChannelDesc<char>(), cudaCreateChannelDesc<float2>(),
    };
    for (auto &g : good) {
        a = nullptr;
        ERR(cudaMallocArray(&a, &g, 4, 4, 0), cudaSuccess);
        if (a) ERR(cudaFreeArray(a), cudaSuccess);
    }
    a = nullptr;
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(64, 64, 0), cudaArraySparse), cudaErrorNotSupported);
    ERR(cudaMalloc3DArray(&a, &f1, make_cudaExtent(64, 64, 0), cudaArrayDeferredMapping), cudaErrorNotSupported);
    ERR(cudaFreeArray(nullptr), cudaSuccess);
    ERR(cudaFreeArray((cudaArray_t)(uintptr_t)0x10), cudaErrorInvalidValue);
    cudaChannelFormatDesc dd; cudaExtent ee; unsigned ff;
    ERR(cudaArrayGetInfo(&dd, &ee, &ff, (cudaArray_t)(uintptr_t)0x10), cudaErrorInvalidValue);
    ERR(cudaGetChannelDesc(&dd, (cudaArray_t)(uintptr_t)0x10), cudaErrorInvalidValue);
    ERR(cudaMallocArray(&a, &f1, 3, 3, 0), cudaSuccess);
    ERR(cudaFreeArray(a), cudaSuccess);
    ERR(cudaArrayGetInfo(&dd, &ee, &ff, a), cudaErrorInvalidValue);   // a freed handle is no longer valid
}

// ============================================================ 3: copies

static void test_array_copy()
{
    const int W = 6, H = 4;                       // floats; row = 24 bytes
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    float img[H][W];
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) img[y][x] = (float)(y * 100 + x);
    cudaArray_t a;
    OK(cudaMallocArray(&a, &f1, W, H, 0));
    float padded[H][8];                           // host pitch 32 bytes
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < 8; ++x) padded[y][x] = x < W ? img[y][x] : -1.0f;
    OK(cudaMemcpy2DToArray(a, 0, 0, padded, 32, 24, H, cudaMemcpyHostToDevice));
    float back[H][10];
    for (auto &r : back) for (auto &v : r) v = -9;
    OK(cudaMemcpy2DFromArray(back, 40, a, 0, 0, 24, H, cudaMemcpyDeviceToHost));
    bool same = true;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) same &= back[y][x] == img[y][x];
    for (int y = 0; y < H; ++y) for (int x = W; x < 10; ++x) same &= back[y][x] == -9;   // pad untouched
    CHECK(same);
    float sub[2][3] = {{-1, -2, -3}, {-4, -5, -6}};   // bytes 8..20 of rows 1..2
    OK(cudaMemcpy2DToArray(a, 8, 1, sub, 12, 12, 2, cudaMemcpyHostToDevice));
    float one[H][W];
    OK(cudaMemcpy2DFromArray(one, 24, a, 0, 0, 24, H, cudaMemcpyDefault));
    same = true;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float want = img[y][x];
            if ((y == 1 || y == 2) && x >= 2 && x < 5) want = sub[y - 1][x - 2];
            same &= one[y][x] == want;
        }
    CHECK(same);
    ERR(cudaMemcpy2DToArray(a, 8, 0, sub, 16, 17, 1, cudaMemcpyHostToDevice), cudaErrorInvalidValue);    // 8+17 > 24
    ERR(cudaMemcpy2DToArray(a, 0, 3, padded, 32, 24, 2, cudaMemcpyHostToDevice), cudaErrorInvalidValue); // 3+2 > 4
    ERR(cudaMemcpy2DToArray(a, 0, 0, padded, 16, 24, 1, cudaMemcpyHostToDevice), cudaErrorInvalidPitchValue);
    ERR(cudaMemcpy2DFromArray(back, 16, a, 0, 0, 24, 1, cudaMemcpyDeviceToHost), cudaErrorInvalidPitchValue);
    {   // width 0 / height 0 change nothing and succeed
        float before[H][W];
        OK(cudaMemcpy2DFromArray(before, 24, a, 0, 0, 24, H, cudaMemcpyDeviceToHost));
        float junk[8] = {7, 7, 7, 7, 7, 7, 7, 7};
        ERR(cudaMemcpy2DToArray(a, 0, 0, junk, 32, 0, 3, cudaMemcpyHostToDevice), cudaSuccess);
        ERR(cudaMemcpy2DToArray(a, 0, 0, junk, 32, 8, 0, cudaMemcpyHostToDevice), cudaSuccess);
        float after[H][W];
        OK(cudaMemcpy2DFromArray(after, 24, a, 0, 0, 24, H, cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(before, after, sizeof before) == 0);
    }
    OK(cudaMemcpyToArray(a, 0, 0, img, sizeof img, cudaMemcpyHostToDevice));
    float flat[H * W];
    OK(cudaMemcpyFromArray(flat, a, 0, 0, sizeof flat, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(flat, img, sizeof img) == 0);
    {   // from (wOffset 8, hOffset 1), 40 bytes: 16 in row 1, 24 in row 2
        float patch[10];
        for (int i = 0; i < 10; ++i) patch[i] = 500.0f + i;
        OK(cudaMemcpyToArray(a, 8, 1, patch, 40, cudaMemcpyHostToDevice));
        float got[H * W];
        OK(cudaMemcpyFromArray(got, a, 0, 0, sizeof got, cudaMemcpyDeviceToHost));
        bool s = true;
        for (int i = 0; i < H * W; ++i) {
            float want = ((float *)img)[i];
            if (i >= W + 2 && i < W + 2 + 10) want = 500.0f + (i - (W + 2));
            s &= got[i] == want;
        }
        CHECK(s);
        float part[10];
        OK(cudaMemcpyFromArray(part, a, 8, 1, 40, cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(part, patch, 40) == 0);
    }
    ERR(cudaMemcpyToArray(a, 0, 3, img, 25, cudaMemcpyHostToDevice), cudaErrorInvalidValue);   // past the end
    ERR(cudaMemcpyFromArray(flat, a, 4, 3, 24, cudaMemcpyDeviceToHost), cudaErrorInvalidValue);
    ERR(cudaMemcpyToArray(a, 0, 3, img, 24, cudaMemcpyHostToDevice), cudaSuccess);              // exactly the last row
    OK(cudaMemcpyToArray(a, 0, 0, img, sizeof img, cudaMemcpyHostToDevice));
    float *dbuf;
    OK(cudaMalloc(&dbuf, sizeof img));
    OK(cudaMemcpyFromArray(dbuf, a, 0, 0, sizeof img, cudaMemcpyDeviceToDevice));
    float chk[H * W];
    OK(cudaMemcpy(chk, dbuf, sizeof chk, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(chk, img, sizeof img) == 0);
    cudaArray_t b;
    OK(cudaMallocArray(&b, &f1, W, H, 0));
    OK(cudaMemcpyToArray(b, 0, 0, dbuf, sizeof img, cudaMemcpyDeviceToDevice));
    OK(cudaMemcpyFromArray(chk, b, 0, 0, sizeof chk, cudaMemcpyDefault));
    CHECK(std::memcmp(chk, img, sizeof img) == 0);
    cudaArray_t c;
    OK(cudaMallocArray(&c, &f1, W, H, 0));
    OK(cudaMemset(dbuf, 0, sizeof img));
    OK(cudaMemcpyToArray(c, 0, 0, dbuf, sizeof img, cudaMemcpyDeviceToDevice));
    OK(cudaMemcpyArrayToArray(c, 0, 0, a, 0, 0, sizeof img, cudaMemcpyDeviceToDevice));
    OK(cudaMemcpyFromArray(chk, c, 0, 0, sizeof chk, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(chk, img, sizeof img) == 0);
    OK(cudaMemcpyToArray(c, 0, 0, dbuf, sizeof img, cudaMemcpyDeviceToDevice));      // zero again
    OK(cudaMemcpy2DArrayToArray(c, 4, 1, a, 8, 2, 12, 2, cudaMemcpyDeviceToDevice));
    OK(cudaMemcpyFromArray(chk, c, 0, 0, sizeof chk, cudaMemcpyDeviceToHost));
    same = true;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float want = 0;
            if (y >= 1 && y < 3 && x >= 1 && x < 4) want = img[y + 1][x + 1];   // dst byte 4 (x 1) row 1 <- src byte 8 (x 2) row 2
            same &= chk[y * W + x] == want;
        }
    CHECK(same);
    ERR(cudaMemcpy2DArrayToArray(c, 4, 1, a, 8, 2, 24, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);
    ERR(cudaMemcpy2DArrayToArray(c, 0, 3, a, 0, 0, 24, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);
    ERR(cudaMemcpy2DArrayToArray(c, 0, 0, a, 0, 3, 24, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    float zero[H * W] = {0};
    OK(cudaMemcpyToArrayAsync(a, 0, 0, zero, sizeof zero, cudaMemcpyHostToDevice, s));
    OK(cudaMemcpyToArrayAsync(a, 0, 0, img, sizeof img, cudaMemcpyHostToDevice, s));
    float asyncback[H * W];
    OK(cudaMemcpyFromArrayAsync(asyncback, a, 0, 0, sizeof img, cudaMemcpyDeviceToHost, s));
    OK(cudaStreamSynchronize(s));
    CHECK(std::memcmp(asyncback, img, sizeof img) == 0);
    OK(cudaMemcpy2DToArrayAsync(b, 0, 0, zero, 24, 24, H, cudaMemcpyHostToDevice, s));
    OK(cudaMemcpy2DToArrayAsync(b, 0, 1, img[1], 24, 24, 2, cudaMemcpyHostToDevice, s));
    OK(cudaMemcpy2DFromArrayAsync(asyncback, 24, b, 0, 0, 24, H, cudaMemcpyDeviceToHost, s));
    OK(cudaStreamSynchronize(s));
    same = true;
    for (int i = 0; i < H * W; ++i) same &= asyncback[i] == ((i >= W && i < 3 * W) ? ((float *)img)[i] : 0.0f);
    CHECK(same);
    OK(cudaStreamDestroy(s));
    {   // cudaMemcpy3D with arrays: x position and width are in elements
        cudaArray_t v;
        cudaChannelFormatDesc u2 = cudaCreateChannelDesc<ushort2>();
        OK(cudaMalloc3DArray(&v, &u2, make_cudaExtent(4, 3, 2), 0));
        ushort2 full[2][3][4];
        for (int z = 0; z < 2; ++z) for (int y = 0; y < 3; ++y) for (int x = 0; x < 4; ++x)
            full[z][y][x] = make_ushort2((unsigned short)(z * 100 + y * 10 + x), (unsigned short)(1000 + z * 100 + y * 10 + x));
        cudaMemcpy3DParms p;
        std::memset(&p, 0, sizeof p);
        p.srcPtr = make_cudaPitchedPtr(full, 4 * sizeof(ushort2), 4, 3);
        p.dstArray = v;
        p.extent = make_cudaExtent(4, 3, 2);
        p.kind = cudaMemcpyHostToDevice;
        ERR(cudaMemcpy3D(&p), cudaSuccess);
        ushort2 outb[1][2][6];   // x = 1..2, y = 1..2, z = 1 into a pitched host buffer (6 texels per row)
        std::memset(outb, 0, sizeof outb);
        cudaMemcpy3DParms q;
        std::memset(&q, 0, sizeof q);
        q.srcArray = v;
        q.srcPos = make_cudaPos(1, 1, 1);
        q.dstPtr = make_cudaPitchedPtr(outb, 6 * sizeof(ushort2), 6, 2);
        q.extent = make_cudaExtent(2, 2, 1);
        q.kind = cudaMemcpyDeviceToHost;
        ERR(cudaMemcpy3D(&q), cudaSuccess);
        same = true;
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 6; ++x) {
                ushort2 w = x < 2 ? full[1][y + 1][x + 1] : make_ushort2(0, 0);
                same &= outb[0][y][x].x == w.x && outb[0][y][x].y == w.y;
            }
        CHECK(same);
        q.srcPos = make_cudaPos(3, 0, 0);
        q.extent = make_cudaExtent(2, 1, 1);
        ERR(cudaMemcpy3D(&q), cudaErrorInvalidValue);       // x 3 + 2 > 4 (elements)
        q.srcPos = make_cudaPos(0, 0, 2);
        q.extent = make_cudaExtent(1, 1, 1);
        ERR(cudaMemcpy3D(&q), cudaErrorInvalidValue);       // slice 2 of 2
        cudaArray_t w4;
        OK(cudaMalloc3DArray(&w4, &f1, make_cudaExtent(4, 3, 2), 0));
        cudaMemcpy3DParms r;
        std::memset(&r, 0, sizeof r);
        r.srcArray = v;
        r.dstArray = w4;
        r.extent = make_cudaExtent(4, 3, 2);
        r.kind = cudaMemcpyDeviceToDevice;
        // ushort2 (4 bytes) and float (4 bytes) have the same element size; use a different size instead
        cudaArray_t w8;
        OK(cudaMalloc3DArray(&w8, &u2 /*placeholder*/, make_cudaExtent(4, 3, 2), 0));
        cudaFreeArray(w8);
        cudaChannelFormatDesc f2 = cudaCreateChannelDesc<float2>();
        OK(cudaMalloc3DArray(&w8, &f2, make_cudaExtent(4, 3, 2), 0));
        r.dstArray = w8;
        ERR(cudaMemcpy3D(&r), cudaErrorInvalidValue);       // 4-byte against 8-byte elements
        cudaArray_t v2;
        OK(cudaMalloc3DArray(&v2, &u2, make_cudaExtent(4, 3, 2), 0));
        r.dstArray = v2;
        ERR(cudaMemcpy3D(&r), cudaSuccess);
        ushort2 got[2][3][4];
        std::memset(&q, 0, sizeof q);
        q.srcArray = v2;
        q.dstPtr = make_cudaPitchedPtr(got, 4 * sizeof(ushort2), 4, 3);
        q.extent = make_cudaExtent(4, 3, 2);
        q.kind = cudaMemcpyDeviceToHost;
        ERR(cudaMemcpy3D(&q), cudaSuccess);
        CHECK(std::memcmp(got, full, sizeof full) == 0);
        r.srcArray = (cudaArray_t)(uintptr_t)0x10;           // an unknown handle
        ERR(cudaMemcpy3D(&r), cudaErrorInvalidValue);
        cudaFreeArray(v); cudaFreeArray(w4); cudaFreeArray(v2); cudaFreeArray(w8);
    }
    cudaFree(dbuf);
    cudaFreeArray(a); cudaFreeArray(b); cudaFreeArray(c);
}

// ============================================================ 4: object creation

static void test_object_create()
{
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    float t4[4] = {1, 2, 3, 4};
    cudaArray_t arr = mkarr(f1, 4, 0, 0, 0, t4, 4);
    cudaResourceDesc rd;
    std::memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    TD td;
    cudaTextureDesc d = mkdesc(td);
    cudaTextureObject_t o = 0;
    ERR(cudaCreateTextureObject(nullptr, &rd, &d, nullptr), cudaErrorInvalidValue);
    ERR(cudaCreateTextureObject(&o, nullptr, &d, nullptr), cudaErrorInvalidValue);
    ERR(cudaCreateTextureObject(&o, &rd, nullptr, nullptr), cudaErrorInvalidValue);
    {
        cudaTextureDesc b = d;
        b.addressMode[0] = (cudaTextureAddressMode)4;
        ERR(cudaCreateTextureObject(&o, &rd, &b, nullptr), cudaErrorInvalidValue);
        b = d; b.addressMode[2] = (cudaTextureAddressMode)7;
        ERR(cudaCreateTextureObject(&o, &rd, &b, nullptr), cudaErrorInvalidValue);
        b = d; b.filterMode = (cudaTextureFilterMode)2;
        ERR(cudaCreateTextureObject(&o, &rd, &b, nullptr), cudaErrorInvalidValue);
        b = d; b.readMode = (cudaTextureReadMode)2;
        ERR(cudaCreateTextureObject(&o, &rd, &b, nullptr), cudaErrorInvalidValue);
        b = d; b.sRGB = 1;
        ERR(cudaCreateTextureObject(&o, &rd, &b, nullptr), cudaErrorNotSupported);
    }
    {   // resource view on an array resource: same format, full range accepted; a format change is not
        cudaResourceViewDesc vd;
        std::memset(&vd, 0, sizeof vd);
        {   // an all-zero view (format None, zero extent) is not a full-range view: the card allows refusing it
            cudaError_t ez = cudaCreateTextureObject(&o, &rd, &d, &vd);
            XCHECK(ez == cudaSuccess, "all-zero resource view desc on an array: card says only none/same format with full range is accepted; zero extent is not full range");
            if (ez == cudaSuccess && o) OK(cudaDestroyTextureObject(o));
        }
        vd.format = cudaResViewFormatFloat1;
        vd.width = 4; vd.height = 0; vd.depth = 0;
        ERR(cudaCreateTextureObject(&o, &rd, &d, &vd), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        vd.format = cudaResViewFormatUnsignedInt1;
        vd.width = 4;
        ERR(cudaCreateTextureObject(&o, &rd, &d, &vd), cudaErrorNotSupported);
    }
    {
        cudaResourceDesc m;
        std::memset(&m, 0, sizeof m);
        m.resType = cudaResourceTypeMipmappedArray;
        ERR(cudaCreateTextureObject(&o, &m, &d, nullptr), cudaErrorNotSupported);
    }
    {   // round trip of the descriptions
        TD t2 = mk(cudaAddressModeMirror, cudaFilterModeLinear, 1, 0.5f);
        t2.am[1] = cudaAddressModeWrap;
        t2.am[2] = cudaAddressModeBorder;
        t2.border[1] = 0.25f; t2.border[2] = -3.f; t2.border[3] = 9.f;
        cudaTextureDesc dd = mkdesc(t2);
        ERR(cudaCreateTextureObject(&o, &rd, &dd, nullptr), cudaSuccess);
        cudaTextureDesc g;
        std::memset(&g, 0xEE, sizeof g);
        ERR(cudaGetTextureObjectTextureDesc(&g, o), cudaSuccess);
        CHECK(g.addressMode[0] == cudaAddressModeMirror && g.addressMode[1] == cudaAddressModeWrap &&
              g.addressMode[2] == cudaAddressModeBorder);
        CHECK(g.filterMode == cudaFilterModeLinear && g.readMode == cudaReadModeElementType);
        CHECK(g.normalizedCoords == 1 && g.sRGB == 0);
        CHECK(g.borderColor[0] == 0.5f && g.borderColor[1] == 0.25f && g.borderColor[2] == -3.f && g.borderColor[3] == 9.f);
        cudaResourceDesc gr;
        std::memset(&gr, 0xEE, sizeof gr);
        ERR(cudaGetTextureObjectResourceDesc(&gr, o), cudaSuccess);
        CHECK(gr.resType == cudaResourceTypeArray && gr.res.array.array == arr);
        cudaResourceViewDesc gv;
        std::memset(&gv, 0xEE, sizeof gv);
        ERR(cudaGetTextureObjectResourceViewDesc(&gv, o), cudaSuccess);   // no view was given: all zero
        CHECK(gv.format == cudaResViewFormatNone && gv.width == 0 && gv.height == 0 && gv.depth == 0 &&
              gv.firstMipmapLevel == 0 && gv.lastMipmapLevel == 0 && gv.firstLayer == 0 && gv.lastLayer == 0);
        ERR(cudaGetTextureObjectTextureDesc(nullptr, o), cudaErrorInvalidValue);
        ERR(cudaDestroyTextureObject(o), cudaSuccess);
        ERR(cudaGetTextureObjectTextureDesc(&g, o), cudaErrorInvalidValue);
        ERR(cudaGetTextureObjectResourceDesc(&gr, o), cudaErrorInvalidValue);
        ERR(cudaDestroyTextureObject(o), cudaErrorInvalidValue);
    }
    float *dl;
    OK(cudaMalloc(&dl, 256));
    {
        cudaResourceDesc lr;
        std::memset(&lr, 0, sizeof lr);
        lr.resType = cudaResourceTypeLinear;
        lr.res.linear.devPtr = dl;
        lr.res.linear.desc = f1;
        lr.res.linear.sizeInBytes = 64;
        TD lin = mk(cudaAddressModeWrap, cudaFilterModeLinear, 1);   // ignored for a linear resource
        cudaTextureDesc ld = mkdesc(lin);
        ERR(cudaCreateTextureObject(&o, &lr, &ld, nullptr), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        lr.res.linear.sizeInBytes = 3;                               // less than one texel
        ERR(cudaCreateTextureObject(&o, &lr, &ld, nullptr), cudaErrorInvalidValue);
        lr.res.linear.sizeInBytes = 64;
        cudaResourceViewDesc vd;
        std::memset(&vd, 0, sizeof vd);
        vd.format = cudaResViewFormatFloat1;
        vd.width = 16;
        ERR(cudaCreateTextureObject(&o, &lr, &ld, &vd), cudaErrorInvalidValue);
        lr.res.linear.desc = cudaCreateChannelDesc<unsigned char>();   // integer + element type + linear filter: filter ignored
        TD li; li.filter = cudaFilterModeLinear;
        cudaTextureDesc lid = mkdesc(li);
        ERR(cudaCreateTextureObject(&o, &lr, &lid, nullptr), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        cudaResourceDesc pr;
        std::memset(&pr, 0, sizeof pr);
        pr.resType = cudaResourceTypePitch2D;
        pr.res.pitch2D.devPtr = dl;
        pr.res.pitch2D.desc = f1;
        pr.res.pitch2D.width = 4;
        pr.res.pitch2D.height = 3;
        pr.res.pitch2D.pitchInBytes = 32;
        TD pt;
        cudaTextureDesc pd = mkdesc(pt);
        ERR(cudaCreateTextureObject(&o, &pr, &pd, nullptr), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        pr.res.pitch2D.pitchInBytes = 15;                            // < width * texel
        ERR(cudaCreateTextureObject(&o, &pr, &pd, nullptr), cudaErrorInvalidValue);
        pr.res.pitch2D.pitchInBytes = 32;
        pr.res.pitch2D.width = 0;
        ERR(cudaCreateTextureObject(&o, &pr, &pd, nullptr), cudaErrorInvalidValue);
        pr.res.pitch2D.width = 4; pr.res.pitch2D.height = 0;
        ERR(cudaCreateTextureObject(&o, &pr, &pd, nullptr), cudaErrorInvalidValue);
        pr.res.pitch2D.height = 3;
        ERR(cudaCreateTextureObject(&o, &pr, &pd, &vd), cudaErrorInvalidValue);
    }
    cudaFree(dl);
    {   // integer texels: element type + linear filter is refused on an array; normalized float is fine
        unsigned char u[4] = {0, 85, 170, 255};
        cudaArray_t ua = mkarr(cudaCreateChannelDesc<unsigned char>(), 4, 0, 0, 0, u, 1);
        TD tl = mk(cudaAddressModeClamp, cudaFilterModeLinear, 0);
        cudaResourceDesc ur;
        std::memset(&ur, 0, sizeof ur);
        ur.resType = cudaResourceTypeArray;
        ur.res.array.array = ua;
        cudaTextureDesc dl_ = mkdesc(tl);
        ERR(cudaCreateTextureObject(&o, &ur, &dl_, nullptr), cudaErrorInvalidValue);
        tl.read = cudaReadModeNormalizedFloat;
        dl_ = mkdesc(tl);
        ERR(cudaCreateTextureObject(&o, &ur, &dl_, nullptr), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        tl.read = cudaReadModeElementType;
        tl.filter = cudaFilterModePoint;
        dl_ = mkdesc(tl);
        ERR(cudaCreateTextureObject(&o, &ur, &dl_, nullptr), cudaSuccess);
        if (o) ERR(cudaDestroyTextureObject(o), cudaSuccess);
        cudaFreeArray(ua);
    }
    cudaFreeArray(arr);
}

// ============================================================ 5: linear weights

static void test_linear_weights()
{
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    float T[4] = {0, 256, 512, 768};
    cudaArray_t a = mkarr(f1, 4, 0, 0, 0, T, 4);
    for (int norm = 0; norm < 2; ++norm) {
        cudaTextureObject_t o = mkobj(a, mk(cudaAddressModeClamp, cudaFilterModeLinear, norm));
        std::vector<float> xs, want, xt, wt, xb, wb;
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 256; ++k) {
                float x = (float)i + 0.5f + (float)k / 256.0f;
                xs.push_back(norm ? x / 4.0f : x);
                want.push_back(T[i] + (float)k);          // weight exactly k/256
                float above = x + 0.5f / 256.0f + 1.0f / 1048576.0f;   // just above the tie: rounds up
                xt.push_back(norm ? above / 4.0f : above);
                wt.push_back(T[i] + (float)(k + 1));
                float below = x + 0.5f / 256.0f - 1.0f / 1048576.0f;   // just below the tie: stays
                xb.push_back(norm ? below / 4.0f : below);
                wb.push_back(T[i] + (float)k);
            }
        auto sweep = [&](const char *what, const std::vector<float> &x, const std::vector<float> &w) {
            std::vector<float> got = s1<float>(o, x);
            int bad = 0;
            for (size_t j = 0; j < x.size(); ++j)
                if (got[j] != w[j]) { if (bad++ < 5) std::printf("  %s norm=%d x=%.9g got %.9g want %.9g\n", what, norm, x[j], got[j], w[j]); }
            CHECK(bad == 0);
        };
        sweep("weight", xs, want);
        sweep("below-tie", xb, wb);
        sweep("above-tie", xt, wt);
        {   // exactly on the tie: the guide does not say; the card says round up
            std::vector<float> xe, we;
            for (int k = 0; k < 255; ++k) {
                float x = 1.5f + ((float)k + 0.5f) / 256.0f;
                xe.push_back(norm ? x / 4.0f : x);
                we.push_back(256.0f + (float)(k + 1));
            }
            std::vector<float> got = s1<float>(o, xe);
            int b2 = 0;
            for (size_t j = 0; j < xe.size(); ++j) if (got[j] != we[j]) ++b2;
            XCHECK(b2 == 0, "exact tie of the 8-bit weight is not specified by the guide; the card says round up");
        }
        std::vector<float> g1 = s1<float>(o, std::vector<float>{norm ? 2.49805f / 4 : 2.49805f});   // the guide's example
        CHECK(g1[0] == 512.0f);
        OK(cudaDestroyTextureObject(o));
    }
    cudaFreeArray(a);

    {   // 2D: 8 x 4 small integers, grid coordinates (weights are multiples of 1/65536)
        const int W = 8, H = 4;
        std::vector<float> img(W * H);
        for (auto &v : img) v = (float)(rnd() % 256);
        cudaArray_t a2 = mkarr(f1, W, H, 0, 0, img.data(), 4);
        for (int norm = 0; norm < 2; ++norm) {
            TD td = mk(cudaAddressModeClamp, cudaFilterModeLinear, norm);
            cudaTextureObject_t o = mkobj(a2, td);
            std::vector<float> xs, ys;
            for (int n = 0; n < 4000; ++n) {
                float x = gridcoord((int)(rnd() % (W * 1024))), y = gridcoord((int)(rnd() % (H * 1024)));
                xs.push_back(norm ? x / W : x);
                ys.push_back(norm ? y / H : y);
            }
            std::vector<float> got = s2<float>(o, xs, ys);
            int bad = 0;
            for (size_t j = 0; j < xs.size(); ++j) {
                float want = refsample(img, W, H, 1, xs[j], ys[j], 0, td, 2, 0);
                if (std::fabs(got[j] - want) > 1e-4f) { if (bad++ < 5) std::printf("  2D (%.6f,%.6f) got %.9g want %.9g\n", xs[j], ys[j], got[j], want); }
            }
            CHECK(bad == 0);
            OK(cudaDestroyTextureObject(o));
        }
        float q[4] = {0, 100, 200, 300};   // by hand
        cudaArray_t qa = mkarr(f1, 2, 2, 0, 0, q, 4);
        cudaTextureObject_t o = mkobj(qa, mk(cudaAddressModeClamp, cudaFilterModeLinear, 0));
        std::vector<float> r = s2<float>(o, {1.0f, 0.75f, 1.25f, 0.5f, 1.0f}, {1.0f, 0.5f, 1.5f, 0.5f, 0.5f});
        CHECK_EQF(r[0], 150.0f);    // centre: all four
        CHECK_EQF(r[1], 25.0f);     // xB = .25, yB = 0: .75*0 + .25*100
        CHECK_EQF(r[2], 275.0f);    // xB = .75, yB = 1 (row 1 only): .25*200 + .75*300
        CHECK_EQF(r[3], 0.0f);      // corner
        CHECK_EQF(r[4], 50.0f);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(qa);
        cudaFreeArray(a2);
    }
    {   // 3D 4 x 4 x 4
        const int W = 4, H = 4, D = 4;
        std::vector<float> vol(W * H * D);
        for (auto &v : vol) v = (float)(rnd() % 16);
        cudaArray_t a3 = mkarr(f1, W, H, D, 0, vol.data(), 4);
        for (int norm = 0; norm < 2; ++norm) {
            TD td = mk(cudaAddressModeClamp, cudaFilterModeLinear, norm);
            cudaTextureObject_t o = mkobj(a3, td);
            std::vector<float> xs, ys, zs;
            for (int n = 0; n < 3000; ++n) {
                float c[3];
                for (int k = 0; k < 3; ++k) c[k] = gridcoord((int)(rnd() % (4 * 1024)));
                xs.push_back(norm ? c[0] / 4 : c[0]);
                ys.push_back(norm ? c[1] / 4 : c[1]);
                zs.push_back(norm ? c[2] / 4 : c[2]);
            }
            std::vector<float> got = s3<float>(o, xs, ys, zs);
            int bad = 0;
            for (size_t j = 0; j < xs.size(); ++j) {
                float want = refsample(vol, W, H, D, xs[j], ys[j], zs[j], td, 3, 0);
                if (std::fabs(got[j] - want) > 2e-4f) { if (bad++ < 5) std::printf("  3D (%.5f,%.5f,%.5f) got %.9g want %.9g\n", xs[j], ys[j], zs[j], got[j], want); }
            }
            CHECK(bad == 0);
            OK(cudaDestroyTextureObject(o));
        }
        TD td = mk(cudaAddressModeBorder, cudaFilterModeLinear, 0, 77);   // texel centres return the texel
        cudaTextureObject_t o = mkobj(a3, td);
        std::vector<float> xs, ys, zs, want;
        for (int z = 0; z < D; ++z) for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            xs.push_back(x + 0.5f); ys.push_back(y + 0.5f); zs.push_back(z + 0.5f);
            want.push_back(vol[(z * H + y) * W + x]);
        }
        CHECK(s3<float>(o, xs, ys, zs) == want);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(a3);
    }
}

// ============================================================ 6: address modes

struct EdgeCase { cudaTextureAddressMode m; int norm; int lin; float x; float want; };

static void test_edges()
{
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    float T[4] = {10, 20, 30, 40};
    const float B = -7;
    cudaArray_t a = mkarr(f1, 4, 0, 0, 0, T, 4);
    const auto C = cudaAddressModeClamp, Bd = cudaAddressModeBorder, Wr = cudaAddressModeWrap, Mi = cudaAddressModeMirror;
    // Expected values worked out by hand from the guide's rules (texels 10 20 30 40, border -7).
    EdgeCase cases[] = {
        // point, clamp
        {C, 0, 0, -0.5f, 10}, {C, 0, 0, -100, 10}, {C, 0, 0, 3.99f, 40}, {C, 0, 0, 4, 40}, {C, 0, 0, 1e30f, 40},
        {C, 0, 0, NANF, 10}, {C, 0, 0, -1e30f, 10}, {C, 0, 0, 0, 10}, {C, 0, 0, 1.99f, 20}, {C, 0, 0, 2, 30},
        // linear, clamp
        {C, 0, 1, -0.5f, 10}, {C, 0, 1, 0.25f, 10}, {C, 0, 1, 1.0f, 15}, {C, 0, 1, 3.75f, 40}, {C, 0, 1, 100, 40},
        {C, 0, 1, 1e30f, 40}, {C, 0, 1, NANF, 10}, {C, 0, 1, 4.25f, 40}, {C, 0, 1, 2.0f, 25},
        // point, border
        {Bd, 0, 0, -0.001f, B}, {Bd, 0, 0, 0, 10}, {Bd, 0, 0, 3.999f, 40}, {Bd, 0, 0, 4, B}, {Bd, 0, 0, 1e30f, B}, {Bd, 0, 0, -1e30f, B},
        // linear, border: neighbours outside the texture blend in the border colour
        {Bd, 0, 1, 0.25f, 0.25f * B + 0.75f * 10}, {Bd, 0, 1, -0.5f, B},
        {Bd, 0, 1, 4.5f, B}, {Bd, 0, 1, 0.5f, 10}, {Bd, 0, 1, 3.5f, 40}, {Bd, 0, 1, 3.999f, 0.5f * 40 + 0.5f * B},
        // normalized, point
        {C, 1, 0, 0.99f, 40}, {C, 1, 0, 1.0f, 40}, {C, 1, 0, 5.0f, 40}, {C, 1, 0, -3.0f, 10}, {C, 1, 0, NANF, 10},
        {Wr, 1, 0, 1.25f, 20}, {Wr, 1, 0, -0.25f, 40}, {Wr, 1, 0, 0, 10}, {Wr, 1, 0, 1.0f, 10}, {Wr, 1, 0, 2.5f, 30}, {Wr, 1, 0, -1.0f, 10},
        {Mi, 1, 0, 1.25f, 40}, {Mi, 1, 0, -0.25f, 20}, {Mi, 1, 0, 2.25f, 20}, {Mi, 1, 0, 0.25f, 20}, {Mi, 1, 0, 1.0f, 40},
        {Mi, 1, 0, 0.0f, 10}, {Mi, 1, 0, -1.25f, 40}, {Mi, 1, 0, 3.75f, 20},
        {Bd, 1, 0, 1.0f, B}, {Bd, 1, 0, -0.01f, B}, {Bd, 1, 0, 0.999f, 40}, {Bd, 1, 0, 0.0f, 10},
        // normalized, linear
        {Wr, 1, 1, 0.0f, 25}, {Wr, 1, 1, 0.25f, 15}, {Wr, 1, 1, 1.25f, 15}, {Wr, 1, 1, -0.25f, 35}, {Wr, 1, 1, 0.875f, 40},
        {Mi, 1, 1, 0.0f, 10}, {Mi, 1, 1, 1.0f, 40}, {Mi, 1, 1, -0.25f, 15}, {Mi, 1, 1, 0.125f, 10},
        {C, 1, 1, -3.0f, 10}, {C, 1, 1, 3.0f, 40}, {C, 1, 1, 0.5f, 25},
        {Bd, 1, 1, 0.0f, 0.5f * B + 0.5f * 10}, {Bd, 1, 1, 0.5f, 25},
        // coordinates not normalized: wrap and mirror act as clamp
        {Wr, 0, 0, 5.5f, 40}, {Wr, 0, 0, -1.5f, 10}, {Mi, 0, 0, 5.5f, 40}, {Mi, 0, 0, -1.5f, 10}, {Wr, 0, 1, 7.0f, 40},
        {Mi, 0, 1, -7.0f, 10}, {Wr, 0, 1, 3.5f, 40}, {Wr, 0, 1, 4.5f, 40}, {Mi, 0, 1, 0.0f, 10},
    };
    int bad = 0, n = 0;
    for (auto &c : cases) {
        TD td = mk(c.m, c.lin ? cudaFilterModeLinear : cudaFilterModePoint, c.norm, B);
        cudaTextureObject_t o = mkobj(a, td);
        float got = s1<float>(o, {c.x})[0];
        ++n;
        ++g_checks;
        if (!(got == c.want || std::fabs(got - c.want) <= 1e-5f)) {
            std::printf("FAIL edge: mode %d norm %d linear %d x=%.9g got %.9g want %.9g\n", (int)c.m, c.norm, c.lin, c.x, got, c.want);
            ++g_fail; ++bad;
        }
        // the host reference must agree with the hand-worked table (guards the reference itself)
        if (c.x == c.x) {
            float ref = refsample(std::vector<float>(T, T + 4), 4, 1, 1, c.x, 0, 0, td, 1, B);
            ++g_checks;
            if (std::fabs(ref - c.want) > 1e-5f) { std::printf("FAIL reference: mode %d norm %d lin %d x=%g ref %g want %g\n", (int)c.m, c.norm, c.lin, c.x, ref, c.want); ++g_fail; }
        }
        OK(cudaDestroyTextureObject(o));
    }
    std::printf("  edge table: %d cases, %d bad\n", n, bad);
    // random sweep against the reference: every mode, point and linear, 1D
    for (int m = 0; m < 4; ++m)
        for (int lin = 0; lin < 2; ++lin)
            for (int norm = 0; norm < 2; ++norm) {
                TD td = mk((cudaTextureAddressMode)m, lin ? cudaFilterModeLinear : cudaFilterModePoint, norm, B);
                cudaTextureObject_t o = mkobj(a, td);
                std::vector<float> xs;
                for (int i = 0; i < 1500; ++i)
                    xs.push_back(gridcoord((int)(rnd() % (16 * 1024)) - 6 * 1024) / (norm ? 4.0f : 1.0f));
                std::vector<float> got = s1<float>(o, xs);
                int b = 0, xamb = 0;
                for (size_t j = 0; j < xs.size(); ++j) {
                    float want = refsample(std::vector<float>(T, T + 4), 4, 1, 1, xs[j], 0, 0, td, 1, B);
                    if (std::fabs(got[j] - want) > 2e-5f && amb_axis(td, 0, xs[j], 4)) { ++xamb; continue; }
                    if (std::fabs(got[j] - want) > 2e-5f) { if (b++ < 4) std::printf("  sweep mode %d lin %d norm %d x=%.9g got %.9g want %.9g\n", m, lin, norm, xs[j], got[j], want); }
                }
                CHECK(b == 0);
                XCHECK(xamb == 0, "linear filter + border mode, coordinate outside the texture: pure border colour returned, a blend of the edge texel and the border colour expected");
                OK(cudaDestroyTextureObject(o));
            }
    {   // the two hand cases of the same kind (x = 4.25: .25 * T[3] + .75 * border; normalized 1.0: half and half)
        TD tb = mk(cudaAddressModeBorder, cudaFilterModeLinear, 0, B);
        cudaTextureObject_t ob = mkobj(a, tb);
        float g1 = s1<float>(ob, {4.25f})[0];
        XCHECK(std::fabs(g1 - (0.25f * 40 + 0.75f * B)) < 1e-5f, "border + linear at x = 4.25 (N = 4): expected 4.75");
        OK(cudaDestroyTextureObject(ob));
        tb.norm = 1;
        ob = mkobj(a, tb);
        g1 = s1<float>(ob, {1.0f})[0];
        XCHECK(std::fabs(g1 - (0.5f * 40 + 0.5f * B)) < 1e-5f, "border + linear at normalized x = 1.0 (N = 4): expected 16.5");
        OK(cudaDestroyTextureObject(ob));
    }
    cudaFreeArray(a);

    {   // per-axis modes: 4 x 2 texture, x wraps (normalized), y is border
        float img[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        cudaArray_t ia = mkarr(f1, 4, 2, 0, 0, img, 4);
        TD td;
        td.norm = 1;
        td.am[0] = Wr;
        td.am[1] = Bd;
        td.border[0] = 100;
        cudaTextureObject_t o = mkobj(ia, td);
        std::vector<float> r = s2<float>(o, {1.25f, 1.25f, -0.25f, 0.125f}, {0.75f, 1.2f, 0.25f, -0.1f});
        CHECK_EQF(r[0], 6.0f);     // x 1.25 -> texel 1; y .75 -> row 1
        CHECK_EQF(r[1], 100.0f);   // y outside
        CHECK_EQF(r[2], 4.0f);     // x -.25 -> .75 -> texel 3, row 0
        CHECK_EQF(r[3], 100.0f);
        OK(cudaDestroyTextureObject(o));
        td.am[0] = C; td.am[1] = Mi;
        o = mkobj(ia, td);
        r = s2<float>(o, {5.0f, -2.0f, 0.1f}, {1.25f, 0.1f, 0.75f});
        CHECK_EQF(r[0], 8.0f);    // x clamps to col 3; y 1.25 mirrors to .75 -> row 1
        CHECK_EQF(r[1], 1.0f);
        CHECK_EQF(r[2], 5.0f);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(ia);
    }
    {   // border colour for integer results is read as float bits (card)
        int iv[2] = {5, 6};
        cudaArray_t ia = mkarr(cudaCreateChannelDesc<int>(), 2, 0, 0, 0, iv, 4);
        TD td = mk(Bd, cudaFilterModePoint, 0, 0);
        float bits;
        int seven = 7;
        std::memcpy(&bits, &seven, 4);
        td.border[0] = bits;
        cudaTextureObject_t o = mkobj(ia, td);
        std::vector<int> r = s1<int>(o, {-1.0f, 0.5f, 2.0f});
        XCHECK(r[0] == 7 && r[1] == 5 && r[2] == 7, "border value of integer results: the documents do not define it; the card says float bits");
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(ia);
    }
    {   // 3D sweep with different modes per axis
        const int W = 4, H = 8, D = 2;
        std::vector<float> vol(W * H * D);
        for (auto &v : vol) v = (float)(rnd() % 100);
        cudaArray_t va = mkarr(f1, W, H, D, 0, vol.data(), 4);
        for (int rep = 0; rep < 8; ++rep) {
            TD td;
            td.filter = (rep & 1) ? cudaFilterModeLinear : cudaFilterModePoint;
            td.norm = (rep >> 1) & 1;
            for (int k = 0; k < 3; ++k) td.am[k] = (cudaTextureAddressMode)((rep + k + (rep >> 2)) % 4);
            td.border[0] = 55;
            cudaTextureObject_t o = mkobj(va, td);
            std::vector<float> xs, ys, zs;
            for (int i = 0; i < 1200; ++i) {
                float c[3];
                for (int k = 0; k < 3; ++k) c[k] = gridcoord((int)(rnd() % (8 * 1024)) - 2 * 1024) / (td.norm ? 4.0f : 1.0f);
                xs.push_back(c[0]); ys.push_back(c[1]); zs.push_back(c[2]);
            }
            std::vector<float> got = s3<float>(o, xs, ys, zs);
            int b = 0, xamb = 0;
            for (size_t j = 0; j < xs.size(); ++j) {
                float want = refsample(vol, W, H, D, xs[j], ys[j], zs[j], td, 3, 55);
                if (std::fabs(got[j] - want) > 2e-3f &&
                    (amb_axis(td, 0, xs[j], W) || amb_axis(td, 1, ys[j], H) || amb_axis(td, 2, zs[j], D))) { ++xamb; continue; }
                if (std::fabs(got[j] - want) > 2e-3f) { if (b++ < 3) std::printf("  3D sweep rep %d (%g,%g,%g) got %.9g want %.9g\n", rep, xs[j], ys[j], zs[j], got[j], want); }
            }
            CHECK(b == 0);
            XCHECK(xamb == 0, "3D linear + border mode with a coordinate outside the texture: pure border colour instead of a blend");
            OK(cudaDestroyTextureObject(o));
        }
        cudaFreeArray(va);
    }
}

// ============================================================ 7: read modes

static void test_read_modes()
{
    {   // 8-bit unsigned
        unsigned char u[6] = {0, 51, 255, 128, 1, 254};
        cudaArray_t a = mkarr(cudaCreateChannelDesc<unsigned char>(), 6, 0, 0, 0, u, 1);
        TD td; td.read = cudaReadModeNormalizedFloat;
        cudaTextureObject_t o = mkobj(a, td);
        std::vector<float> r = s1<float>(o, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
        for (int i = 0; i < 6; ++i) CHECK_NEAR(r[i], u[i] / 255.0f, 1e-7f);
        CHECK_EQF(r[0], 0.0f);
        CHECK_EQF(r[2], 1.0f);
        CHECK_NEAR(r[1], 0.2f, 1e-7f);
        OK(cudaDestroyTextureObject(o));
        TD te;   // the same texels, element type: raw values
        o = mkobj(a, te);
        std::vector<unsigned char> raw = s1<unsigned char>(o, {0.5f, 1.5f, 2.5f, 3.5f});
        CHECK(raw[0] == 0 && raw[1] == 51 && raw[2] == 255 && raw[3] == 128);
        OK(cudaDestroyTextureObject(o));
        td.filter = cudaFilterModeLinear;   // filtering of normalized values
        o = mkobj(a, td);
        r = s1<float>(o, {1.0f, 2.0f, 5.5f, 0.0f});
        CHECK_NEAR(r[0], 25.5f / 255.0f, 1e-6f);
        CHECK_NEAR(r[1], 153.0f / 255.0f, 1e-6f);
        CHECK_NEAR(r[2], 254.0f / 255.0f, 1e-7f);
        CHECK_NEAR(r[3], 0.0f, 1e-7f);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(a);
    }
    {   // 8-bit signed
        signed char s[6] = {-128, -127, 127, 0, -1, 64};
        cudaArray_t a = mkarr(cudaCreateChannelDesc<signed char>(), 6, 0, 0, 0, s, 1);
        TD td; td.read = cudaReadModeNormalizedFloat;
        cudaTextureObject_t o = mkobj(a, td);
        std::vector<float> r = s1<float>(o, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f});
        CHECK_EQF(r[0], -1.0f);
        CHECK_EQF(r[1], -1.0f);
        CHECK_EQF(r[2], 1.0f);
        CHECK_EQF(r[3], 0.0f);
        CHECK_NEAR(r[4], -1.0f / 127.0f, 1e-7f);
        CHECK_NEAR(r[5], 64.0f / 127.0f, 1e-7f);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(a);
    }
    {   // 16-bit
        unsigned short u[3] = {0, 65535, 1000};
        short s[4] = {-32768, -32767, 32767, 1000};
        cudaArray_t a = mkarr(cudaCreateChannelDesc<unsigned short>(), 3, 0, 0, 0, u, 2);
        cudaArray_t b = mkarr(cudaCreateChannelDesc<short>(), 4, 0, 0, 0, s, 2);
        TD td; td.read = cudaReadModeNormalizedFloat;
        cudaTextureObject_t o = mkobj(a, td), p = mkobj(b, td);
        std::vector<float> r = s1<float>(o, {0.5f, 1.5f, 2.5f});
        CHECK_EQF(r[0], 0.0f);
        CHECK_EQF(r[1], 1.0f);
        CHECK_NEAR(r[2], 1000.0f / 65535.0f, 1e-7f);
        r = s1<float>(p, {0.5f, 1.5f, 2.5f, 3.5f});
        CHECK_EQF(r[0], -1.0f);
        CHECK_EQF(r[1], -1.0f);
        CHECK_EQF(r[2], 1.0f);
        CHECK_NEAR(r[3], 1000.0f / 32767.0f, 1e-7f);
        OK(cudaDestroyTextureObject(o)); OK(cudaDestroyTextureObject(p));
        cudaFreeArray(a); cudaFreeArray(b);
    }
    {   // vector texels
        uchar4 v[2] = {make_uchar4(0, 51, 255, 102), make_uchar4(255, 0, 255, 0)};
        cudaArray_t a = mkarr(cudaCreateChannelDesc<uchar4>(), 2, 0, 0, 0, v, 4);
        TD td; td.read = cudaReadModeNormalizedFloat;
        cudaTextureObject_t o = mkobj(a, td);
        std::vector<float4> r = s1<float4>(o, {0.5f, 1.5f});
        CHECK_NEAR(r[0].x, 0.0f, 1e-7f); CHECK_NEAR(r[0].y, 0.2f, 1e-7f); CHECK_NEAR(r[0].z, 1.0f, 1e-7f); CHECK_NEAR(r[0].w, 0.4f, 1e-7f);
        CHECK_NEAR(r[1].x, 1.0f, 1e-7f); CHECK_NEAR(r[1].y, 0.0f, 1e-7f);
        OK(cudaDestroyTextureObject(o));
        TD te;
        o = mkobj(a, te);
        std::vector<uchar4> q = s1<uchar4>(o, {0.5f, 1.5f});
        CHECK(q[0].x == 0 && q[0].y == 51 && q[0].z == 255 && q[0].w == 102 && q[1].x == 255 && q[1].w == 0);
        std::vector<uchar2> q2 = s1<uchar2>(o, {0.5f});   // fewer components than the texel
        CHECK(q2[0].x == 0 && q2[0].y == 51);
        std::vector<unsigned char> q1 = s1<unsigned char>(o, {1.5f});
        CHECK(q1[0] == 255);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(a);
    }
    {   // 32-bit integers are not converted
        int iv[3] = {-5, 0x7FFFFFFF, (int)0x80000000};
        unsigned uv[3] = {0xFFFFFFFFu, 1u, 0x80000000u};
        cudaArray_t a = mkarr(cudaCreateChannelDesc<int>(), 3, 0, 0, 0, iv, 4);
        cudaArray_t b = mkarr(cudaCreateChannelDesc<unsigned>(), 3, 0, 0, 0, uv, 4);
        for (int rm = 0; rm < 2; ++rm) {
            TD td; td.read = rm ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
            cudaTextureObject_t o = mkobj(a, td), p = mkobj(b, td);
            std::vector<int> r = s1<int>(o, {0.5f, 1.5f, 2.5f});
            CHECK(r[0] == -5 && r[1] == 0x7FFFFFFF && r[2] == (int)0x80000000);
            std::vector<unsigned> ru = s1<unsigned>(p, {0.5f, 1.5f, 2.5f});
            CHECK(ru[0] == 0xFFFFFFFFu && ru[1] == 1u && ru[2] == 0x80000000u);
            OK(cudaDestroyTextureObject(o)); OK(cudaDestroyTextureObject(p));
        }
        cudaFreeArray(a); cudaFreeArray(b);
    }
    {   // half precision: normal, subnormal, infinity, NaN; promoted to float before filtering
        unsigned short h[8] = {0x3C00, 0xC000, 0x0001, 0x03FF, 0x7C00, 0xFC00, 0x7E00, 0x0000};
        cudaArray_t a = mkarr(cudaCreateChannelDescHalf(), 8, 0, 0, 0, h, 2);
        TD td;
        cudaTextureObject_t o = mkobj(a, td);
        std::vector<float> r = s1<float>(o, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f});
        CHECK_EQF(r[0], 1.0f);
        CHECK_EQF(r[1], -2.0f);
        CHECK_EQF(r[2], std::ldexp(1.0f, -24));
        CHECK_EQF(r[3], 1023.0f * std::ldexp(1.0f, -24));
        CHECK(std::isinf(r[4]) && r[4] > 0);
        CHECK(std::isinf(r[5]) && r[5] < 0);
        CHECK(std::isnan(r[6]));
        CHECK(r[7] == 0.0f);
        OK(cudaDestroyTextureObject(o));
        td.filter = cudaFilterModeLinear;
        o = mkobj(a, td);
        r = s1<float>(o, {1.0f, 2.0f, 3.0f});        // midpoints of (1,-2), (-2,2^-24), (2^-24, 1023*2^-24)
        CHECK_EQF(r[0], -0.5f);
        CHECK_NEAR(r[1], -1.0f, 1e-6f);
        CHECK_NEAR(r[2], 512.0f * std::ldexp(1.0f, -24), 1e-12f);
        OK(cudaDestroyTextureObject(o));
        cudaFreeArray(a);
    }
    {   // normalised channel kinds are normalised whatever the read mode (card item 7)
        cudaChannelFormatDesc nd = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsignedNormalized8X1);
        unsigned char u[2] = {255, 51};
        cudaArray_t a = nullptr;
        cudaError_t e = cudaMallocArray(&a, &nd, 2, 0, 0);
        ++g_checks;
        if (e != cudaSuccess) {
            std::printf("EXPECTED-FAIL %s:%d: cudaMallocArray(UnsignedNormalized8X1) -> %s (card item 7 expects the kind to be usable)\n", __FILE__, __LINE__, cudaGetErrorString(e));
            ++g_xfail;
        } else {
            OK(cudaMemcpyToArray(a, 0, 0, u, 2, cudaMemcpyHostToDevice));
            TD td;   // element-type read mode
            cudaTextureObject_t o = mkobj(a, td);
            std::vector<float> r = s1<float>(o, {0.5f, 1.5f});
            XCHECK(std::fabs(r[0] - 1.0f) < 1e-7f && std::fabs(r[1] - 0.2f) < 1e-7f, "Normalized8X1 channel is not normalised in element-type mode");
            if (o) OK(cudaDestroyTextureObject(o));
            cudaFreeArray(a);
        }
    }
}

// ============================================================ 8: tex1Dfetch

static void test_fetch()
{
    float lin[6] = {10, 20, 30, 40, 50, 60};
    float *d;
    OK(cudaMalloc(&d, sizeof lin));
    OK(cudaMemcpy(d, lin, sizeof lin, cudaMemcpyHostToDevice));
    cudaResourceDesc rd;
    std::memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = d;
    rd.res.linear.desc = cudaCreateChannelDesc<float>();
    rd.res.linear.sizeInBytes = sizeof lin;
    TD td = mk(cudaAddressModeWrap, cudaFilterModeLinear, 1);   // settings that must not matter
    cudaTextureDesc d1 = mkdesc(td);
    cudaTextureObject_t o;
    OK(cudaCreateTextureObject(&o, &rd, &d1, nullptr));
    std::vector<float> r = sf<float>(o, {0, 1, 5, 6, 7, -1, -100, 1000000, 2147483647, -2147483647 - 1});
    CHECK_EQF(r[0], 10.0f); CHECK_EQF(r[1], 20.0f); CHECK_EQF(r[2], 60.0f);
    CHECK_EQF(r[3], 0.0f); CHECK_EQF(r[4], 0.0f); CHECK_EQF(r[5], 0.0f);
    CHECK_EQF(r[6], 0.0f); CHECK_EQF(r[7], 0.0f); CHECK_EQF(r[8], 0.0f); CHECK_EQF(r[9], 0.0f);
    OK(cudaDestroyTextureObject(o));
    float4 f4[3] = {make_float4(1, 2, 3, 4), make_float4(5, 6, 7, 8), make_float4(9, 10, 11, 12)};
    uint4 u4[2] = {make_uint4(1, 2, 3, 4), make_uint4(0xFFFFFFFFu, 0, 1, 2)};
    char2 c2[3] = {make_char2(-1, 2), make_char2(3, -4), make_char2(-128, 127)};
    void *bufs[3];
    size_t sizes[3] = {sizeof f4, sizeof u4, sizeof c2};
    const void *srcs[3] = {f4, u4, c2};
    cudaChannelFormatDesc cds[3] = {cudaCreateChannelDesc<float4>(), cudaCreateChannelDesc<uint4>(), cudaCreateChannelDesc<char2>()};
    cudaTextureObject_t os[3];
    for (int i = 0; i < 3; ++i) {
        OK(cudaMalloc(&bufs[i], sizes[i]));
        OK(cudaMemcpy(bufs[i], srcs[i], sizes[i], cudaMemcpyHostToDevice));
        cudaResourceDesc q;
        std::memset(&q, 0, sizeof q);
        q.resType = cudaResourceTypeLinear;
        q.res.linear.devPtr = bufs[i];
        q.res.linear.desc = cds[i];
        q.res.linear.sizeInBytes = sizes[i];
        TD t0;
        cudaTextureDesc dd = mkdesc(t0);
        OK(cudaCreateTextureObject(&os[i], &q, &dd, nullptr));
    }
    std::vector<float4> rf = sf<float4>(os[0], {2, 0, 3, -1});
    CHECK(rf[0].x == 9 && rf[0].w == 12 && rf[1].y == 2 && rf[2].x == 0 && rf[2].w == 0 && rf[3].z == 0);
    std::vector<uint4> ru = sf<uint4>(os[1], {1, 2});
    CHECK(ru[0].x == 0xFFFFFFFFu && ru[0].w == 2 && ru[1].x == 0 && ru[1].y == 0 && ru[1].z == 0 && ru[1].w == 0);
    std::vector<char2> rc = sf<char2>(os[2], {0, 2, 3});
    CHECK(rc[0].x == -1 && rc[0].y == 2 && rc[1].x == -128 && rc[1].y == 127 && rc[2].x == 0 && rc[2].y == 0);
    for (int i = 0; i < 3; ++i) { OK(cudaDestroyTextureObject(os[i])); cudaFree(bufs[i]); }
    {   // normalized-float read mode with tex1Dfetch of 8-bit data
        unsigned char u[3] = {0, 51, 255};
        unsigned char *du;
        OK(cudaMalloc(&du, 3));
        OK(cudaMemcpy(du, u, 3, cudaMemcpyHostToDevice));
        cudaResourceDesc q;
        std::memset(&q, 0, sizeof q);
        q.resType = cudaResourceTypeLinear;
        q.res.linear.devPtr = du;
        q.res.linear.desc = cudaCreateChannelDesc<unsigned char>();
        q.res.linear.sizeInBytes = 3;
        TD t0; t0.read = cudaReadModeNormalizedFloat;
        cudaTextureDesc dd = mkdesc(t0);
        cudaTextureObject_t qo;
        OK(cudaCreateTextureObject(&qo, &q, &dd, nullptr));
        std::vector<float> g = sf<float>(qo, {0, 1, 2, 3});
        CHECK_EQF(g[0], 0.0f); CHECK_NEAR(g[1], 0.2f, 1e-7f); CHECK_EQF(g[2], 1.0f); CHECK_EQF(g[3], 0.0f);
        OK(cudaDestroyTextureObject(qo));
        cudaFree(du);
    }
    {   // pitch2D resource sampled with a 2D fetch: rows are a pitch apart
        float pitched[3][8];
        for (int y = 0; y < 3; ++y) for (int x = 0; x < 8; ++x) pitched[y][x] = x < 4 ? (float)(y * 10 + x) : -99.0f;
        float *dp;
        OK(cudaMalloc(&dp, sizeof pitched));
        OK(cudaMemcpy(dp, pitched, sizeof pitched, cudaMemcpyHostToDevice));
        cudaResourceDesc q;
        std::memset(&q, 0, sizeof q);
        q.resType = cudaResourceTypePitch2D;
        q.res.pitch2D.devPtr = dp;
        q.res.pitch2D.desc = cudaCreateChannelDesc<float>();
        q.res.pitch2D.width = 4;
        q.res.pitch2D.height = 3;
        q.res.pitch2D.pitchInBytes = 32;
        TD t0 = mk(cudaAddressModeClamp, cudaFilterModePoint, 0);
        cudaTextureDesc dd = mkdesc(t0);
        cudaTextureObject_t qo;
        OK(cudaCreateTextureObject(&qo, &q, &dd, nullptr));
        std::vector<float> g = s2<float>(qo, {0.5f, 3.5f, 2.5f, 9.0f}, {0.5f, 1.5f, 2.5f, 2.5f});
        CHECK_EQF(g[0], 0.0f); CHECK_EQF(g[1], 13.0f); CHECK_EQF(g[2], 22.0f); CHECK_EQF(g[3], 23.0f);
        OK(cudaDestroyTextureObject(qo));
        cudaFree(dp);
    }
    cudaFree(d);
}

// ============================================================ 10: layered

static void test_layered()
{
    cudaChannelFormatDesc f1 = cudaCreateChannelDesc<float>();
    float l1[3][4];   // 1D layered: 4 texels x 3 layers; layer values far apart so mixing shows
    for (int l = 0; l < 3; ++l) for (int x = 0; x < 4; ++x) l1[l][x] = (float)(l * 1000 + x * 10);
    cudaArray_t a = mkarr(f1, 4, 0, 3, cudaArrayLayered, l1, 4);
    {
        cudaChannelFormatDesc dd; cudaExtent ex; unsigned fl;
        OK(cudaArrayGetInfo(&dd, &ex, &fl, a));
        CHECK(ex.width == 4 && ex.height == 0 && ex.depth == 3 && (fl & cudaArrayLayered));
    }
    for (int lin = 0; lin < 2; ++lin) {
        TD td = mk(cudaAddressModeClamp, lin ? cudaFilterModeLinear : cudaFilterModePoint, 0);
        cudaTextureObject_t o = mkobj(a, td);
        std::vector<float> xs;
        std::vector<int> ls;
        for (int l = 0; l < 3; ++l) for (int k = 0; k < 40; ++k) { xs.push_back(-1.0f + k * 0.15f); ls.push_back(l); }
        Dev<float> dx(xs); Dev<int> dl(ls); Dev<float> out(xs.size());
        k_t1l<float><<<grid(xs.size()), 64>>>(o, dx.p, dl.p, out.p, (int)xs.size());
        OK(cudaDeviceSynchronize());
        std::vector<float> got = out.get();
        int bad = 0;
        for (size_t j = 0; j < xs.size(); ++j) {
            std::vector<float> layer(l1[ls[j]], l1[ls[j]] + 4);
            float want = refsample(layer, 4, 1, 1, xs[j], 0, 0, td, 1, 0);
            if (std::fabs(got[j] - want) > 1e-3f) { if (bad++ < 4) std::printf("  1Dlayered lin %d x=%g layer %d got %g want %g\n", lin, xs[j], ls[j], got[j], want); }
        }
        CHECK(bad == 0);
        std::vector<int> bl = {-1, 3, 100};   // out-of-range layers must not crash (value unspecified)
        Dev<float> bx(std::vector<float>{1.0f, 1.0f, 1.0f}); Dev<int> bli(bl); Dev<float> bo(3);
        k_t1l<float><<<1, 3>>>(o, bx.p, bli.p, bo.p, 3);
        OK(cudaDeviceSynchronize());
        OK(cudaDestroyTextureObject(o));
    }
    cudaFreeArray(a);
    {   // 2D layered: 4 x 2 x 3, layers never mix
        float l2[3][2][4];
        for (int l = 0; l < 3; ++l) for (int y = 0; y < 2; ++y) for (int x = 0; x < 4; ++x) l2[l][y][x] = (float)(l * 1000 + y * 100 + x * 10);
        cudaArray_t b = mkarr(f1, 4, 2, 3, cudaArrayLayered, l2, 4);
        for (int variant = 0; variant < 4; ++variant) {
            TD td = mk(variant < 2 ? cudaAddressModeBorder : cudaAddressModeWrap, (variant & 1) ? cudaFilterModeLinear : cudaFilterModePoint, variant >= 2, 5000);
            cudaTextureObject_t o = mkobj(b, td);
            std::vector<float> xs, ys;
            std::vector<int> ls;
            for (int i = 0; i < 300; ++i) {
                float x = gridcoord((int)(rnd() % 6144) - 1024), y = gridcoord((int)(rnd() % 3072) - 512);
                xs.push_back(td.norm ? x / 4 : x);
                ys.push_back(td.norm ? y / 2 : y);
                ls.push_back(i % 3);
            }
            Dev<float> dx(xs), dy(ys), out(xs.size());
            Dev<int> dl(ls);
            k_t2l<float><<<grid(xs.size()), 64>>>(o, dx.p, dy.p, dl.p, out.p, (int)xs.size());
            OK(cudaDeviceSynchronize());
            std::vector<float> got = out.get();
            int bad = 0, xamb = 0;
            for (size_t j = 0; j < xs.size(); ++j) {
                std::vector<float> layer((float *)l2[ls[j]], (float *)l2[ls[j]] + 8);
                float want = refsample(layer, 4, 2, 1, xs[j], ys[j], 0, td, 2, 5000);
                if (std::fabs(got[j] - want) > 2e-2f && (amb_axis(td, 0, xs[j], 4) || amb_axis(td, 1, ys[j], 2))) { ++xamb; continue; }
                if (std::fabs(got[j] - want) > 2e-2f) { if (bad++ < 4) std::printf("  2Dlayered variant %d (%g,%g) layer %d got %g want %g\n", variant, xs[j], ys[j], ls[j], got[j], want); }
            }
            CHECK(bad == 0);
            XCHECK(xamb == 0, "2D layered linear + border mode with a coordinate outside the texture: pure border colour instead of a blend");
            OK(cudaDestroyTextureObject(o));
        }
        cudaFreeArray(b);
    }
}

// ============================================================ 9: references

#if HAVE_REFS

texture<float, 1, cudaReadModeElementType> r_a;
texture<float, 1, cudaReadModeElementType> r_b;
texture<float, 2, cudaReadModeElementType> r_img;
texture<unsigned char, 1, cudaReadModeNormalizedFloat> r_n8;
texture<uchar4, 1, cudaReadModeNormalizedFloat> r_n4;
texture<float, 1, cudaReadModeElementType> r_lin1;
texture<float, 1, cudaReadModeElementType> r_unbound;

__global__ void kr_a(float *o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(r_a, i); }
__global__ void kr_slow(float *o, int n, volatile int *spin)
{
    int i = threadIdx.x;
    float v = 0;
    for (int k = 0; k < 20000; ++k) v += 0.0f * (float)spin[0];
    if (i < n) o[i] = tex1Dfetch(r_a, i) + v;
}
__global__ void kr_b(float *o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(r_b, i); }
__global__ void kr_img(const float *x, const float *y, float *o, int n)
{
    int i = threadIdx.x;
    if (i < n) o[i] = tex2D(r_img, x[i], y[i]);
}
__global__ void kr_lin1(const float *x, float *o, int n)
{
    int i = threadIdx.x;
    if (i < n) o[i] = tex1D(r_lin1, x[i]);
}
__global__ void kr_n8(float *o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(r_n8, i); }
__global__ void kr_n4(float4 *o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(r_n4, i); }
__global__ void kr_unb(float *o, int n) { int i = threadIdx.x; if (i < n) o[i] = tex1Dfetch(r_unbound, i); }

static void test_references()
{
    float ha[4] = {1, 2, 3, 4}, hb[4] = {10, 20, 30, 40};
    float *da, *db;
    OK(cudaMalloc(&da, 16));
    OK(cudaMalloc(&db, 16));
    OK(cudaMemcpy(da, ha, 16, cudaMemcpyHostToDevice));
    OK(cudaMemcpy(db, hb, 16, cudaMemcpyHostToDevice));
    float *o1, *o2;
    OK(cudaMalloc(&o1, 64));
    OK(cudaMalloc(&o2, 64));
    size_t off = 99;
    ERR(cudaBindTexture(&off, r_a, da, 16), cudaSuccess);          // channel from the reference
    CHECK(off == 0);
    kr_a<<<1, 4>>>(o1, 4);
    OK(cudaDeviceSynchronize());
    float g[4];
    OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 1 && g[1] == 2 && g[2] == 3 && g[3] == 4);
    ERR(cudaBindTexture(0, r_a, db, cudaCreateChannelDesc<float>(), 16), cudaSuccess);   // explicit channel
    kr_a<<<1, 4>>>(o1, 4);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 10 && g[3] == 40);
    ERR(cudaBindTexture(0, r_a, da), cudaSuccess);                  // default size: the whole allocation
    kr_a<<<1, 4>>>(o1, 4);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 1 && g[3] == 4);
    ERR(cudaBindTexture(&off, (const textureReference *)&r_b, db, &r_b.channelDesc, 8), cudaSuccess);   // C form, first two texels
    kr_b<<<1, 4>>>(o2, 4);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(g, o2, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 10 && g[1] == 20 && g[2] == 0 && g[3] == 0);       // past the bound size reads zero
    {
        const textureReference *p = nullptr;
        ERR(cudaGetTextureReference(&p, &r_a), cudaSuccess);
        CHECK(p == (const textureReference *)&r_a);
        size_t o_ = 5;
        ERR(cudaGetTextureAlignmentOffset(&o_, &r_a), cudaSuccess);
        CHECK(o_ == 0);
        ERR(cudaGetTextureAlignmentOffset(&o_, &r_unbound), cudaErrorInvalidTextureBinding);
        ERR(cudaBindTexture(0, (const textureReference *)nullptr, da, &r_a.channelDesc, 16), cudaErrorInvalidTexture);
        ERR(cudaUnbindTexture((const textureReference *)nullptr), cudaErrorInvalidTexture);
    }
    ERR(cudaUnbindTexture(&r_a), cudaSuccess);                      // unbound: fetches read zero
    OK(cudaMemset(o1, 0x7F, 16));
    kr_a<<<1, 4>>>(o1, 4);
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 0 && g[1] == 0 && g[2] == 0 && g[3] == 0);
    {
        size_t o_;
        ERR(cudaGetTextureAlignmentOffset(&o_, &r_a), cudaErrorInvalidTextureBinding);
    }
    OK(cudaMemset(o1, 0x7F, 16));
    kr_unb<<<1, 4>>>(o1, 4);       // never bound
    OK(cudaDeviceSynchronize());
    OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
    CHECK(g[0] == 0 && g[3] == 0);

    {   // launch ordering: a kernel launched before a rebind sees the old binding
        ERR(cudaBindTexture(0, r_a, da, 16), cudaSuccess);
        int *spin;
        OK(cudaMalloc(&spin, 4));
        OK(cudaMemset(spin, 0, 4));
        cudaStream_t s;
        OK(cudaStreamCreate(&s));
        OK(cudaMemset(o1, 0, 16));
        kr_slow<<<1, 4, 0, s>>>(o1, 4, spin);
        ERR(cudaBindTexture(0, r_a, db, 16), cudaSuccess);       // waits for queued work first
        kr_a<<<1, 4, 0, s>>>(o2, 4);
        OK(cudaStreamSynchronize(s));
        float g1[4], g2[4];
        OK(cudaMemcpy(g1, o1, 16, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(g2, o2, 16, cudaMemcpyDeviceToHost));
        CHECK(g1[0] == 1 && g1[1] == 2 && g1[2] == 3 && g1[3] == 4);       // old binding
        CHECK(g2[0] == 10 && g2[3] == 40);                                  // new binding
        ERR(cudaUnbindTexture(&r_a), cudaSuccess);
        OK(cudaStreamDestroy(s));
        cudaFree(spin);
    }
    {   // 2D bound to a CUDA array; host changes of the mutable members reach the next kernel
        float img[3][4];
        for (int y = 0; y < 3; ++y) for (int x = 0; x < 4; ++x) img[y][x] = (float)(y * 10 + x);
        cudaArray_t arr = mkarr(cudaCreateChannelDesc<float>(), 4, 3, 0, 0, img, 4);
        ERR(cudaBindTextureToArray(r_img, arr), cudaSuccess);
        Dev<float> dx(std::vector<float>{0.5f, -1.0f, 4.5f, 1.0f, 0.25f, 1.25f});
        Dev<float> dy(std::vector<float>{0.5f, 0.5f, 2.5f, 1.0f, 0.5f, 0.5f});
        Dev<float> out(6);
        auto run = [&]() { kr_img<<<1, 6>>>(dx.p, dy.p, out.p, 6); OK(cudaDeviceSynchronize()); return out.get(); };
        r_img.filterMode = cudaFilterModePoint; r_img.normalized = 0; r_img.addressMode[0] = r_img.addressMode[1] = cudaAddressModeClamp;
        std::vector<float> r = run();
        CHECK(r[0] == 0 && r[1] == 0 && r[2] == 23 && r[3] == 11 && r[4] == 0 && r[5] == 1);
        r_img.addressMode[0] = cudaAddressModeBorder;               // x = -1 and 4.5 leave the texture: zero
        r = run();
        CHECK(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 11 && r[5] == 1);
        r_img.addressMode[0] = cudaAddressModeClamp;
        r_img.filterMode = cudaFilterModeLinear;
        r = run();
        CHECK_NEAR(r[3], 5.5f, 1e-4f);       // (1,1): centre of texels 0, 1, 10, 11
        CHECK_NEAR(r[0], 0.0f, 1e-4f);
        r_img.normalized = 1;                // x*4, y*3
        r_img.filterMode = cudaFilterModePoint;
        r_img.addressMode[0] = r_img.addressMode[1] = cudaAddressModeWrap;
        Dev<float> nx(std::vector<float>{0.25f, 1.25f, -0.25f, 0.0f, 0.5f, 0.99f});
        Dev<float> ny(std::vector<float>{0.5f, 0.5f, 0.9f, 1.0f, 0.34f, 0.0f});
        kr_img<<<1, 6>>>(nx.p, ny.p, out.p, 6);
        OK(cudaDeviceSynchronize());
        r = out.get();
        CHECK(r[0] == 11 && r[1] == 11 && r[2] == 23 && r[3] == 0 && r[4] == 12 && r[5] == 3);
        r_img.normalized = 0;
        ERR(cudaUnbindTexture(&r_img), cudaSuccess);
        cudaFreeArray(arr);
        float *dp;
        OK(cudaMalloc(&dp, 128));
        ERR(cudaBindTexture2D(&off, r_img, dp, cudaCreateChannelDesc<float>(), 4, 3, 15), cudaErrorInvalidValue);
        ERR(cudaBindTexture2D(&off, r_img, dp, cudaCreateChannelDesc<float>(), 4, 3, 32), cudaSuccess);
        r_img.filterMode = cudaFilterModePoint;
        kr_img<<<1, 6>>>(dx.p, dy.p, out.p, 6);
        OK(cudaDeviceSynchronize());
        ERR(cudaUnbindTexture(&r_img), cudaSuccess);
        cudaFree(dp);
    }
    {   // 1D linear filtering through a reference bound to a CUDA array (linear memory is fetched with tex1Dfetch only)
        cudaArray_t la1 = mkarr(cudaCreateChannelDesc<float>(), 4, 0, 0, 0, ha, 4);
        ERR(cudaBindTextureToArray(r_lin1, la1), cudaSuccess);
        r_lin1.filterMode = cudaFilterModeLinear;
        Dev<float> x(std::vector<float>{0.5f, 1.0f, 1.75f, 4.0f}), out(4);
        kr_lin1<<<1, 4>>>(x.p, out.p, 4);
        OK(cudaDeviceSynchronize());
        std::vector<float> r = out.get();
        CHECK_EQF(r[0], 1.0f); CHECK_EQF(r[1], 1.5f); CHECK_EQF(r[2], 2.25f); CHECK_EQF(r[3], 4.0f);
        r_lin1.filterMode = cudaFilterModePoint;
        kr_lin1<<<1, 4>>>(x.p, out.p, 4);
        OK(cudaDeviceSynchronize());
        r = out.get();
        CHECK_EQF(r[0], 1.0f); CHECK_EQF(r[1], 2.0f); CHECK_EQF(r[2], 2.0f); CHECK_EQF(r[3], 4.0f);
        ERR(cudaUnbindTexture(&r_lin1), cudaSuccess);
        cudaFreeArray(la1);
    }
    {   // normalized-float references: uchar -> float, uchar4 -> float4
        unsigned char u[4] = {0, 51, 255, 128};
        uchar4 u4[2] = {make_uchar4(0, 51, 255, 102), make_uchar4(255, 255, 0, 0)};
        unsigned char *d8;
        uchar4 *d4;
        OK(cudaMalloc(&d8, 4));
        OK(cudaMalloc(&d4, sizeof u4));
        OK(cudaMemcpy(d8, u, 4, cudaMemcpyHostToDevice));
        OK(cudaMemcpy(d4, u4, sizeof u4, cudaMemcpyHostToDevice));
        ERR(cudaBindTexture(0, r_n8, d8, 4), cudaSuccess);
        ERR(cudaBindTexture(0, r_n4, d4, sizeof u4), cudaSuccess);
        kr_n8<<<1, 4>>>(o1, 4);
        Dev<float4> o4(2);
        kr_n4<<<1, 2>>>(o4.p, 2);
        OK(cudaDeviceSynchronize());
        OK(cudaMemcpy(g, o1, 16, cudaMemcpyDeviceToHost));
        CHECK_EQF(g[0], 0.0f); CHECK_NEAR(g[1], 0.2f, 1e-7f); CHECK_EQF(g[2], 1.0f); CHECK_NEAR(g[3], 128.0f / 255.0f, 1e-7f);
        std::vector<float4> v = o4.get();
        CHECK_NEAR(v[0].y, 0.2f, 1e-7f); CHECK_NEAR(v[0].w, 0.4f, 1e-7f); CHECK_EQF(v[1].x, 1.0f);
        ERR(cudaUnbindTexture(&r_n8), cudaSuccess);
        ERR(cudaUnbindTexture(&r_n4), cudaSuccess);
        cudaFree(d8); cudaFree(d4);
    }
    cudaFree(da); cudaFree(db); cudaFree(o1); cudaFree(o2);
}

#endif

int main()
{
    test_channel_desc();
    test_array_alloc();
    test_array_copy();
    test_object_create();
    test_linear_weights();
    test_edges();
    test_read_modes();
    test_fetch();
    test_layered();
#if HAVE_REFS
    test_references();
#endif
    std::printf("rt_texture_full: %d checks, %d failed, %d expected-fail\n", g_checks, g_fail, g_xfail);
    return g_fail ? 1 : 0;
}
