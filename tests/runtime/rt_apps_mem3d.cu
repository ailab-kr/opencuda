// 3D memory, device limits and profiler control, checked against the public
// CUDA Runtime API reference.
//
// Address model used for every expectation (CUDA C++ Programming Guide,
// "Linear Memory": element (x, y, z) of a pitched 3D allocation lives at
//   base + (z * ysize + y) * pitch + x          (x in bytes)
// cudaMemset3D sets exactly extent.width bytes of each of extent.height
// rows of each of extent.depth slices, nothing else (the pitch padding and
// everything outside the extent keep their bytes). cudaMemcpy3D copies the
// same box from (srcPos) in the source to (dstPos) in the destination.
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) do { ++g_checks; cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); ++g_fail; } } while (0)
#define EXPECT(call, want) do { ++g_checks; cudaError_t e_ = (call); if (e_ != (want)) { std::printf("FAIL %s:%d: %s -> %s, want %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_), cudaGetErrorName(want)); ++g_fail; } } while (0)
#define FAILS(call) do { ++g_checks; cudaError_t e_ = (call); if (e_ == cudaSuccess) { std::printf("FAIL %s:%d: %s succeeded, an error was expected\n", __FILE__, __LINE__, #call); ++g_fail; } } while (0)

// Known disagreements are listed as expected failures: EXPECT_BUG prints a
// line starting with "XFAIL" when the call misbehaves (does not fail the
// run), and "XPASS" + failure when the call behaves (so the marker has to
// be removed). Used only where a defect was confirmed.
#define XFAIL_COND(cond, why) do { ++g_checks; if (cond) { std::printf("XPASS (bug fixed, remove marker) %s:%d: %s\n", __FILE__, __LINE__, why); ++g_fail; } else std::printf("XFAIL %s:%d: %s\n", __FILE__, __LINE__, why); } while (0)

typedef std::vector<unsigned char> Bytes;

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 11); }
static size_t rndn(size_t lo, size_t hi) { return lo + rnd() % (hi - lo + 1); }

// A refused call must leave the destination as it was; the check re-seeds the
// buffer so the next case starts clean and a failure names the call.
static void put(const struct Buf &b, const Bytes &v);
static Bytes get(const struct Buf &b);
#define CHECK_KEEP(buf, orig, what) do { ++g_checks; if (get(buf) != (orig)) { ++g_fail; \
    std::printf("FAIL %s:%d: destination modified by a refused call: %s\n", __FILE__, __LINE__, what); put(buf, orig); } } while (0)

// Confirmed defect, kept as an expected failure: the call is refused but
// the destination was already (partly) written. XFAIL when it still
// happens; when it stops happening the run fails with XPASS so the marker
// gets removed (and the check becomes a plain CHECK_KEEP).
#define XKEEP(buf, orig, what) do { ++g_checks; if (get(buf) != (orig)) { std::printf("XFAIL %s:%d: %s\n", __FILE__, __LINE__, what); put(buf, orig); } \
    else { ++g_fail; std::printf("XPASS %s:%d: defect no longer present, turn into CHECK_KEEP: %s\n", __FILE__, __LINE__, what); } } while (0)

// ---------------------------------------------------------------- buffers
struct Buf {
    bool dev = false;
    unsigned char *p = nullptr;
    size_t pitch = 0, ysize = 0, depth = 0;
    size_t bytes() const { return pitch * ysize * depth; }
    cudaPitchedPtr pp() const { return make_cudaPitchedPtr(p, pitch, pitch, ysize); }
};

// width in bytes of the logical row; ysize rows per slice; depth slices.
static bool alloc_buf(Buf &b, bool dev, size_t width, size_t ysz, size_t dep, size_t padw, size_t padh)
{
    b.dev = dev;
    b.depth = dep;
    if (dev) {
        cudaPitchedPtr p;
        if (cudaMalloc3D(&p, make_cudaExtent(width, ysz, dep)) != cudaSuccess) return false;
        b.p = (unsigned char *)p.ptr; b.pitch = p.pitch; b.ysize = p.ysize;
        if (p.xsize != width || p.ysize != ysz || p.pitch < width) return false;
    } else {
        b.pitch = width + padw; b.ysize = ysz + padh;
        b.p = (unsigned char *)std::malloc(b.bytes());
    }
    return b.p != nullptr;
}
static void free_buf(Buf &b)
{
    if (b.dev) cudaFree(b.p); else std::free(b.p);
    b.p = nullptr;
}
static void put(const Buf &b, const Bytes &v)
{
    if (b.dev) { if (cudaMemcpy(b.p, v.data(), v.size(), cudaMemcpyHostToDevice) != cudaSuccess) std::printf("FAIL put\n"); }
    else std::memcpy(b.p, v.data(), v.size());
}
static Bytes get(const Buf &b)
{
    Bytes v(b.bytes());
    if (b.dev) { if (cudaMemcpy(v.data(), b.p, v.size(), cudaMemcpyDeviceToHost) != cudaSuccess) std::printf("FAIL get\n"); }
    else std::memcpy(v.data(), b.p, v.size());
    return v;
}
static Bytes randbytes(size_t n)
{
    Bytes v(n);
    for (auto &x : v) x = (unsigned char)rnd();
    return v;
}

// ------------------------------------------------------------ malloc3d
static void test_malloc3d()
{
    struct { size_t w, h, d; } cases[] = {
        {1, 1, 1}, {20, 5, 4}, {255, 7, 3}, {256, 1, 1}, {257, 2, 2}, {1000, 50, 30},
        {4096, 3, 2}, {3, 1000, 1}, {1, 1, 1000}, {512, 512, 2}};
    for (auto &c : cases) {
        cudaPitchedPtr p;
        std::memset(&p, 0xcd, sizeof p);
        OK(cudaMalloc3D(&p, make_cudaExtent(c.w, c.h, c.d)));
        CHECK(p.ptr != nullptr);
        CHECK(p.pitch >= c.w);
        CHECK(p.xsize == c.w);
        CHECK(p.ysize == c.h);
        // the whole pitch * height * depth block is usable: write and read it back
        size_t total = p.pitch * c.h * c.d;
        OK(cudaMemset(p.ptr, 0x5a, total));
        Bytes back(total);
        OK(cudaMemcpy(back.data(), p.ptr, total, cudaMemcpyDeviceToHost));
        bool all = true;
        for (unsigned char x : back) all = all && x == 0x5a;
        CHECK(all);
        // device pointer is known to the runtime as a device allocation
        cudaPointerAttributes at;
        OK(cudaPointerGetAttributes(&at, p.ptr));
        CHECK(at.type == cudaMemoryTypeDevice);
        OK(cudaFree(p.ptr));
    }
    // pitch is consistent between two allocations of the same width
    cudaPitchedPtr a, b;
    OK(cudaMalloc3D(&a, make_cudaExtent(100, 4, 2)));
    OK(cudaMalloc3D(&b, make_cudaExtent(100, 9, 5)));
    CHECK(a.pitch == b.pitch);
    cudaFree(a.ptr); cudaFree(b.ptr);
    // the pitch is usable by kernels' row addressing: two allocations do not overlap
    cudaPitchedPtr x, y;
    OK(cudaMalloc3D(&x, make_cudaExtent(64, 8, 8)));
    OK(cudaMalloc3D(&y, make_cudaExtent(64, 8, 8)));
    unsigned char *xa = (unsigned char *)x.ptr, *ya = (unsigned char *)y.ptr;
    CHECK(xa + x.pitch * 8 * 8 <= ya || ya + y.pitch * 8 * 8 <= xa);
    cudaFree(x.ptr); cudaFree(y.ptr);

    // invalid arguments
    EXPECT(cudaMalloc3D(nullptr, make_cudaExtent(8, 8, 8)), cudaErrorInvalidValue);
    // impossible sizes fail (no wraparound to a small allocation)
    cudaPitchedPtr big;
    std::memset(&big, 0, sizeof big);
    FAILS(cudaMalloc3D(&big, make_cudaExtent((size_t)1 << 40, (size_t)1 << 40, (size_t)1 << 40)));
    FAILS(cudaMalloc3D(&big, make_cudaExtent(~(size_t)0, 1, 1)));
    FAILS(cudaMalloc3D(&big, make_cudaExtent(1, ~(size_t)0, 1)));
    FAILS(cudaMalloc3D(&big, make_cudaExtent(1, 1, ~(size_t)0)));
    FAILS(cudaMalloc3D(&big, make_cudaExtent(((size_t)1 << 33) + 1, ((size_t)1 << 33) + 1, 4)));
    (void)cudaGetLastError();
    // after the failures the runtime still works
    OK(cudaMalloc3D(&a, make_cudaExtent(10, 10, 10)));
    OK(cudaFree(a.ptr));

    // zero extents: the documentation is silent; the runtime must either
    // refuse cleanly or hand back something that can be freed.
    size_t zeros[][3] = {{0, 4, 4}, {4, 0, 4}, {4, 4, 0}, {0, 0, 0}};
    for (auto &z : zeros) {
        cudaPitchedPtr p;
        std::memset(&p, 0, sizeof p);
        cudaError_t e = cudaMalloc3D(&p, make_cudaExtent(z[0], z[1], z[2]));
        std::printf("note: cudaMalloc3D(%zu,%zu,%zu) -> %s ptr=%p pitch=%zu xsize=%zu ysize=%zu\n", z[0], z[1], z[2],
                    cudaGetErrorName(e), p.ptr, p.pitch, p.xsize, p.ysize);
        CHECK(e == cudaSuccess || e == cudaErrorInvalidValue);
        if (e == cudaSuccess) {
            CHECK(p.xsize == z[0] && p.ysize == z[1]);
            OK(cudaFree(p.ptr));      // freeing the returned pointer (even null) must succeed
        }
        (void)cudaGetLastError();
    }
}

// ------------------------------------------------------------ memset3d
static void test_memset3d()
{
    const size_t W = 37, H = 6, D = 5;
    cudaPitchedPtr p;
    OK(cudaMalloc3D(&p, make_cudaExtent(W, H, D)));
    Buf b; b.dev = true; b.p = (unsigned char *)p.ptr; b.pitch = p.pitch; b.ysize = H; b.depth = D;
    CHECK(p.pitch > W || p.pitch == W); // padding may or may not exist; both are legal

    struct Box { size_t w, h, d; unsigned char val; int arg; };
    Box boxes[] = {
        {10, 3, 2, 0x11, 0x11}, {W, H, D, 0x22, 0x22}, {1, 1, 1, 0x33, 0x33}, {W, 1, 1, 0x44, 0x44},
        {1, H, D, 0x55, 0x55}, {5, 6, 1, 0xab, 0x1ab}, {5, 2, 5, 0xff, -1}, {4, 4, 4, 0x00, 0x100}};
    for (auto &bx : boxes) {
        Bytes init(b.bytes(), 0xaa);
        put(b, init);
        OK(cudaMemset3D(p, bx.arg, make_cudaExtent(bx.w, bx.h, bx.d)));
        Bytes got = get(b), want = init;
        for (size_t z = 0; z < bx.d; ++z)
            for (size_t y = 0; y < bx.h; ++y)
                for (size_t x = 0; x < bx.w; ++x) want[(z * H + y) * p.pitch + x] = bx.val;
        CHECK(got == want);
        if (got != want) {
            size_t i = 0; while (got[i] == want[i]) ++i;
            std::printf("  box %zux%zux%zu val=%#x: first diff at byte %zu got %02x want %02x (pitch %zu)\n",
                        bx.w, bx.h, bx.d, bx.arg, i, got[i], want[i], p.pitch);
        }
    }
    // padding: with a pitch larger than the width, the padding bytes are untouched by a full-width box
    if (p.pitch > W) {
        Bytes init(b.bytes(), 0x77);
        put(b, init);
        OK(cudaMemset3D(p, 0x01, make_cudaExtent(W, H, D)));
        Bytes got = get(b);
        bool pad_ok = true, body_ok = true;
        for (size_t z = 0; z < D; ++z)
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < p.pitch; ++x) {
                    unsigned char v = got[(z * H + y) * p.pitch + x];
                    if (x < W) body_ok = body_ok && v == 0x01; else pad_ok = pad_ok && v == 0x77;
                }
        CHECK(body_ok);
        CHECK(pad_ok);
    }
    // zero extents succeed and change nothing
    size_t zx[][3] = {{0, 3, 3}, {3, 0, 3}, {3, 3, 0}, {0, 0, 0}};
    for (auto &z : zx) {
        Bytes init(b.bytes(), 0x9c);
        put(b, init);
        OK(cudaMemset3D(p, 0, make_cudaExtent(z[0], z[1], z[2])));
        CHECK(get(b) == init);
    }
    // an extent wider than the pitch, or deeper than the allocation, is refused and changes nothing
    {
        Bytes init(b.bytes(), 0x3c);
        put(b, init);
        FAILS(cudaMemset3D(p, 0, make_cudaExtent(p.pitch + 1, 1, 1))); CHECK_KEEP(b, init, "memset3D width pitch+1");
        FAILS(cudaMemset3D(p, 0, make_cudaExtent(1, 1, D + 100000))); XKEEP(b, init, "memset3D depth D+100000 beyond the allocation: error returned but slices were already written");
        FAILS(cudaMemset3D(p, 0, make_cudaExtent(~(size_t)0, ~(size_t)0, ~(size_t)0))); CHECK_KEEP(b, init, "memset3D all SIZE_MAX");
        (void)cudaGetLastError();
    }
    // null pointer
    EXPECT(cudaMemset3D(make_cudaPitchedPtr(nullptr, 64, 64, 4), 0, make_cudaExtent(4, 1, 1)), cudaErrorInvalidValue);
    // a host pointer is not a device allocation
    {
        unsigned char hostbuf[64];
        FAILS(cudaMemset3D(make_cudaPitchedPtr(hostbuf, 16, 16, 4), 0, make_cudaExtent(16, 4, 1)));
        (void)cudaGetLastError();
    }

    // async: same result, ordered in the stream
    cudaStream_t s;
    OK(cudaStreamCreate(&s));
    {
        Bytes init(b.bytes(), 0xaa);
        put(b, init);
        OK(cudaMemset3DAsync(p, 0x12, make_cudaExtent(9, 2, 3), s));
        OK(cudaMemset3DAsync(p, 0x34, make_cudaExtent(3, 2, 2), s)); // overwrites part of the first box
        OK(cudaStreamSynchronize(s));
        Bytes want = init;
        for (size_t z = 0; z < 3; ++z) for (size_t y = 0; y < 2; ++y) for (size_t x = 0; x < 9; ++x) want[(z * H + y) * p.pitch + x] = 0x12;
        for (size_t z = 0; z < 2; ++z) for (size_t y = 0; y < 2; ++y) for (size_t x = 0; x < 3; ++x) want[(z * H + y) * p.pitch + x] = 0x34;
        CHECK(get(b) == want);
        // default stream
        put(b, init);
        OK(cudaMemset3DAsync(p, 0x66, make_cudaExtent(2, 2, 2), 0));
        OK(cudaStreamSynchronize(0));
        Bytes w2 = init;
        for (size_t z = 0; z < 2; ++z) for (size_t y = 0; y < 2; ++y) for (size_t x = 0; x < 2; ++x) w2[(z * H + y) * p.pitch + x] = 0x66;
        CHECK(get(b) == w2);
        EXPECT(cudaMemset3DAsync(p, 0, make_cudaExtent(0, 1, 1), s), cudaSuccess);
        OK(cudaStreamSynchronize(s));
    }
    // cudaMemset2DAsync: width bytes of height rows at the pitch
    {
        Bytes init(b.bytes(), 0xaa);
        put(b, init);
        OK(cudaMemset2DAsync(p.ptr, p.pitch, 0x5e, 11, 4, s));
        OK(cudaStreamSynchronize(s));
        Bytes want = init;
        for (size_t y = 0; y < 4; ++y) for (size_t x = 0; x < 11; ++x) want[y * p.pitch + x] = 0x5e;
        CHECK(get(b) == want);
        OK(cudaMemset2DAsync(p.ptr, p.pitch, 0, 0, 4, s));   // empty
        OK(cudaMemset2DAsync(p.ptr, p.pitch, 0, 4, 0, s));
        OK(cudaStreamSynchronize(s));
        CHECK(get(b) == want);
        FAILS(cudaMemset2DAsync(p.ptr, p.pitch, 0, p.pitch + 1, 1, s));
        (void)cudaGetLastError();
    }
    OK(cudaStreamDestroy(s));
    OK(cudaFree(p.ptr));
}

// ------------------------------------------------------------ memcpy3d
static cudaMemcpyKind kind_of(bool sdev, bool ddev)
{
    return sdev ? (ddev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost)
                : (ddev ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost);
}

static Bytes model_copy(const Bytes &dst0, const Buf &d, const Bytes &src, const Buf &s, size_t sx, size_t sy, size_t sz,
                        size_t dx, size_t dy, size_t dz, size_t w, size_t h, size_t dep)
{
    Bytes out = dst0;
    for (size_t z = 0; z < dep; ++z)
        for (size_t y = 0; y < h; ++y)
            for (size_t x = 0; x < w; ++x)
                out[((dz + z) * d.ysize + dy + y) * d.pitch + dx + x] = src[((sz + z) * s.ysize + sy + y) * s.pitch + sx + x];
    return out;
}

static int g_random_bad;

static void one_random_copy(int iter, bool async, cudaStream_t st)
{
    bool sdev = rnd() & 1, ddev = rnd() & 1;
    size_t w = rndn(1, 90), h = rndn(1, 9), dep = rndn(1, 6);
    size_t sx = rnd() % 3 ? rndn(0, 40) : 0, sy = rnd() % 3 ? rndn(0, 4) : 0, sz = rnd() % 3 ? rndn(0, 3) : 0;
    size_t dx = rnd() % 3 ? rndn(0, 40) : 0, dy = rnd() % 3 ? rndn(0, 4) : 0, dz = rnd() % 3 ? rndn(0, 3) : 0;
    // allocation logical widths and heights (rows per slice) differ between source and destination
    size_t sw = sx + w + rndn(0, 50), sh = sy + h + rndn(0, 5), sd = sz + dep + rndn(0, 3);
    size_t dw = dx + w + rndn(0, 50), dh = dy + h + rndn(0, 5), dd = dz + dep + rndn(0, 3);
    Buf s, d;
    if (!alloc_buf(s, sdev, sw, sh, sd, rndn(0, 33), 0) || !alloc_buf(d, ddev, dw, dh, dd, rndn(0, 33), 0)) {
        std::printf("FAIL allocation in random copy %d\n", iter);
        ++g_fail;
        return;
    }
    // host buffers get a ysize larger than the data rows by a random amount
    if (!sdev) { size_t pad = rndn(0, 4); s.p = (unsigned char *)std::realloc(s.p, s.pitch * (sh + pad) * sd); s.ysize = sh + pad; }
    if (!ddev) { size_t pad = rndn(0, 4); d.p = (unsigned char *)std::realloc(d.p, d.pitch * (dh + pad) * dd); d.ysize = dh + pad; }
    Bytes sbytes = randbytes(s.bytes()), dbytes = randbytes(d.bytes());
    put(s, sbytes);
    put(d, dbytes);

    cudaMemcpy3DParms pr;
    std::memset(&pr, 0, sizeof pr);
    pr.srcPtr = make_cudaPitchedPtr(s.p, s.pitch, s.pitch, s.ysize);
    pr.dstPtr = make_cudaPitchedPtr(d.p, d.pitch, d.pitch, d.ysize);
    pr.srcPos = make_cudaPos(sx, sy, sz);
    pr.dstPos = make_cudaPos(dx, dy, dz);
    pr.extent = make_cudaExtent(w, h, dep);
    pr.kind = (rnd() % 4 == 0) ? cudaMemcpyDefault : kind_of(sdev, ddev);
    cudaError_t e = async ? cudaMemcpy3DAsync(&pr, st) : cudaMemcpy3D(&pr);
    if (async) (void)cudaStreamSynchronize(st);
    Bytes want = model_copy(dbytes, d, sbytes, s, sx, sy, sz, dx, dy, dz, w, h, dep);
    Bytes got = get(d);
    bool good = e == cudaSuccess && got == want && get(s) == sbytes;
    ++g_checks;
    if (!good) {
        ++g_fail;
        if (++g_random_bad <= 8)
            std::printf("FAIL random copy %d (%s): %s->%s kind=%d ext=%zux%zux%zu src pos=(%zu,%zu,%zu) pitch=%zu ysize=%zu "
                        "dst pos=(%zu,%zu,%zu) pitch=%zu ysize=%zu -> %s%s\n",
                        iter, async ? "async" : "sync", sdev ? "dev" : "host", ddev ? "dev" : "host", (int)pr.kind, w, h, dep,
                        sx, sy, sz, s.pitch, s.ysize, dx, dy, dz, d.pitch, d.ysize, cudaGetErrorName(e),
                        got == want ? "" : " (content differs)");
    }
    free_buf(s);
    free_buf(d);
}

static void test_memcpy3d()
{
    cudaStream_t st;
    OK(cudaStreamCreate(&st));
    for (int i = 0; i < 400; ++i) one_random_copy(i, false, st);
    for (int i = 0; i < 100; ++i) one_random_copy(1000 + i, true, st);
    std::printf("note: random 3D copies: %d bad of 500\n", g_random_bad);

    // Fixed documented example: dense host cube to device and back, with a hole check.
    {
        const size_t w = 20, h = 5, d = 4;
        Buf dv;
        CHECK(alloc_buf(dv, true, w, h, d, 0, 0));
        Bytes zero(dv.bytes(), 0xee);
        put(dv, zero);
        Bytes src(w * h * d);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (unsigned char)(i * 7 + 1);
        cudaMemcpy3DParms up;
        std::memset(&up, 0, sizeof up);
        up.srcPtr = make_cudaPitchedPtr(src.data(), w, w, h);
        up.dstPtr = make_cudaPitchedPtr(dv.p, dv.pitch, w, h);
        up.extent = make_cudaExtent(w, h, d);
        up.kind = cudaMemcpyHostToDevice;
        OK(cudaMemcpy3D(&up));
        Bytes raw = get(dv);
        bool ok = true, pad_ok = true;
        for (size_t z = 0; z < d; ++z)
            for (size_t y = 0; y < h; ++y)
                for (size_t x = 0; x < dv.pitch; ++x) {
                    unsigned char v = raw[(z * h + y) * dv.pitch + x];
                    if (x < w) ok = ok && v == src[(z * h + y) * w + x]; else pad_ok = pad_ok && v == 0xee;
                }
        CHECK(ok);
        CHECK(pad_ok);
        // device-to-device into a second allocation then back out, with a different pitch
        Buf dv2;
        CHECK(alloc_buf(dv2, true, w + 300, h + 2, d + 1, 0, 0));
        put(dv2, Bytes(dv2.bytes(), 0));
        cudaMemcpy3DParms dd = up;
        dd.srcPtr = make_cudaPitchedPtr(dv.p, dv.pitch, w, h);
        dd.dstPtr = make_cudaPitchedPtr(dv2.p, dv2.pitch, w + 300, h + 2);
        dd.dstPos = make_cudaPos(5, 1, 1);
        dd.kind = cudaMemcpyDeviceToDevice;
        OK(cudaMemcpy3D(&dd));
        Bytes back(w * h * d, 0);
        cudaMemcpy3DParms out;
        std::memset(&out, 0, sizeof out);
        out.srcPtr = make_cudaPitchedPtr(dv2.p, dv2.pitch, w + 300, h + 2);
        out.srcPos = make_cudaPos(5, 1, 1);
        out.dstPtr = make_cudaPitchedPtr(back.data(), w, w, h);
        out.extent = make_cudaExtent(w, h, d);
        out.kind = cudaMemcpyDeviceToHost;
        OK(cudaMemcpy3D(&out));
        CHECK(back == src);
        free_buf(dv); free_buf(dv2);
    }

    // Zero extents: success, destination untouched.
    {
        Buf s, d;
        CHECK(alloc_buf(s, true, 30, 4, 3, 0, 0));
        CHECK(alloc_buf(d, true, 30, 4, 3, 0, 0));
        Bytes sb = randbytes(s.bytes()), db = randbytes(d.bytes());
        put(s, sb); put(d, db);
        size_t ex[][3] = {{0, 2, 2}, {5, 0, 2}, {5, 2, 0}, {0, 0, 0}};
        for (auto &e : ex) {
            cudaMemcpy3DParms pr;
            std::memset(&pr, 0, sizeof pr);
            pr.srcPtr = make_cudaPitchedPtr(s.p, s.pitch, 30, 4);
            pr.dstPtr = make_cudaPitchedPtr(d.p, d.pitch, 30, 4);
            pr.extent = make_cudaExtent(e[0], e[1], e[2]);
            pr.kind = cudaMemcpyDeviceToDevice;
            OK(cudaMemcpy3D(&pr));
            CHECK(get(d) == db);
            OK(cudaMemcpy3DAsync(&pr, st));
            OK(cudaStreamSynchronize(st));
            CHECK(get(d) == db);
        }

        // Errors. Extent beyond the source allocation (depth) -> invalid value, destination unchanged.
        cudaMemcpy3DParms bad;
        std::memset(&bad, 0, sizeof bad);
        bad.srcPtr = make_cudaPitchedPtr(s.p, s.pitch, 30, 4);
        bad.dstPtr = make_cudaPitchedPtr(d.p, d.pitch, 30, 4);
        bad.kind = cudaMemcpyDeviceToDevice;
        bad.extent = make_cudaExtent(30, 4, 100);
        EXPECT(cudaMemcpy3D(&bad), cudaErrorInvalidValue);
        XKEEP(d, db, "memcpy3D depth 100 > allocation: error returned but the first slices were already copied");
        // offsets that push the box out of the destination
        bad.extent = make_cudaExtent(30, 4, 3);
        bad.dstPos = make_cudaPos(0, 0, 1);
        EXPECT(cudaMemcpy3D(&bad), cudaErrorInvalidValue);
        XKEEP(d, db, "memcpy3D dstPos z=1 + depth 3 > dst depth 3: error returned but 2 slices already copied");
        bad.dstPos = make_cudaPos(0, 0, 0);
        bad.srcPos = make_cudaPos(0, 1, 0);       // 4 rows from row 1 exceeds ysize 4
        EXPECT(cudaMemcpy3D(&bad), cudaErrorInvalidValue);
        CHECK_KEEP(d, db, "memcpy3D srcPos y=1 + height 4 > src ysize 4");
        bad.srcPos = make_cudaPos(0, 0, 0);
        bad.extent = make_cudaExtent(~(size_t)0, 1, 1);
        FAILS(cudaMemcpy3D(&bad));
        bad.extent = make_cudaExtent(1, ~(size_t)0, 2);
        FAILS(cudaMemcpy3D(&bad));
        (void)cudaGetLastError();
        // pitch smaller than the rows to copy
        bad.extent = make_cudaExtent(30, 2, 1);
        bad.srcPtr = make_cudaPitchedPtr(s.p, 16, 16, 4);
        FAILS(cudaMemcpy3D(&bad));
        (void)cudaGetLastError();
        CHECK_KEEP(d, db, "memcpy3D src pitch 16 < width 30");
        // null pointers, null parameter block
        EXPECT(cudaMemcpy3D(nullptr), cudaErrorInvalidValue);
        EXPECT(cudaMemcpy3DAsync(nullptr, st), cudaErrorInvalidValue);
        cudaMemcpy3DParms nul;
        std::memset(&nul, 0, sizeof nul);
        nul.extent = make_cudaExtent(4, 1, 1);
        nul.kind = cudaMemcpyDeviceToDevice;
        EXPECT(cudaMemcpy3D(&nul), cudaErrorInvalidValue);
        nul.srcPtr = make_cudaPitchedPtr(s.p, s.pitch, 30, 4);
        EXPECT(cudaMemcpy3D(&nul), cudaErrorInvalidValue);   // destination missing
        nul.srcPtr = make_cudaPitchedPtr(nullptr, s.pitch, 30, 4);
        nul.dstPtr = make_cudaPitchedPtr(d.p, d.pitch, 30, 4);
        EXPECT(cudaMemcpy3D(&nul), cudaErrorInvalidValue);   // source missing
        // arrays are not supported (documented in the project card): explicit "not supported"
        cudaMemcpy3DParms arr;
        std::memset(&arr, 0, sizeof arr);
        arr.srcPtr = make_cudaPitchedPtr(s.p, s.pitch, 30, 4);
        arr.dstArray = reinterpret_cast<cudaArray_t>(0x10);
        arr.extent = make_cudaExtent(8, 2, 1);
        arr.kind = cudaMemcpyDeviceToDevice;
        EXPECT(cudaMemcpy3D(&arr), cudaErrorNotSupported);
        EXPECT(cudaMemcpy3DAsync(&arr, st), cudaErrorNotSupported);
        arr.dstArray = nullptr;
        arr.dstPtr = make_cudaPitchedPtr(d.p, d.pitch, 30, 4);
        arr.srcPtr = make_cudaPitchedPtr(nullptr, 0, 0, 0);
        arr.srcArray = reinterpret_cast<cudaArray_t>(0x10);
        EXPECT(cudaMemcpy3D(&arr), cudaErrorNotSupported);
        CHECK_KEEP(d, db, "memcpy3D with array argument");
        (void)cudaGetLastError();
        free_buf(s); free_buf(d);
    }

    // Stream order: memset -> H2D -> D2D -> D2H in one stream, no synchronisation between
    {
        const size_t w = 41, h = 7, d = 3;
        Buf a, b;
        CHECK(alloc_buf(a, true, w, h, d, 0, 0));
        CHECK(alloc_buf(b, true, w, h, d, 0, 0));
        Bytes src = randbytes(w * h * d), out(w * h * d, 0);
        OK(cudaMemset3DAsync(a.pp(), 0, make_cudaExtent(w, h, d), st));
        cudaMemcpy3DParms up;
        std::memset(&up, 0, sizeof up);
        up.srcPtr = make_cudaPitchedPtr(src.data(), w, w, h);
        up.dstPtr = make_cudaPitchedPtr(a.p, a.pitch, w, h);
        up.extent = make_cudaExtent(w, h, d);
        up.kind = cudaMemcpyHostToDevice;
        OK(cudaMemcpy3DAsync(&up, st));
        cudaMemcpy3DParms dd = up;
        dd.srcPtr = up.dstPtr;
        dd.dstPtr = make_cudaPitchedPtr(b.p, b.pitch, w, h);
        dd.kind = cudaMemcpyDeviceToDevice;
        OK(cudaMemcpy3DAsync(&dd, st));
        cudaMemcpy3DParms dn;
        std::memset(&dn, 0, sizeof dn);
        dn.srcPtr = dd.dstPtr;
        dn.dstPtr = make_cudaPitchedPtr(out.data(), w, w, h);
        dn.extent = up.extent;
        dn.kind = cudaMemcpyDeviceToHost;
        OK(cudaMemcpy3DAsync(&dn, st));
        OK(cudaStreamSynchronize(st));
        CHECK(out == src);
        free_buf(a); free_buf(b);
    }

    // Large single copy
    {
        const size_t w = 1000, h = 50, d = 30;
        Buf a;
        CHECK(alloc_buf(a, true, w, h, d, 0, 0));
        Bytes src = randbytes(w * h * d), out(w * h * d, 0);
        cudaMemcpy3DParms up;
        std::memset(&up, 0, sizeof up);
        up.srcPtr = make_cudaPitchedPtr(src.data(), w, w, h);
        up.dstPtr = a.pp();
        up.extent = make_cudaExtent(w, h, d);
        up.kind = cudaMemcpyHostToDevice;
        OK(cudaMemcpy3D(&up));
        cudaMemcpy3DParms dn;
        std::memset(&dn, 0, sizeof dn);
        dn.srcPtr = a.pp();
        dn.dstPtr = make_cudaPitchedPtr(out.data(), w, w, h);
        dn.extent = up.extent;
        dn.kind = cudaMemcpyDeviceToHost;
        OK(cudaMemcpy3D(&dn));
        CHECK(out == src);
        free_buf(a);
    }
    OK(cudaStreamDestroy(st));
}

// ------------------------------------------------------------ limits
static void test_limits()
{
    size_t v = 12345;
    // defaults exist
    OK(cudaDeviceGetLimit(&v, cudaLimitStackSize));
    CHECK(v > 0);
    OK(cudaDeviceGetLimit(&v, cudaLimitPrintfFifoSize));
    CHECK(v > 0);
    OK(cudaDeviceGetLimit(&v, cudaLimitMallocHeapSize));
    CHECK(v > 0);

    // set / get round trip (values are multiples of 1 MiB or 1 KiB so a granularity rounding is not an issue)
    size_t stack_vals[] = {2048, 4096, 8192, 16384, 1u << 20};
    for (size_t s : stack_vals) {
        OK(cudaDeviceSetLimit(cudaLimitStackSize, s));
        OK(cudaDeviceGetLimit(&v, cudaLimitStackSize));
        CHECK(v == s);
    }
    size_t mib[] = {1u << 20, 4u << 20, 16u << 20, 64u << 20, 128u << 20};
    for (size_t s : mib) {
        OK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, s));
        OK(cudaDeviceGetLimit(&v, cudaLimitMallocHeapSize));
        CHECK(v == s);
        OK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, s));
        OK(cudaDeviceGetLimit(&v, cudaLimitPrintfFifoSize));
        CHECK(v == s);
    }
    // limits are independent of each other
    OK(cudaDeviceSetLimit(cudaLimitStackSize, 3072));
    OK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2u << 20));
    OK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 24u << 20));
    size_t a, b, c;
    OK(cudaDeviceGetLimit(&a, cudaLimitStackSize));
    OK(cudaDeviceGetLimit(&b, cudaLimitPrintfFifoSize));
    OK(cudaDeviceGetLimit(&c, cudaLimitMallocHeapSize));
    CHECK(a == 3072 && b == (2u << 20) && c == (24u << 20));
    // set twice the same value
    OK(cudaDeviceSetLimit(cudaLimitStackSize, 3072));
    OK(cudaDeviceGetLimit(&a, cudaLimitStackSize));
    CHECK(a == 3072);

    // zero is not a valid size; a refused set does not change the stored value
    EXPECT(cudaDeviceSetLimit(cudaLimitStackSize, 0), cudaErrorInvalidValue);
    EXPECT(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0), cudaErrorInvalidValue);
    EXPECT(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0), cudaErrorInvalidValue);
    OK(cudaDeviceGetLimit(&a, cudaLimitStackSize));
    OK(cudaDeviceGetLimit(&b, cudaLimitPrintfFifoSize));
    OK(cudaDeviceGetLimit(&c, cudaLimitMallocHeapSize));
    CHECK(a == 3072 && b == (2u << 20) && c == (24u << 20));

    // unknown enumerators
    int bogus[] = {7, 8, 99, 0x100, -1, 0x7fffffff, (int)0x80000000};
    for (int e : bogus) {
        size_t w = 77;
        EXPECT(cudaDeviceGetLimit(&w, (cudaLimit)e), cudaErrorInvalidValue);
        CHECK(w == 77);                              // output untouched on error
        EXPECT(cudaDeviceSetLimit((cudaLimit)e, 4096), cudaErrorInvalidValue);
    }
    // limits that this runtime does not support
    cudaLimit unsup[] = {cudaLimitDevRuntimeSyncDepth, cudaLimitDevRuntimePendingLaunchCount, cudaLimitPersistingL2CacheSize};
    for (cudaLimit l : unsup) {
        size_t w = 77;
        EXPECT(cudaDeviceGetLimit(&w, l), cudaErrorUnsupportedLimit);
        EXPECT(cudaDeviceSetLimit(l, 1), cudaErrorUnsupportedLimit);
        CHECK(w == 77);
    }
    // L2 fetch granularity is a hint: queried as 0
    v = 99;
    OK(cudaDeviceGetLimit(&v, cudaLimitMaxL2FetchGranularity));
    CHECK(v == 0);
    // null output pointer
    EXPECT(cudaDeviceGetLimit(nullptr, cudaLimitStackSize), cudaErrorInvalidValue);
    EXPECT(cudaDeviceGetLimit(nullptr, cudaLimitPersistingL2CacheSize), cudaErrorInvalidValue);

    // refused calls recorded an error; reading it clears it, and nothing is left afterwards
    { cudaError_t le = cudaGetLastError(); CHECK(le == cudaErrorInvalidValue || le == cudaErrorUnsupportedLimit); }
    OK(cudaGetLastError());
    OK(cudaDeviceSynchronize());
    OK(cudaDeviceGetLimit(&a, cudaLimitStackSize));
    CHECK(a == 3072);
    // concurrent get from several host threads sees the stored value
    std::vector<std::thread> th;
    int bad = 0;
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&bad] {
            for (int k = 0; k < 100; ++k) {
                size_t x = 0;
                if (cudaDeviceGetLimit(&x, cudaLimitStackSize) != cudaSuccess || x != 3072) __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
            }
        });
    for (auto &t : th) t.join();
    CHECK(bad == 0);
}

// ------------------------------------------------------------ profiler
__global__ void touch(int *p) { *p += 1; }

static void test_profiler()
{
    OK(cudaProfilerStart());
    OK(cudaProfilerStart());          // repeating is harmless
    OK(cudaProfilerStop());
    OK(cudaProfilerStop());
    OK(cudaProfilerStop());
    OK(cudaProfilerStart());
    int *d;
    OK(cudaMalloc(&d, sizeof(int)));
    OK(cudaMemset(d, 0, sizeof(int)));
    touch<<<1, 1>>>(d);
    OK(cudaProfilerStop());           // between a launch and its completion
    OK(cudaDeviceSynchronize());
    touch<<<1, 1>>>(d);
    OK(cudaProfilerStart());
    OK(cudaDeviceSynchronize());
    int h = 0;
    OK(cudaMemcpy(&h, d, sizeof h, cudaMemcpyDeviceToHost));
    CHECK(h == 2);                    // no effect on the work itself
    OK(cudaGetLastError());
    cudaFree(d);
    std::vector<std::thread> th;
    int bad = 0;
    for (int i = 0; i < 4; ++i)
        th.emplace_back([&bad, i] {
            for (int k = 0; k < 50; ++k)
                if ((i & 1 ? cudaProfilerStart() : cudaProfilerStop()) != cudaSuccess) __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
        });
    for (auto &t : th) t.join();
    CHECK(bad == 0);
}

int main()
{
    test_malloc3d();
    test_memset3d();
    test_memcpy3d();
    test_limits();
    test_profiler();
#ifdef OCU_MUTANT
    CHECK(false);
#endif
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
