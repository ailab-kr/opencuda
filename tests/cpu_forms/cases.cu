// Cases for the CPU path's two kernel forms (fiber per thread, whole-block
// loop; docs/cpu-region-loops.md, stage 1; card backend-cpu, list "블록
// 통째 반복문"). One program, many small cases; every case prints
//     case <name> <hash of its output>
// so a driver can run the program under OCU_CPU_FORM=fiber and =loop (and
// every order, seed, warp size and OS-thread count) and compare. Cases that
// have a value known without running (index maths, early returns, atomics,
// floating-point state) also check it on the host: "FAIL ..." lines and a
// non-zero exit.
//
// Kernel names carry their expected form (tests/cpu_forms/golden_check.cmake):
//   k_lp_*  must convert (loop)       k_fb_*  must stay on fibers
//   k_ptr_* are launched through a pointer (runtime decision)
//
// usage: cases [group ...]   groups: core idx early locals fp mem order ptr
//                                    sync stack big (default: all but stack, big)
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

extern __device__ int ext_dev(int x);          // other translation unit
extern __global__ void k_ext(int *o);          // other translation unit
void launch_inl_ext(int *o, int n);             // other translation unit: the same inline kernel
unsigned long long run_many(bool *ok);          // third translation unit: 300 kernels of one type, launched through pointers
#include "inline_kernel.cuh"

static int g_fail, g_checks, g_xfail;
#define XCHECK(c, why) do { ++g_checks; if (!(c)) { std::printf("EXPECTED-FAIL %s:%d: %s -- %s\n", __FILE__, __LINE__, #c, why); ++g_xfail; } } while (0)
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) do { cudaError_t e_ = (call); ++g_checks; if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); ++g_fail; } } while (0)

static uint64_t fnv(const void *p, size_t n, uint64_t h = 1469598103934665603ull)
{
    const unsigned char *c = (const unsigned char *)p;
    for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}
static void report(const char *name, const void *p, size_t n)
{
    std::printf("case %s %016llx\n", name, (unsigned long long)fnv(p, n));
}
template <class T> static void report(const char *name, const std::vector<T> &v) { report(name, v.data(), v.size() * sizeof(T)); }

template <class T> struct Dev {
    T *p = nullptr;
    size_t n;
    explicit Dev(size_t n_, int fill = 0) : n(n_)
    {
        OK(cudaMalloc((void **)&p, (n ? n : 1) * sizeof(T)));
        OK(cudaMemset(p, fill, (n ? n : 1) * sizeof(T)));
    }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const
    {
        std::vector<T> v(n);
        OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};
static void sync() { OK(cudaDeviceSynchronize()); }

// ------------------------------------------------------------ core, indices

__global__ void k_lp_axpy(float a, const float *x, float *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

static unsigned pack3(unsigned x, unsigned y, unsigned z) { return x | (y << 11) | (z << 22); }
__host__ __device__ static unsigned pk3(unsigned x, unsigned y, unsigned z) { return x | (y << 11) | (z << 22); }

__global__ void k_lp_idx(unsigned *o)
{
    unsigned bs = blockDim.x * blockDim.y * blockDim.z;
    unsigned blk = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    unsigned tl = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned g = blk * bs + tl;
    o[g * 4 + 0] = pk3(threadIdx.x, threadIdx.y, threadIdx.z);
    o[g * 4 + 1] = pk3(blockIdx.x, blockIdx.y, blockIdx.z);
    o[g * 4 + 2] = pk3(blockDim.x, blockDim.y, blockDim.z) ^ (pk3(gridDim.x, gridDim.y, gridDim.z) * 31u);
    o[g * 4 + 3] = (unsigned)warpSize;
}

static void idx_case(const char *name, dim3 grid, dim3 block)
{
    unsigned bs = block.x * block.y * block.z, nb = grid.x * grid.y * grid.z;
    Dev<unsigned> o((size_t)bs * nb * 4, 0xFF);
    k_lp_idx<<<grid, block>>>(o.p);
    sync();
    std::vector<unsigned> v = o.get();
    bool ok = true;
    unsigned warp = v.empty() ? 0 : v[3];
    CHECK(warp == 32 || warp == 64);
    for (unsigned b = 0; b < nb && ok; ++b)
        for (unsigned t = 0; t < bs && ok; ++t) {
            unsigned bx = b % grid.x, by = (b / grid.x) % grid.y, bz = b / (grid.x * grid.y);
            unsigned tx = t % block.x, ty = (t / block.x) % block.y, tz = t / (block.x * block.y);
            size_t g = ((size_t)b * bs + t) * 4;
            ok = v[g] == pk3(tx, ty, tz) && v[g + 1] == pk3(bx, by, bz) &&
                 v[g + 2] == (pk3(block.x, block.y, block.z) ^ (pk3(grid.x, grid.y, grid.z) * 31u)) && v[g + 3] == warp;
        }
    CHECK(ok);
    report(name, v);
}

static void group_idx()
{
    idx_case("idx_3x5x7", dim3(2, 3, 2), dim3(3, 5, 7));
    idx_case("idx_1x1x64", dim3(3), dim3(1, 1, 64));
    idx_case("idx_1024", dim3(2), dim3(1024));
    idx_case("idx_1x1024", dim3(2), dim3(1, 1024, 1));
    idx_case("idx_1thread", dim3(5, 2), dim3(1));
    idx_case("idx_33", dim3(3), dim3(33));        // short last warp
    idx_case("idx_100", dim3(2), dim3(100));
    idx_case("idx_65", dim3(2), dim3(65));        // short last warp for 64-lane warps
    idx_case("idx_32x2x2", dim3(1, 2, 3), dim3(32, 2, 2));
    idx_case("idx_7x1x9", dim3(4, 1, 2), dim3(7, 1, 9));
}

__global__ void k_lp_activemask(unsigned *o)
{
    unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
    o[g] = __activemask() & 0x1u ? 1u : 2u;
}

static void group_core()
{
    {
        const int n = 1000;
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; ++i) { x[i] = 0.25f * i; y[i] = 1.0f + i; }
        Dev<float> dx(n), dy(n);
        OK(cudaMemcpy(dx.p, x.data(), n * 4, cudaMemcpyHostToDevice));
        OK(cudaMemcpy(dy.p, y.data(), n * 4, cudaMemcpyHostToDevice));
        k_lp_axpy<<<(n + 127) / 128, 128>>>(2.5f, dx.p, dy.p, n);
        sync();
        std::vector<float> r = dy.get();
        bool ok = true;
        for (int i = 0; i < n; ++i) ok &= r[i] == 2.5f * x[i] + y[i];
        CHECK(ok);
        report("axpy", r);
    }
    {
        Dev<unsigned> o(256);
        k_lp_activemask<<<2, 128>>>(o.p);
        sync();
        report("activemask", o.get());
    }
}

// ------------------------------------------------------------ early return

__host__ __device__ static unsigned early_val(int mode, unsigned t, unsigned n)
{
    switch (mode) {
    case 0: return t % 3 == 0 ? 0xFFFFFFFFu : t * 7 + 1;        // returns on the thread index
    default: return t;
    }
}

__global__ void k_lp_early_tid(unsigned *o, unsigned n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (t % 3 == 0) return;
    o[t] = t * 7 + 1;
}
__global__ void k_lp_early_loop(unsigned *o, unsigned n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned s = 0;
    for (unsigned i = 0; i < 40; ++i) {
        s += i * t;
        if (s > 5000 + t) { o[t] = s; return; }     // returns inside a loop
    }
    o[t] = s + 1000000u;
}
__global__ void k_lp_early_all(unsigned *o)
{
    o[0] = 77;     // written by every thread; then all return at once
    return;
}
__global__ void k_lp_early_all2(unsigned *o, int never)
{
    if (never != 1234567) return;      // every thread returns before writing
    o[blockIdx.x * blockDim.x + threadIdx.x] = 5;
}
__global__ void k_lp_early_after(unsigned *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    o[t] = 11;
    if (t & 1) return;
    o[t] = 22;
}
__global__ void k_lp_early_before(unsigned *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t & 1) return;
    o[t] = 22;
}

static void group_early()
{
    const unsigned n = 300;
    {
        Dev<unsigned> o(n);
        k_lp_early_tid<<<5, 64>>>(o.p, n);
        sync();
        std::vector<unsigned> v = o.get();
        bool ok = true;
        for (unsigned t = 0; t < n; ++t) ok &= v[t] == (t % 3 == 0 ? 0u : t * 7 + 1);
        CHECK(ok);
        report("early_tid", v);
    }
    {
        Dev<unsigned> o(n);
        k_lp_early_loop<<<5, 64>>>(o.p, n);
        sync();
        std::vector<unsigned> v = o.get();
        bool ok = true;
        for (unsigned t = 0; t < n; ++t) {
            unsigned s = 0, want = 0;
            bool done = false;
            for (unsigned i = 0; i < 40; ++i) {
                s += i * t;
                if (s > 5000 + t) { want = s; done = true; break; }
            }
            if (!done) want = s + 1000000u;
            ok &= v[t] == want;
        }
        CHECK(ok);
        report("early_loop", v);
    }
    {
        Dev<unsigned> o(4);
        k_lp_early_all<<<3, 40>>>(o.p);
        sync();
        std::vector<unsigned> v = o.get();
        CHECK(v[0] == 77 && v[1] == 0);
        Dev<unsigned> o2(240, 0x11);
        k_lp_early_all2<<<3, 80>>>(o2.p, 0);
        sync();
        std::vector<unsigned> v2 = o2.get();
        bool ok = true;
        for (unsigned x : v2) ok &= x == 0x11111111u;
        CHECK(ok);
        report("early_all", v);
        report("early_all2", v2);
    }
    {
        Dev<unsigned> a(128), b(128, 0x33);
        k_lp_early_after<<<2, 64>>>(a.p);
        k_lp_early_before<<<2, 64>>>(b.p);
        sync();
        std::vector<unsigned> va = a.get(), vb = b.get();
        bool ok = true;
        for (unsigned t = 0; t < 128; ++t) { ok &= va[t] == ((t & 1) ? 11u : 22u); ok &= vb[t] == ((t & 1) ? 0x33333333u : 22u); }
        CHECK(ok);
        report("early_after", va);
        report("early_before", vb);
    }
    (void)early_val;
}

// ------------------------------------------------------------ locals, stack

__device__ __noinline__ static unsigned f_noinline(unsigned x) { return x * 2654435761u + (x >> 3); }

__global__ void k_lp_locals(unsigned *o, int n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned buf[64];
    for (int i = 0; i < 64; ++i) buf[i] = t * (unsigned)(i + 1);
    unsigned s = 0;
    for (int i = 0; i < n; ++i) s += buf[(i * 5 + t) & 63] ^ f_noinline(i);
    o[t] = s;
}
__host__ __device__ static unsigned f_noinline_h(unsigned x) { return x * 2654435761u + (x >> 3); }
static unsigned locals_ref(unsigned t, int n)
{
    unsigned buf[64], s = 0;
    for (int i = 0; i < 64; ++i) buf[i] = t * (unsigned)(i + 1);
    for (int i = 0; i < n; ++i) s += buf[(i * 5 + t) & 63] ^ f_noinline_h(i);
    return s;
}

__global__ void k_lp_vla(unsigned *o, int n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    int len = 4 + (int)(t % 5) + n;
    unsigned vla[len];
    for (int i = 0; i < len; ++i) vla[i] = t + i * i;
    unsigned s = 0;
    for (int i = 0; i < len; ++i) s = s * 3 + vla[len - 1 - i];
    o[t] = s;
}
__global__ void k_lp_alloca(unsigned *o, int n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    int len = 3 + (int)(t % 7) + n;
    unsigned *p = (unsigned *)__builtin_alloca(len * sizeof(unsigned));
    for (int i = 0; i < len; ++i) p[i] = t ^ (unsigned)(i * 40503);
    unsigned s = 17;
    for (int i = 0; i < len; ++i) s = s * 31 + p[i];
    o[t] = s;
}

// the address of a local is written to global memory and read back by the same thread
__global__ void k_lp_escape(unsigned *o, unsigned **slots)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned local[4];
    local[0] = t; local[1] = t * 2; local[2] = t * 3; local[3] = t * 4;
    slots[t] = &local[1];                 // escapes to global memory
    unsigned *back = slots[t];
    back[1] += 5;                         // local[2]
    o[t] = local[0] + local[1] + local[2] + local[3] + (unsigned)(back - local);
    slots[t] = nullptr;                   // no stack address stays behind in device memory (the two forms use different stacks)
}
__global__ void k_lp_escape_shared(unsigned *o)
{
    unsigned t = threadIdx.x;
    __shared__ unsigned *g_sh_slot[256];
    unsigned local[3] = {t, t + 100, t + 200};
    g_sh_slot[t] = &local[2];
    unsigned *back = g_sh_slot[t];
    *back += 1;
    o[blockIdx.x * blockDim.x + t] = local[0] + local[1] + local[2];
}
// an uninitialised local: only that nothing crashes (undefined value)
__global__ void k_lp_uninit(unsigned *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned u[8];
    if (t == 0xFFFFFFFFu) u[3] = 1;
    unsigned x = u[(t) & 7];
    o[t] = (x & 0) + 1;
}
// a local array that uses most of the stack when OCU_CPU_STACK_KB is small
template <int KB>
__global__ void k_lp_bigstack(unsigned *o, int rounds)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    volatile unsigned char buf[KB * 1024];
    for (int i = 0; i < KB * 1024; i += 61) buf[i] = (unsigned char)(i + t + rounds);
    unsigned s = 0;
    for (int i = 0; i < KB * 1024; i += 61) s += buf[i];
    o[t] = s;
}

static void group_locals()
{
    const int blocks = 4, bs = 64, n = 100;
    Dev<unsigned> o(blocks * 128);
    k_lp_locals<<<blocks, bs>>>(o.p, n);
    sync();
    std::vector<unsigned> v = o.get();
    bool ok = true;
    for (unsigned t = 0; t < blocks * bs; ++t) ok &= v[t] == locals_ref(t, n);
    CHECK(ok);
    report("locals", v);
    k_lp_vla<<<blocks, bs>>>(o.p, 2);
    sync();
    v = o.get();
    ok = true;
    for (unsigned t = 0; t < blocks * bs; ++t) {
        int len = 4 + (int)(t % 5) + 2;
        std::vector<unsigned> vla(len);
        for (int i = 0; i < len; ++i) vla[i] = t + i * i;
        unsigned s = 0;
        for (int i = 0; i < len; ++i) s = s * 3 + vla[len - 1 - i];
        ok &= v[t] == s;
    }
    CHECK(ok);
    report("vla", v);
    k_lp_alloca<<<blocks, bs>>>(o.p, 1);
    sync();
    v = o.get();
    ok = true;
    for (unsigned t = 0; t < blocks * bs; ++t) {
        int len = 3 + (int)(t % 7) + 1;
        unsigned s = 17;
        for (int i = 0; i < len; ++i) s = s * 31 + (t ^ (unsigned)(i * 40503));
        ok &= v[t] == s;
    }
    CHECK(ok);
    report("alloca", v);
    {
        Dev<unsigned*> slots(blocks * bs);
        k_lp_escape<<<blocks, bs>>>(o.p, slots.p);
        sync();
        v = o.get();
        ok = true;
        for (unsigned t = 0; t < blocks * bs; ++t) ok &= v[t] == t + t * 2 + (t * 3 + 5) + t * 4 + 1u;
        CHECK(ok);
        report("escape", v);
    }
    k_lp_escape_shared<<<blocks, 128>>>(o.p);
    sync();
    v = o.get();
    ok = true;
    for (unsigned b = 0; b < blocks; ++b)
        for (unsigned t = 0; t < 128; ++t) ok &= v[b * 128 + t] == t + (t + 100) + (t + 200 + 1);
    CHECK(ok);
    report("escape_shared", v);
    k_lp_uninit<<<blocks, bs>>>(o.p);
    sync();
    v = o.get();
    ok = true;
    for (unsigned t = 0; t < blocks * bs; ++t) ok &= v[t] == 1;
    CHECK(ok);
    report("uninit", v);
}

// Stack overflow (run with OCU_CPU_STACK_KB=64): the launch fails in both
// forms; the next launch of the same shape is fine.
static void group_stack()
{
    const char *kb = std::getenv("OCU_CPU_STACK_KB");
    CHECK(kb != nullptr);
    Dev<unsigned> o(64);
    k_lp_bigstack<8><<<2, 32>>>(o.p, 1);           // fits
    sync();
    std::vector<unsigned> fit = o.get();
    unsigned s0 = 0;
    for (int i = 0; i < 8 * 1024; i += 61) s0 += (unsigned char)(i + 0 + 1);
    CHECK(fit[0] == s0);
    report("stack_fit", fit);
    k_lp_bigstack<200><<<2, 32>>>(o.p, 1);         // overflows a 64 KiB stack
    cudaError_t e1 = cudaDeviceSynchronize();
    const char *ord = std::getenv("OCU_CPU_ORDER");
    if (ord && std::string(ord) == "random") {
        XCHECK(e1 != cudaSuccess, "an overflow of the fiber stack is reported as a launch failure (doc item 6); in random order the loop form does not notice it (fixed order does, and the fiber form does in both orders)");
    } else {
        CHECK(e1 != cudaSuccess);
    }
    std::printf("stack overflow reported: %d\n", e1 != cudaSuccess);
    (void)cudaGetLastError();
    k_lp_bigstack<8><<<2, 32>>>(o.p, 1);           // same shape again
    cudaError_t e2 = cudaDeviceSynchronize();
    XCHECK(e2 == cudaSuccess, "after a reported stack overflow the next launch of the same shape runs normally (card item 4, canary recovery); here it fails with the earlier error");
    if (e2 != cudaSuccess) std::printf("  next launch after the overflow: %s (first: %s)\n", cudaGetErrorString(e2), cudaGetErrorString(e1));
    if (e2 == cudaSuccess) {
        std::vector<unsigned> again = o.get();
        CHECK(again[0] == s0);
        report("stack_after", again);
    }
}

// ------------------------------------------------------------ order

__global__ void k_lp_lastwriter(unsigned *o)
{
    o[0] = threadIdx.x + 1000u * blockIdx.x;
}
__global__ void k_lp_printf(int base)
{
    printf("pf b%d t%d\n", (int)blockIdx.x, (int)threadIdx.x + base);
}
// result depends on the order threads run in: shared memory read without a barrier
__global__ void k_lp_order_shared(unsigned *o)
{
    extern __shared__ unsigned s[];               // dynamic: fresh (0xA5) for every block; static shared memory keeps the previous block's values
    unsigned t = threadIdx.x;
    s[t] = t + 1;
    o[blockIdx.x * 64 + t] = s[(t + 63) & 63];
}
__global__ void k_lp_order_global(unsigned *o)
{
    unsigned t = threadIdx.x;
    unsigned old = o[blockIdx.x];
    o[blockIdx.x] = old * 31 + t + 1;       // races on purpose: value depends on the thread order
}

static void group_order()
{
    {
        Dev<unsigned> o(2);
        k_lp_lastwriter<<<1, 64>>>(o.p);
        sync();
        std::vector<unsigned> v = o.get();
        CHECK(v[0] < 64);
        std::printf("lastwriter %u\n", v[0]);
        report("lastwriter", v);
    }
    {
        k_lp_printf<<<1, 5>>>(0);
        k_lp_printf<<<1, 40>>>(100);
        sync();
        std::fflush(stdout);
    }
    {
        Dev<unsigned> o(3 * 64);
        k_lp_order_shared<<<3, 64, 64 * 4>>>(o.p);
        sync();
        report("order_shared", o.get());
        Dev<unsigned> g(4);
        k_lp_order_global<<<4, 48>>>(g.p);
        sync();
        report("order_global", g.get());
    }
}

// ------------------------------------------------------------ memory

__global__ void k_lp_shared_static_init(unsigned *o)
{
    __shared__ unsigned s[96];
    unsigned t = threadIdx.x;
    unsigned seen = s[t];                         // read before any write: the fresh pattern
    o[blockIdx.x * blockDim.x + t] = blockIdx.x == 0 ? seen : (seen >= 0x1000u && seen < 0x2000u ? 1u : 0u);   // later blocks: 1 if they see an earlier block's write
    s[t] = 0x1000u + t + blockIdx.x;              // written for the next block, which must not see it
}
__global__ void k_lp_shared_dyn_init(unsigned *o)
{
    extern __shared__ unsigned d[];
    unsigned t = threadIdx.x;
    o[blockIdx.x * blockDim.x + t] = d[t];
    d[t] = 0x2000u + t;
}
__global__ void k_lp_shared_mine(unsigned *o)
{
    __shared__ unsigned mine[128];
    extern __shared__ unsigned dd[];
    unsigned t = threadIdx.x;
    mine[t] = t * 3;                  // each thread only reads its own slot: no barrier needed
    dd[t] = t * 5;
    o[blockIdx.x * blockDim.x + t] = mine[t] + dd[t];
}
__global__ void k_lp_atomic(unsigned *counter, unsigned *hist, unsigned long long *sum, int n)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if ((int)t >= n) return;
    unsigned old = atomicAdd(counter, 1u);
    atomicAdd(&hist[t % 16], 1u);
    atomicAdd(sum, (unsigned long long)t);
    atomicMax(&hist[16], t);
    atomicMin(&hist[17], t);
    (void)old;
}

// Static __shared__ memory at the start of a block. Run with one OS thread
// only: with several, which block follows which on a thread is not fixed.
static void group_sshared()
{
    {
        Dev<unsigned> o(5 * 96);
        k_lp_shared_static_init<<<5, 96>>>(o.p);
        sync();
        std::vector<unsigned> v = o.get();
        bool ok = true;
        for (size_t i = 0; i < v.size(); ++i) {
            bool good = i < 96 ? v[i] == v[0] : v[i] == 0;            // fresh per block: no block sees what an earlier block wrote
            if (!good && ok) std::fprintf(stderr, "  note: static shared: a later block sees what an earlier block wrote\n");
            ok &= good;
        }
        XCHECK(ok, "docs/cpu-region-loops.md test 8: static __shared__ is fresh for every block; here a block starts with what the previous block on the same OS thread wrote (both forms; dynamic shared memory is fresh)");
        XCHECK(v[0] == 0xA5A5A5A5u, "docs/cpu-region-loops.md test 8: static __shared__ holds 0xA5 at the start of a block (observed: a different constant; dynamic shared memory does hold 0xA5)");
        report("shared_static_init_block0", v.data(), 96 * 4);     // later blocks start from the previous block's values (see above)
    }
}

static void group_mem()
{
    {
        std::vector<unsigned> v;
        bool ok = true;
        Dev<unsigned> d(4 * 64);
        k_lp_shared_dyn_init<<<4, 64, 64 * 4>>>(d.p);
        sync();
        v = d.get();
        ok = true;
        for (unsigned x : v) ok &= x == 0xA5A5A5A5u;
        CHECK(ok);
        report("shared_dyn_init", v);
        Dev<unsigned> m(3 * 128);
        k_lp_shared_mine<<<3, 128, 128 * 4>>>(m.p);
        sync();
        v = m.get();
        ok = true;
        for (unsigned t = 0; t < 128; ++t) ok &= v[t] == t * 8 && v[128 + t] == t * 8 && v[256 + t] == t * 8;
        CHECK(ok);
        report("shared_mine", v);
    }
    const int n = 20000;
    auto check_atomic = [&](const char *name, unsigned *counter, unsigned *hist, unsigned long long *sum) {
        k_lp_atomic<<<(n + 127) / 128, 128>>>(counter, hist, sum, n);
        sync();
        unsigned c = *(volatile unsigned *)counter;
        unsigned long long s = *(volatile unsigned long long *)sum;
        CHECK(c == (unsigned)n);
        CHECK(s == (unsigned long long)n * (n - 1) / 2);
        unsigned h0 = 0;
        for (int k = 0; k < 16; ++k) h0 += hist[k];
        CHECK(h0 == (unsigned)n);
        CHECK(hist[0] == (unsigned)(n / 16));
        CHECK(hist[16] == (unsigned)(n - 1));
        CHECK(hist[17] == 0);
        std::printf("case %s %u %llu %u %u\n", name, c, s, hist[16], hist[17]);
    };
    {   // device memory
        Dev<unsigned> c(1), h(18);
        Dev<unsigned long long> s(1);
        unsigned init[18];
        for (int i = 0; i < 18; ++i) init[i] = 0;
        init[17] = 0xFFFFFFFFu;
        OK(cudaMemcpy(h.p, init, sizeof init, cudaMemcpyHostToDevice));
        k_lp_atomic<<<(n + 127) / 128, 128>>>(c.p, h.p, s.p, n);
        sync();
        std::vector<unsigned> ch = c.get(), hh = h.get();
        std::vector<unsigned long long> sh = s.get();
        CHECK(ch[0] == (unsigned)n);
        CHECK(sh[0] == (unsigned long long)n * (n - 1) / 2);
        unsigned tot = 0;
        for (int k = 0; k < 16; ++k) tot += hh[k];
        CHECK(tot == (unsigned)n && hh[16] == (unsigned)(n - 1) && hh[17] == 0);
        report("atomic_device", ch);
        report("atomic_device_hist", hh);
    }
    {   // pinned memory
        unsigned *c, *h;
        unsigned long long *s;
        OK(cudaHostAlloc((void **)&c, 4, cudaHostAllocMapped));
        OK(cudaHostAlloc((void **)&h, 18 * 4, cudaHostAllocMapped));
        OK(cudaHostAlloc((void **)&s, 8, cudaHostAllocMapped));
        *c = 0; *s = 0;
        for (int i = 0; i < 18; ++i) h[i] = 0;
        h[17] = 0xFFFFFFFFu;
        unsigned *dc, *dh;
        unsigned long long *ds;
        OK(cudaHostGetDevicePointer((void **)&dc, c, 0));
        OK(cudaHostGetDevicePointer((void **)&dh, h, 0));
        OK(cudaHostGetDevicePointer((void **)&ds, s, 0));
        check_atomic("atomic_pinned", dc, dh, ds);
        cudaFreeHost(c); cudaFreeHost(h); cudaFreeHost(s);
    }
    {   // managed memory
        unsigned *c, *h;
        unsigned long long *s;
        OK(cudaMallocManaged((void **)&c, 4));
        OK(cudaMallocManaged((void **)&h, 18 * 4));
        OK(cudaMallocManaged((void **)&s, 8));
        *c = 0; *s = 0;
        for (int i = 0; i < 18; ++i) h[i] = 0;
        h[17] = 0xFFFFFFFFu;
        check_atomic("atomic_managed", c, h, s);
        cudaFree(c); cudaFree(h); cudaFree(s);
    }
    {   // axpy on pinned and managed buffers
        const int m = 777;
        float *x, *y;
        OK(cudaMallocManaged((void **)&x, m * 4));
        OK(cudaMallocManaged((void **)&y, m * 4));
        for (int i = 0; i < m; ++i) { x[i] = (float)i; y[i] = 2.0f; }
        k_lp_axpy<<<(m + 63) / 64, 64>>>(3.0f, x, y, m);
        sync();
        bool ok = true;
        for (int i = 0; i < m; ++i) ok &= y[i] == 3.0f * i + 2.0f;
        CHECK(ok);
        report("axpy_managed", y, m * 4);
        cudaFree(x); cudaFree(y);
    }
}

// ------------------------------------------------------------ floating-point state

__global__ void k_lp_fpmode(float *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    volatile float one = 1.0f, tiny = 1e-10f, sub = 1e-38f, half = 0.5f;
    float up = one + tiny;                 // 1 under round-to-nearest, 1+eps rounding upward
    float dn = -one - tiny;                // -1 to nearest, -(1+eps) rounding downward
    float den = sub * half;                // a subnormal; 0 with flush-to-zero
    o[t * 3 + 0] = up;
    o[t * 3 + 1] = dn;
    o[t * 3 + 2] = den;
}
// a kernel that changes the rounding mode itself (refused: fibers)
__global__ void k_fb_fesetround(float *o, int mode)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    std::fesetround(mode);
    volatile float one = 1.0f, tiny = 1e-10f;
    o[t] = one + tiny;
}
__global__ void k_fb_fesetenv(float *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    fenv_t env;
    std::fegetenv(&env);
    std::fesetround(FE_UPWARD);
    volatile float one = 1.0f, tiny = 1e-10f;
    o[t] = one + tiny;
    std::fesetenv(&env);
}

static void group_fp()
{
    const int n = 96;
    Dev<float> o(n * 3);
    int saved = std::fegetround();
    for (int pass = 0; pass < 3; ++pass) {
        // pass 0: nearest, 1: upward, 2: downward
        int mode = pass == 0 ? FE_TONEAREST : (pass == 1 ? FE_UPWARD : FE_DOWNWARD);
        std::fesetround(mode);
        k_lp_fpmode<<<3, 32>>>(o.p);
        sync();
        int after = std::fegetround();
        std::fesetround(saved);
        CHECK(after == mode);                   // the host's value is unchanged by the launch
        std::vector<float> v = o.get();
        float wantUp = pass == 1 ? 1.00000012f : 1.0f;
        float wantDn = pass == 2 ? -1.00000012f : -1.0f;
        bool ok = true;
        int bad = 0;
        for (int t = 0; t < n; ++t)
            if (v[t * 3] != wantUp || v[t * 3 + 1] != wantDn) { ok = false; ++bad; }
        if (!ok) std::printf("  fp mode %d: %d of %d threads did not see the host's rounding mode (thread 0: %.9g %.9g, want %.9g %.9g)\n", pass, bad, n, v[0], v[1], wantUp, wantDn);
        XCHECK(ok, "docs/cpu-region-loops.md test 7: a launch starts every thread with the host's rounding mode");
        char name[32];
        std::snprintf(name, sizeof name, "fp_round_%d", pass);
        report(name, v);
    }
#if defined(__x86_64__)
    {   // flush-to-zero and denormals-are-zero set by the host
        unsigned csr = _mm_getcsr();
        _mm_setcsr(csr | 0x8000u | 0x0040u);
        k_lp_fpmode<<<3, 32>>>(o.p);
        sync();
        unsigned after = _mm_getcsr();
        _mm_setcsr(csr);
        CHECK((after & 0x8040u) == 0x8040u);
        std::vector<float> v = o.get();
        bool ok = true;
        for (int t = 0; t < n; ++t) ok &= v[t * 3 + 2] == 0.0f;
        if (!ok) std::printf("  FTZ: thread 0 gave %.9g, expected 0 with flush-to-zero set by the host\n", v[2]);
        XCHECK(ok, "docs/cpu-region-loops.md test 7: a launch starts every thread with the host's flush-to-zero setting");
        report("fp_ftz", v);
        k_lp_fpmode<<<3, 32>>>(o.p);
        sync();
        v = o.get();
        ok = true;
        for (int t = 0; t < n; ++t) ok &= v[t * 3 + 2] == 0.5e-38f;
        CHECK(ok);          // and back to normal arithmetic afterwards
        report("fp_noftz", v);
    }
#endif
    // kernels that change the mode themselves: refused (fibers), same behaviour as before
    {
        Dev<float> r(96);
        k_fb_fesetround<<<3, 32>>>(r.p, FE_UPWARD);
        sync();
        std::vector<float> v = r.get();
        bool ok = true;
        for (float x : v) ok &= x == 1.00000012f;
        CHECK(ok);
        report("fesetround_up", v);
        CHECK(std::fegetround() == saved);          // the host's own mode is not changed by the kernel
        k_fb_fesetenv<<<3, 32>>>(r.p);
        sync();
        v = r.get();
        ok = true;
        for (float x : v) ok &= x == 1.00000012f;
        CHECK(ok);
        report("fesetenv", v);
        CHECK(std::fegetround() == saved);
        // after a kernel changed the mode, a plain kernel in the same process sees nothing leftover
        k_lp_fpmode<<<3, 32>>>(o.p);
        sync();
        v = o.get();
        ok = true;
        for (int t = 0; t < n; ++t) ok &= v[t * 3] == 1.0f && v[t * 3 + 1] == -1.0f;
        CHECK(ok);
        report("fp_after_change", v);
    }
}

// ------------------------------------------------------------ kernels with synchronisation (fibers)

__global__ void k_fb_sync(unsigned *o)
{
    __shared__ unsigned s[128];
    unsigned t = threadIdx.x;
    s[t] = t * 3 + blockIdx.x;
    __syncthreads();
    o[blockIdx.x * blockDim.x + t] = s[blockDim.x - 1 - t];
}
__device__ static void helper_sync_inl() { __syncthreads(); }
__global__ void k_fb_sync_helper(unsigned *o)
{
    __shared__ unsigned s[128];
    unsigned t = threadIdx.x;
    s[t] = t + 1;
    helper_sync_inl();
    o[blockIdx.x * blockDim.x + t] = s[(t + 1) % blockDim.x];
}
__device__ __noinline__ static void helper_sync_noinl() { __syncthreads(); }
__global__ void k_fb_sync_noinline(unsigned *o)
{
    __shared__ unsigned s[128];
    unsigned t = threadIdx.x;
    s[t] = t + 7;
    helper_sync_noinl();
    o[blockIdx.x * blockDim.x + t] = s[(t + 2) % blockDim.x];
}
__global__ void k_fb_sync_if(unsigned *o, int n)
{
    __shared__ unsigned s[128];
    unsigned t = threadIdx.x;
    s[t] = t + 9;
    if (n > 0) __syncthreads();                     // uniform: n is a kernel argument
    o[blockIdx.x * blockDim.x + t] = s[(t + 3) % blockDim.x];
}
__global__ void k_fb_sync_dead(unsigned *o)
{
    unsigned t = threadIdx.x;
    o[blockIdx.x * blockDim.x + t] = t + 4;
    return;
    __syncthreads();                                // unreachable
}
__global__ void k_fb_sync_dead_arg(unsigned *o, int never)
{
    unsigned t = threadIdx.x;
    o[blockIdx.x * blockDim.x + t] = t + 4;
    if (never == 987654321) {
        __syncthreads();
        o[0] = 1;
    }
}
__global__ void k_fb_sync_count(unsigned *o)
{
    unsigned t = threadIdx.x;
    int c = __syncthreads_count((int)(t % 3 == 0));
    o[blockIdx.x * blockDim.x + t] = (unsigned)c;
}
__global__ void k_fb_sync_and(unsigned *o)
{
    unsigned t = threadIdx.x;
    int c = __syncthreads_and((int)(t < 1000));
    o[blockIdx.x * blockDim.x + t] = (unsigned)c;
}
__global__ void k_fb_sync_or(unsigned *o)
{
    unsigned t = threadIdx.x;
    int c = __syncthreads_or((int)(t == 5));
    o[blockIdx.x * blockDim.x + t] = (unsigned)c;
}
template <int W> __global__ void k_fb_shfl_idx(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __shfl_sync(0xffffffffu, t * 3 + 1, 1, W); }
template <int W> __global__ void k_fb_shfl_up(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __shfl_up_sync(0xffffffffu, t * 3 + 1, 1, W); }
template <int W> __global__ void k_fb_shfl_down(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __shfl_down_sync(0xffffffffu, t * 3 + 1, 1, W); }
template <int W> __global__ void k_fb_shfl_xor(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __shfl_xor_sync(0xffffffffu, t * 3 + 1, 1, W); }
__global__ void k_fb_ballot(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __ballot_sync(0xffffffffu, t % 3 == 0); }
__global__ void k_fb_any(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __any_sync(0xffffffffu, t == 7); }
__global__ void k_fb_all(unsigned *o) { unsigned t = threadIdx.x; o[blockIdx.x * blockDim.x + t] = __all_sync(0xffffffffu, t < 1000); }
__global__ void k_fb_syncwarp(unsigned *o)
{
    __shared__ unsigned s[128];
    unsigned t = threadIdx.x;
    s[t] = t + 1;
    __syncwarp();
    o[blockIdx.x * blockDim.x + t] = s[t ^ 1];
}
__global__ void k_fb_amask_shfl(unsigned *o)
{
    unsigned t = threadIdx.x;
    unsigned m = __activemask();
    o[blockIdx.x * blockDim.x + t] = __shfl_down_sync(m, t, 1) + (m & 1);
}
__global__ void k_fb_cub_block_scan(int *o)
{
    typedef cub::BlockScan<int, 64> S;
    __shared__ typename S::TempStorage ts;
    int v = threadIdx.x + 1, r;
    S(ts).ExclusiveSum(v, r);
    o[blockIdx.x * 64 + threadIdx.x] = r;
}
__global__ void k_fb_cub_block_scan_incl(int *o)
{
    typedef cub::BlockScan<int, 64> S;
    __shared__ typename S::TempStorage ts;
    int v = threadIdx.x + 1, r;
    S(ts).InclusiveSum(v, r);
    o[blockIdx.x * 64 + threadIdx.x] = r;
}
__global__ void k_fb_asm(unsigned *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = t * 5;
#if defined(__x86_64__)
    asm volatile("addl $3, %0" : "+r"(x));
#else
    asm volatile("" : "+r"(x));
    x += 3;
#endif
    o[t] = x;
}
__global__ void k_fb_extcall(int *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    o[t] = ext_dev((int)t);
}
struct VBase { __device__ virtual int f(int x) const = 0; __device__ virtual ~VBase() {} };
struct VA : VBase { __device__ int f(int x) const override { return x + 1; } };
struct VB : VBase { __device__ int f(int x) const override { return x * 2; } };
__global__ void k_fb_virtual(int *o, int pick)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    VA a;
    VB b;
    const VBase *p = (pick + (int)t) & 1 ? (const VBase *)&a : (const VBase *)&b;
    o[t] = p->f((int)t);
}
__device__ __attribute__((weak)) int weak_dev(int x) { return x + 100; }
__global__ void k_fb_weak(int *o)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    o[t] = weak_dev((int)t);
}


static void group_sync()
{
    const int B = 3, N = 64;
    Dev<unsigned> o(B * N);
    auto rep = [&](const char *name) { sync(); report(name, o.get()); };
    k_fb_sync<<<B, N>>>(o.p);
    sync();
    {
        std::vector<unsigned> v = o.get();
        bool ok = true;
        for (int b = 0; b < B; ++b) for (int t = 0; t < N; ++t) ok &= v[b * N + t] == (unsigned)((N - 1 - t) * 3 + b);
        CHECK(ok);
        report("sync", v);
    }
    k_fb_sync_helper<<<B, N>>>(o.p); rep("sync_helper");
    k_fb_sync_noinline<<<B, N>>>(o.p); rep("sync_noinline");
    k_fb_sync_if<<<B, N>>>(o.p, 1); rep("sync_if");
    k_fb_sync_dead<<<B, N>>>(o.p); rep("sync_dead");
    k_fb_sync_dead_arg<<<B, N>>>(o.p, 0); rep("sync_dead_arg");
    k_fb_sync_count<<<B, N>>>(o.p);
    sync();
    {
        std::vector<unsigned> v = o.get();
        CHECK(v[0] == (unsigned)((N + 2) / 3) && v[N + 5] == (unsigned)((N + 2) / 3));
        report("sync_count", v);
    }
    k_fb_sync_and<<<B, N>>>(o.p); rep("sync_and");
    k_fb_sync_or<<<B, N>>>(o.p); rep("sync_or");
    k_fb_shfl_idx<2><<<B, N>>>(o.p); rep("shfl_idx_2");
    k_fb_shfl_idx<4><<<B, N>>>(o.p); rep("shfl_idx_4");
    k_fb_shfl_idx<8><<<B, N>>>(o.p); rep("shfl_idx_8");
    k_fb_shfl_idx<16><<<B, N>>>(o.p); rep("shfl_idx_16");
    k_fb_shfl_idx<32><<<B, N>>>(o.p); rep("shfl_idx_32");
    k_fb_shfl_up<2><<<B, N>>>(o.p); rep("shfl_up_2");
    k_fb_shfl_up<8><<<B, N>>>(o.p); rep("shfl_up_8");
    k_fb_shfl_up<32><<<B, N>>>(o.p); rep("shfl_up_32");
    k_fb_shfl_down<4><<<B, N>>>(o.p); rep("shfl_down_4");
    k_fb_shfl_down<16><<<B, N>>>(o.p); rep("shfl_down_16");
    k_fb_shfl_down<32><<<B, N>>>(o.p); rep("shfl_down_32");
    k_fb_shfl_xor<2><<<B, N>>>(o.p); rep("shfl_xor_2");
    k_fb_shfl_xor<16><<<B, N>>>(o.p); rep("shfl_xor_16");
    k_fb_shfl_xor<32><<<B, N>>>(o.p); rep("shfl_xor_32");
    k_fb_ballot<<<B, N>>>(o.p); rep("ballot");
    k_fb_any<<<B, N>>>(o.p); rep("any");
    k_fb_all<<<B, N>>>(o.p); rep("all");
    k_fb_syncwarp<<<B, N>>>(o.p); rep("syncwarp");
    {   const char *wenv = std::getenv("OCU_CPU_WARP"); int dw = wenv ? std::atoi(wenv) : 32;
        if (dw == 32) { k_fb_amask_shfl<<<B, N>>>(o.p); rep("amask_shfl"); }   // a 32-bit mask cannot name the lanes of a 64-lane warp: undefined there
    }
    Dev<int> oi(B * N);
    auto repi = [&](const char *name) { sync(); report(name, oi.get()); };
    k_fb_cub_block_scan<<<B, 64>>>(oi.p);
    sync();
    {
        std::vector<int> v = oi.get();
        bool ok = true;
        for (int b = 0; b < B; ++b) { int s = 0; for (int t = 0; t < 64; ++t) { ok &= v[b * 64 + t] == s; s += t + 1; } }
        CHECK(ok);
        report("cub_block_scan", v);
    }
    k_fb_cub_block_scan_incl<<<B, 64>>>(oi.p); repi("cub_block_scan_incl");
    k_fb_asm<<<B, N>>>(o.p);
    sync();
    {
        std::vector<unsigned> v = o.get();
        bool ok = true;
        for (unsigned t = 0; t < B * N; ++t) ok &= v[t] == t * 5 + 3;
        CHECK(ok);
        report("asm", v);
    }
    k_fb_extcall<<<B, N>>>(oi.p);
    sync();
    {
        std::vector<int> v = oi.get();
        bool ok = true;
        for (int t = 0; t < B * N; ++t) ok &= v[t] == t * 3 + 1;
        CHECK(ok);
        report("extcall", v);
    }
    k_fb_virtual<<<B, N>>>(oi.p, 0);
    sync();
    {
        std::vector<int> v = oi.get();
        bool ok = true;
        for (int t = 0; t < B * N; ++t) ok &= v[t] == ((t & 1) ? t + 1 : t * 2);
        CHECK(ok);
        report("virtual", v);
    }
    k_fb_weak<<<B, N>>>(oi.p);
    sync();
    {
        std::vector<int> v = oi.get();
        bool ok = true;
        for (int t = 0; t < B * N; ++t) ok &= v[t] == t + 100;
        CHECK(ok);
        report("weak", v);
    }
}

// ------------------------------------------------------------ launches through a pointer

__global__ void k_ptr_a(int *o) { o[blockIdx.x * blockDim.x + threadIdx.x] = (int)threadIdx.x * 2 + 1; }
__global__ void k_ptr_b(int *o) { o[blockIdx.x * blockDim.x + threadIdx.x] = (int)threadIdx.x * 3 + 2; }
__global__ void k_ptr_sync(int *o)       // same type as k_ptr_a, but with a barrier: declines at run time
{
    __shared__ int s[256];
    s[threadIdx.x] = (int)threadIdx.x * 5;
    __syncthreads();
    o[blockIdx.x * blockDim.x + threadIdx.x] = s[blockDim.x - 1 - threadIdx.x];
}
__global__ void k_ptr_axpy(float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    y[i] += a * x[i];
}

template <class K, class... A>
static void launch_k(K k, dim3 g, dim3 b, A... a) { k<<<g, b>>>(a...); }

static int g_side;
static void launch_side(void (*k)(int *), int *o)
{
    g_side++;                       // more than copying arguments before the call
    int local = g_side * 2;
    (void)local;
    k<<<2, 32>>>(o);
}

typedef void (*kfn)(int *);

static void group_ptr()
{
    const int B = 2, N = 32;
    Dev<int> o(B * N);
    auto expect = [&](const char *name, int mul, int add) {
        sync();
        std::vector<int> v = o.get();
        bool ok = true;
        for (int b = 0; b < B; ++b) for (int t = 0; t < N; ++t) ok &= v[b * N + t] == t * mul + add;
        CHECK(ok);
        report(name, v);
    };
    {   // cudaLaunchKernel
        void *args[] = {&o.p};
        OK(cudaLaunchKernel(k_ptr_a, dim3(B), dim3(N), args, 0, 0));
        expect("ptr_cudaLaunchKernel", 2, 1);
        OK(cudaLaunchKernel(k_ptr_b, dim3(B), dim3(N), args, 0, 0));
        expect("ptr_cudaLaunchKernel_b", 3, 2);
    }
    launch_k(k_ptr_a, dim3(B), dim3(N), o.p);          // a template helper taking the kernel
    expect("ptr_template_helper", 2, 1);
    launch_k(k_ptr_b, dim3(B), dim3(N), o.p);
    expect("ptr_template_helper_b", 3, 2);
    for (int c = 0; c < 2; ++c) {                       // auto k = c ? k1 : k2
        auto k = c ? k_ptr_b : k_ptr_a;
        k<<<B, N>>>(o.p);
        expect(c ? "ptr_ternary_b" : "ptr_ternary_a", c ? 3 : 2, c ? 2 : 1);
    }
    {   // a kernel pointer kept in memory (a table in host memory)
        struct Table { kfn fn[3]; } tab = {{k_ptr_a, k_ptr_b, k_ptr_sync}};
        for (int i = 0; i < 2; ++i) {
            kfn kt = tab.fn[i];
            kt<<<B, N>>>(o.p);
            expect(i ? "ptr_table_b" : "ptr_table_a", i ? 3 : 2, i ? 2 : 1);
        }
    }
    {   // two kernels of one type, one with a barrier: it declines at run time and runs on fibers
        k_ptr_a<<<B, N>>>(o.p);
        expect("ptr_mixed_a", 2, 1);
        kfn k = k_ptr_sync;
        k<<<B, N>>>(o.p);
        sync();
        std::vector<int> v = o.get();
        bool ok = true;
        for (int b = 0; b < B; ++b) for (int t = 0; t < N; ++t) ok &= v[b * N + t] == (N - 1 - t) * 5;
        CHECK(ok);
        report("ptr_mixed_sync", v);
        k = k_ptr_b;
        k<<<B, N>>>(o.p);
        expect("ptr_mixed_b", 3, 2);
    }
    {   // a helper that does more than forward its arguments
        launch_side(k_ptr_a, o.p);
        expect("ptr_side_effect", 2, 1);
        launch_side(k_ptr_sync, o.p);
        sync();
        report("ptr_side_effect_sync", o.get());
    }
    {   // a kernel of another translation unit, through a pointer and directly
        void (*pe)(int *) = k_ext;
        pe<<<B, N>>>(o.p);
        sync();
        std::vector<int> v = o.get();
        bool ok = true;
        for (int b = 0; b < B; ++b) for (int t = 0; t < N; ++t) ok &= v[b * N + t] == t * 7 + b;
        CHECK(ok);
        report("ptr_other_tu", v);
        k_ext<<<B, N>>>(o.p);
        sync();
        report("direct_other_tu", o.get());
    }
    {   // the same inline kernel launched from this object and from the other
        Dev<int> a(64), b(64);
        launch_inl(a.p, 64);
        launch_inl_ext(b.p, 64);
        sync();
        std::vector<int> va = a.get(), vb = b.get();
        bool ok = true;
        for (int t = 0; t < 64; ++t) ok &= va[t] == t * 11 + 3 && vb[t] == t * 11 + 3;
        CHECK(ok);
        report("inline_both", va);
    }
}

// ------------------------------------------------------------ large grids

__global__ void k_lp_big(unsigned *sum)
{
    atomicAdd(sum, blockIdx.x ^ threadIdx.x);
}

static void group_big()
{
    Dev<unsigned> s(1);
    const unsigned nb = 300000;
    k_lp_big<<<nb, 2>>>(s.p);
    sync();
    unsigned want = 0;
    for (unsigned b = 0; b < nb; ++b) want += (b ^ 0u) + (b ^ 1u);
    std::vector<unsigned> v = s.get();
    CHECK(v[0] == want);
    report("big_grid", v);
}

int main(int argc, char **argv)
{
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) groups.push_back(argv[i]);
    if (groups.empty()) groups = {"core", "idx", "early", "locals", "fp", "mem", "order", "ptr", "sync"};
    for (auto &g : groups) {
        std::printf("group %s\n", g.c_str());
        if (g == "core") group_core();
        else if (g == "idx") group_idx();
        else if (g == "early") group_early();
        else if (g == "locals") group_locals();
        else if (g == "fp") group_fp();
        else if (g == "mem") group_mem();
        else if (g == "sshared") group_sshared();
        else if (g == "order") group_order();
        else if (g == "ptr") group_ptr();
        else if (g == "many") { bool ok = false; unsigned long long h = run_many(&ok); CHECK(ok); std::printf("case ptr_many %016llx\n", h); }
        else if (g == "sync") group_sync();
        else if (g == "stack") group_stack();
        else if (g == "big") group_big();
        else { std::printf("unknown group %s\n", g.c_str()); return 2; }
    }
    std::printf("cases: %d checks, %d failed, %d expected-fail\n", g_checks, g_fail, g_xfail);
    return g_fail ? 1 : 0;
}
