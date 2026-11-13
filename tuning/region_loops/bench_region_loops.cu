// bench_region_loops: stage 0 of docs/cpu-region-loops.md. Measures one
// GPU thread of llama.cpp's mul_mat_vec_q (Q8_0 x Q8_1, one column, see
// mmvq_rl.h) in three forms on the CPU path:
//   fiber   the kernel below, compiled by the CPU path and launched through
//           the runtime (one fiber per GPU thread, as today);
//   loop    the same kernel written by hand in region-loop form
//           (mmvq_loop.cpp), compiled with vectorisation off ("scalar"),
//           with the compiler's default vectoriser ("vec"), and for
//           x86-64-v3 ("avx2");
//   disp    the region functions run by the pc dispatcher (the fallback).
// Every form gets the same inputs and must give the same bits.
//
//   OCU_CPU_THREADS=n bench_region_loops [reps] [loop threads...]
//   (default 41 reps; loop forms on n threads, like the fiber form)
//
// Prints one line per shape, form and thread count: median and slow-10 %
// (90th percentile) time of one launch, the same per GPU thread, and the
// fixed launch cost measured with an empty launch.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mmvq_rl.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---- the kernel (fiber form) -------------------------------------------------

struct blk_q8_0 { half d; int8_t qs[MMVQ_QK]; };
struct blk_q8_1 { half2 ds; int8_t qs[MMVQ_QK]; };
static_assert(sizeof(blk_q8_0) == sizeof(mmvq_q8_0), "layout");
static_assert(sizeof(blk_q8_1) == sizeof(mmvq_q8_1), "layout");

static __device__ __forceinline__ int int_b2(const void *x, int i32)
{
    const uint16_t *x16 = (const uint16_t *)x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}

static __device__ __forceinline__ int int_b4(const void *x, int i32) { return ((const int *)x)[i32]; }

// dp4a as the CPU path compiles it (compute capability 5.0: byte products)
static __device__ __forceinline__ int dp4a_bytes(int a, int b, int c)
{
    const int8_t *a8 = (const int8_t *)&a;
    const int8_t *b8 = (const int8_t *)&b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(const void *vbq, const blk_q8_1 *bq8_1,
                                                          int kbx, int iqs)
{
    const blk_q8_0 *bq8_0 = (const blk_q8_0 *)vbq + kbx;
    int v[MMVQ_VDR], u[MMVQ_VDR];
#pragma unroll
    for (int i = 0; i < MMVQ_VDR; ++i) {
        v[i] = int_b2(bq8_0->qs, iqs + i);
        u[i] = int_b4(bq8_1->qs, iqs + i);
    }
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < MMVQ_VDR; ++i) sumi = dp4a_bytes(v[i], u[i], sumi);
    const float d8_0 = __half2float(bq8_0->d);
    const float d8_1 = __half2float(__low2half(bq8_1->ds));
    return d8_0 * d8_1 * ((float)sumi);
}

__global__ void mmvq_q8_0_kernel(const void *vx, const void *vy, float *dst, int ncols_x, int stride_row_x)
{
    constexpr int qi = MMVQ_QI, vdr = MMVQ_VDR, nwarps = MMVQ_NWARPS, warp_size = MMVQ_WARP;
    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / MMVQ_QK;
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;

    float tmp = 0.0f;
    const blk_q8_1 *y = (const blk_q8_1 *)vy;
    const int kbx_offset = row0 * stride_row_x;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx;
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot_q8_0_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    __shared__ float tmp_shared[nwarps - 1][warp_size];
    if (threadIdx.y > 0) tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int l = 0; l < nwarps - 1; ++l) tmp += tmp_shared[l][threadIdx.x];
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        tmp += __shfl_xor_sync(0xffffffff, tmp, offset, warp_size);
    if (threadIdx.x == 0) dst[row0] = tmp;
}

__global__ void empty_kernel() {}

// ---- driver ------------------------------------------------------------------

namespace {

using clk = std::chrono::steady_clock;

double now_ns() { return std::chrono::duration<double, std::nano>(clk::now().time_since_epoch()).count(); }

uint16_t f32_to_f16_bits(float f)
{   // round to nearest even, normal range only (scales are in [1e-3, 1e-1])
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, exp = (x >> 23) & 0xffu, man = x & 0x7fffffu;
    int e = int(exp) - 127 + 15;
    uint32_t m = man >> 13, rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (m & 1u))) ++m;
    if (m == 0x400u) { m = 0; ++e; }
    return uint16_t(sign | (uint32_t(e) << 10) | m);
}

struct shape { const char *name; int nrows, ncols; };

struct stat2 { double med, p90; };

stat2 stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    auto pct = [&](double f) {
        double i = f * double(v.size() - 1);
        size_t lo = size_t(i);
        size_t hi = std::min(lo + 1, v.size() - 1);
        return v[lo] + (v[hi] - v[lo]) * (i - double(lo));
    };
    return {pct(0.5), pct(0.9)};
}

int env_threads()
{
    const char *s = std::getenv("OCU_CPU_THREADS");
    return s ? std::atoi(s) : 0;
}

} // namespace

int main(int argc, char **argv)
{
    int reps = argc > 1 ? std::atoi(argv[1]) : 41;
    std::vector<int> threads;
    for (int i = 2; i < argc; ++i) threads.push_back(std::atoi(argv[i]));
    // The fiber form uses the runtime's pool, sized by OCU_CPU_THREADS: run
    // one thread count per process (the loop forms default to the same).
    const int fiber_threads = env_threads();
    if (threads.empty()) threads.push_back(fiber_threads > 0 ? fiber_threads : 1);

    // Qwen2.5-0.5B shapes: attention q/o (896 x 896), FFN up/gate
    // (4864 x 896), FFN down (896 x 4864).
    const shape shapes[] = {{"q896x896", 896, 896}, {"up4864x896", 4864, 896}, {"down896x4864", 896, 4864}};

    std::printf("# shape form threads launch_med_us launch_p90_us ns_per_thread_med ns_per_thread_p90 "
                "fallback_blocks\n");
    for (const shape &s : shapes) {
        const int bpr = s.ncols / MMVQ_QK;
        std::vector<mmvq_q8_0> hx(size_t(s.nrows) * bpr);
        std::vector<mmvq_q8_1> hy(bpr);
        uint32_t seed = 12345u + uint32_t(s.nrows) * 7u + uint32_t(s.ncols);
        auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        for (auto &b : hx) {
            b.d = f32_to_f16_bits(1e-3f + float(rnd() % 1000) * 1e-4f);
            for (auto &q : b.qs) q = int8_t(int(rnd() % 255) - 127);
        }
        for (auto &b : hy) {
            b.d = f32_to_f16_bits(1e-3f + float(rnd() % 1000) * 1e-4f);
            b.s = 0;
            for (auto &q : b.qs) q = int8_t(int(rnd() % 255) - 127);
        }
        const double nthr = double(s.nrows) * MMVQ_THREADS;

        // fiber form
        void *dx, *dy;
        float *dd;
        cudaMalloc(&dx, hx.size() * sizeof(mmvq_q8_0));
        cudaMalloc(&dy, hy.size() * sizeof(mmvq_q8_1));
        cudaMalloc((void **)&dd, size_t(s.nrows) * sizeof(float));
        cudaMemcpy(dx, hx.data(), hx.size() * sizeof(mmvq_q8_0), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy.data(), hy.size() * sizeof(mmvq_q8_1), cudaMemcpyHostToDevice);
        std::vector<float> ref(s.nrows);
        {
            std::vector<double> t, te;
            for (int r = 0; r < reps + 3; ++r) {
                double t0 = now_ns();
                mmvq_q8_0_kernel<<<s.nrows, dim3(MMVQ_WARP, MMVQ_NWARPS)>>>(dx, dy, dd, s.ncols, bpr);
                if (cudaDeviceSynchronize() != cudaSuccess) { std::fprintf(stderr, "launch failed\n"); return 1; }
                double t1 = now_ns();
                empty_kernel<<<1, 1>>>();
                cudaDeviceSynchronize();
                double t2 = now_ns();
                if (r >= 3) { t.push_back(t1 - t0); te.push_back(t2 - t1); }
            }
            cudaMemcpy(ref.data(), dd, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
            stat2 a = stats(t), e = stats(te);
            std::printf("%s fiber %d %.1f %.1f %.2f %.2f 0\n", s.name, fiber_threads, a.med / 1e3, a.p90 / 1e3,
                        a.med / nthr, a.p90 / nthr);
            std::printf("%s fiber-empty-launch %d %.1f %.1f - - -\n", s.name, fiber_threads, e.med / 1e3,
                        e.p90 / 1e3);
        }
        cudaFree(dx);
        cudaFree(dy);
        cudaFree(dd);

        // loop forms
        struct lf { const char *name; long (*fn)(const mmvq_args *, int, int); int form; };
        const lf forms[] = {{"loop-scalar", mmvq_loop_scalar, 0},
                            {"loop-vec", mmvq_loop_vec, 0},
                            {"loop-avx2", mmvq_loop_avx2, 0},
                            {"disp-scalar", mmvq_loop_scalar, 1}};
        std::vector<float> out(s.nrows);
        mmvq_args a = {hx.data(), hy.data(), out.data(), s.ncols, bpr, s.nrows};
        for (int nt : threads) {
            for (const lf &f : forms) {
                std::vector<double> t;
                long fb = 0;
                for (int r = 0; r < reps + 3; ++r) {
                    std::fill(out.begin(), out.end(), -1.0f);
                    double t0 = now_ns();
                    fb = f.fn(&a, nt, f.form);
                    double t1 = now_ns();
                    if (r >= 3) t.push_back(t1 - t0);
                }
                if (std::memcmp(out.data(), ref.data(), out.size() * sizeof(float)) != 0) {
                    size_t i = 0;
                    while (i < out.size() && std::memcmp(&out[i], &ref[i], 4) == 0) ++i;
                    std::printf("MISMATCH %s %s row %zu: %.9g vs fiber %.9g\n", s.name, f.name, i,
                                double(out[i]), double(ref[i]));
                    return 1;
                }
                stat2 st = stats(t);
                std::printf("%s %s %d %.1f %.1f %.2f %.2f %ld\n", s.name, f.name, nt, st.med / 1e3,
                            st.p90 / 1e3, st.med / nthr, st.p90 / nthr, fb);
            }
            std::vector<double> te;
            for (int r = 0; r < reps + 3; ++r) {
                double t0 = now_ns();
                mmvq_pool_empty(nt);
                double t1 = now_ns();
                if (r >= 3) te.push_back(t1 - t0);
            }
            stat2 e = stats(te);
            std::printf("%s loop-empty-launch %d %.1f %.1f - - -\n", s.name, nt, e.med / 1e3, e.p90 / 1e3);
        }
        std::printf("%s sample dst[0..3] = %.9g %.9g %.9g %.9g (bit-identical across forms)\n", s.name,
                    double(ref[0]), double(ref[1]), double(ref[2]), double(ref[3]));
        std::fflush(stdout);
    }
    return 0;
}
