// mmvq_loop.cpp -- the kernel of bench_region_loops.cu written by hand in
// the form the region-loop pass would produce (docs/cpu-region-loops.md,
// "Review decisions"):
//
//   - the body is cut at the block barrier and at each warp shuffle into
//     regions R0..R6, each compiled once as `next_pc = region_k(ctx, tid)`;
//   - values live across a cut are kept in one array per value and cut
//     (structure of arrays, index = thread);
//   - a shuffle `x = __shfl_xor_sync(full, v, o)` is the region before
//     writing v into its cut array and the region after reading
//     cut[lane ^ o] of the same warp (own value for a source past a short
//     last warp, as the fiber path's helpers do);
//   - driver 1, the uniform fast loop: one loop over the live threads per
//     region, the region inlined into it; each thread's next pc is stored
//     and compared once with the expected one (a thread that returns
//     becomes inactive, "exit" is always accepted). On a mismatch the
//     block continues in the dispatcher;
//   - driver 2, the pc dispatcher: `pc[t] = R[pc[t]](ctx, t)` per thread in
//     thread order, round by round. (The arrival checks that make it a
//     full stand-in for the fiber schedule are stage 3 and not modelled;
//     for this kernel every thread reaches the same cuts.)
//
// Compiled three times (MMVQ_LOOP_NAME): vectoriser off, default, and
// x86-64-v3; see tuning/CMakeLists.txt.
#include "mmvq_rl.h"

#include <atomic>
#include <cstdint>
#include <cstring>

#ifndef MMVQ_LOOP_NAME
#error "define MMVQ_LOOP_NAME"
#endif

#define RL_INLINE inline __attribute__((always_inline))

namespace {

constexpr int NT = MMVQ_THREADS, NW = MMVQ_NWARPS, W = MMVQ_WARP;
constexpr int8_t PC_EXIT = -1;
constexpr int NREGIONS = 7;

struct alignas(64) ctx {
    // uniform: kernel arguments and block index
    const mmvq_q8_0 *vx;
    const mmvq_q8_1 *vy;
    float *dst;
    int ncols_x, stride_row_x, bx;
    // __shared__ float tmp_shared[nwarps - 1][warp_size]
    alignas(64) float tmp_shared[NW - 1][W];
    // cut arrays: tmp across the barrier, x across each of the 5 shuffles
    alignas(64) float c_bar_tmp[NT];
    alignas(64) float c_shfl[5][NT];
    // driver state
    alignas(64) int8_t pc[NT];
    uint32_t live[NW];
};

RL_INLINE float f16_to_f32(uint16_t h)
{
    uint32_t s = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) {
            x = s;
        } else { // subnormal: normalise
            int ee = 1;
            while (!(m & 0x400u)) { m <<= 1; --ee; }
            x = s | (uint32_t(ee + 112) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) {
        x = s | 0x7f800000u | (m << 13);
    } else {
        x = s | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

RL_INLINE int load_i32(const int8_t *p, int i32)
{
    int v;
    std::memcpy(&v, p + 4 * i32, 4);
    return v;
}

RL_INLINE int dp4a_bytes(int a, int b, int c)
{
    int8_t a8[4], b8[4];
    std::memcpy(a8, &a, 4);
    std::memcpy(b8, &b, 4);
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

RL_INLINE float vec_dot(const mmvq_q8_0 *bx, const mmvq_q8_1 *by, int iqs)
{
    int sumi = 0;
    for (int i = 0; i < MMVQ_VDR; ++i)
        sumi = dp4a_bytes(load_i32(bx->qs, iqs + i), load_i32(by->qs, iqs + i), sumi);
    const float d8_0 = f16_to_f32(bx->d), d8_1 = f16_to_f32(by->d);
    return d8_0 * d8_1 * float(sumi);
}

// Source lane of a xor shuffle with width 32 in a warp of `nlanes` lanes.
RL_INLINE int xor_src(int lane, int o, int nlanes)
{
    int src = lane ^ o;
    return src < nlanes ? src : lane;
}

// ---- regions -----------------------------------------------------------------

// R0: entry to the barrier.
RL_INLINE int8_t region0(ctx &c, int tid)
{
    constexpr int qi = MMVQ_QI, vdr = MMVQ_VDR;
    constexpr int blocks_per_iter = vdr * NW * W / qi;
    const int tx = tid % W, ty = tid / W;
    const int blocks_per_row_x = c.ncols_x / MMVQ_QK;
    const int kbx_offset = c.bx * c.stride_row_x;
    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot(c.vx + kbx_offset + kbx, c.vy + kbx, kqs);
    }
    if (ty > 0) c.tmp_shared[ty - 1][tx] = tmp;
    c.c_bar_tmp[tid] = tmp;
    return 1;
}

// R1: after the barrier; threads of warps 1..3 return; sum the partial
// sums, up to the first shuffle.
RL_INLINE int8_t region1(ctx &c, int tid)
{
    const int tx = tid % W, ty = tid / W;
    if (ty > 0) return PC_EXIT;
    float tmp = c.c_bar_tmp[tid];
    for (int l = 0; l < NW - 1; ++l) tmp += c.tmp_shared[l][tx];
    c.c_shfl[0][tid] = tmp;
    return 2;
}

// R2..R5: tmp += shfl_xor(tmp, 16 >> k), up to the next shuffle.
template <int K>
RL_INLINE int8_t region_shfl(ctx &c, int tid)
{
    const int lane = tid % W, base = tid - lane;
    const float tmp = c.c_shfl[K][tid] + c.c_shfl[K][base + xor_src(lane, 16 >> K, W)];
    c.c_shfl[K + 1][tid] = tmp;
    return int8_t(K + 3);
}

// R6: the last shuffle and the store.
RL_INLINE int8_t region6(ctx &c, int tid)
{
    const int lane = tid % W, base = tid - lane;
    const float tmp = c.c_shfl[4][tid] + c.c_shfl[4][base + xor_src(lane, 1, W)];
    if (lane == 0) c.dst[c.bx] = tmp;
    return PC_EXIT;
}

RL_INLINE int8_t run_region(int k, ctx &c, int tid)
{
    switch (k) {
    case 0: return region0(c, tid);
    case 1: return region1(c, tid);
    case 2: return region_shfl<0>(c, tid);
    case 3: return region_shfl<1>(c, tid);
    case 4: return region_shfl<2>(c, tid);
    case 5: return region_shfl<3>(c, tid);
    default: return region6(c, tid);
    }
}

// ---- driver 1: uniform fast loop ----------------------------------------------

// One region over all live threads. Returns false when two live threads
// asked for different next regions (then c.pc holds every thread's pc).
template <typename R>
RL_INLINE bool region_loop(ctx &c, int8_t expected, R region)
{
    unsigned bad = 0;
    for (int w = 0; w < NW; ++w) {
        const uint32_t live = c.live[w];
        if (live == 0) continue;
        uint32_t gone = 0;
        if (live == 0xffffffffu) {
            for (int l = 0; l < W; ++l) {
                const int8_t pc = region(c, w * W + l);
                c.pc[w * W + l] = pc;
                gone |= uint32_t(pc == PC_EXIT) << l;
                bad |= unsigned(pc != expected) & unsigned(pc != PC_EXIT);
            }
        } else {
            for (int l = 0; l < W; ++l) {
                if (!((live >> l) & 1u)) continue;
                const int8_t pc = region(c, w * W + l);
                c.pc[w * W + l] = pc;
                gone |= uint32_t(pc == PC_EXIT) << l;
                bad |= unsigned(pc != expected) & unsigned(pc != PC_EXIT);
            }
        }
        c.live[w] = live & ~gone;
    }
    return bad == 0;
}

// ---- driver 2: pc dispatcher --------------------------------------------------

__attribute__((noinline)) int8_t region_call(int k, ctx &c, int tid) { return run_region(k, c, tid); }

void dispatch(ctx &c)
{
    for (;;) {
        bool any = false;
        for (int t = 0; t < NT; ++t) {
            if (c.pc[t] == PC_EXIT) continue;
            c.pc[t] = region_call(c.pc[t], c, t);
            any = true;
        }
        if (!any) return;
    }
}

// ---- one block, the grid --------------------------------------------------------

// Returns 1 when the block fell back to the dispatcher.
int run_block(ctx &c, int form)
{
    for (int w = 0; w < NW; ++w) c.live[w] = 0xffffffffu;
    if (form == 1) {
        std::memset(c.pc, 0, sizeof(c.pc));
        dispatch(c);
        return 1;
    }
    // The static successor of each region is unique here, so `expected`
    // is a constant; a region with several successors would take the
    // first live thread's pc instead.
    if (!region_loop(c, 1, [](ctx &cc, int t) { return region0(cc, t); })) goto fallback;
    if (!region_loop(c, 2, [](ctx &cc, int t) { return region1(cc, t); })) goto fallback;
    if (!region_loop(c, 3, [](ctx &cc, int t) { return region_shfl<0>(cc, t); })) goto fallback;
    if (!region_loop(c, 4, [](ctx &cc, int t) { return region_shfl<1>(cc, t); })) goto fallback;
    if (!region_loop(c, 5, [](ctx &cc, int t) { return region_shfl<2>(cc, t); })) goto fallback;
    if (!region_loop(c, 6, [](ctx &cc, int t) { return region_shfl<3>(cc, t); })) goto fallback;
    if (!region_loop(c, PC_EXIT, [](ctx &cc, int t) { return region6(cc, t); })) goto fallback;
    return 0;
fallback:
    for (int w = 0; w < NW; ++w)
        for (int l = 0; l < W; ++l)
            if (!((c.live[w] >> l) & 1u)) c.pc[w * W + l] = PC_EXIT;
    dispatch(c);
    return 1;
}

struct grid_job {
    const mmvq_args *a;
    int form, chunk;
    std::atomic<int> next{0};
    std::atomic<long> fallbacks{0};
};

void grid_worker(void *p)
{
    grid_job &j = *static_cast<grid_job *>(p);
    static thread_local ctx c;
    c.vx = j.a->vx;
    c.vy = j.a->vy;
    c.dst = j.a->dst;
    c.ncols_x = j.a->ncols_x;
    c.stride_row_x = j.a->stride_row_x;
    long fb = 0;
    for (;;) {
        int b0 = j.next.fetch_add(j.chunk, std::memory_order_relaxed);
        if (b0 >= j.a->nrows) break;
        int b1 = b0 + j.chunk < j.a->nrows ? b0 + j.chunk : j.a->nrows;
        for (int b = b0; b < b1; ++b) {
            c.bx = b;
            fb += run_block(c, j.form);
        }
    }
    if (fb) j.fallbacks.fetch_add(fb, std::memory_order_relaxed);
}

static_assert(NREGIONS == 7, "regions");

} // namespace

extern "C" long MMVQ_LOOP_NAME(const mmvq_args *a, int nthreads, int form)
{
    grid_job j;
    j.a = a;
    j.form = form;
    int per = a->nrows / (nthreads * 8);
    j.chunk = per < 1 ? 1 : per;
    mmvq_pool_run(nthreads, grid_worker, &j);
    return j.fallbacks.load();
}
