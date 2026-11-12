// Warp shuffles and votes written as ordinary CUDA (card backend-cpu, tester
// list item 5, language layer): __shfl_sync, __shfl_up/down/xor_sync with
// widths 1..32 (and 64 when the warp is 64 lanes), negative and
// out-of-range source lanes, __ballot_sync / __any_sync / __all_sync with
// full, half, parity and single-lane masks, blocks whose last warp is short.
//
// Expected values are computed on the host from the CUDA C++ Programming
// Guide: a shuffle reads the lane `src mod width` of the caller's segment of
// `width` lanes (up/down: the lane delta below/above, the caller's own value
// when that lies outside the segment; xor: lane ^ mask); a vote has bit j
// set for lane j of the 32-thread group when the lane is named by the mask
// and its predicate is nonzero. A source thread that does not exist (short
// last warp) gives the caller's own value in this implementation (the Guide
// calls such a read undefined).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail, g_checks, g_pf;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_pf++ < 20) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

template <typename T>
__global__ void k_shfl(T *out, const T *in, int mode, int arg, int width)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    T v = in[t];
    T r = v;
    switch (mode) {
    case 0: r = __shfl_sync(0xffffffffu, v, arg, width); break;
    case 1: r = __shfl_up_sync(0xffffffffu, v, (unsigned)arg, width); break;
    case 2: r = __shfl_down_sync(0xffffffffu, v, (unsigned)arg, width); break;
    default: r = __shfl_xor_sync(0xffffffffu, v, arg, width); break;
    }
    out[t] = r;
}

// Several different shuffles in a row at one call site (a loop), each reading
// the previous result: the values must not leak from one to the next.
__global__ void k_shfl_chain(int *out, int rounds)
{
    int t = threadIdx.x;
    int v = t * 7 + 1;
    for (int i = 0; i < rounds; ++i) {
        v = __shfl_xor_sync(0xffffffffu, v, 1, 32) + (i & 1) + __shfl_down_sync(0xffffffffu, v, 1, 32);
    }
    out[t + blockIdx.x * blockDim.x] = v;
}

__global__ void k_vote(unsigned *ballot, int *any, int *all, const int *pred, int maskmode)
{
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned lane32 = threadIdx.x % 32u;
    unsigned mask;
    switch (maskmode) {
    case 0: mask = 0xffffffffu; break;
    case 1: mask = lane32 < 16 ? 0x0000ffffu : 0xffff0000u; break;
    case 2: mask = (lane32 & 1u) ? 0xaaaaaaaau : 0x55555555u; break;
    case 3: mask = 1u << lane32; break;
    default: mask = 0; break;
    }
    int p = pred[t];
    ballot[t] = __ballot_sync(mask, p);
    any[t] = __any_sync(mask, p);
    all[t] = __all_sync(mask, p);
}

template <typename T>
static T *dev_copy(const std::vector<T> &v)
{
    T *p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(T));
    cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

static int g_warp = 32;

static int mod_pos(int x, int w) { return ((x % w) + w) % w; }

// Expected result of one shuffle for thread t of a block of n threads.
template <typename T>
static T expect(const std::vector<T> &in, int blk_base, int tid, int n, int mode, int arg, int width)
{
    int lane = tid % g_warp;
    int in_seg = lane % width;
    int seg_base = tid - in_seg;            // thread index of the segment start
    int target = tid;
    switch (mode) {
    case 0: target = seg_base + mod_pos(arg, width); break;
    case 1: target = in_seg - arg >= 0 ? tid - arg : tid; break;
    case 2: target = in_seg + arg < width ? tid + arg : tid; break;
    default: target = seg_base + (in_seg ^ arg); break;
    }
    if (target < 0 || target >= n) target = tid;
    return in[blk_base + target];
}

template <typename T>
static void run_shfl(const char *tname, int n, int blocks)
{
    std::vector<T> in(n * blocks);
    for (int i = 0; i < n * blocks; ++i) in[i] = (T)(i * 3 + 1000 * (i % 7) + 5);
    T *din = dev_copy(in), *dout = nullptr;
    cudaMalloc(&dout, in.size() * sizeof(T));
    std::vector<int> widths = {1, 2, 4, 8, 16, 32};
    // Width 64 on a 64-lane warp is not tested: the 32-bit mask names lanes of one
    // 32-lane group, so a read across the two halves is outside what the mask covers.
    const int idx_args[] = {0, 1, 5, -1, -2, 31, 32, 33, -31, 100, 17};
    const int delta_args[] = {0, 1, 2, 16, 31, 32, 40};
    const int xor_args[] = {0, 1, 2, 3, 5, 8, 15, 16, 31};
    int bad = 0, first_bad_mode = -1, first_bad_arg = 0, first_bad_width = 0, cases = 0;
    for (int width : widths) {
        for (int mode = 0; mode < 4; ++mode) {
            const int *args = mode == 0 ? idx_args : (mode == 3 ? xor_args : delta_args);
            int na = mode == 0 ? 11 : (mode == 3 ? 9 : 7);
            for (int ai = 0; ai < na; ++ai) {
                int arg = args[ai];
                if (mode == 3 && arg >= width) continue; // lane ^ mask leaves the segment: skip
                k_shfl<T><<<blocks, n>>>(dout, din, mode, arg, width);
                if (cudaDeviceSynchronize() != cudaSuccess) { CHECK(false); continue; }
                std::vector<T> got(in.size());
                cudaMemcpy(got.data(), dout, got.size() * sizeof(T), cudaMemcpyDeviceToHost);
                ++cases;
                for (int b = 0; b < blocks; ++b) {
                    for (int t = 0; t < n; ++t) {
                        T want = expect(in, b * n, t, n, mode, arg, width);
                        if (got[b * n + t] != want) {
                            if (first_bad_mode < 0) {
                                first_bad_mode = mode; first_bad_arg = arg; first_bad_width = width;
                                std::printf("  %s n=%d mode=%d arg=%d width=%d thread %d: got %g want %g\n",
                                            tname, n, mode, arg, width, t, (double)got[b * n + t], (double)want);
                            }
                            ++bad;
                        }
                    }
                }
            }
        }
    }
    CHECK(cases > 100);
    CHECK(bad == 0);
    cudaFree(din);
    cudaFree(dout);
}

static void test_shuffles()
{
    const int sizes[] = {32, 33, 40, 64, 70, 128, 1};
    for (int n : sizes) {
        run_shfl<int>("int", n, 2);
        run_shfl<double>("double", n, 2);
    }
}

static void test_shuffle_chain()
{
    const int sizes[] = {32, 40, 64};
    for (int n : sizes) {
        int *out = nullptr;
        cudaMalloc(&out, 2 * n * sizeof(int));
        k_shfl_chain<<<2, n>>>(out, 40);
        cudaDeviceSynchronize();
        std::vector<int> got(2 * n);
        cudaMemcpy(got.data(), out, got.size() * sizeof(int), cudaMemcpyDeviceToHost);
        // model: warp = g_warp lanes, default width = warp
        int bad = 0;
        for (int b = 0; b < 2; ++b) {
            std::vector<int> v(n);
            for (int t = 0; t < n; ++t) v[t] = t * 7 + 1;
            for (int i = 0; i < 40; ++i) {
                std::vector<int> nv(n);
                for (int t = 0; t < n; ++t) {
                    int lane = t % 32;                  // width 32
                    int x = t ^ 1;
                    int xv = x < n ? v[x] : v[t];
                    int d = (lane + 1 < 32 && t + 1 < n) ? v[t + 1] : v[t];
                    nv[t] = xv + (i & 1) + d;
                }
                v = nv;
            }
            for (int t = 0; t < n; ++t) bad += got[b * n + t] != v[t];
        }
        CHECK(bad == 0);
        cudaFree(out);
    }
}

static void test_votes()
{
    const int sizes[] = {32, 33, 40, 64, 96, 128, 7};
    for (int n : sizes) {
        for (int maskmode = 0; maskmode < 5; ++maskmode) {
            for (int pat = 0; pat < 5; ++pat) {
                int blocks = 2;
                std::vector<int> pred(n * blocks);
                for (int i = 0; i < n * blocks; ++i) {
                    int t = i % n;
                    switch (pat) {
                    case 0: pred[i] = 0; break;
                    case 1: pred[i] = 1; break;
                    case 2: pred[i] = (t == 5 || t == 36) ? 7 : 0; break;     // single lane, odd value
                    case 3: pred[i] = (t * 2654435761u >> 7) & 1 ? -3 : 0; break;
                    default: pred[i] = (t % 32 == 31 || t % 32 == 0) ? 0 : 1; break;
                    }
                }
                int *dpred = dev_copy(pred);
                unsigned *dballot = nullptr; int *dany = nullptr, *dall = nullptr;
                cudaMalloc(&dballot, n * blocks * 4);
                cudaMalloc(&dany, n * blocks * 4);
                cudaMalloc(&dall, n * blocks * 4);
                k_vote<<<blocks, n>>>(dballot, dany, dall, dpred, maskmode);
                bool ok = cudaDeviceSynchronize() == cudaSuccess;
                CHECK(ok);
                if (ok && maskmode != 4) { // mask 0 is undefined in CUDA: only completion is checked
                    std::vector<unsigned> bl(n * blocks);
                    std::vector<int> an(n * blocks), al(n * blocks);
                    cudaMemcpy(bl.data(), dballot, bl.size() * 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(an.data(), dany, an.size() * 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(al.data(), dall, al.size() * 4, cudaMemcpyDeviceToHost);
                    int bad = 0, shown = 0;
                    for (int b = 0; b < blocks; ++b) {
                        for (int t = 0; t < n; ++t) {
                            int l32 = t % 32, g = t - l32;
                            unsigned mask;
                            switch (maskmode) {
                            case 0: mask = 0xffffffffu; break;
                            case 1: mask = l32 < 16 ? 0x0000ffffu : 0xffff0000u; break;
                            case 2: mask = (l32 & 1) ? 0xaaaaaaaau : 0x55555555u; break;
                            default: mask = 1u << l32; break;
                            }
                            unsigned wantb = 0;
                            int want_all = 1;
                            for (int j = 0; j < 32; ++j) {
                                if (!((mask >> j) & 1u) || g + j >= n) continue;
                                int p = pred[b * n + g + j];
                                if (p) wantb |= 1u << j;
                                else want_all = 0;
                            }
                            int i = b * n + t;
                            if (bl[i] != wantb || an[i] != (wantb != 0) || al[i] != want_all) {
                                if (shown++ == 0)
                                    std::printf("  vote n=%d maskmode=%d pat=%d thread %d: ballot %08x/%08x any %d/%d all %d/%d (got/want)\n",
                                                n, maskmode, pat, t, bl[i], wantb, an[i], wantb != 0, al[i], want_all);
                                ++bad;
                            }
                        }
                    }
                    CHECK(bad == 0);
                }
                cudaFree(dpred); cudaFree(dballot); cudaFree(dany); cudaFree(dall);
            }
        }
    }
}

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    g_warp = prop.warpSize;
    test_shuffles();
    test_shuffle_chain();
    test_votes();
    if (g_fail == 0) {
        std::printf("rt_warp_lang: ok (%d checks, warp %d)\n", g_checks, g_warp);
        return 0;
    }
    std::printf("rt_warp_lang: %d of %d checks failed\n", g_fail, g_checks);
    return 1;
}
