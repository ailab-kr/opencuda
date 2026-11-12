// CUB block-wide primitives (card libs-cub-subset, test list items 12-13):
// BlockScan, BlockLoad, BlockStore. Expected results follow the public CUB
// API documentation (blocked / striped arrangements, aggregate outputs,
// valid_items and out-of-bounds defaults). Runs on the CPU path and on the
// NVIDIA path.
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_fail < 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECKM(c, ...) do { ++g_checks; if (!(c)) { if (g_fail < 40) { std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } ++g_fail; } } while (0)

template <class T>
struct Dev {
    T *p = nullptr;
    size_t n;
    explicit Dev(size_t count) : n(count) { cudaMalloc((void **)&p, (count ? count : 1) * sizeof(T)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    void up(const std::vector<T> &v) { cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    std::vector<T> down() const
    {
        std::vector<T> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

// ---- BlockScan, one item per thread -----------------------------------------

struct Slots { // per-thread result slots
    enum { INC_SUM, INC_SUM_AGG, EXC_SUM, EXC_SUM_AGG, INC_MAX, EXC_SCAN_INIT, EXC_SCAN_INIT_AGG, INC_SCAN_AGG, REUSE, NSLOT };
};

template <class T, int X, int Y, int Z>
__global__ void scan_kernel(const T *in, T *out, T *agg)
{
    typedef cub::BlockScan<T, X, cub::BLOCK_SCAN_RAKING, Y, Z> Scan;
    __shared__ typename Scan::TempStorage ts;
    const int threads = X * Y * Z;
    const int tid = threadIdx.x + X * (threadIdx.y + Y * threadIdx.z);
    const T v = in[blockIdx.x * threads + tid];
    T *o = out + (size_t)blockIdx.x * threads * Slots::NSLOT;
    T *a = agg + blockIdx.x * Slots::NSLOT;
    T r, g;

    Scan(ts).InclusiveSum(v, r);                        o[Slots::INC_SUM * threads + tid] = r;
    __syncthreads();
    Scan(ts).InclusiveSum(v, r, g);                     o[Slots::INC_SUM_AGG * threads + tid] = r;
    if (tid == 0) a[Slots::INC_SUM_AGG] = g;
    __syncthreads();
    Scan(ts).ExclusiveSum(v, r);                        o[Slots::EXC_SUM * threads + tid] = r;
    __syncthreads();
    Scan(ts).ExclusiveSum(v, r, g);                     o[Slots::EXC_SUM_AGG * threads + tid] = r;
    if (tid == 0) a[Slots::EXC_SUM_AGG] = g;
    __syncthreads();
    Scan(ts).InclusiveScan(v, r, cub::Max());           o[Slots::INC_MAX * threads + tid] = r;
    __syncthreads();
    Scan(ts).ExclusiveScan(v, r, T(100), cub::Sum());   o[Slots::EXC_SCAN_INIT * threads + tid] = r;
    __syncthreads();
    Scan(ts).ExclusiveScan(v, r, T(100), cub::Max(), g); o[Slots::EXC_SCAN_INIT_AGG * threads + tid] = r;
    if (tid == 0) a[Slots::EXC_SCAN_INIT_AGG] = g;
    __syncthreads();
    Scan(ts).InclusiveScan(v, r, cub::Min(), g);        o[Slots::INC_SCAN_AGG * threads + tid] = r;
    if (tid == 0) a[Slots::INC_SCAN_AGG] = g;
    __syncthreads();
    // The same temporary storage again, many times, with changing input.
    T w = v;
    for (int k = 0; k < 3; ++k) {
        Scan(ts).InclusiveSum(w, r);
        __syncthreads();
        w = r - v;      // shifts the data each round
    }
    o[Slots::REUSE * threads + tid] = r;
}

template <class T, int X, int Y, int Z>
static void test_block_scan(const char *name, int blocks)
{
    const int threads = X * Y * Z;
    std::mt19937 g(threads * 31 + blocks);
    std::vector<T> in(blocks * threads);
    for (auto &x : in) x = T((int)(g() % 9) - 3);
    Dev<T> din(in.size()), dout(in.size() * Slots::NSLOT), dagg(blocks * Slots::NSLOT);
    din.up(in);
    std::vector<T> zero(dout.n, T(-1234));
    dout.up(zero);
    dim3 blk(X, Y, Z);
    scan_kernel<T, X, Y, Z><<<blocks, blk>>>(din.p, dout.p, dagg.p);
    cudaError_t e = cudaDeviceSynchronize();
    CHECKM(e == cudaSuccess, "%s kernel err %d", name, (int)e);
    std::vector<T> out = dout.down(), agg = dagg.down();
    for (int b = 0; b < blocks; ++b) {
        const T *v = &in[b * threads];
        const T *o = &out[(size_t)b * threads * Slots::NSLOT];
        const T *a = &agg[b * Slots::NSLOT];
        std::vector<T> incs(threads), excs(threads), mx(threads), ex100(threads), exmax(threads), inmin(threads), reuse(threads);
        T run = 0, m = v[0], mn = v[0], em = 100, e100 = 100;
        for (int t = 0; t < threads; ++t) {
            excs[t] = run;
            run = T(run + v[t]);
            incs[t] = run;
            m = std::max(m, v[t]); mx[t] = m;
            mn = std::min(mn, v[t]); inmin[t] = mn;
            ex100[t] = e100; e100 = T(e100 + v[t]);
            exmax[t] = em; em = std::max(em, v[t]);
        }
        T total = run;
        // reuse: w0=v; r1=prefix(v); w1=r1-v; r2=prefix(w1); w2=r2-v; r3=prefix(w2)
        std::vector<T> w(v, v + threads), r(threads);
        for (int k = 0; k < 3; ++k) {
            T s = 0;
            for (int t = 0; t < threads; ++t) { s = T(s + w[t]); r[t] = s; }
            for (int t = 0; t < threads; ++t) w[t] = T(r[t] - v[t]);
        }
        reuse = r;
        int bad[Slots::NSLOT] = {};
        for (int t = 0; t < threads; ++t) {
            bad[Slots::INC_SUM] += !(o[Slots::INC_SUM * threads + t] == incs[t]);
            bad[Slots::INC_SUM_AGG] += !(o[Slots::INC_SUM_AGG * threads + t] == incs[t]);
            bad[Slots::EXC_SUM] += !(o[Slots::EXC_SUM * threads + t] == excs[t]);
            bad[Slots::EXC_SUM_AGG] += !(o[Slots::EXC_SUM_AGG * threads + t] == excs[t]);
            bad[Slots::INC_MAX] += !(o[Slots::INC_MAX * threads + t] == mx[t]);
            bad[Slots::EXC_SCAN_INIT] += !(o[Slots::EXC_SCAN_INIT * threads + t] == ex100[t]);
            bad[Slots::EXC_SCAN_INIT_AGG] += !(o[Slots::EXC_SCAN_INIT_AGG * threads + t] == exmax[t]);
            bad[Slots::INC_SCAN_AGG] += !(o[Slots::INC_SCAN_AGG * threads + t] == inmin[t]);
            bad[Slots::REUSE] += !(o[Slots::REUSE * threads + t] == reuse[t]);
        }
        for (int s = 0; s < Slots::NSLOT; ++s) CHECKM(bad[s] == 0, "%s block %d slot %d: %d wrong", name, b, s, bad[s]);
        CHECKM(a[Slots::INC_SUM_AGG] == total, "%s agg InclusiveSum %g want %g", name, (double)a[Slots::INC_SUM_AGG], (double)total);
        CHECKM(a[Slots::EXC_SUM_AGG] == total, "%s agg ExclusiveSum %g want %g", name, (double)a[Slots::EXC_SUM_AGG], (double)total);
        // The public docs call the aggregate the reduction of the inputs; whether
        // the initial value joins it is not stated, so both are accepted.
        CHECKM(a[Slots::EXC_SCAN_INIT_AGG] == mx[threads - 1] || a[Slots::EXC_SCAN_INIT_AGG] == std::max<T>(100, mx[threads - 1]), "%s agg ExclusiveScan(init,Max) = %g", name, (double)a[Slots::EXC_SCAN_INIT_AGG]);
        CHECKM(a[Slots::INC_SCAN_AGG] == mn, "%s agg InclusiveScan(Min)", name);
    }
}

// ---- BlockScan, several items per thread (blocked arrangement) ---------------

template <int X, int ITEMS>
__global__ void scan_items_kernel(const int *in, int *out, int *agg)
{
    typedef cub::BlockScan<int, X> Scan;
    __shared__ typename Scan::TempStorage ts;
    const int tile = X * ITEMS;
    int v[ITEMS], r[ITEMS];
    for (int i = 0; i < ITEMS; ++i) v[i] = in[blockIdx.x * tile + threadIdx.x * ITEMS + i];   // blocked
    int *o = out + (size_t)blockIdx.x * tile * 5;
    int g;
    Scan(ts).InclusiveSum(v, r);
    for (int i = 0; i < ITEMS; ++i) o[0 * tile + threadIdx.x * ITEMS + i] = r[i];
    __syncthreads();
    Scan(ts).ExclusiveSum(v, r);
    for (int i = 0; i < ITEMS; ++i) o[1 * tile + threadIdx.x * ITEMS + i] = r[i];
    __syncthreads();
    Scan(ts).ExclusiveSum(v, r, g);
    for (int i = 0; i < ITEMS; ++i) o[2 * tile + threadIdx.x * ITEMS + i] = r[i];
    if (threadIdx.x == 0) agg[blockIdx.x * 2] = g;
    __syncthreads();
    Scan(ts).InclusiveScan(v, r, cub::Max());
    for (int i = 0; i < ITEMS; ++i) o[3 * tile + threadIdx.x * ITEMS + i] = r[i];
    __syncthreads();
    Scan(ts).ExclusiveScan(v, r, 7, cub::Sum(), g);
    for (int i = 0; i < ITEMS; ++i) o[4 * tile + threadIdx.x * ITEMS + i] = r[i];
    if (threadIdx.x == 0) agg[blockIdx.x * 2 + 1] = g;
}

template <int X, int ITEMS>
static void test_block_scan_items(int blocks)
{
    const int tile = X * ITEMS;
    std::mt19937 g(X * 7 + ITEMS);
    std::vector<int> in(blocks * tile);
    for (auto &x : in) x = (int)(g() % 11) - 4;
    Dev<int> din(in.size()), dout(in.size() * 5), dagg(blocks * 2);
    din.up(in);
    scan_items_kernel<X, ITEMS><<<blocks, X>>>(din.p, dout.p, dagg.p);
    cudaError_t e = cudaDeviceSynchronize();
    CHECKM(e == cudaSuccess, "scan items err %d", (int)e);
    std::vector<int> out = dout.down(), agg = dagg.down();
    for (int b = 0; b < blocks; ++b) {
        const int *v = &in[b * tile];
        const int *o = &out[(size_t)b * tile * 5];
        int bad[5] = {};
        int run = 0, mx = v[0], ex = 7;
        for (int k = 0; k < tile; ++k) {
            int excl = run;
            run += v[k];
            mx = std::max(mx, v[k]);
            bad[0] += o[0 * tile + k] != run;
            bad[1] += o[1 * tile + k] != excl;
            bad[2] += o[2 * tile + k] != excl;
            bad[3] += o[3 * tile + k] != mx;
            bad[4] += o[4 * tile + k] != ex;
            ex += v[k];
        }
        for (int s = 0; s < 5; ++s) CHECKM(bad[s] == 0, "scan items X=%d N=%d block %d slot %d: %d wrong", X, ITEMS, b, s, bad[s]);
        CHECKM(agg[b * 2] == run, "scan items aggregate %d want %d", agg[b * 2], run);
        CHECKM(agg[b * 2 + 1] == run || agg[b * 2 + 1] == run + 7, "scan items aggregate with init %d (inputs sum %d)", agg[b * 2 + 1], run);
    }
}

// ---- BlockLoad / BlockStore ----------------------------------------------------

template <cub::BlockLoadAlgorithm LA, bool STRIPED, int VALID_MODE>
__global__ void load_kernel(const float *in, float *dump, int valid, float dflt)
{
    typedef cub::BlockLoad<float, 128, 16, LA> Load;
    __shared__ typename Load::TempStorage ts;
    float v[16];
    const float *src = in + blockIdx.x * 2048;
    for (int i = 0; i < 16; ++i) v[i] = -555.0f;
    if (VALID_MODE == 0) Load(ts).Load(src, v);
    else if (VALID_MODE == 1) Load(ts).Load(src, v, valid);
    else Load(ts).Load(src, v, valid, dflt);
    // Position (t, i) -> where it sits in the tile: blocked or striped.
    for (int i = 0; i < 16; ++i) dump[blockIdx.x * 2048 + threadIdx.x * 16 + i] = v[i];
}

template <cub::BlockLoadAlgorithm LA, bool STRIPED, int VALID_MODE>
static void test_load_one(const char *name, int valid)
{
    const int blocks = 2;
    std::vector<float> in(blocks * 2048);
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(i) + 0.25f;
    Dev<float> din(in.size()), dump(in.size());
    din.up(in);
    dump.up(std::vector<float>(in.size(), -999.0f));
    // For partial loads the second block gets the same valid count; the
    // input of a partial load must not be read past valid_items, so give
    // the first block a buffer that is exactly the valid part when possible.
    load_kernel<LA, STRIPED, VALID_MODE><<<blocks, 128>>>(din.p, dump.p, valid, -7.0f);
    cudaError_t e = cudaDeviceSynchronize();
    CHECKM(e == cudaSuccess, "%s kernel err %d", name, (int)e);
    std::vector<float> d = dump.down();
    int bad = 0, first = -1;
    for (int b = 0; b < blocks; ++b)
        for (int t = 0; t < 128; ++t)
            for (int i = 0; i < 16; ++i) {
                int k = STRIPED ? i * 128 + t : t * 16 + i;       // index in the tile
                float got = d[b * 2048 + t * 16 + i];
                bool in_range = VALID_MODE == 0 || k < valid;
                if (in_range) {
                    if (got != in[b * 2048 + k]) { if (first < 0) first = b * 2048 + t * 16 + i; ++bad; }
                } else if (VALID_MODE == 2) {
                    if (got != -7.0f) { if (first < 0) first = b * 2048 + t * 16 + i; ++bad; }
                }
            }
    CHECKM(bad == 0, "%s valid=%d: %d wrong, first at dump[%d]=%g", name, valid, bad, first, first >= 0 ? d[first] : 0.0f);
}

template <cub::BlockLoadAlgorithm LA, bool STRIPED>
static void test_load_alg(const char *name)
{
    test_load_one<LA, STRIPED, 0>(name, 2048);
    for (int valid : {0, 1, 15, 16, 17, 127, 129, 777, 1000, 2047, 2048}) {
        test_load_one<LA, STRIPED, 1>(name, valid);
        test_load_one<LA, STRIPED, 2>(name, valid);
    }
}

template <cub::BlockStoreAlgorithm SA, bool STRIPED, bool PARTIAL>
__global__ void store_kernel(float *out, int valid)
{
    typedef cub::BlockStore<float, 128, 16, SA> Store;
    __shared__ typename Store::TempStorage ts;
    float v[16];
    // Item i of thread t holds the number of its tile position.
    for (int i = 0; i < 16; ++i) v[i] = STRIPED ? float(i * 128 + threadIdx.x) : float(threadIdx.x * 16 + i);
    float *dst = out + blockIdx.x * 2048;
    if (PARTIAL) Store(ts).Store(dst, v, valid);
    else Store(ts).Store(dst, v);
}

template <cub::BlockStoreAlgorithm SA, bool STRIPED, bool PARTIAL>
static void test_store_one(const char *name, int valid)
{
    const int blocks = 2;
    Dev<float> dout(blocks * 2048);
    dout.up(std::vector<float>(dout.n, -1.0f));
    store_kernel<SA, STRIPED, PARTIAL><<<blocks, 128>>>(dout.p, valid);
    cudaError_t e = cudaDeviceSynchronize();
    CHECKM(e == cudaSuccess, "%s kernel err %d", name, (int)e);
    std::vector<float> o = dout.down();
    int bad = 0, first = -1;
    for (int b = 0; b < blocks; ++b)
        for (int k = 0; k < 2048; ++k) {
            float want = (!PARTIAL || k < valid) ? float(k) : -1.0f;   // tile position k holds value k
            if (o[b * 2048 + k] != want) { if (first < 0) first = b * 2048 + k; ++bad; }
        }
    CHECKM(bad == 0, "%s valid=%d: %d wrong, first at out[%d]=%g", name, valid, bad, first, first >= 0 ? o[first] : 0.0f);
}

template <cub::BlockStoreAlgorithm SA, bool STRIPED>
static void test_store_alg(const char *name)
{
    test_store_one<SA, STRIPED, false>(name, 2048);
    for (int valid : {0, 1, 15, 16, 17, 127, 129, 777, 1000, 2047, 2048}) test_store_one<SA, STRIPED, true>(name, valid);
}

// llama.cpp ssm-scan shape: Load and Store temp storage in one union, the
// same block loads, transforms and stores, with a barrier between.
__global__ void load_store_union_kernel(const float *in, float *out)
{
    typedef cub::BlockLoad<float, 128, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE> Load;
    typedef cub::BlockStore<float, 128, 16, cub::BLOCK_STORE_WARP_TRANSPOSE> Store;
    __shared__ union {
        typename Load::TempStorage load;
        typename Store::TempStorage store;
    } ts;
    for (int round = 0; round < 3; ++round) {
        float v[16];
        const float *src = in + (size_t)(blockIdx.x * 3 + round) * 2048;
        Load(ts.load).Load(src, v);
        __syncthreads();
        for (int i = 0; i < 16; ++i) v[i] = v[i] * 2.0f + float(round);
        Store(ts.store).Store(out + (size_t)(blockIdx.x * 3 + round) * 2048, v);
        __syncthreads();
    }
}

static void test_union()
{
    const int blocks = 3;
    std::vector<float> in(blocks * 3 * 2048);
    std::mt19937 g(5);
    for (auto &x : in) x = float((int)(g() % 1000) - 500);
    Dev<float> din(in.size()), dout(in.size());
    din.up(in);
    dout.up(std::vector<float>(in.size(), -1.0f));
    load_store_union_kernel<<<blocks, 128>>>(din.p, dout.p);
    cudaError_t e = cudaDeviceSynchronize();
    CHECKM(e == cudaSuccess, "union kernel err %d", (int)e);
    std::vector<float> o = dout.down();
    int bad = 0;
    for (size_t k = 0; k < in.size(); ++k) {
        int round = (int)((k / 2048) % 3);
        bad += o[k] != in[k] * 2.0f + float(round);
    }
    CHECKM(bad == 0, "union load/transform/store: %d wrong", bad);
}

int main()
{
    test_block_scan<int, 128, 1, 1>("BlockScan<int,128>", 3);
    test_block_scan<int, 32, 1, 1>("BlockScan<int,32>", 2);
    test_block_scan<int, 64, 1, 1>("BlockScan<int,64>", 2);
    test_block_scan<int, 1, 1, 1>("BlockScan<int,1>", 2);
    test_block_scan<int, 96, 1, 1>("BlockScan<int,96>", 2);
    test_block_scan<float, 256, 1, 1>("BlockScan<float,256>", 3);
    test_block_scan<int, 16, 8, 1>("BlockScan<int,16,Y=8>", 3);
    test_block_scan<float, 16, 8, 1>("BlockScan<float,16,Y=8>", 2);
    test_block_scan<int, 8, 4, 2>("BlockScan<int,8,4,2>", 2);
    test_block_scan_items<64, 4>(3);
    test_block_scan_items<128, 1>(2);
    test_block_scan_items<32, 7>(2);
    test_block_scan_items<256, 2>(2);

    test_load_alg<cub::BLOCK_LOAD_DIRECT, false>("Load DIRECT");
    test_load_alg<cub::BLOCK_LOAD_STRIPED, true>("Load STRIPED");
    test_load_alg<cub::BLOCK_LOAD_VECTORIZE, false>("Load VECTORIZE");
    test_load_alg<cub::BLOCK_LOAD_TRANSPOSE, false>("Load TRANSPOSE");
    test_load_alg<cub::BLOCK_LOAD_WARP_TRANSPOSE, false>("Load WARP_TRANSPOSE");
    test_load_alg<cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, false>("Load WARP_TRANSPOSE_TIMESLICED");

    test_store_alg<cub::BLOCK_STORE_DIRECT, false>("Store DIRECT");
    test_store_alg<cub::BLOCK_STORE_STRIPED, true>("Store STRIPED");
    test_store_alg<cub::BLOCK_STORE_VECTORIZE, false>("Store VECTORIZE");
    test_store_alg<cub::BLOCK_STORE_TRANSPOSE, false>("Store TRANSPOSE");
    test_store_alg<cub::BLOCK_STORE_WARP_TRANSPOSE, false>("Store WARP_TRANSPOSE");
    test_store_alg<cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, false>("Store WARP_TRANSPOSE_TIMESLICED");
    test_union();

    std::printf("cub_block: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
