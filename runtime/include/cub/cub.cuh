/*
 * cub/cub.cuh -- a small subset of the CUB parallel-primitives API.
 *
 * Written for this project from the public CUB API documentation (class
 * and function signatures, documented semantics: two-phase temporary
 * storage, stability of radix sorts, ordering of floating-point keys,
 * blocked and striped data arrangements). Not derived from CUB's source.
 *
 * What is here (namespace cub):
 *   Sum, Max, Min                               binary operators
 *   BlockScan<T, BLOCK_DIM_X, ALG, Y, Z>        InclusiveSum, ExclusiveSum,
 *                                               InclusiveScan, ExclusiveScan
 *                                               (one item or an array per thread)
 *   BlockLoad / BlockStore<T, X, ITEMS, ALG>    blocked and striped arrangements
 *   BlockReduce<T, BLOCK_DIM_X, ALG, Y, Z>      Reduce, Sum (one item, an array per
 *                                               thread, or the first num_valid threads)
 *   WarpReduce<T, LOGICAL_WARP_THREADS>         Reduce, Sum (optionally valid_items)
 *   DeviceReduce                                Sum, Max, Min, Reduce
 *   DeviceScan                                  InclusiveSum, ExclusiveSum,
 *                                               InclusiveScan, ExclusiveScan
 *   DeviceRadixSort                             SortPairs(Descending), SortKeys(Descending)
 *   DeviceSegmentedRadixSort, DeviceSegmentedSort
 *                                               SortPairs(Descending), SortKeys(Descending),
 *                                               StableSort* (segmented sort)
 *
 * Every algorithm is plain CUDA C++ in this header; it is compiled with
 * the program for whatever target the program is built for.
 *
 * Device-wide calls follow the documented protocol: with d_temp_storage ==
 * nullptr only temp_storage_bytes is written (always at least 1, so an
 * allocation of that size is never a null "query" pointer by accident);
 * otherwise the work is queued on `stream` and the launch status is
 * returned. Nothing is synchronised.
 *
 * Limits of this subset: element types must be trivially copyable;
 * BlockScan's temporary storage holds 2 * BLOCK_THREADS values;
 * the version macros of the CCCL distribution (CCCL_MAJOR_VERSION, ...)
 * are not defined, so code that needs a particular CCCL release does not
 * take that path.
 */
#ifndef OCU_CUB_CUB_CUH
#define OCU_CUB_CUB_CUH

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <iterator>
#include <limits>
#include <type_traits>

namespace cub {

/* ---- operators ------------------------------------------------------- */

struct Sum {
    template <typename T, typename U>
    __host__ __device__ __forceinline__ auto operator()(const T &a, const U &b) const
        -> decltype(a + b)
    {
        return a + b;
    }
};

struct Max {
    template <typename T, typename U>
    __host__ __device__ __forceinline__ typename std::common_type<T, U>::type
    operator()(const T &a, const U &b) const
    {
        return (a < b) ? b : a;
    }
};

struct Min {
    template <typename T, typename U>
    __host__ __device__ __forceinline__ typename std::common_type<T, U>::type
    operator()(const T &a, const U &b) const
    {
        return (b < a) ? b : a;
    }
};

struct NullType {};

/* ---- algorithm selectors (they change speed, not results) ------------ */

enum BlockScanAlgorithm {
    BLOCK_SCAN_RAKING,
    BLOCK_SCAN_RAKING_MEMOIZE,
    BLOCK_SCAN_WARP_SCANS
};

enum BlockLoadAlgorithm {
    BLOCK_LOAD_DIRECT,
    BLOCK_LOAD_STRIPED,
    BLOCK_LOAD_VECTORIZE,
    BLOCK_LOAD_TRANSPOSE,
    BLOCK_LOAD_WARP_TRANSPOSE,
    BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED
};

enum BlockStoreAlgorithm {
    BLOCK_STORE_DIRECT,
    BLOCK_STORE_STRIPED,
    BLOCK_STORE_VECTORIZE,
    BLOCK_STORE_TRANSPOSE,
    BLOCK_STORE_WARP_TRANSPOSE,
    BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED
};

namespace ocu_detail {

template <int X, int Y, int Z>
__device__ __forceinline__ int linear_tid()
{
    return (Y == 1 && Z == 1) ? (int)threadIdx.x
                              : (int)(threadIdx.x + X * (threadIdx.y + Y * threadIdx.z));
}

/* Uninitialised storage for N values of T (usable as a __shared__ member). */
template <typename T, int N>
struct raw_array {
    alignas(alignof(T) > 16 ? alignof(T) : 16) unsigned char bytes[sizeof(T) * N];
    __device__ __forceinline__ T *data() { return reinterpret_cast<T *>(bytes); }
};

/*
 * Inclusive scan of one value per thread over `threads` threads (all of
 * them must call), Hillis-Steele with two buffers. Returns the inclusive
 * result; `buf` then holds every thread's inclusive result (the buffer
 * pointer returned through `last`).
 */
template <typename T, typename Op>
__device__ __forceinline__ T block_inclusive(T *buf0, T *buf1, int threads, int tid, T v, Op op,
                                             T *&last)
{
    T *a = buf0;
    T *b = buf1;
    a[tid] = v;
    __syncthreads();
    for (int d = 1; d < threads; d <<= 1) {
        T x = a[tid];
        if (tid >= d) x = op(a[tid - d], x);
        b[tid] = x;
        __syncthreads();
        T *t = a;
        a = b;
        b = t;
    }
    last = a;
    return a[tid];
}

/*
 * The global side of a transposing load/store: the tile moved between global
 * and shared memory in 16-byte pieces, neighbouring threads taking
 * neighbouring pieces, so each access instruction covers whole cache lines
 * (item-wide accesses of 1- or 2-byte types cover a fraction of one, and
 * thread-contiguous wide ones leave gaps between lanes). True when done.
 */
template <int BYTES, int THREADS>
__device__ __forceinline__ bool copy_striped16(void *dst, const void *src, int tid)
{
    if (BYTES % 16 != 0 || (((uintptr_t)dst | (uintptr_t)src) & 15) != 0) return false;
    for (int c = tid; c < BYTES / 16; c += THREADS)
        reinterpret_cast<uint4 *>(dst)[c] = reinterpret_cast<const uint4 *>(src)[c];
    return true;
}

} // namespace ocu_detail

/* ---- BlockScan ------------------------------------------------------- */

template <typename T, int BLOCK_DIM_X, BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockScan {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

public:
    /* Declare as __shared__; reuse after a __syncthreads(). */
    struct TempStorage {
        ocu_detail::raw_array<T, 2 * BLOCK_THREADS> buf;
    };

    __device__ __forceinline__ BlockScan(TempStorage &temp_storage)
        : s_(temp_storage),
          tid_(ocu_detail::linear_tid<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>())
    {
    }

    /* -- one item per thread -- */

    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(T input, T &output, ScanOp scan_op,
                                                  T &block_aggregate)
    {
        T *last;
        output = incl(input, scan_op, last);
        block_aggregate = last[BLOCK_THREADS - 1];
    }

    template <typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(T input, T &output, ScanOp scan_op)
    {
        T *last;
        output = incl(input, scan_op, last);
    }

    __device__ __forceinline__ void InclusiveSum(T input, T &output) { InclusiveScan(input, output, Sum()); }

    __device__ __forceinline__ void InclusiveSum(T input, T &output, T &block_aggregate)
    {
        InclusiveScan(input, output, Sum(), block_aggregate);
    }

    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(T input, T &output, T initial_value,
                                                  ScanOp scan_op, T &block_aggregate)
    {
        T *last;
        incl(input, scan_op, last);
        output = tid_ == 0 ? initial_value : (T)scan_op(initial_value, last[tid_ - 1]);
        block_aggregate = last[BLOCK_THREADS - 1];
    }

    template <typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(T input, T &output, T initial_value,
                                                  ScanOp scan_op)
    {
        T agg;
        ExclusiveScan(input, output, initial_value, scan_op, agg);
    }

    __device__ __forceinline__ void ExclusiveSum(T input, T &output)
    {
        ExclusiveScan(input, output, T(0), Sum());
    }

    __device__ __forceinline__ void ExclusiveSum(T input, T &output, T &block_aggregate)
    {
        ExclusiveScan(input, output, T(0), Sum(), block_aggregate);
    }

    /* -- ITEMS items per thread, blocked arrangement (thread i holds
          items i*ITEMS ... i*ITEMS+ITEMS-1 of the block's sequence) -- */

    template <int ITEMS, typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(T (&input)[ITEMS], T (&output)[ITEMS],
                                                  ScanOp scan_op, T &block_aggregate)
    {
        T local = input[0];
        for (int i = 1; i < ITEMS; ++i) local = scan_op(local, input[i]);
        T *last;
        incl(local, scan_op, last);
        block_aggregate = last[BLOCK_THREADS - 1];
        T run = tid_ == 0 ? input[0] : (T)scan_op(last[tid_ - 1], input[0]);
        output[0] = run;
        for (int i = 1; i < ITEMS; ++i) {
            run = scan_op(run, input[i]);
            output[i] = run;
        }
    }

    template <int ITEMS, typename ScanOp>
    __device__ __forceinline__ void InclusiveScan(T (&input)[ITEMS], T (&output)[ITEMS],
                                                  ScanOp scan_op)
    {
        T agg;
        InclusiveScan(input, output, scan_op, agg);
    }

    template <int ITEMS>
    __device__ __forceinline__ void InclusiveSum(T (&input)[ITEMS], T (&output)[ITEMS])
    {
        InclusiveScan(input, output, Sum());
    }

    template <int ITEMS>
    __device__ __forceinline__ void InclusiveSum(T (&input)[ITEMS], T (&output)[ITEMS],
                                                 T &block_aggregate)
    {
        InclusiveScan(input, output, Sum(), block_aggregate);
    }

    template <int ITEMS, typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(T (&input)[ITEMS], T (&output)[ITEMS],
                                                  T initial_value, ScanOp scan_op,
                                                  T &block_aggregate)
    {
        T local = input[0];
        for (int i = 1; i < ITEMS; ++i) local = scan_op(local, input[i]);
        T *last;
        incl(local, scan_op, last);
        block_aggregate = last[BLOCK_THREADS - 1];
        T run = tid_ == 0 ? initial_value : (T)scan_op(initial_value, last[tid_ - 1]);
        for (int i = 0; i < ITEMS; ++i) {
            T next = scan_op(run, input[i]);
            output[i] = run;
            run = next;
        }
    }

    template <int ITEMS, typename ScanOp>
    __device__ __forceinline__ void ExclusiveScan(T (&input)[ITEMS], T (&output)[ITEMS],
                                                  T initial_value, ScanOp scan_op)
    {
        T agg;
        ExclusiveScan(input, output, initial_value, scan_op, agg);
    }

    template <int ITEMS>
    __device__ __forceinline__ void ExclusiveSum(T (&input)[ITEMS], T (&output)[ITEMS])
    {
        ExclusiveScan(input, output, T(0), Sum());
    }

    template <int ITEMS>
    __device__ __forceinline__ void ExclusiveSum(T (&input)[ITEMS], T (&output)[ITEMS],
                                                 T &block_aggregate)
    {
        ExclusiveScan(input, output, T(0), Sum(), block_aggregate);
    }

private:
    template <typename ScanOp>
    __device__ __forceinline__ T incl(T v, ScanOp op, T *&last)
    {
        T *b = s_.buf.data();
        return ocu_detail::block_inclusive<T>(b, b + BLOCK_THREADS, BLOCK_THREADS, tid_, v, op,
                                              last);
    }

    TempStorage &s_;
    int tid_;
};

/* ---- BlockLoad / BlockStore ------------------------------------------ */

/*
 * The result arrangement is what the documentation gives for each
 * algorithm: BLOCK_LOAD_STRIPED gives a striped arrangement (item i of
 * thread t is block_itr[i * BLOCK_THREADS + t]); every other algorithm
 * gives the blocked one (block_itr[t * ITEMS + i]). The transposing
 * algorithms read through shared memory; the others read directly.
 */
template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD,
          BlockLoadAlgorithm ALGORITHM = BLOCK_LOAD_DIRECT, int BLOCK_DIM_Y = 1,
          int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockLoad {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr bool TRANSPOSE = ALGORITHM == BLOCK_LOAD_TRANSPOSE ||
                                      ALGORITHM == BLOCK_LOAD_WARP_TRANSPOSE ||
                                      ALGORITHM == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;

public:
    struct TempStorage {
        ocu_detail::raw_array<T, TRANSPOSE ? TILE : 1> buf;
    };

    __device__ __forceinline__ BlockLoad(TempStorage &temp_storage)
        : s_(temp_storage),
          tid_(ocu_detail::linear_tid<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>())
    {
    }

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(InputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
        load(block_itr, items, TILE, false, T());
    }

    template <typename InputIteratorT>
    __device__ __forceinline__ void Load(InputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD],
                                         int valid_items)
    {
        load(block_itr, items, valid_items, false, T());
    }

    template <typename InputIteratorT, typename DefaultT>
    __device__ __forceinline__ void Load(InputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD],
                                         int valid_items, DefaultT oob_default)
    {
        load(block_itr, items, valid_items, true, (T)oob_default);
    }

private:
    template <typename It>
    __device__ __forceinline__ void load(It it, T (&items)[ITEMS_PER_THREAD], int valid,
                                         bool fill, T dflt)
    {
        if (ALGORITHM == BLOCK_LOAD_STRIPED) {
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int k = i * BLOCK_THREADS + tid_;
                if (k < valid) items[i] = it[k];
                else if (fill) items[i] = dflt;
            }
        } else if (TRANSPOSE) {
            T *sm = s_.buf.data();
            bool wide = false;
            if constexpr (std::is_pointer<It>::value && std::is_trivially_copyable<T>::value) {
                if (valid >= TILE)
                    wide = ocu_detail::copy_striped16<(int)sizeof(T) * TILE, BLOCK_THREADS>(sm, (const void *)it, tid_);
            }
            if (!wide) {
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    int k = i * BLOCK_THREADS + tid_;
                    if (k < valid) sm[k] = it[k];
                }
            }
            __syncthreads();
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int k = tid_ * ITEMS_PER_THREAD + i;
                if (k < valid) items[i] = sm[k];
                else if (fill) items[i] = dflt;
            }
        } else {
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int k = tid_ * ITEMS_PER_THREAD + i;
                if (k < valid) items[i] = it[k];
                else if (fill) items[i] = dflt;
            }
        }
    }

    TempStorage &s_;
    int tid_;
};

template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD,
          BlockStoreAlgorithm ALGORITHM = BLOCK_STORE_DIRECT, int BLOCK_DIM_Y = 1,
          int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockStore {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr int TILE = BLOCK_THREADS * ITEMS_PER_THREAD;
    static constexpr bool TRANSPOSE = ALGORITHM == BLOCK_STORE_TRANSPOSE ||
                                      ALGORITHM == BLOCK_STORE_WARP_TRANSPOSE ||
                                      ALGORITHM == BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED;

public:
    struct TempStorage {
        ocu_detail::raw_array<T, TRANSPOSE ? TILE : 1> buf;
    };

    __device__ __forceinline__ BlockStore(TempStorage &temp_storage)
        : s_(temp_storage),
          tid_(ocu_detail::linear_tid<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>())
    {
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD])
    {
        store(block_itr, items, TILE);
    }

    template <typename OutputIteratorT>
    __device__ __forceinline__ void Store(OutputIteratorT block_itr, T (&items)[ITEMS_PER_THREAD],
                                          int valid_items)
    {
        store(block_itr, items, valid_items);
    }

private:
    template <typename It>
    __device__ __forceinline__ void store(It it, T (&items)[ITEMS_PER_THREAD], int valid)
    {
        if (ALGORITHM == BLOCK_STORE_STRIPED) {
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int k = i * BLOCK_THREADS + tid_;
                if (k < valid) it[k] = items[i];
            }
        } else if (TRANSPOSE) {
            T *sm = s_.buf.data();
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) sm[tid_ * ITEMS_PER_THREAD + i] = items[i];
            __syncthreads();
            bool wide = false;
            if constexpr (std::is_pointer<It>::value && std::is_trivially_copyable<T>::value) {
                if (valid >= TILE)
                    wide = ocu_detail::copy_striped16<(int)sizeof(T) * TILE, BLOCK_THREADS>((void *)it, sm, tid_);
            }
            if (!wide) {
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    int k = i * BLOCK_THREADS + tid_;
                    if (k < valid) it[k] = sm[k];
                }
            }
        } else {
            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int k = tid_ * ITEMS_PER_THREAD + i;
                if (k < valid) it[k] = items[i];
            }
        }
    }

    TempStorage &s_;
    int tid_;
};

/* ---- device-wide: shared pieces -------------------------------------- */

namespace ocu_detail {

constexpr int kThreads = 256;           /* threads per block of every device-wide kernel */
constexpr int kScanItems = 8;           /* items per thread in a scan tile */
constexpr int kScanTile = kThreads * kScanItems;
constexpr int kReduceMaxBlocks = 256;
constexpr int kReduceSingleItems = 32;  /* up to 256*32 items: one block, one launch */
constexpr int kScanSingleTiles = 4;     /* up to 4 tiles: one block walks them in order */
constexpr int kSortItems = 4;           /* items per thread in a sort tile */
constexpr int kSortTile = kThreads * kSortItems;
constexpr int kRadixBits = 4;
constexpr int kRadixBins = 1 << kRadixBits;

inline size_t align_bytes(size_t x) { return (x + 255) & ~(size_t)255; }

template <typename It>
using value_of = typename std::remove_cv<typename std::iterator_traits<It>::value_type>::type;

/* Bump allocator over the caller's temporary storage. */
struct carve {
    unsigned char *base;
    size_t used;
    template <typename T>
    T *take(size_t n)
    {
        T *p = base ? reinterpret_cast<T *>(base + used) : nullptr;
        used += align_bytes(n * sizeof(T));
        return p;
    }
};

inline cudaError_t finish_query(size_t &temp_storage_bytes, size_t bytes)
{
    temp_storage_bytes = bytes > 0 ? bytes : 1;
    return cudaSuccess;
}

inline cudaError_t check_space(size_t have, size_t need)
{
    return have < (need > 0 ? need : 1) ? cudaErrorInvalidValue : cudaSuccess;
}

namespace {

/* -- reduction -- */

/* Block b combines items b*256+t, stepping by grid*256. Every block has at
   least one item (the grid is never larger than ceil(n / 256)). */
template <typename InIt, typename Acc, typename Op>
__global__ void reduce_blocks(InIt in, long long n, Acc *partial, Op op)
{
    __shared__ raw_array<Acc, kThreads> vs;
    __shared__ int fs[kThreads];
    Acc *v = vs.data();
    const int t = threadIdx.x;
    const long long stride = (long long)gridDim.x * kThreads;
    long long i = (long long)blockIdx.x * kThreads + t;
    Acc acc{};
    int have = 0;
    for (; i < n; i += stride) {
        Acc x = (Acc)in[i];
        acc = have ? (Acc)op(acc, x) : x;
        have = 1;
    }
    v[t] = acc;
    fs[t] = have;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (t < s && fs[t + s]) {
            v[t] = fs[t] ? (Acc)op(v[t], v[t + s]) : v[t + s];
            fs[t] = 1;
        }
        __syncthreads();
    }
    if (t == 0) partial[blockIdx.x] = v[0];
}

/* One block combines `count` values (block partials, or the whole input
   when it is small) and writes op(init, result), or init when empty. */
template <typename InIt, typename OutIt, typename Acc, typename Op, typename Init>
__global__ void reduce_final(InIt partial, long long count, OutIt out, Op op, Init init)
{
    __shared__ raw_array<Acc, kThreads> vs;
    __shared__ int fs[kThreads];
    Acc *v = vs.data();
    const int t = threadIdx.x;
    Acc acc{};
    int have = 0;
    for (long long i = t; i < count; i += kThreads) {
        Acc x = (Acc)partial[i];
        acc = have ? (Acc)op(acc, x) : x;
        have = 1;
    }
    v[t] = acc;
    fs[t] = have;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (t < s && fs[t + s]) {
            v[t] = fs[t] ? (Acc)op(v[t], v[t + s]) : v[t + s];
            fs[t] = 1;
        }
        __syncthreads();
    }
    if (t == 0) {
        Acc r = fs[0] ? (Acc)op((Acc)init, v[0]) : (Acc)init;
        *out = r;
    }
}

/* -- scan: tile aggregates, scan of the aggregates, tiles -- */

template <typename Acc, typename InIt>
__device__ __forceinline__ int load_tile(InIt in, long long n, long long base, int t,
                                         Acc (&x)[kScanItems])
{
    int valid = 0;
    for (int j = 0; j < kScanItems; ++j) {
        long long k = base + (long long)t * kScanItems + j;
        if (k < n) {
            x[j] = (Acc)in[k];
            valid = j + 1;
        }
    }
    return valid;
}

template <typename InIt, typename Acc, typename Op>
__global__ void scan_tile_aggregates(InIt in, long long n, Acc *agg, Op op)
{
    __shared__ raw_array<Acc, 2 * kThreads> sb;
    const int t = threadIdx.x;
    const long long base = (long long)blockIdx.x * kScanTile;
    Acc x[kScanItems];
    int valid = load_tile<Acc>(in, n, base, t, x);
    /* Threads past the end sit after every valid thread, so any value
       there leaves the valid prefix unchanged; use the tile's first item. */
    Acc local = valid ? x[0] : (Acc)in[base];
    for (int j = 1; j < valid; ++j) local = op(local, x[j]);
    Acc *last;
    block_inclusive<Acc>(sb.data(), sb.data() + kThreads, kThreads, t, local, op, last);
    long long rem = n - base;
    int threads_used = (int)((rem + kScanItems - 1) / kScanItems);
    if (threads_used > kThreads) threads_used = kThreads;
    if (t == 0) agg[blockIdx.x] = last[threads_used - 1];
}

/* One block: inclusive scan of `count` tile aggregates, in chunks. */
template <typename Acc, typename Op>
__global__ void scan_aggregates(const Acc *agg, Acc *incl, long long count, Op op)
{
    __shared__ raw_array<Acc, 2 * kThreads> sb;
    __shared__ raw_array<Acc, 1> carry_s;
    const int t = threadIdx.x;
    for (long long c = 0; c < count; c += kThreads) {
        long long k = c + t;
        Acc v = k < count ? agg[k] : agg[c];
        Acc *last;
        Acc r = block_inclusive<Acc>(sb.data(), sb.data() + kThreads, kThreads, t, v, op, last);
        if (c > 0) r = op(carry_s.data()[0], r);
        if (k < count) incl[k] = r;
        int n_here = (int)((count - c) < kThreads ? (count - c) : kThreads);
        __syncthreads();
        if (t == n_here - 1) carry_s.data()[0] = r;
        __syncthreads();
    }
}

template <bool EXCLUSIVE, typename InIt, typename OutIt, typename Acc, typename Op,
          typename Init>
__global__ void scan_tiles(InIt in, OutIt out, long long n, const Acc *tile_incl, Op op,
                           Init init)
{
    __shared__ raw_array<Acc, 2 * kThreads> sb;
    const int t = threadIdx.x;
    const long long base = (long long)blockIdx.x * kScanTile;
    Acc x[kScanItems];
    int valid = load_tile<Acc>(in, n, base, t, x);
    Acc local = valid ? x[0] : (Acc)in[base];
    for (int j = 1; j < valid; ++j) local = op(local, x[j]);
    Acc *last;
    block_inclusive<Acc>(sb.data(), sb.data() + kThreads, kThreads, t, local, op, last);
    /* Prefix of everything before this thread's first item. */
    bool have = false;
    Acc before{};
    if (blockIdx.x > 0) {
        before = tile_incl[blockIdx.x - 1];
        have = true;
    }
    if (t > 0) {
        before = have ? (Acc)op(before, last[t - 1]) : last[t - 1];
        have = true;
    }
    for (int j = 0; j < valid; ++j) {
        long long k = base + (long long)t * kScanItems + j;
        if (EXCLUSIVE) {
            out[k] = have ? (Acc)op((Acc)init, before) : (Acc)init;
            before = have ? (Acc)op(before, x[j]) : x[j];
        } else {
            before = have ? (Acc)op(before, x[j]) : x[j];
            out[k] = before;
        }
        have = true;
    }
}

/* Small scans: one block walks the tiles in order, carrying the prefix. */
template <bool EXCLUSIVE, typename InIt, typename OutIt, typename Acc, typename Op,
          typename Init>
__global__ void scan_single(InIt in, OutIt out, long long n, Op op, Init init)
{
    __shared__ raw_array<Acc, 2 * kThreads> sb;
    __shared__ raw_array<Acc, 1> carry_s;
    const int t = threadIdx.x;
    for (long long base = 0; base < n; base += kScanTile) {
        Acc x[kScanItems];
        int valid = load_tile<Acc>(in, n, base, t, x);
        Acc local = valid ? x[0] : (Acc)in[base];
        for (int j = 1; j < valid; ++j) local = op(local, x[j]);
        Acc *last;
        block_inclusive<Acc>(sb.data(), sb.data() + kThreads, kThreads, t, local, op, last);
        bool have = false;
        Acc before{};
        if (base > 0) {
            before = carry_s.data()[0];
            have = true;
        }
        if (t > 0) {
            before = have ? (Acc)op(before, last[t - 1]) : last[t - 1];
            have = true;
        }
        for (int j = 0; j < valid; ++j) {
            long long k = base + (long long)t * kScanItems + j;
            if (EXCLUSIVE) {
                out[k] = have ? (Acc)op((Acc)init, before) : (Acc)init;
                before = have ? (Acc)op(before, x[j]) : x[j];
            } else {
                before = have ? (Acc)op(before, x[j]) : x[j];
                out[k] = before;
            }
            have = true;
        }
        long long rem = n - base;
        int used = (int)((rem + kScanItems - 1) / kScanItems);
        if (used > kThreads) used = kThreads;
        Acc tile_total = last[used - 1];
        __syncthreads();
        if (t == 0) carry_s.data()[0] = base > 0 ? (Acc)op(carry_s.data()[0], tile_total) : tile_total;
        __syncthreads();
    }
}

} // namespace

/* Temporary storage of a scan over n items of Acc. */
template <typename Acc>
inline size_t scan_temp_bytes(long long n)
{
    long long tiles = (n + kScanTile - 1) / kScanTile;
    return 2 * align_bytes((size_t)tiles * sizeof(Acc));
}

template <bool EXCLUSIVE, typename Acc, typename InIt, typename OutIt, typename Op,
          typename Init>
inline cudaError_t scan_run(void *temp, InIt in, OutIt out, long long n, Op op, Init init,
                            cudaStream_t stream)
{
    if (n <= 0) return cudaSuccess;
    long long tiles = (n + kScanTile - 1) / kScanTile;
    if (tiles <= kScanSingleTiles) {
        scan_single<EXCLUSIVE, InIt, OutIt, Acc, Op, Init><<<1, kThreads, 0, stream>>>(
            in, out, n, op, init);
        return cudaPeekAtLastError();
    }
    carve c{static_cast<unsigned char *>(temp), 0};
    Acc *agg = c.take<Acc>((size_t)tiles);
    Acc *incl = c.take<Acc>((size_t)tiles);
    if (tiles > 1) {
        scan_tile_aggregates<InIt, Acc, Op><<<(unsigned)tiles, kThreads, 0, stream>>>(in, n, agg, op);
        scan_aggregates<Acc, Op><<<1, kThreads, 0, stream>>>(agg, incl, tiles, op);
    }
    scan_tiles<EXCLUSIVE, InIt, OutIt, Acc, Op, Init><<<(unsigned)tiles, kThreads, 0, stream>>>(
        in, out, n, incl, op, init);
    return cudaPeekAtLastError();
}

/* -- radix sort -- */

/* Keys as unsigned integers whose order is the key order (documented:
   negative floats fully inverted, positive floats with the sign bit set;
   -0.0 is treated as +0.0). */
template <typename K, typename = void>
struct radix_key;

template <typename K>
struct radix_key<K, typename std::enable_if<std::is_integral<K>::value>::type> {
    using U = typename std::make_unsigned<K>::type;
    __device__ __forceinline__ static U bits(K k)
    {
        U u = (U)k;
        if (std::is_signed<K>::value) u ^= (U)((U)1 << (sizeof(U) * 8 - 1));
        return u;
    }
};

template <typename K>
struct radix_key<K, typename std::enable_if<std::is_floating_point<K>::value>::type> {
    using U = typename std::conditional<sizeof(K) == 4, uint32_t, uint64_t>::type;
    __device__ __forceinline__ static U bits(K k)
    {
        U u;
        __builtin_memcpy(&u, &k, sizeof(U));
        const U sign = (U)((U)1 << (sizeof(U) * 8 - 1));
        if (u == sign) u = 0; /* -0.0 sorts as +0.0 */
        return (u & sign) ? (U)~u : (U)(u | sign);
    }
};

template <typename K, bool DESCENDING>
__device__ __forceinline__ int radix_digit(K k, int shift, unsigned mask)
{
    auto u = radix_key<K>::bits(k);
    if (DESCENDING) u = ~u;
    return (int)((u >> shift) & mask);
}

/* 16 counters of 16 bits each, packed in four 64-bit words. */
struct counts16 {
    unsigned long long w[4];
};

struct counts_add {
    __device__ __forceinline__ counts16 operator()(const counts16 &a, const counts16 &b) const
    {
        counts16 r;
        for (int i = 0; i < 4; ++i) r.w[i] = a.w[i] + b.w[i];
        return r;
    }
};

__device__ __forceinline__ unsigned count_of(const counts16 &c, int d)
{
    return (unsigned)((c.w[d >> 2] >> (16 * (d & 3))) & 0xffffu);
}

__device__ __forceinline__ void count_add(counts16 &c, int d)
{
    c.w[d >> 2] += 1ull << (16 * (d & 3));
}

struct rank_storage {
    raw_array<counts16, 2 * kThreads> buf;
};

/*
 * Stable ranks within a tile of kSortTile items in blocked arrangement
 * (thread t holds tile items t*kSortItems + j; digit -1 = no item).
 * rank[j] = number of tile items before it with the same digit;
 * total = items per digit in the tile. All threads must call.
 */
__device__ __forceinline__ void rank_tile(rank_storage &s, const int (&digit)[kSortItems],
                                          unsigned (&rank)[kSortItems], counts16 &total)
{
    const int t = threadIdx.x;
    counts16 mine = {{0, 0, 0, 0}};
    unsigned local[kSortItems];
    for (int j = 0; j < kSortItems; ++j) {
        if (digit[j] >= 0) {
            local[j] = count_of(mine, digit[j]);
            count_add(mine, digit[j]);
        }
    }
    counts16 *last;
    counts16 incl = block_inclusive<counts16>(s.buf.data(), s.buf.data() + kThreads, kThreads, t,
                                              mine, counts_add(), last);
    total = last[kThreads - 1];
    for (int j = 0; j < kSortItems; ++j) {
        if (digit[j] >= 0) {
            int d = digit[j];
            rank[j] = count_of(incl, d) - count_of(mine, d) + local[j];
        }
    }
}

namespace {

template <typename K, bool DESCENDING>
__global__ void radix_histogram(const K *keys, long long n, int shift, unsigned mask,
                                unsigned *hist, long long tiles)
{
    __shared__ rank_storage rs;
    const int t = threadIdx.x;
    const long long base = (long long)blockIdx.x * kSortTile;
    int digit[kSortItems];
    unsigned rank[kSortItems];
    for (int j = 0; j < kSortItems; ++j) {
        long long k = base + (long long)t * kSortItems + j;
        digit[j] = k < n ? radix_digit<K, DESCENDING>(keys[k], shift, mask) : -1;
    }
    counts16 total;
    rank_tile(rs, digit, rank, total);
    if (t < kRadixBins) hist[(long long)t * tiles + blockIdx.x] = count_of(total, t);
}

template <typename K, typename V, bool HAS_VALUES, bool DESCENDING>
__global__ void radix_scatter(const K *keys_in, K *keys_out, const V *vals_in, V *vals_out,
                              long long n, int shift, unsigned mask, const unsigned *offsets,
                              long long tiles)
{
    __shared__ rank_storage rs;
    const int t = threadIdx.x;
    const long long base = (long long)blockIdx.x * kSortTile;
    int digit[kSortItems];
    unsigned rank[kSortItems];
    K key[kSortItems];
    for (int j = 0; j < kSortItems; ++j) {
        long long k = base + (long long)t * kSortItems + j;
        if (k < n) {
            key[j] = keys_in[k];
            digit[j] = radix_digit<K, DESCENDING>(key[j], shift, mask);
        } else {
            digit[j] = -1;
        }
    }
    counts16 total;
    rank_tile(rs, digit, rank, total);
    for (int j = 0; j < kSortItems; ++j) {
        if (digit[j] < 0) continue;
        long long dst = (long long)offsets[(long long)digit[j] * tiles + blockIdx.x] + rank[j];
        keys_out[dst] = key[j];
        if (HAS_VALUES) vals_out[dst] = vals_in[base + (long long)t * kSortItems + j];
    }
}

template <typename K, typename V, bool HAS_VALUES>
__global__ void radix_copy(const K *keys_in, K *keys_out, const V *vals_in, V *vals_out,
                           long long n)
{
    for (long long i = (long long)blockIdx.x * kThreads + threadIdx.x; i < n;
         i += (long long)gridDim.x * kThreads) {
        keys_out[i] = keys_in[i];
        if (HAS_VALUES) vals_out[i] = vals_in[i];
    }
}

/* One block per segment: every digit pass of the segment in one kernel. */
template <typename K, typename V, bool HAS_VALUES, bool DESCENDING, typename BeginIt,
          typename EndIt>
__global__ void segmented_radix(const K *keys_in, K *keys_out, K *keys_alt, const V *vals_in,
                                V *vals_out, V *vals_alt, BeginIt begin_offsets,
                                EndIt end_offsets, int begin_bit, int end_bit)
{
    __shared__ rank_storage rs;
    __shared__ unsigned hist[kRadixBins];
    const int t = threadIdx.x;
    const long long seg_begin = (long long)begin_offsets[blockIdx.x];
    const long long seg_end = (long long)end_offsets[blockIdx.x];
    if (seg_end <= seg_begin) return; /* the whole block leaves: no barrier skipped */
    const long long len = seg_end - seg_begin;
    const int passes = (end_bit - begin_bit + kRadixBits - 1) / kRadixBits;
    if (passes <= 0) {
        for (long long i = seg_begin + t; i < seg_end; i += kThreads) {
            keys_out[i] = keys_in[i];
            if (HAS_VALUES) vals_out[i] = vals_in[i];
        }
        return;
    }
    const K *ksrc = keys_in;
    const V *vsrc = vals_in;
    for (int p = 0; p < passes; ++p) {
        K *kdst = ((passes - 1 - p) % 2 == 0) ? keys_out : keys_alt;
        V *vdst = ((passes - 1 - p) % 2 == 0) ? vals_out : vals_alt;
        const int shift = begin_bit + p * kRadixBits;
        const int width = (end_bit - shift) < kRadixBits ? (end_bit - shift) : kRadixBits;
        const unsigned mask = (1u << width) - 1u;
        if (t < kRadixBins) hist[t] = 0;
        __syncthreads();
        for (long long i = seg_begin + t; i < seg_end; i += kThreads) {
            atomicAdd(&hist[radix_digit<K, DESCENDING>(ksrc[i], shift, mask)], 1u);
        }
        __syncthreads();
        if (t == 0) {
            unsigned run = 0;
            for (int d = 0; d < kRadixBins; ++d) {
                unsigned c = hist[d];
                hist[d] = run;
                run += c;
            }
        }
        __syncthreads();
        for (long long tb = 0; tb < len; tb += kSortTile) {
            int digit[kSortItems];
            unsigned rank[kSortItems];
            K key[kSortItems];
            for (int j = 0; j < kSortItems; ++j) {
                long long k = tb + (long long)t * kSortItems + j;
                if (k < len) {
                    key[j] = ksrc[seg_begin + k];
                    digit[j] = radix_digit<K, DESCENDING>(key[j], shift, mask);
                } else {
                    digit[j] = -1;
                }
            }
            counts16 total;
            rank_tile(rs, digit, rank, total);
            for (int j = 0; j < kSortItems; ++j) {
                if (digit[j] < 0) continue;
                long long dst = seg_begin + (long long)hist[digit[j]] + rank[j];
                kdst[dst] = key[j];
                if (HAS_VALUES) vdst[dst] = vsrc[seg_begin + tb + (long long)t * kSortItems + j];
            }
            __syncthreads();
            if (t < kRadixBins) hist[t] += count_of(total, t);
            __syncthreads();
        }
        ksrc = kdst;
        vsrc = vdst;
    }
}

} // namespace

template <typename K, typename V, bool HAS_VALUES>
inline size_t radix_temp_bytes(long long n)
{
    long long tiles = (n + kSortTile - 1) / kSortTile;
    size_t b = align_bytes((size_t)n * sizeof(K));
    if (HAS_VALUES) b += align_bytes((size_t)n * sizeof(V));
    b += 2 * align_bytes((size_t)(tiles * kRadixBins) * sizeof(unsigned));
    b += scan_temp_bytes<unsigned>(tiles * kRadixBins);
    return b;
}

template <bool DESCENDING, typename K, typename V, bool HAS_VALUES>
inline cudaError_t radix_sort(void *d_temp_storage, size_t &temp_storage_bytes, const K *keys_in,
                              K *keys_out, const V *vals_in, V *vals_out, long long n,
                              int begin_bit, int end_bit, cudaStream_t stream)
{
    const size_t need = radix_temp_bytes<K, V, HAS_VALUES>(n);
    if (d_temp_storage == nullptr) return finish_query(temp_storage_bytes, need);
    cudaError_t e = check_space(temp_storage_bytes, need);
    if (e != cudaSuccess) return e;
    if (n <= 0) return cudaSuccess;
    if (begin_bit < 0 || end_bit > (int)(sizeof(K) * 8) || begin_bit > end_bit) {
        return cudaErrorInvalidValue;
    }
    const long long tiles = (n + kSortTile - 1) / kSortTile;
    carve c{static_cast<unsigned char *>(d_temp_storage), 0};
    K *kalt = c.take<K>((size_t)n);
    V *valt = HAS_VALUES ? c.take<V>((size_t)n) : nullptr;
    unsigned *hist = c.take<unsigned>((size_t)(tiles * kRadixBins));
    unsigned *offs = c.take<unsigned>((size_t)(tiles * kRadixBins));
    void *scan_temp = c.base + c.used;
    const int passes = (end_bit - begin_bit + kRadixBits - 1) / kRadixBits;
    const unsigned grid = (unsigned)tiles;
    if (passes == 0) {
        radix_copy<K, V, HAS_VALUES><<<grid, kThreads, 0, stream>>>(keys_in, keys_out, vals_in,
                                                                    vals_out, n);
        return cudaPeekAtLastError();
    }
    const K *ksrc = keys_in;
    const V *vsrc = vals_in;
    for (int p = 0; p < passes; ++p) {
        K *kdst = ((passes - 1 - p) % 2 == 0) ? keys_out : kalt;
        V *vdst = ((passes - 1 - p) % 2 == 0) ? vals_out : valt;
        const int shift = begin_bit + p * kRadixBits;
        const int width = (end_bit - shift) < kRadixBits ? (end_bit - shift) : kRadixBits;
        const unsigned mask = (1u << width) - 1u;
        radix_histogram<K, DESCENDING><<<grid, kThreads, 0, stream>>>(ksrc, n, shift, mask, hist,
                                                                      tiles);
        e = scan_run<true, unsigned>(scan_temp, (const unsigned *)hist, offs, tiles * kRadixBins,
                                     Sum(), 0u, stream);
        if (e != cudaSuccess) return e;
        radix_scatter<K, V, HAS_VALUES, DESCENDING><<<grid, kThreads, 0, stream>>>(
            ksrc, kdst, vsrc, vdst, n, shift, mask, offs, tiles);
        e = cudaPeekAtLastError();
        if (e != cudaSuccess) return e;
        ksrc = kdst;
        vsrc = vdst;
    }
    return cudaSuccess;
}

template <bool DESCENDING, typename K, typename V, bool HAS_VALUES, typename BeginIt,
          typename EndIt>
inline cudaError_t segmented_sort(void *d_temp_storage, size_t &temp_storage_bytes,
                                  const K *keys_in, K *keys_out, const V *vals_in, V *vals_out,
                                  long long n, long long num_segments, BeginIt begin_offsets,
                                  EndIt end_offsets, int begin_bit, int end_bit,
                                  cudaStream_t stream)
{
    size_t need = align_bytes((size_t)n * sizeof(K));
    if (HAS_VALUES) need += align_bytes((size_t)n * sizeof(V));
    if (d_temp_storage == nullptr) return finish_query(temp_storage_bytes, need);
    cudaError_t e = check_space(temp_storage_bytes, need);
    if (e != cudaSuccess) return e;
    if (n <= 0 || num_segments <= 0) return cudaSuccess;
    if (begin_bit < 0 || end_bit > (int)(sizeof(K) * 8) || begin_bit > end_bit) {
        return cudaErrorInvalidValue;
    }
    carve c{static_cast<unsigned char *>(d_temp_storage), 0};
    K *kalt = c.take<K>((size_t)n);
    V *valt = HAS_VALUES ? c.take<V>((size_t)n) : nullptr;
    segmented_radix<K, V, HAS_VALUES, DESCENDING, BeginIt, EndIt><<<(unsigned)num_segments,
                                                                   kThreads, 0, stream>>>(
        keys_in, keys_out, kalt, vals_in, vals_out, valt, begin_offsets, end_offsets, begin_bit,
        end_bit);
    return cudaPeekAtLastError();
}

} // namespace ocu_detail

/* ---- DeviceReduce ---------------------------------------------------- */

struct DeviceReduce {
    template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT,
              typename T, typename NumItemsT>
    static cudaError_t Reduce(void *d_temp_storage, size_t &temp_storage_bytes,
                              InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items,
                              ReductionOpT reduction_op, T init, cudaStream_t stream = 0)
    {
        using namespace ocu_detail;
        using Acc = value_of<InputIteratorT>;
        const long long n = (long long)num_items;
        /* Small inputs: one block, one launch. */
        const bool single = n <= (long long)kThreads * kReduceSingleItems;
        long long blocks = single ? 0 : (n + kThreads - 1) / kThreads;
        if (blocks > kReduceMaxBlocks) blocks = kReduceMaxBlocks;
        const size_t need = align_bytes((size_t)blocks * sizeof(Acc));
        if (d_temp_storage == nullptr) return finish_query(temp_storage_bytes, need);
        cudaError_t e = check_space(temp_storage_bytes, need);
        if (e != cudaSuccess) return e;
        if (single) {
            reduce_final<InputIteratorT, OutputIteratorT, Acc, ReductionOpT, T><<<1, kThreads, 0,
                                                                              stream>>>(
                d_in, n, d_out, reduction_op, init);
            return cudaPeekAtLastError();
        }
        Acc *partial = static_cast<Acc *>(d_temp_storage);
        reduce_blocks<InputIteratorT, Acc, ReductionOpT><<<(unsigned)blocks, kThreads, 0,
                                                           stream>>>(d_in, n, partial,
                                                                     reduction_op);
        reduce_final<const Acc *, OutputIteratorT, Acc, ReductionOpT, T><<<1, kThreads, 0,
                                                                         stream>>>(
            partial, blocks, d_out, reduction_op, init);
        return cudaPeekAtLastError();
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT d_in,
                           OutputIteratorT d_out, NumItemsT num_items, cudaStream_t stream = 0)
    {
        using Acc = ocu_detail::value_of<InputIteratorT>;
        return Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cub::Sum(),
                      Acc(0), stream);
    }

    /* Max and Min start from the lowest / highest value of the type
       (std::numeric_limits), which is what an empty input gives. */
    template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT d_in,
                           OutputIteratorT d_out, NumItemsT num_items, cudaStream_t stream = 0)
    {
        using Acc = ocu_detail::value_of<InputIteratorT>;
        return Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cub::Max(),
                      lowest<Acc>(), stream);
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, InputIteratorT d_in,
                           OutputIteratorT d_out, NumItemsT num_items, cudaStream_t stream = 0)
    {
        using Acc = ocu_detail::value_of<InputIteratorT>;
        return Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, cub::Min(),
                      highest<Acc>(), stream);
    }

private:
    template <typename T>
    static T lowest()
    {
        return std::numeric_limits<T>::lowest();
    }
    template <typename T>
    static T highest()
    {
        return std::numeric_limits<T>::max();
    }
};

/* ---- DeviceScan ------------------------------------------------------ */

struct DeviceScan {
    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT,
              typename NumItemsT>
    static cudaError_t InclusiveScan(void *d_temp_storage, size_t &temp_storage_bytes,
                                     InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op,
                                     NumItemsT num_items, cudaStream_t stream = 0)
    {
        using namespace ocu_detail;
        using Acc = value_of<InputIteratorT>;
        const long long n = (long long)num_items;
        const size_t need = scan_temp_bytes<Acc>(n);
        if (d_temp_storage == nullptr) return finish_query(temp_storage_bytes, need);
        cudaError_t e = check_space(temp_storage_bytes, need);
        if (e != cudaSuccess) return e;
        return scan_run<false, Acc>(d_temp_storage, d_in, d_out, n, scan_op, Acc(), stream);
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT,
              typename InitValueT, typename NumItemsT>
    static cudaError_t ExclusiveScan(void *d_temp_storage, size_t &temp_storage_bytes,
                                     InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op,
                                     InitValueT init_value, NumItemsT num_items,
                                     cudaStream_t stream = 0)
    {
        using namespace ocu_detail;
        using Acc = value_of<InputIteratorT>;
        const long long n = (long long)num_items;
        const size_t need = scan_temp_bytes<Acc>(n);
        if (d_temp_storage == nullptr) return finish_query(temp_storage_bytes, need);
        cudaError_t e = check_space(temp_storage_bytes, need);
        if (e != cudaSuccess) return e;
        return scan_run<true, Acc>(d_temp_storage, d_in, d_out, n, scan_op, init_value, stream);
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes,
                                    InputIteratorT d_in, OutputIteratorT d_out,
                                    NumItemsT num_items, cudaStream_t stream = 0)
    {
        return InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, cub::Sum(),
                             num_items, stream);
    }

    template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes,
                                    InputIteratorT d_in, OutputIteratorT d_out,
                                    NumItemsT num_items, cudaStream_t stream = 0)
    {
        using Acc = ocu_detail::value_of<InputIteratorT>;
        return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, cub::Sum(), Acc(0),
                             num_items, stream);
    }
};

/* ---- DeviceRadixSort ------------------------------------------------- */

/* Stable, ascending or descending, over key bits [begin_bit, end_bit) of
   the order-preserving key transform (after the descending inversion).
   Input and output ranges must not overlap. */
struct DeviceRadixSort {
    template <typename KeyT, typename ValueT, typename NumItemsT>
    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes,
                                 const KeyT *d_keys_in, KeyT *d_keys_out,
                                 const ValueT *d_values_in, ValueT *d_values_out,
                                 NumItemsT num_items, int begin_bit = 0,
                                 int end_bit = sizeof(KeyT) * 8, cudaStream_t stream = 0)
    {
        return ocu_detail::radix_sort<false, KeyT, ValueT, true>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
            (long long)num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename ValueT, typename NumItemsT>
    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes,
                                           const KeyT *d_keys_in, KeyT *d_keys_out,
                                           const ValueT *d_values_in, ValueT *d_values_out,
                                           NumItemsT num_items, int begin_bit = 0,
                                           int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0)
    {
        return ocu_detail::radix_sort<true, KeyT, ValueT, true>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
            (long long)num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename NumItemsT>
    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes,
                                const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items,
                                int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                cudaStream_t stream = 0)
    {
        return ocu_detail::radix_sort<false, KeyT, NullType, false>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, (const NullType *)nullptr,
            (NullType *)nullptr, (long long)num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename NumItemsT>
    static cudaError_t SortKeysDescending(void *d_temp_storage, size_t &temp_storage_bytes,
                                          const KeyT *d_keys_in, KeyT *d_keys_out,
                                          NumItemsT num_items, int begin_bit = 0,
                                          int end_bit = sizeof(KeyT) * 8,
                                          cudaStream_t stream = 0)
    {
        return ocu_detail::radix_sort<true, KeyT, NullType, false>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, (const NullType *)nullptr,
            (NullType *)nullptr, (long long)num_items, begin_bit, end_bit, stream);
    }
};

/* ---- segmented sorts ------------------------------------------------- */

/* Segment i is [d_begin_offsets[i], d_end_offsets[i]); an empty or
   reversed range is an empty segment. Items outside every segment are
   neither read nor written. Sorting is stable in every variant here. */
struct DeviceSegmentedRadixSort {
    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT, typename NumItemsT, typename NumSegmentsT>
    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes,
                                 const KeyT *d_keys_in, KeyT *d_keys_out,
                                 const ValueT *d_values_in, ValueT *d_values_out,
                                 NumItemsT num_items, NumSegmentsT num_segments,
                                 BeginOffsetIteratorT d_begin_offsets,
                                 EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                 int end_bit = sizeof(KeyT) * 8, cudaStream_t stream = 0)
    {
        return ocu_detail::segmented_sort<false, KeyT, ValueT, true>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
            (long long)num_items, (long long)num_segments, d_begin_offsets, d_end_offsets,
            begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT,
              typename EndOffsetIteratorT, typename NumItemsT, typename NumSegmentsT>
    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes,
                                           const KeyT *d_keys_in, KeyT *d_keys_out,
                                           const ValueT *d_values_in, ValueT *d_values_out,
                                           NumItemsT num_items, NumSegmentsT num_segments,
                                           BeginOffsetIteratorT d_begin_offsets,
                                           EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                           int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0)
    {
        return ocu_detail::segmented_sort<true, KeyT, ValueT, true>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out,
            (long long)num_items, (long long)num_segments, d_begin_offsets, d_end_offsets,
            begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
              typename NumItemsT, typename NumSegmentsT>
    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes,
                                const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items,
                                NumSegmentsT num_segments, BeginOffsetIteratorT d_begin_offsets,
                                EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                int end_bit = sizeof(KeyT) * 8, cudaStream_t stream = 0)
    {
        return ocu_detail::segmented_sort<false, KeyT, NullType, false>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, (const NullType *)nullptr,
            (NullType *)nullptr, (long long)num_items, (long long)num_segments, d_begin_offsets,
            d_end_offsets, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,
              typename NumItemsT, typename NumSegmentsT>
    static cudaError_t SortKeysDescending(void *d_temp_storage, size_t &temp_storage_bytes,
                                          const KeyT *d_keys_in, KeyT *d_keys_out,
                                          NumItemsT num_items, NumSegmentsT num_segments,
                                          BeginOffsetIteratorT d_begin_offsets,
                                          EndOffsetIteratorT d_end_offsets, int begin_bit = 0,
                                          int end_bit = sizeof(KeyT) * 8,
                                          cudaStream_t stream = 0)
    {
        return ocu_detail::segmented_sort<true, KeyT, NullType, false>(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, (const NullType *)nullptr,
            (NullType *)nullptr, (long long)num_items, (long long)num_segments, d_begin_offsets,
            d_end_offsets, begin_bit, end_bit, stream);
    }
};

struct DeviceSegmentedSort {
#define OCU_CUB_SEGSORT_PAIRS(NAME, DESC)                                                       \
    template <typename KeyT, typename ValueT, typename BeginOffsetIteratorT,                   \
              typename EndOffsetIteratorT, typename NumItemsT, typename NumSegmentsT>          \
    static cudaError_t NAME(void *d_temp_storage, size_t &temp_storage_bytes,                 \
                            const KeyT *d_keys_in, KeyT *d_keys_out,                           \
                            const ValueT *d_values_in, ValueT *d_values_out,                   \
                            NumItemsT num_items, NumSegmentsT num_segments,                    \
                            BeginOffsetIteratorT d_begin_offsets,                              \
                            EndOffsetIteratorT d_end_offsets, cudaStream_t stream = 0)         \
    {                                                                                          \
        return ocu_detail::segmented_sort<DESC, KeyT, ValueT, true>(                           \
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,            \
            d_values_out, (long long)num_items, (long long)num_segments, d_begin_offsets,      \
            d_end_offsets, 0, (int)sizeof(KeyT) * 8, stream);                                  \
    }
#define OCU_CUB_SEGSORT_KEYS(NAME, DESC)                                                        \
    template <typename KeyT, typename BeginOffsetIteratorT, typename EndOffsetIteratorT,       \
              typename NumItemsT, typename NumSegmentsT>                                       \
    static cudaError_t NAME(void *d_temp_storage, size_t &temp_storage_bytes,                 \
                            const KeyT *d_keys_in, KeyT *d_keys_out, NumItemsT num_items,      \
                            NumSegmentsT num_segments, BeginOffsetIteratorT d_begin_offsets,   \
                            EndOffsetIteratorT d_end_offsets, cudaStream_t stream = 0)         \
    {                                                                                          \
        return ocu_detail::segmented_sort<DESC, KeyT, NullType, false>(                        \
            d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out,                         \
            (const NullType *)nullptr, (NullType *)nullptr, (long long)num_items,              \
            (long long)num_segments, d_begin_offsets, d_end_offsets, 0,                        \
            (int)sizeof(KeyT) * 8, stream);                                                    \
    }
    OCU_CUB_SEGSORT_PAIRS(SortPairs, false)
    OCU_CUB_SEGSORT_PAIRS(SortPairsDescending, true)
    OCU_CUB_SEGSORT_PAIRS(StableSortPairs, false)
    OCU_CUB_SEGSORT_PAIRS(StableSortPairsDescending, true)
    OCU_CUB_SEGSORT_KEYS(SortKeys, false)
    OCU_CUB_SEGSORT_KEYS(SortKeysDescending, true)
    OCU_CUB_SEGSORT_KEYS(StableSortKeys, false)
    OCU_CUB_SEGSORT_KEYS(StableSortKeysDescending, true)
#undef OCU_CUB_SEGSORT_PAIRS
#undef OCU_CUB_SEGSORT_KEYS
};

/* ---- BlockReduce, WarpReduce ------------------------------------------ */

/*
 * As documented: the aggregate is returned to thread 0 (lane 0 of each
 * logical warp for WarpReduce); other threads' return values are
 * undefined (here: also the aggregate). Inputs combine in thread order,
 * op(lower, higher), so the operator need only be associative. Every
 * thread of the block (warp) must call; TempStorage may be reused after a
 * __syncthreads() (__syncwarp()).
 */
enum BlockReduceAlgorithm {
    BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
    BLOCK_REDUCE_RAKING,
    BLOCK_REDUCE_WARP_REDUCTIONS
};

template <typename T, int BLOCK_DIM_X, BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockReduce {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

public:
    struct TempStorage {
        ocu_detail::raw_array<T, BLOCK_THREADS> buf;
    };

    __device__ __forceinline__ BlockReduce(TempStorage &temp_storage)
        : s_(temp_storage), tid_(ocu_detail::linear_tid<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z>())
    {
    }

    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(T input, ReductionOp op)
    {
        return reduce(input, op, BLOCK_THREADS);
    }

    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(T input, ReductionOp op, int num_valid)
    {
        return reduce(input, op, num_valid < BLOCK_THREADS ? num_valid : BLOCK_THREADS);
    }

    template <int ITEMS_PER_THREAD, typename ReductionOp>
    __device__ __forceinline__ T Reduce(T (&inputs)[ITEMS_PER_THREAD], ReductionOp op)
    {
        T v = inputs[0];
        for (int i = 1; i < ITEMS_PER_THREAD; ++i) v = op(v, inputs[i]);
        return reduce(v, op, BLOCK_THREADS);
    }

    __device__ __forceinline__ T Sum(T input) { return Reduce(input, cub::Sum()); }
    __device__ __forceinline__ T Sum(T input, int num_valid) { return Reduce(input, cub::Sum(), num_valid); }
    template <int ITEMS_PER_THREAD>
    __device__ __forceinline__ T Sum(T (&inputs)[ITEMS_PER_THREAD]) { return Reduce(inputs, cub::Sum()); }

private:
    TempStorage &s_;
    int tid_;

    template <typename ReductionOp>
    __device__ __forceinline__ T reduce(T v, ReductionOp op, int n)
    {
        T *b = s_.buf.data();
        b[tid_] = v;
        __syncthreads();
        for (int d = 1; d < BLOCK_THREADS; d *= 2) {
            if (tid_ % (2 * d) == 0 && tid_ + d < n) b[tid_] = op(b[tid_], b[tid_ + d]);
            __syncthreads();
        }
        return b[0];
    }
};

template <typename T, int LOGICAL_WARP_THREADS = 32, int LEGACY_PTX_ARCH = 0>
class WarpReduce {
public:
    struct TempStorage {
        ocu_detail::raw_array<T, LOGICAL_WARP_THREADS> buf;
    };

    __device__ __forceinline__ WarpReduce(TempStorage &temp_storage)
        : s_(temp_storage), lane_((int)(ocu_detail::linear_tid<1024, 1, 1>() % LOGICAL_WARP_THREADS))
    {
        /* linear thread id over x, y, z (the block's dims are not template parameters here) */
        lane_ = (int)((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % LOGICAL_WARP_THREADS);
    }

    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(T input, ReductionOp op) { return reduce(input, op, LOGICAL_WARP_THREADS); }

    template <typename ReductionOp>
    __device__ __forceinline__ T Reduce(T input, ReductionOp op, int valid_items)
    {
        return reduce(input, op, valid_items < LOGICAL_WARP_THREADS ? valid_items : LOGICAL_WARP_THREADS);
    }

    __device__ __forceinline__ T Sum(T input) { return Reduce(input, cub::Sum()); }
    __device__ __forceinline__ T Sum(T input, int valid_items) { return Reduce(input, cub::Sum(), valid_items); }

private:
    TempStorage &s_;
    int lane_;

    template <typename ReductionOp>
    __device__ __forceinline__ T reduce(T v, ReductionOp op, int n)
    {
        T *b = s_.buf.data();
        b[lane_] = v;
        __syncwarp();
        for (int d = 1; d < LOGICAL_WARP_THREADS; d *= 2) {
            T mine = b[lane_];
            T other = lane_ + d < LOGICAL_WARP_THREADS ? b[lane_ + d] : mine;
            __syncwarp();
            if (lane_ % (2 * d) == 0 && lane_ + d < n) b[lane_] = op(mine, other);
            __syncwarp();
        }
        return b[0];
    }
};

} // namespace cub

#endif /* OCU_CUB_CUB_CUH */
