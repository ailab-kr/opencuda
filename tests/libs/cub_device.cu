// CUB device-wide algorithms (card libs-cub-subset, test list items 7-11):
// DeviceReduce, DeviceScan, DeviceRadixSort, DeviceSegmentedRadixSort,
// DeviceSegmentedSort. Expected results come from host reference
// computations and from the public CUB API documentation (two-phase
// temporary storage, stability of radix sort, floating-point key order).
// Runs on the CPU path and on the NVIDIA path.
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

static int g_fail, g_checks;
static const int kMaxPrint = 40;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_fail < kMaxPrint) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
// Check with a context string.
#define CHECKM(c, ...) do { ++g_checks; if (!(c)) { if (g_fail < kMaxPrint) { std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } ++g_fail; } } while (0)

template <class T>
struct Dev {
    T *p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { cudaMalloc((void **)&p, (count ? count : 1) * sizeof(T)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev &) = delete;
    Dev &operator=(const Dev &) = delete;
    void up(const std::vector<T> &v) { if (!v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    void fill(T v) { up(std::vector<T>(n, v)); }
    std::vector<T> down() const
    {
        std::vector<T> h(n);
        if (n) cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

// Two-phase call. f(void *temp, size_t &bytes) -> cudaError_t.
template <class F>
static cudaError_t two_phase(F f)
{
    size_t bytes = 0;
    cudaError_t e = f(nullptr, bytes);
    if (e != cudaSuccess) return e;
    if (bytes == 0) return cudaErrorInvalidValue;
    Dev<unsigned char> tmp(bytes);
    size_t b = bytes;
    e = f(tmp.p, b);
    if (e != cudaSuccess) return e;
    return cudaDeviceSynchronize();
}

// Item 7: the temporary storage protocol of a call.
template <class F, class Out>
static void protocol(const char *name, F f, Dev<Out> &out, Out sentinel)
{
    out.fill(sentinel);
    size_t q1 = 12345, q2 = 54321;
    CHECKM(f(nullptr, q1) == cudaSuccess, "%s query", name);
    CHECKM(f(nullptr, q2) == cudaSuccess, "%s query 2", name);
    CHECKM(q1 > 0, "%s bytes %zu", name, q1);
    CHECKM(q1 == q2, "%s bytes differ %zu %zu", name, q1, q2);
    cudaDeviceSynchronize();
    std::vector<Out> r = out.down();
    bool same = true;
    for (Out v : r) same = same && std::memcmp(&v, &sentinel, sizeof(Out)) == 0;
    CHECKM(same, "%s query changed the output", name);
    if (q1 >= 1) {
        Dev<unsigned char> tmp(q1);
        size_t small = q1 - 1;
        cudaError_t e = f(tmp.p, small);
        CHECKM(e == cudaErrorInvalidValue, "%s short storage -> %d", name, (int)e);
        (void)cudaGetLastError();
        cudaDeviceSynchronize();
        (void)cudaGetLastError();
        // The failed call must not have produced output.
        r = out.down();
        same = true;
        for (Out v : r) same = same && std::memcmp(&v, &sentinel, sizeof(Out)) == 0;
        CHECKM(same, "%s short storage wrote output", name);
        size_t exact = q1;
        CHECKM(f(tmp.p, exact) == cudaSuccess, "%s exact storage", name);
        cudaDeviceSynchronize();
    }
}

static const size_t kSizes[] = {0, 1, 2, 31, 32, 33, 255, 256, 257, 1000, 2047, 2048, 2049,
                                4095, 4096, 4097, 8191, 8192, 8193, 16385, 100000, 1000000};

// ---- reduce ---------------------------------------------------------------

struct MulMod7 { // not used for ordering: commutative, associative
    __host__ __device__ int operator()(int a, int b) const { return (a * b) % 7; }
};
struct XorOp {
    __host__ __device__ unsigned operator()(unsigned a, unsigned b) const { return a ^ b; }
};

template <class T>
static std::vector<T> gen_small(size_t n, int mod, int shift, uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::vector<T> v(n);
    for (auto &x : v) x = T((int)(g() % mod) - shift);
    return v;
}

template <class T>
static void test_reduce_type(const char *tn, int mod, int shift)
{
    for (size_t n : kSizes) {
        std::vector<T> h = gen_small<T>(n, mod, shift, 100 + n);
        Dev<T> in(n), out(1);
        in.up(h);
        T ref_sum = T(0);
        for (T v : h) ref_sum = T(ref_sum + v);
        // Sum
        out.fill(T(-77));
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Sum(t, b, in.p, out.p, (int)n); });
        CHECKM(e == cudaSuccess, "%s Sum n=%zu err %d", tn, n, (int)e);
        T got = out.down()[0];
        CHECKM(got == ref_sum, "%s Sum n=%zu got %g want %g", tn, n, (double)got, (double)ref_sum);
        // Max / Min
        T mx = n ? *std::max_element(h.begin(), h.end()) : std::numeric_limits<T>::lowest();
        T mn = n ? *std::min_element(h.begin(), h.end()) : std::numeric_limits<T>::max();
        out.fill(T(-77));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Max(t, b, in.p, out.p, (int)n); });
        got = out.down()[0];
        CHECKM(e == cudaSuccess && got == mx, "%s Max n=%zu got %g want %g", tn, n, (double)got, (double)mx);
        out.fill(T(-77));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Min(t, b, in.p, out.p, (int)n); });
        got = out.down()[0];
        CHECKM(e == cudaSuccess && got == mn, "%s Min n=%zu got %g want %g", tn, n, (double)got, (double)mn);
        // Reduce(op, init): init enters the result once.
        out.fill(T(-77));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, in.p, out.p, (int)n, cub::Sum(), T(5)); });
        got = out.down()[0];
        CHECKM(e == cudaSuccess && got == T(ref_sum + T(5)), "%s Reduce(Sum,5) n=%zu got %g want %g", tn, n, (double)got, (double)(T)(ref_sum + T(5)));
        T big = T(1000), small = T(-1000);
        if (std::is_unsigned<T>::value) small = T(0);
        out.fill(T(-77));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, in.p, out.p, (int)n, cub::Max(), big); });
        got = out.down()[0];
        CHECKM(e == cudaSuccess && got == big, "%s Reduce(Max,1000) n=%zu got %g", tn, n, (double)got);
        out.fill(T(-77));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, in.p, out.p, (int)n, cub::Min(), small); });
        got = out.down()[0];
        CHECKM(e == cudaSuccess && got == small, "%s Reduce(Min,-1000) n=%zu got %g", tn, n, (double)got);
    }
}

static void test_reduce()
{
    test_reduce_type<int>("int", 201, 100);
    test_reduce_type<unsigned>("uint", 50, 0);
    test_reduce_type<long long>("int64", 1000, 500);
    test_reduce_type<float>("float", 4, 0);     // small integers: exact in any order
    test_reduce_type<double>("double", 7, 3);

    // Empty input: documented identities of Max / Min through the Reduce form
    // and the returned init.
    {
        Dev<int> in(1), out(1);
        out.fill(-7);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Sum(t, b, in.p, out.p, 0); });
        CHECK(e == cudaSuccess && out.down()[0] == 0);
        out.fill(-7);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, in.p, out.p, 0, cub::Sum(), 42); });
        CHECK(e == cudaSuccess && out.down()[0] == 42);
    }
    // Custom functors; 64-bit item count; extremes; float specials.
    {
        size_t n = 70001;
        std::vector<unsigned> h(n);
        std::mt19937 g(5);
        unsigned x = 0;
        for (auto &v : h) { v = g(); x ^= v; }
        Dev<unsigned> in(n), out(1);
        in.up(h);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, in.p, out.p, (long long)n, XorOp(), 0u); });
        CHECK(e == cudaSuccess && out.down()[0] == x);
        std::vector<int> hi(n);
        int m = 1;
        for (size_t i = 0; i < n; ++i) { hi[i] = (int)(g() % 6) + 1; m = (m * hi[i]) % 7; }
        Dev<int> ini(n), outi(1);
        ini.up(hi);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, ini.p, outi.p, (int)n, MulMod7(), 1); });
        CHECK(e == cudaSuccess && outi.down()[0] == m);
        // Extremes with Max/Min.
        std::vector<int> ex(5000, 0);
        ex[4321] = std::numeric_limits<int>::max();
        ex[17] = std::numeric_limits<int>::min();
        Dev<int> ine(ex.size()), oute(1);
        ine.up(ex);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Max(t, b, ine.p, oute.p, 5000); });
        CHECK(e == cudaSuccess && oute.down()[0] == std::numeric_limits<int>::max());
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Min(t, b, ine.p, oute.p, 5000); });
        CHECK(e == cudaSuccess && oute.down()[0] == std::numeric_limits<int>::min());
        // Floats with infinities and signed zeros.
        std::vector<float> f(3000, -0.0f);
        f[2000] = INFINITY;
        f[5] = -INFINITY;
        Dev<float> inf_(f.size()), outf(1);
        inf_.up(f);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Max(t, b, inf_.p, outf.p, 3000); });
        CHECK(e == cudaSuccess && outf.down()[0] == INFINITY);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Min(t, b, inf_.p, outf.p, 3000); });
        CHECK(e == cudaSuccess && outf.down()[0] == -INFINITY);
        f[2000] = -0.0f; f[5] = 0.0f;
        inf_.up(f);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Max(t, b, inf_.p, outf.p, 3000); });
        CHECK(e == cudaSuccess && outf.down()[0] == 0.0f);   // -0 == +0: either sign
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceReduce::Sum(t, b, inf_.p, outf.p, 3000); });
        CHECK(e == cudaSuccess && outf.down()[0] == 0.0f);
    }
    // Item 7 for reduce.
    {
        size_t n = 3000;
        std::vector<int> h(n, 1);
        Dev<int> in(n), out(1);
        in.up(h);
        for (size_t nn : {(size_t)0, (size_t)1, n, (size_t)100000}) {
            Dev<int> big(nn);
            big.fill(1);
            protocol("Sum", [&](void *t, size_t &b) { return cub::DeviceReduce::Sum(t, b, big.p, out.p, (int)nn); }, out, -12345);
            protocol("Max", [&](void *t, size_t &b) { return cub::DeviceReduce::Max(t, b, big.p, out.p, (int)nn); }, out, -12345);
            protocol("Min", [&](void *t, size_t &b) { return cub::DeviceReduce::Min(t, b, big.p, out.p, (int)nn); }, out, -12345);
            protocol("Reduce", [&](void *t, size_t &b) { return cub::DeviceReduce::Reduce(t, b, big.p, out.p, (int)nn, cub::Sum(), 0); }, out, -12345);
        }
    }
}

// ---- scan -------------------------------------------------------------------

struct M2 { // 2x2 matrix, modular arithmetic; associative, not commutative
    uint32_t a, b, c, d;
};
static const uint64_t kMod = 1000000007ull;
struct MatMul {
    __host__ __device__ M2 operator()(const M2 &x, const M2 &y) const
    {
        M2 r;
        r.a = (uint32_t)(((uint64_t)x.a * y.a + (uint64_t)x.b * y.c) % 1000000007ull);
        r.b = (uint32_t)(((uint64_t)x.a * y.b + (uint64_t)x.b * y.d) % 1000000007ull);
        r.c = (uint32_t)(((uint64_t)x.c * y.a + (uint64_t)x.d * y.c) % 1000000007ull);
        r.d = (uint32_t)(((uint64_t)x.c * y.b + (uint64_t)x.d * y.d) % 1000000007ull);
        return r;
    }
};
static bool eq(const M2 &x, const M2 &y) { return x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d; }

template <class T>
static void test_scan_type(const char *tn, int mod, int shift)
{
    for (size_t n : kSizes) {
        std::vector<T> h = gen_small<T>(n, mod, shift, 900 + n);
        std::vector<T> inc(n), exc(n);
        T run = T(0);
        for (size_t i = 0; i < n; ++i) { exc[i] = run; run = T(run + h[i]); inc[i] = run; }
        Dev<T> in(n), out(n);
        in.up(h);
        const T sent = T(-99);
        // InclusiveSum
        out.fill(sent);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveSum(t, b, in.p, out.p, (int)n); });
        std::vector<T> r = out.down();
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < n; ++i) if (!(r[i] == inc[i])) { if (!bad) first = i; ++bad; }
        CHECKM(e == cudaSuccess && bad == 0, "%s InclusiveSum n=%zu bad=%zu first=%zu got %g want %g", tn, n, bad, first, bad ? (double)r[first] : 0.0, bad ? (double)inc[first] : 0.0);
        // ExclusiveSum
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveSum(t, b, in.p, out.p, (int)n); });
        r = out.down();
        bad = 0;
        for (size_t i = 0; i < n; ++i) if (!(r[i] == exc[i])) { if (!bad) first = i; ++bad; }
        CHECKM(e == cudaSuccess && bad == 0, "%s ExclusiveSum n=%zu bad=%zu first=%zu got %g want %g", tn, n, bad, first, bad ? (double)r[first] : 0.0, bad ? (double)exc[first] : 0.0);
        // In place.
        Dev<T> ip(n);
        ip.up(h);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveSum(t, b, ip.p, ip.p, (int)n); });
        CHECKM(e == cudaSuccess && ip.down() == inc, "%s InclusiveSum in place n=%zu", tn, n);
        ip.up(h);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveSum(t, b, ip.p, ip.p, (int)n); });
        CHECKM(e == cudaSuccess && ip.down() == exc, "%s ExclusiveSum in place n=%zu", tn, n);
        // ExclusiveScan(Sum, init) and InclusiveScan(Max)
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveScan(t, b, in.p, out.p, cub::Sum(), T(10), (int)n); });
        r = out.down();
        bad = 0;
        for (size_t i = 0; i < n; ++i) if (!(r[i] == T(exc[i] + T(10)))) { if (!bad) first = i; ++bad; }
        CHECKM(e == cudaSuccess && bad == 0, "%s ExclusiveScan(Sum,10) n=%zu bad=%zu", tn, n, bad);
        if (n) CHECKM(r[0] == T(10), "%s first output of ExclusiveScan is init", tn);
        std::vector<T> mxs(n);
        T m = std::numeric_limits<T>::lowest();
        for (size_t i = 0; i < n; ++i) { m = std::max(m, h[i]); mxs[i] = m; }
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveScan(t, b, in.p, out.p, cub::Max(), (int)n); });
        CHECKM(e == cudaSuccess && out.down() == mxs, "%s InclusiveScan(Max) n=%zu", tn, n);
        std::vector<T> mxe(n);
        m = T(50);
        for (size_t i = 0; i < n; ++i) { mxe[i] = m; m = std::max(m, h[i]); }
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveScan(t, b, in.p, out.p, cub::Max(), T(50), (int)n); });
        CHECKM(e == cudaSuccess && out.down() == mxe, "%s ExclusiveScan(Max,50) n=%zu", tn, n);
    }
}

static void test_scan_noncommutative()
{
    const size_t ns[] = {0, 1, 2, 255, 257, 2049, 8193, 20000, 100000};
    for (size_t n : ns) {
        std::vector<M2> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = M2{1, (uint32_t)(i % 5), (uint32_t)(i % 3), 1 + (uint32_t)(i % 2)};
        MatMul mul;
        M2 ident{1, 0, 0, 1};
        std::vector<M2> inc(n), exc(n);
        M2 acc = ident;
        for (size_t i = 0; i < n; ++i) { exc[i] = acc; acc = mul(acc, h[i]); inc[i] = acc; }
        Dev<M2> in(n), out(n);
        in.up(h);
        M2 sent{7, 7, 7, 7};
        out.fill(sent);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveScan(t, b, in.p, out.p, MatMul(), (int)n); });
        std::vector<M2> r = out.down();
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) bad += !eq(r[i], inc[i]);
        CHECKM(e == cudaSuccess && bad == 0, "InclusiveScan(MatMul) n=%zu bad=%zu", n, bad);
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveScan(t, b, in.p, out.p, MatMul(), ident, (int)n); });
        r = out.down();
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += !eq(r[i], exc[i]);
        CHECKM(e == cudaSuccess && bad == 0, "ExclusiveScan(MatMul) n=%zu bad=%zu", n, bad);
        if (n) CHECK(eq(r[0], ident));
        // Non-identity init: first output is init, the rest are init * prefix.
        M2 init{2, 3, 5, 7};
        out.fill(sent);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveScan(t, b, in.p, out.p, MatMul(), init, (int)n); });
        r = out.down();
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += !eq(r[i], mul(init, exc[i]));
        CHECKM(e == cudaSuccess && bad == 0, "ExclusiveScan(MatMul, init) n=%zu bad=%zu", n, bad);
        // In place.
        Dev<M2> ip(n);
        ip.up(h);
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveScan(t, b, ip.p, ip.p, MatMul(), (int)n); });
        r = ip.down();
        bad = 0;
        for (size_t i = 0; i < n; ++i) bad += !eq(r[i], inc[i]);
        CHECKM(e == cudaSuccess && bad == 0, "InclusiveScan(MatMul) in place n=%zu bad=%zu", n, bad);
    }
}

static void test_scan()
{
    test_scan_type<int>("int", 7, 3);
    test_scan_type<unsigned>("uint", 5, 0);
    test_scan_type<long long>("int64", 1001, 500);
    test_scan_type<float>("float", 4, 0);
    test_scan_type<double>("double", 6, 2);
    test_scan_noncommutative();
    // Protocol, n = 0 leaves the output untouched.
    for (size_t nn : {(size_t)0, (size_t)1, (size_t)5000, (size_t)100000}) {
        Dev<int> in(nn), out(nn ? nn : 1);
        in.fill(1);
        protocol("InclusiveSum", [&](void *t, size_t &b) { return cub::DeviceScan::InclusiveSum(t, b, in.p, out.p, (int)nn); }, out, -321);
        protocol("ExclusiveSum", [&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveSum(t, b, in.p, out.p, (int)nn); }, out, -321);
        protocol("ExclusiveScan", [&](void *t, size_t &b) { return cub::DeviceScan::ExclusiveScan(t, b, in.p, out.p, cub::Sum(), 0, (int)nn); }, out, -321);
        protocol("InclusiveScan", [&](void *t, size_t &b) { return cub::DeviceScan::InclusiveScan(t, b, in.p, out.p, cub::Max(), (int)nn); }, out, -321);
    }
    {
        Dev<int> in(10), out(10);
        in.fill(1);
        out.fill(-5);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceScan::InclusiveSum(t, b, in.p, out.p, 0); });
        std::vector<int> r = out.down();
        CHECK(e == cudaSuccess && std::all_of(r.begin(), r.end(), [](int v) { return v == -5; }));
    }
}

// ---- radix sort ---------------------------------------------------------------

template <class K> struct Bits;
template <> struct Bits<int32_t> { typedef uint32_t U; };
template <> struct Bits<uint32_t> { typedef uint32_t U; };
template <> struct Bits<int64_t> { typedef uint64_t U; };
template <> struct Bits<uint64_t> { typedef uint64_t U; };
template <> struct Bits<float> { typedef uint32_t U; };
template <> struct Bits<double> { typedef uint64_t U; };

template <class K>
static bool same_bits(const K &a, const K &b) { return std::memcmp(&a, &b, sizeof(K)) == 0; }

template <class K, class = void> struct is_fp : std::false_type {};
template <class K> struct is_fp<K, typename std::enable_if<std::is_floating_point<K>::value>::type> : std::true_type {};

template <class K>
static K gen_key(std::mt19937_64 &g, int style)
{
    if (style == 1) return K((int)(g() % 17) - 8);                 // many equal keys
    if constexpr (std::is_floating_point<K>::value) {
        const K sp[] = {K(0.0), -K(0.0), std::numeric_limits<K>::infinity(), -std::numeric_limits<K>::infinity(),
                        std::numeric_limits<K>::max(), -std::numeric_limits<K>::max(),
                        std::numeric_limits<K>::min(), -std::numeric_limits<K>::min(),
                        std::numeric_limits<K>::denorm_min(), -std::numeric_limits<K>::denorm_min(), K(1), K(-1)};
        if (g() % 8 == 0) return sp[g() % (sizeof sp / sizeof sp[0])];
        for (;;) {
            typename Bits<K>::U u = (typename Bits<K>::U)g();
            K k;
            std::memcpy(&k, &u, sizeof k);
            if (!std::isnan(k)) return k;
        }
    }
    typename Bits<K>::U u = (typename Bits<K>::U)g();
    K k;
    std::memcpy(&k, &u, sizeof k);
    return k;
}

// Reference order. bits [b,e): when the range is the whole key, the natural
// order of K (floating-point: -0.0 == +0.0).
template <class K>
static uint64_t subkey(K k, int b, int e)
{
    typename Bits<K>::U u;
    std::memcpy(&u, &k, sizeof k);
    uint64_t w = (uint64_t)u >> b;
    int width = e - b;
    return width >= 64 ? w : (w & ((1ull << width) - 1));
}

template <class K>
static void ref_sort(std::vector<size_t> &idx, const std::vector<K> &keys, bool desc, int b, int e)
{
    bool full = b == 0 && e == (int)sizeof(K) * 8;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        if (full) return desc ? keys[x] > keys[y] : keys[x] < keys[y];
        uint64_t sx = subkey(keys[x], b, e), sy = subkey(keys[y], b, e);
        return desc ? sx > sy : sx < sy;
    });
}

template <class K>
static void sort_case(const char *tn, std::vector<K> keys, bool pairs, bool desc, int b, int e, const char *tag)
{
    size_t n = keys.size();
    std::vector<int> vals(n);
    std::iota(vals.begin(), vals.end(), 0);
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    ref_sort(idx, keys, desc, b, e);
    Dev<K> kin(n), kout(n);
    Dev<int> vin(n), vout(n);
    kin.up(keys);
    vin.up(vals);
    K ksent;
    std::memset(&ksent, 0x5a, sizeof ksent);
    kout.fill(ksent);
    vout.fill(-1);
    cudaError_t err;
    if (pairs) {
        if (desc) err = two_phase([&](void *t, size_t &bb) { return cub::DeviceRadixSort::SortPairsDescending(t, bb, kin.p, kout.p, vin.p, vout.p, (int)n, b, e); });
        else err = two_phase([&](void *t, size_t &bb) { return cub::DeviceRadixSort::SortPairs(t, bb, kin.p, kout.p, vin.p, vout.p, (int)n, b, e); });
    } else {
        if (desc) err = two_phase([&](void *t, size_t &bb) { return cub::DeviceRadixSort::SortKeysDescending(t, bb, kin.p, kout.p, (int)n, b, e); });
        else err = two_phase([&](void *t, size_t &bb) { return cub::DeviceRadixSort::SortKeys(t, bb, kin.p, kout.p, (int)n, b, e); });
    }
    CHECKM(err == cudaSuccess, "%s %s n=%zu err %d", tn, tag, n, (int)err);
    std::vector<K> ko = kout.down();
    std::vector<int> vo = vout.down();
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < n; ++i) {
        bool ok = same_bits(ko[i], keys[idx[i]]) && (!pairs || vo[i] == (int)idx[i]);
        if (!ok) { if (!bad) first = i; ++bad; }
    }
    if (bad)
        CHECKM(false, "%s %s n=%zu bad=%zu first=%zu got key %.9g val %d want key %.9g val %zu", tn, tag, n, bad, first,
               (double)ko[first], pairs ? vo[first] : -1, (double)keys[idx[first]], idx[first]);
    else
        CHECK(true);
    // Input unchanged.
    CHECKM(kin.down().size() == n && std::equal(keys.begin(), keys.end(), kin.down().begin(), same_bits<K>), "%s %s input modified", tn, tag);
}

template <class K>
static void test_sort_type(const char *tn, const std::vector<size_t> &sizes)
{
    for (size_t n : sizes) {
        for (int style = 0; style < 2; ++style) {
            std::mt19937_64 g(31 * n + style);
            std::vector<K> keys(n);
            for (auto &k : keys) k = gen_key<K>(g, style);
            char tag[64];
            std::snprintf(tag, sizeof tag, "style%d", style);
            const int W = sizeof(K) * 8;
            sort_case<K>(tn, keys, true, false, 0, W, tag);
            sort_case<K>(tn, keys, true, true, 0, W, tag);
            sort_case<K>(tn, keys, false, false, 0, W, tag);
            sort_case<K>(tn, keys, false, true, 0, W, tag);
        }
    }
}

static void test_sort_special_floats()
{
    // -0.0 / +0.0 are equal keys (input order kept); infinities at the ends.
    const float nz = -0.0f, pz = 0.0f, inf = INFINITY;
    std::vector<float> k = {pz, nz, 1.0f, -1.0f, nz, pz, -inf, inf, nz, 0.5f, -0.5f, pz, inf, -inf, 1e-45f, -1e-45f};
    sort_case<float>("float", k, true, false, 0, 32, "zeros");
    sort_case<float>("float", k, true, true, 0, 32, "zeros");
    sort_case<float>("float", k, false, false, 0, 32, "zeros");
    std::vector<double> kd(k.begin(), k.end());
    for (size_t i = 0; i < k.size(); ++i) kd[i] = (double)k[i];
    sort_case<double>("double", kd, true, false, 0, 64, "zeros");
    sort_case<double>("double", kd, true, true, 0, 64, "zeros");
    // All zeros of both signs, large enough for several tiles: order unchanged.
    std::vector<float> z(5000);
    for (size_t i = 0; i < z.size(); ++i) z[i] = (i * 7 % 3 == 0) ? nz : pz;
    sort_case<float>("float", z, true, false, 0, 32, "allzeros");
    sort_case<float>("float", z, true, true, 0, 32, "allzeros");

    // NaN: the position of NaNs is not asserted. Everything else must be in
    // order and the output must be a permutation of the input.
    for (size_t n : {(size_t)20, (size_t)1500, (size_t)5000}) {
        for (int desc = 0; desc < 2; ++desc) {
            std::mt19937_64 g(77 + n);
            std::vector<float> keys(n);
            for (auto &x : keys) x = gen_key<float>(g, 0);
            for (size_t i = 0; i < n; i += 7) keys[i] = std::numeric_limits<float>::quiet_NaN();
            for (size_t i = 3; i < n; i += 31) keys[i] = -std::numeric_limits<float>::quiet_NaN();
            std::vector<int> vals(n);
            std::iota(vals.begin(), vals.end(), 0);
            Dev<float> kin(n), kout(n);
            Dev<int> vin(n), vout(n);
            kin.up(keys); vin.up(vals);
            cudaError_t e = desc ? two_phase([&](void *t, size_t &b) { return cub::DeviceRadixSort::SortPairsDescending(t, b, kin.p, kout.p, vin.p, vout.p, (int)n); })
                                 : two_phase([&](void *t, size_t &b) { return cub::DeviceRadixSort::SortPairs(t, b, kin.p, kout.p, vin.p, vout.p, (int)n); });
            CHECK(e == cudaSuccess);
            std::vector<float> ko = kout.down();
            std::vector<int> vo = vout.down();
            std::vector<int> seen(n, 0);
            bool perm = true;
            for (size_t i = 0; i < n; ++i) {
                if (vo[i] < 0 || (size_t)vo[i] >= n || seen[vo[i]]++) { perm = false; break; }
                if (!same_bits(ko[i], keys[vo[i]])) { perm = false; break; }
            }
            CHECKM(perm, "NaN sort n=%zu desc=%d: output is not a permutation", n, desc);
            if (!perm) continue;
            // Non-NaN subsequence: sorted and stable.
            std::vector<size_t> ref;
            for (size_t i = 0; i < n; ++i) if (!std::isnan(keys[i])) ref.push_back(i);
            std::stable_sort(ref.begin(), ref.end(), [&](size_t x, size_t y) { return desc ? keys[x] > keys[y] : keys[x] < keys[y]; });
            std::vector<int> got;
            for (size_t i = 0; i < n; ++i) if (!std::isnan(ko[i])) got.push_back(vo[i]);
            bool ok = got.size() == ref.size();
            for (size_t i = 0; ok && i < got.size(); ++i) ok = got[i] == (int)ref[i];
            CHECKM(ok, "NaN sort n=%zu desc=%d: non-NaN keys out of order", n, desc);
        }
    }
}

template <class K>
static void test_sort_bits(const char *tn, std::vector<size_t> sizes, std::vector<std::pair<int, int>> ranges)
{
    for (size_t n : sizes) {
        std::mt19937_64 g(1234 + n);
        std::vector<K> keys(n);
        for (auto &k : keys) k = gen_key<K>(g, 0);
        for (auto r : ranges) {
            char tag[64];
            std::snprintf(tag, sizeof tag, "bits[%d,%d)", r.first, r.second);
            sort_case<K>(tn, keys, true, false, r.first, r.second, tag);
            sort_case<K>(tn, keys, true, true, r.first, r.second, tag);
            sort_case<K>(tn, keys, false, false, r.first, r.second, tag);
            sort_case<K>(tn, keys, false, true, r.first, r.second, tag);
        }
    }
}

static void test_sort_protocol_errors()
{
    size_t n = 5000;
    Dev<uint32_t> kin(n), kout(n);
    Dev<int> vin(n), vout(n);
    kin.fill(3);
    vin.fill(1);
    for (size_t nn : {(size_t)0, (size_t)1, (size_t)1025, n}) {
        protocol("SortPairs", [&](void *t, size_t &b) { return cub::DeviceRadixSort::SortPairs(t, b, kin.p, kout.p, vin.p, vout.p, (int)nn); }, kout, 0xdeadbeefu);
        protocol("SortPairsDescending", [&](void *t, size_t &b) { return cub::DeviceRadixSort::SortPairsDescending(t, b, kin.p, kout.p, vin.p, vout.p, (int)nn); }, kout, 0xdeadbeefu);
        protocol("SortKeys", [&](void *t, size_t &b) { return cub::DeviceRadixSort::SortKeys(t, b, kin.p, kout.p, (int)nn); }, kout, 0xdeadbeefu);
        protocol("SortKeysDescending", [&](void *t, size_t &b) { return cub::DeviceRadixSort::SortKeysDescending(t, b, kin.p, kout.p, (int)nn); }, kout, 0xdeadbeefu);
    }
    // Bit ranges outside the key width / reversed (card: cudaErrorInvalidValue).
    Dev<unsigned char> tmp(1 << 20);
    size_t bytes = 0;
    CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, kin.p, kout.p, (int)n) == cudaSuccess);
    bytes = tmp.n;
    CHECK(cub::DeviceRadixSort::SortKeys(tmp.p, bytes, kin.p, kout.p, (int)n, 0, 33) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    bytes = tmp.n;
    CHECK(cub::DeviceRadixSort::SortKeys(tmp.p, bytes, kin.p, kout.p, (int)n, 9, 8) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    bytes = tmp.n;
    CHECK(cub::DeviceRadixSort::SortKeys(tmp.p, bytes, kin.p, kout.p, (int)n, -1, 8) == cudaErrorInvalidValue);
    (void)cudaGetLastError();
    // begin_bit == end_bit: nothing to compare, the input order is kept.
    {
        std::mt19937 g(3);
        std::vector<uint32_t> keys(3000);
        for (auto &k : keys) k = g();
        sort_case<uint32_t>("uint32", keys, true, false, 5, 5, "empty-range");
        sort_case<uint32_t>("uint32", keys, true, true, 5, 5, "empty-range");
    }
    // 64-bit item count type.
    {
        std::vector<uint32_t> keys(2000);
        std::mt19937 g(4);
        for (auto &k : keys) k = g();
        Dev<uint32_t> ki(2000), ko(2000);
        ki.up(keys);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceRadixSort::SortKeys(t, b, ki.p, ko.p, (long long)2000); });
        std::vector<uint32_t> ref = keys;
        std::sort(ref.begin(), ref.end());
        CHECK(e == cudaSuccess && ko.down() == ref);
    }
    // n = 0 writes nothing.
    {
        Dev<uint32_t> ki(4), ko(4);
        ki.fill(9); ko.fill(7);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceRadixSort::SortKeys(t, b, ki.p, ko.p, 0); });
        CHECK(e == cudaSuccess && ko.down() == std::vector<uint32_t>(4, 7));
    }
}

static void test_sort()
{
    std::vector<size_t> small = {0, 1, 2, 1023, 1024, 1025, 5000, 200000};
    test_sort_type<int32_t>("int32", small);
    test_sort_type<uint32_t>("uint32", small);
    test_sort_type<int64_t>("int64", {0, 1, 1023, 1024, 1025, 5000, 70000});
    test_sort_type<uint64_t>("uint64", {1, 1025, 5000});
    test_sort_type<float>("float", small);
    test_sort_type<double>("double", {0, 1, 1023, 1024, 1025, 5000, 70000});
    // ~1e6 items, one run per direction.
    {
        std::mt19937_64 g(2024);
        std::vector<uint32_t> keys(1000000);
        for (auto &k : keys) k = (uint32_t)(g() % 5000);
        sort_case<uint32_t>("uint32", keys, true, false, 0, 32, "1e6");
        sort_case<uint32_t>("uint32", keys, true, true, 0, 32, "1e6");
        std::vector<int32_t> ks(1000003);
        for (auto &k : ks) k = (int32_t)g();
        sort_case<int32_t>("int32", ks, false, false, 0, 32, "1e6");
        sort_case<int32_t>("int32", ks, true, true, 0, 32, "1e6");
    }
    test_sort_special_floats();
    test_sort_bits<uint32_t>("uint32", {1, 1025, 5000, 70000}, {{0, 8}, {8, 16}, {4, 13}, {24, 32}, {0, 1}, {31, 32}});
    test_sort_bits<int32_t>("int32", {1025, 5000}, {{0, 8}, {3, 20}});
    test_sort_bits<uint64_t>("uint64", {1025, 5000}, {{0, 8}, {32, 40}, {0, 64}, {56, 64}});
    test_sort_protocol_errors();
}

// ---- segmented sorts ------------------------------------------------------------

enum SegAPI { RADIX, SEG, SEG_STABLE };

template <class K>
static cudaError_t seg_call(SegAPI api, bool pairs, bool desc, void *t, size_t &b, const K *ki, K *ko, const int *vi, int *vo,
                            int n, int nseg, const int *bo, const int *eo, int bb, int ee)
{
    if (api == RADIX) {
        if (pairs) return desc ? cub::DeviceSegmentedRadixSort::SortPairsDescending(t, b, ki, ko, vi, vo, n, nseg, bo, eo, bb, ee)
                               : cub::DeviceSegmentedRadixSort::SortPairs(t, b, ki, ko, vi, vo, n, nseg, bo, eo, bb, ee);
        return desc ? cub::DeviceSegmentedRadixSort::SortKeysDescending(t, b, ki, ko, n, nseg, bo, eo, bb, ee)
                    : cub::DeviceSegmentedRadixSort::SortKeys(t, b, ki, ko, n, nseg, bo, eo, bb, ee);
    }
    if (api == SEG) {
        if (pairs) return desc ? cub::DeviceSegmentedSort::SortPairsDescending(t, b, ki, ko, vi, vo, n, nseg, bo, eo)
                               : cub::DeviceSegmentedSort::SortPairs(t, b, ki, ko, vi, vo, n, nseg, bo, eo);
        return desc ? cub::DeviceSegmentedSort::SortKeysDescending(t, b, ki, ko, n, nseg, bo, eo)
                    : cub::DeviceSegmentedSort::SortKeys(t, b, ki, ko, n, nseg, bo, eo);
    }
    if (pairs) return desc ? cub::DeviceSegmentedSort::StableSortPairsDescending(t, b, ki, ko, vi, vo, n, nseg, bo, eo)
                           : cub::DeviceSegmentedSort::StableSortPairs(t, b, ki, ko, vi, vo, n, nseg, bo, eo);
    return desc ? cub::DeviceSegmentedSort::StableSortKeysDescending(t, b, ki, ko, n, nseg, bo, eo)
                : cub::DeviceSegmentedSort::StableSortKeys(t, b, ki, ko, n, nseg, bo, eo);
}

// begins/ends describe the segments (host). gap items are those in no segment.
template <class K>
static void seg_case(const char *tn, SegAPI api, bool pairs, bool desc, const std::vector<K> &keys,
                     const std::vector<int> &begins, const std::vector<int> &ends, bool contiguous,
                     int bb, int ee, const char *tag)
{
    int n = (int)keys.size(), nseg = (int)begins.size();
    std::vector<int> vals(n);
    std::iota(vals.begin(), vals.end(), 0);
    Dev<K> kin(n), kout(n);
    Dev<int> vin(n), vout(n);
    kin.up(keys); vin.up(vals);
    K ksent;
    std::memset(&ksent, 0x6b, sizeof ksent);
    kout.fill(ksent);
    vout.fill(-7);
    // Offsets: contiguous form uses offsets / offsets+1 of one array.
    std::vector<int> off(nseg + 1, 0);
    Dev<int> dbeg(nseg + 1), dend(nseg + 1), doff(nseg + 1);
    const int *pb, *pe;
    if (contiguous) {
        for (int i = 0; i < nseg; ++i) off[i] = begins[i];
        off[nseg] = nseg ? ends[nseg - 1] : 0;
        doff.up(off);
        pb = doff.p; pe = doff.p + 1;
    } else {
        dbeg.up(begins); dend.up(ends);
        pb = dbeg.p; pe = dend.p;
    }
    cudaError_t e = two_phase([&](void *t, size_t &b) { return seg_call<K>(api, pairs, desc, t, b, kin.p, kout.p, vin.p, vout.p, n, nseg, pb, pe, bb, ee); });
    CHECKM(e == cudaSuccess, "%s %s n=%d nseg=%d err %d", tn, tag, n, nseg, (int)e);
    std::vector<K> ko = kout.down();
    std::vector<int> vo = vout.down();
    std::vector<char> covered(n, 0);
    size_t bad = 0, badgap = 0;
    for (int s = 0; s < nseg; ++s) {
        int b = begins[s], en = ends[s];
        if (en <= b) continue;
        std::vector<size_t> idx(en - b);
        std::iota(idx.begin(), idx.end(), (size_t)b);
        ref_sort(idx, keys, desc, bb, ee);
        for (int i = b; i < en; ++i) covered[i] = 1;
        bool stable = api != SEG || !pairs;       // keys-only output is the same either way
        if (api == SEG) stable = false;
        for (int i = b; i < en; ++i) {
            size_t want = idx[i - b];
            if (!same_bits(ko[i], keys[want])) { ++bad; continue; }
            if (pairs) {
                if (api == SEG) {   // not stable by contract: key and value must belong together
                    if (vo[i] < b || vo[i] >= en || !same_bits(keys[vo[i]], ko[i])) ++bad;
                } else if (vo[i] != (int)want) ++bad;
            }
        }
        if (api == SEG && pairs) {  // values of the segment form a permutation
            std::vector<int> v(vo.begin() + b, vo.begin() + en);
            std::sort(v.begin(), v.end());
            for (int i = 0; i < en - b; ++i) if (v[i] != b + i) { ++bad; break; }
        }
        (void)stable;
    }
    for (int i = 0; i < n; ++i)
        if (!covered[i] && (!same_bits(ko[i], ksent) || (pairs && vo[i] != -7))) ++badgap;
    CHECKM(bad == 0, "%s %s api=%d pairs=%d desc=%d n=%d nseg=%d bad=%zu", tn, tag, (int)api, (int)pairs, (int)desc, n, nseg, bad);
    CHECKM(badgap == 0, "%s %s api=%d: %zu gap items modified", tn, tag, (int)api, badgap);
}

template <class K>
static void seg_all(const char *tn, const std::vector<K> &keys, const std::vector<int> &begins, const std::vector<int> &ends,
                    bool contiguous, const char *tag, bool bitrange_too)
{
    const int W = sizeof(K) * 8;
    for (int api = 0; api < 3; ++api)
        for (int pairs = 0; pairs < 2; ++pairs)
            for (int desc = 0; desc < 2; ++desc)
                seg_case<K>(tn, (SegAPI)api, pairs, desc, keys, begins, ends, contiguous, 0, W, tag);
    if (bitrange_too && !is_fp<K>::value)
        for (int pairs = 0; pairs < 2; ++pairs)
            for (int desc = 0; desc < 2; ++desc)
                seg_case<K>(tn, RADIX, pairs, desc, keys, begins, ends, contiguous, 0, 8, "bits0-8");
}

static void test_segmented()
{
    std::mt19937_64 g(99);
    // One segment covering everything, sizes around tile edges.
    for (int n : {1, 5, 1024, 1025, 4000, 9000}) {
        std::vector<uint32_t> keys(n);
        for (auto &k : keys) k = (uint32_t)(g() % 300);
        seg_all<uint32_t>("uint32", keys, {0}, {n}, true, "one", true);
    }
    // Many segments: sizes random 0..50 with empty ones, and a few larger.
    for (int nseg : {7, 1000}) {
        std::vector<int> begins, ends;
        int pos = 0;
        for (int s = 0; s < nseg; ++s) {
            int len = (s % 5 == 0) ? 0 : (int)(g() % 51);
            if (s % 97 == 3) len = 1500 + (int)(g() % 700);
            begins.push_back(pos);
            pos += len;
            ends.push_back(pos);
        }
        std::vector<int32_t> ki(pos);
        for (auto &k : ki) k = (int32_t)(g() % 40) - 20;       // duplicates for stability
        seg_all<int32_t>("int32", ki, begins, ends, true, "many-contig", true);
        std::vector<float> kf(pos);
        for (auto &k : kf) k = gen_key<float>(g, g() % 2);
        seg_all<float>("float", kf, begins, ends, true, "many-contig", false);
        std::vector<double> kd(pos);
        for (auto &k : kd) k = gen_key<double>(g, 0);
        seg_all<double>("double", kd, begins, ends, true, "many-contig", false);
        std::vector<int64_t> kl(pos);
        for (auto &k : kl) k = (int64_t)g();
        seg_all<int64_t>("int64", kl, begins, ends, true, "many-contig", false);
    }
    // Gaps between the segments (separate begin and end arrays): gap
    // outputs must stay as they were.
    {
        std::vector<int> begins, ends;
        int pos = 3;
        for (int s = 0; s < 120; ++s) {
            int len = (s % 4 == 0) ? 0 : (int)(g() % 80);
            int gap = (int)(g() % 6);
            begins.push_back(pos);
            ends.push_back(pos + len);
            pos += len + gap;
        }
        std::vector<uint32_t> keys(pos + 2);
        for (auto &k : keys) k = (uint32_t)(g() % 50);
        seg_all<uint32_t>("uint32", keys, begins, ends, false, "gaps", true);
        // Segments given out of order (offsets need not be monotonic across segments).
        std::vector<int> b2(begins.rbegin(), begins.rend()), e2(ends.rbegin(), ends.rend());
        seg_all<uint32_t>("uint32", keys, b2, e2, false, "gaps-reversed-order", false);
    }
    // No segments, no items.
    {
        Dev<int> off(2), ki(4), ko(4);
        off.fill(0); ki.fill(1); ko.fill(9);
        cudaError_t e = two_phase([&](void *t, size_t &b) { return cub::DeviceSegmentedRadixSort::SortKeys(t, b, ki.p, ko.p, 0, 0, off.p, off.p + 1); });
        CHECK(e == cudaSuccess && ko.down() == std::vector<int>(4, 9));
        e = two_phase([&](void *t, size_t &b) { return cub::DeviceSegmentedSort::SortKeys(t, b, ki.p, ko.p, 0, 0, off.p, off.p + 1); });
        CHECK(e == cudaSuccess && ko.down() == std::vector<int>(4, 9));
    }
    // Protocol.
    {
        int n = 3000;
        Dev<int> ki(n), ko(n), vi(n), vo(n), off(4);
        ki.fill(2); vi.fill(2);
        off.up({0, 1000, 1000, 3000});
        for (int nseg : {0, 3}) {
            protocol("SegRadix.SortPairs", [&](void *t, size_t &b) { return cub::DeviceSegmentedRadixSort::SortPairs(t, b, ki.p, ko.p, vi.p, vo.p, n, nseg, off.p, off.p + 1); }, ko, 0x7eadbeef);
            protocol("SegRadix.SortKeysDescending", [&](void *t, size_t &b) { return cub::DeviceSegmentedRadixSort::SortKeysDescending(t, b, ki.p, ko.p, n, nseg, off.p, off.p + 1); }, ko, 0x7eadbeef);
            protocol("SegSort.SortPairs", [&](void *t, size_t &b) { return cub::DeviceSegmentedSort::SortPairs(t, b, ki.p, ko.p, vi.p, vo.p, n, nseg, off.p, off.p + 1); }, ko, 0x7eadbeef);
            protocol("SegSort.StableSortKeys", [&](void *t, size_t &b) { return cub::DeviceSegmentedSort::StableSortKeys(t, b, ki.p, ko.p, n, nseg, off.p, off.p + 1); }, ko, 0x7eadbeef);
        }
    }
}

int main()
{
    test_reduce();
    test_scan();
    test_sort();
    test_segmented();
    std::printf("cub_device: %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "ok", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
