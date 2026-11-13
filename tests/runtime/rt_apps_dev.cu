// Device-side additions checked against the public CUDA C++ Programming
// Guide:
//   * "Load Functions Using Cache Hints" / "Store Functions Using Cache
//     Hints": __ldcg __ldca __ldcs __ldlu __ldcv and __stcg __stcs __stwb
//     __stwt. The hints do not change the value read or written.
//   * "Atomic Functions": atomicOp_block and atomicOp_system. Same
//     arithmetic as atomicOp, returning the old value. (The _system
//     variants are only built where the target provides them.)
//   * Cooperative Groups: thread_block_tile::meta_group_rank / meta_group_size.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

// Line 154 stores a double literal through an int hint on purpose: it checks
// that the stored value converts to the pointee type.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

namespace cg = cooperative_groups;

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define OK(call) do { ++g_checks; cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); ++g_fail; } } while (0)

// The system-scope atomics exist only on the path where every thread is a
// host thread; the other path stops the build with a message (see the
// compile-case test). The header announces the former with this macro.
#if defined(OCU_SCOPED_SYSTEM_ATOMICS)
#define HAVE_SYSTEM_SCOPE 1
#else
#define HAVE_SYSTEM_SCOPE 0
#endif

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 11); }

// ====================================================================
// cache hints
// ====================================================================
template <typename T>
__global__ void hint_k(const T *in, T *cg_, T *ca, T *cs, T *lu, T *cv, T *scg, T *scs, T *swb, T *swt, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cg_[i] = __ldcg(in + i);
    ca[i] = __ldca(in + i);
    cs[i] = __ldcs(in + i);
    lu[i] = __ldlu(in + i);
    cv[i] = __ldcv(in + i);
    T v = in[i];
    __stcg(scg + i, v);
    __stcs(scs + i, v);
    __stwb(swb + i, v);
    __stwt(swt + i, v);
}

// no NaN payloads: float values are sanitised to keep bit comparisons honest
template <typename T> static void sane(T &) {}
static void sane(float &f) { if (f != f) f = -0.0f; }
static void sane(double &d) { if (d != d) d = -0.0; }
static void sane(float2 &v) { sane(v.x); sane(v.y); }
static void sane(float4 &v) { sane(v.x); sane(v.y); sane(v.z); sane(v.w); }
static void sane(double2 &v) { sane(v.x); sane(v.y); }

template <typename T>
static void test_hint_type(const char *name)
{
    const int n = 301;                               // not a multiple of the block size
    std::vector<T> in(n);
    for (auto &x : in) {
        unsigned char *b = (unsigned char *)&x;
        for (size_t k = 0; k < sizeof(T); ++k) b[k] = (unsigned char)rnd();
        sane(x);
    }
    T *d[10];
    for (auto &p : d) {
        OK(cudaMalloc(&p, n * sizeof(T)));
        OK(cudaMemset(p, 0xee, n * sizeof(T)));
    }
    OK(cudaMemcpy(d[0], in.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    hint_k<T><<<(n + 63) / 64, 64>>>(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], n);
    OK(cudaDeviceSynchronize());
    const char *fn[] = {"", "__ldcg", "__ldca", "__ldcs", "__ldlu", "__ldcv", "__stcg", "__stcs", "__stwb", "__stwt"};
    for (int k = 1; k < 10; ++k) {
        std::vector<T> out(n);
        OK(cudaMemcpy(out.data(), d[k], n * sizeof(T), cudaMemcpyDeviceToHost));
        ++g_checks;
        if (std::memcmp(out.data(), in.data(), n * sizeof(T)) != 0) {
            ++g_fail;
            int i = 0;
            while (std::memcmp(&out[i], &in[i], sizeof(T)) == 0) ++i;
            std::printf("FAIL %s on %s: element %d differs\n", fn[k], name, i);
        }
    }
    for (auto p : d) cudaFree(p);
}

#define HT(T) test_hint_type<T>(#T)
static void test_hints()
{
    HT(char); HT(signed char); HT(unsigned char); HT(short); HT(unsigned short);
    HT(int); HT(unsigned int); HT(long); HT(unsigned long); HT(long long); HT(unsigned long long);
    HT(float); HT(double);
    HT(char2); HT(char4); HT(uchar2); HT(uchar4); HT(short2); HT(short4); HT(ushort2); HT(ushort4);
    HT(int2); HT(int4); HT(uint2); HT(uint4); HT(longlong2); HT(ulonglong2);
    HT(float2); HT(float4); HT(double2);
}

// Special float values pass through bit-exactly (-0, infinities, subnormal, max).
__global__ void hint_special(const float *fin, const double *din, float *fo, double *dout)
{
    int i = threadIdx.x;
    fo[i] = __ldcg(fin + i);
    fo[8 + i] = 0.f;
    __stcs(fo + 8 + i, fin[i]);
    dout[i] = __ldcs(din + i);
    dout[8 + i] = 1.0;
    __stwt(dout + 8 + i, din[i]);
}
static void test_hint_special()
{
    float fin[8] = {-0.0f, 0.0f, 1.0e-45f, 3.4028235e38f, -3.4028235e38f, __builtin_inff(), -__builtin_inff(), 1.17549435e-38f};
    double din[8] = {-0.0, 0.0, 4.9e-324, 1.7976931348623157e308, -1.7976931348623157e308, __builtin_inf(), -__builtin_inf(), 2.2250738585072014e-308};
    float *df, *dfo; double *dd, *ddo;
    OK(cudaMalloc(&df, sizeof fin)); OK(cudaMalloc(&dfo, 16 * sizeof(float)));
    OK(cudaMalloc(&dd, sizeof din)); OK(cudaMalloc(&ddo, 16 * sizeof(double)));
    OK(cudaMemcpy(df, fin, sizeof fin, cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dd, din, sizeof din, cudaMemcpyHostToDevice));
    hint_special<<<1, 8>>>(df, dd, dfo, ddo);
    OK(cudaDeviceSynchronize());
    float fo[16]; double dout[16];
    OK(cudaMemcpy(fo, dfo, sizeof fo, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(dout, ddo, sizeof dout, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(fo, fin, sizeof fin) == 0);
    CHECK(std::memcmp(fo + 8, fin, sizeof fin) == 0);
    CHECK(std::memcmp(dout, din, sizeof din) == 0);
    CHECK(std::memcmp(dout + 8, din, sizeof din) == 0);
    cudaFree(df); cudaFree(dfo); cudaFree(dd); cudaFree(ddo);
}

// The value operand is converted to the pointee type ("__stcs(float *, 0)").
__global__ void hint_convert(float *f, double *d, unsigned long long *u, int *i, short *s, unsigned char *c, long long *ll)
{
    __stcs(f, 0);
    __stcg(f + 1, 2.5);          // double to float
    __stwb(f + 2, 7);
    __stwt(f + 3, 'a');
    __stcs(d, 1);
    __stcg(d + 1, 0.1f);
    __stwb(u, -1);               // int to unsigned long long: 2^64 - 1
    __stwt(u + 1, 5u);
    __stcs(i, 3.9);              // double to int truncates
    __stcg(i + 1, 'z');
    __stwb(s, 65);
    __stwt(c, 200);
    __stcs(ll, -5);
    __stcg(ll + 1, 1u << 31);
}
static void test_hint_convert()
{
    float *f; double *d; unsigned long long *u; int *i; short *s; unsigned char *c; long long *ll;
    OK(cudaMalloc(&f, 4 * sizeof(float))); OK(cudaMalloc(&d, 2 * sizeof(double))); OK(cudaMalloc(&u, 16));
    OK(cudaMalloc(&i, 8)); OK(cudaMalloc(&s, 2)); OK(cudaMalloc(&c, 1)); OK(cudaMalloc(&ll, 16));
    OK(cudaMemset(f, 0xff, 16));
    hint_convert<<<1, 1>>>(f, d, u, i, s, c, ll);
    OK(cudaDeviceSynchronize());
    float hf[4]; double hd[2]; unsigned long long hu[2]; int hi[2]; short hs; unsigned char hc; long long hl[2];
    OK(cudaMemcpy(hf, f, 16, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(hd, d, 16, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(hu, u, 16, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(hi, i, 8, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&hs, s, 2, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(&hc, c, 1, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(hl, ll, 16, cudaMemcpyDeviceToHost));
    CHECK(hf[0] == 0.0f && !__builtin_signbit(hf[0]));
    CHECK(hf[1] == 2.5f && hf[2] == 7.0f && hf[3] == 97.0f);
    CHECK(hd[0] == 1.0 && hd[1] == (double)0.1f);
    CHECK(hu[0] == ~0ull && hu[1] == 5ull);
    CHECK(hi[0] == 3 && hi[1] == 'z');
    CHECK(hs == 65 && hc == 200);
    CHECK(hl[0] == -5 && hl[1] == (1ll << 31));
    cudaFree(f); cudaFree(d); cudaFree(u); cudaFree(i); cudaFree(s); cudaFree(c); cudaFree(ll);
}

// __ldcv re-reads memory another thread wrote between two reads in the same
// thread; a hint load of a const-qualified pointer works too.
__global__ void ldcv_k(int *x, int *r1, int *r2, int *r3)
{
    int t = threadIdx.x;
    int v1 = __ldcv(x + t);
    __syncthreads();
    x[(t + 1) % blockDim.x] = 100 + t;     // every element is written by its left neighbour
    __syncthreads();
    int v2 = __ldcv(x + t);
    __syncthreads();
    if (t == 0) for (int k = 0; k < (int)blockDim.x; ++k) x[k] = -k;
    __syncthreads();
    int v3 = __ldcv((const int *)x + t);
    r1[t] = v1; r2[t] = v2; r3[t] = v3;
}
static void test_ldcv()
{
    const int n = 64;
    int *x, *r[3];
    OK(cudaMalloc(&x, n * 4));
    for (auto &p : r) OK(cudaMalloc(&p, n * 4));
    std::vector<int> init(n);
    for (int i = 0; i < n; ++i) init[i] = 7 * i + 1;
    OK(cudaMemcpy(x, init.data(), n * 4, cudaMemcpyHostToDevice));
    ldcv_k<<<1, n>>>(x, r[0], r[1], r[2]);
    OK(cudaDeviceSynchronize());
    std::vector<int> a(n), b(n), c(n);
    OK(cudaMemcpy(a.data(), r[0], n * 4, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(b.data(), r[1], n * 4, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(c.data(), r[2], n * 4, cudaMemcpyDeviceToHost));
    bool oa = true, ob = true, oc = true;
    for (int t = 0; t < n; ++t) {
        oa = oa && a[t] == init[t];
        ob = ob && b[t] == 100 + (t + n - 1) % n;
        oc = oc && c[t] == -t;
    }
    CHECK(oa); CHECK(ob); CHECK(oc);
    cudaFree(x); for (auto p : r) cudaFree(p);
}

// ====================================================================
// scoped atomics
// ====================================================================
enum Op { ADD, SUB, EXCH, MIN, MAX, AND, OR, XOR, CAS, INC, DEC };
enum Scope { PLAIN, BLOCK, SYSTEM };

template <Scope S, Op O, typename T>
__device__ T apply(T *p, T v, T cmp = T())
{
#define SC(base) (S == PLAIN ? atomic##base(p, v) : S == BLOCK ? atomic##base##_block(p, v) : atomic##base##_system(p, v))
    if constexpr (S == SYSTEM && !HAVE_SYSTEM_SCOPE) { return T(); }
    else if constexpr (O == ADD) { if constexpr (S == PLAIN) return atomicAdd(p, v); else if constexpr (S == BLOCK) return atomicAdd_block(p, v); else return atomicAdd_system(p, v); }
    else if constexpr (O == SUB) { if constexpr (S == PLAIN) return atomicSub(p, v); else if constexpr (S == BLOCK) return atomicSub_block(p, v); else return atomicSub_system(p, v); }
    else if constexpr (O == EXCH) { if constexpr (S == PLAIN) return atomicExch(p, v); else if constexpr (S == BLOCK) return atomicExch_block(p, v); else return atomicExch_system(p, v); }
    else if constexpr (O == MIN) { if constexpr (S == PLAIN) return atomicMin(p, v); else if constexpr (S == BLOCK) return atomicMin_block(p, v); else return atomicMin_system(p, v); }
    else if constexpr (O == MAX) { if constexpr (S == PLAIN) return atomicMax(p, v); else if constexpr (S == BLOCK) return atomicMax_block(p, v); else return atomicMax_system(p, v); }
    else if constexpr (O == AND) { if constexpr (S == PLAIN) return atomicAnd(p, v); else if constexpr (S == BLOCK) return atomicAnd_block(p, v); else return atomicAnd_system(p, v); }
    else if constexpr (O == OR) { if constexpr (S == PLAIN) return atomicOr(p, v); else if constexpr (S == BLOCK) return atomicOr_block(p, v); else return atomicOr_system(p, v); }
    else if constexpr (O == XOR) { if constexpr (S == PLAIN) return atomicXor(p, v); else if constexpr (S == BLOCK) return atomicXor_block(p, v); else return atomicXor_system(p, v); }
    else if constexpr (O == CAS) { if constexpr (S == PLAIN) return atomicCAS(p, cmp, v); else if constexpr (S == BLOCK) return atomicCAS_block(p, cmp, v); else return atomicCAS_system(p, cmp, v); }
    else if constexpr (O == INC) { if constexpr (S == PLAIN) return atomicInc(p, v); else if constexpr (S == BLOCK) return atomicInc_block(p, v); else return atomicInc_system(p, v); }
    else { if constexpr (S == PLAIN) return atomicDec(p, v); else if constexpr (S == BLOCK) return atomicDec_block(p, v); else return atomicDec_system(p, v); }
#undef SC
}

// One thread, a fixed sequence of K operations: exact return values.
template <Scope S, Op O, typename T>
__global__ void seq_k(T *tgt, const T *vals, const T *cmps, T *olds, int k)
{
    for (int i = 0; i < k; ++i) olds[i] = apply<S, O, T>(tgt, vals[i], cmps[i]);
}

// Many threads, block-private targets (block scope is only defined per block)
// or one shared target (device and system scope).
template <Scope S, Op O, typename T>
__global__ void par_k(T *tgt, const T *vals, T *olds, int per_block_target)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    T *t = per_block_target ? tgt + blockIdx.x : tgt;
    olds[g] = apply<S, O, T>(t, vals[g], T());
}

// reference semantics (written from the documented definitions)
template <Op O, typename T>
static T ref_apply(T &mem, T v, T cmp)
{
    T old = mem;
    switch (O) {
    case ADD: mem = (T)(old + v); break;
    case SUB: mem = (T)(old - v); break;
    case EXCH: mem = v; break;
    case MIN: mem = v < old ? v : old; break;
    case MAX: mem = v > old ? v : old; break;
    case CAS: mem = (old == cmp) ? v : old; break;
    default: break;
    }
    return old;
}
template <Op O, typename T>
static T ref_bits(T &mem, T v)
{
    T old = mem;
    if (O == AND) mem = old & v; else if (O == OR) mem = old | v; else mem = old ^ v;
    return old;
}
static unsigned ref_inc(unsigned &mem, unsigned lim) { unsigned o = mem; mem = (o >= lim) ? 0u : o + 1; return o; }
static unsigned ref_dec(unsigned &mem, unsigned lim) { unsigned o = mem; mem = (o == 0 || o > lim) ? lim : o - 1; return o; }

template <typename T> static T mk(long long x) { return (T)x; }

template <typename T> static const char *tname();
template <> const char *tname<int>() { return "int"; }
template <> const char *tname<unsigned>() { return "unsigned"; }
template <> const char *tname<unsigned long long>() { return "unsigned long long"; }
template <> const char *tname<long long>() { return "long long"; }
template <> const char *tname<float>() { return "float"; }
template <> const char *tname<double>() { return "double"; }

static const char *oname(Op o)
{
    static const char *n[] = {"Add", "Sub", "Exch", "Min", "Max", "And", "Or", "Xor", "CAS", "Inc", "Dec"};
    return n[o];
}
static const char *sname(Scope s) { return s == PLAIN ? "" : s == BLOCK ? "_block" : "_system"; }

template <typename T> static std::vector<T> interesting();
template <> std::vector<int> interesting<int>() { return {0, 1, -1, 7, INT_MAX, INT_MIN, 123456, -98765, 2, INT_MAX, 1, 1, -5, 100000, INT_MIN}; }
template <> std::vector<unsigned> interesting<unsigned>() { return {0, 1, 0xffffffffu, 7, 0x80000000u, 0x7fffffffu, 123456, 3, 0xfffffff0u, 5, 5, 1, 99, 0x12345678u, 0}; }
template <> std::vector<unsigned long long> interesting<unsigned long long>() { return {0, 1, ~0ull, 7, 1ull << 63, (1ull << 63) - 1, 1ull << 32, 0xffffffffull, 0x123456789abcdefull, 5, 5, 1, 99, ~0ull - 1, 0}; }
template <> std::vector<long long> interesting<long long>() { return {0, 1, -1, 7, LLONG_MAX, LLONG_MIN, 1ll << 32, -(1ll << 40), 12345678901ll, 5, 5, 1, -99, LLONG_MAX, 0}; }
template <> std::vector<float> interesting<float>() { return {0.f, 1.f, -1.f, 0.5f, 1024.f, -2048.f, 16777216.f, 3.f, 0.25f, 5.f, -0.5f, 7.f, 1.f, 100.f, 2.f}; }
template <> std::vector<double> interesting<double>() { return {0., 1., -1., 0.5, 1024., -2048., 9007199254740992., 3., 0.25, 5., -0.5, 7., 1., 100., 2.}; }

template <typename T> static bool same(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <Scope S, Op O, typename T>
static void test_seq(T init)
{
    if (S == SYSTEM && !HAVE_SYSTEM_SCOPE) return;
    std::vector<T> vals = interesting<T>();
    int k = (int)vals.size();
    std::vector<T> cmps(k);
    // CAS: compare values chosen so that about half hit: cmp tracks the model's current memory every other step
    T mem = init;
    std::vector<T> want_old(k);
    for (int i = 0; i < k; ++i) {
        T v = vals[i];
        T cmp = T();
        if (O == CAS) cmp = (i % 2 == 0) ? mem : (T)(mem + 1);   // hit on even steps, miss on odd steps
        cmps[i] = cmp;
        if (O == AND || O == OR || O == XOR) {
            if constexpr (std::is_integral<T>::value) want_old[i] = ref_bits<O, T>(mem, v);
        } else if (O != INC && O != DEC) {
            want_old[i] = ref_apply<O, T>(mem, v, cmp);
        }
    }
    T *dt, *dv, *dc, *dold;
    OK(cudaMalloc(&dt, sizeof(T))); OK(cudaMalloc(&dv, k * sizeof(T))); OK(cudaMalloc(&dc, k * sizeof(T))); OK(cudaMalloc(&dold, k * sizeof(T)));
    OK(cudaMemcpy(dt, &init, sizeof(T), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dv, vals.data(), k * sizeof(T), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dc, cmps.data(), k * sizeof(T), cudaMemcpyHostToDevice));
    seq_k<S, O, T><<<1, 1>>>(dt, dv, dc, dold, k);
    OK(cudaDeviceSynchronize());
    std::vector<T> got(k);
    T fin;
    OK(cudaMemcpy(got.data(), dold, k * sizeof(T), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&fin, dt, sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = same(fin, mem);
    int bad = -1;
    for (int i = 0; i < k; ++i) if (!same(got[i], want_old[i])) { ok = false; if (bad < 0) bad = i; }
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL atomic%s%s<%s> sequence: step %d returned %.17g want %.17g; final %.17g want %.17g\n", oname(O), sname(S),
                    tname<T>(), bad, bad >= 0 ? (double)got[bad] : 0.0, bad >= 0 ? (double)want_old[bad] : 0.0, (double)fin, (double)mem);
    }
    cudaFree(dt); cudaFree(dv); cudaFree(dc); cudaFree(dold);
}

// Inc/Dec have their own sequence kernel and reference.
template <Scope S, Op O>
static void test_incdec()
{
    if (S == SYSTEM && !HAVE_SYSTEM_SCOPE) return;
    struct C { unsigned init; unsigned lim; };
    C cs[] = {{0, 3}, {2, 3}, {3, 3}, {4, 3}, {0, 0}, {7, 0}, {0xffffffffu, 0xffffffffu}, {0xfffffffeu, 0xffffffffu},
              {5, 1000}, {1000, 1000}, {1001, 1000}, {0x80000000u, 0x7fffffffu}};
    for (auto c : cs) {
        std::vector<unsigned> lims(12, c.lim), want(12), cmps(12, 0);
        unsigned mem = c.init;
        for (int i = 0; i < 12; ++i) want[i] = (O == INC) ? ref_inc(mem, c.lim) : ref_dec(mem, c.lim);
        unsigned *dt, *dv, *dc, *dold;
        OK(cudaMalloc(&dt, 4)); OK(cudaMalloc(&dv, 48)); OK(cudaMalloc(&dc, 48)); OK(cudaMalloc(&dold, 48));
        OK(cudaMemcpy(dt, &c.init, 4, cudaMemcpyHostToDevice));
        OK(cudaMemcpy(dv, lims.data(), 48, cudaMemcpyHostToDevice));
        OK(cudaMemcpy(dc, cmps.data(), 48, cudaMemcpyHostToDevice));
        seq_k<S, O, unsigned><<<1, 1>>>(dt, dv, dc, dold, 12);
        OK(cudaDeviceSynchronize());
        std::vector<unsigned> got(12);
        unsigned fin;
        OK(cudaMemcpy(got.data(), dold, 48, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&fin, dt, 4, cudaMemcpyDeviceToHost));
        ++g_checks;
        if (got != want || fin != mem) {
            ++g_fail;
            std::printf("FAIL atomic%s%s sequence init=%u limit=%u: got", oname(O), sname(S), c.init, c.lim);
            for (auto x : got) std::printf(" %u", x);
            std::printf(" final %u; want", fin);
            for (auto x : want) std::printf(" %u", x);
            std::printf(" final %u\n", mem);
        }
        cudaFree(dt); cudaFree(dv); cudaFree(dc); cudaFree(dold);
    }
}

// Parallel: 2 blocks x 64 threads. Final values of the commutative operations
// are checked against the model; Exch and CAS by their invariants.
template <Scope S, Op O, typename T>
static void test_par(T init)
{
    if (S == SYSTEM && !HAVE_SYSTEM_SCOPE) return;
    const int B = 2, N = 64, G = B * N;
    int per_block = (S == BLOCK);
    int ntg = per_block ? B : 1;
    std::vector<T> vals(G);
    for (int i = 0; i < G; ++i) {
        if (std::is_floating_point<T>::value) vals[i] = (T)((i % 17) - 8) * (T)0.5;
        else if (O == AND) vals[i] = (T)~(unsigned long long)(1ull << (i % (int)(8 * sizeof(T) - 1)));
        else if (O == OR || O == XOR) vals[i] = (T)(1ull << (i % (int)(8 * sizeof(T) - 1)));
        else if (O == MIN || O == MAX) vals[i] = (T)(((i * 2654435761u) % 20011) - 10000) * (std::is_signed<T>::value ? 1 : 1);
        else vals[i] = (T)(i * 3 + 1);
        if (O == CAS || O == EXCH) vals[i] = (T)(i + 1);
    }
    std::vector<T> tinit(ntg, init);
    if (O == CAS) for (auto &x : tinit) x = (T)0;
    T *dt, *dv, *dold;
    OK(cudaMalloc(&dt, ntg * sizeof(T))); OK(cudaMalloc(&dv, G * sizeof(T))); OK(cudaMalloc(&dold, G * sizeof(T)));
    OK(cudaMemcpy(dt, tinit.data(), ntg * sizeof(T), cudaMemcpyHostToDevice));
    OK(cudaMemcpy(dv, vals.data(), G * sizeof(T), cudaMemcpyHostToDevice));
    par_k<S, O, T><<<B, N>>>(dt, dv, dold, per_block);
    OK(cudaDeviceSynchronize());
    std::vector<T> fin(ntg), olds(G);
    OK(cudaMemcpy(fin.data(), dt, ntg * sizeof(T), cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(olds.data(), dold, G * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int t = 0; t < ntg; ++t) {
        int lo = per_block ? t * N : 0, hi = per_block ? (t + 1) * N : G;
        T mem = tinit[t];
        if (O == CAS) {
            // exactly one thread sees the initial value and swaps; everyone else sees the winner's value
            int winners = 0;
            T win = fin[t];
            for (int i = lo; i < hi; ++i) { if (olds[i] == (T)0) ++winners; else ok = ok && olds[i] == win; }
            ok = ok && winners == 1 && win != (T)0;
        } else if (O == EXCH) {
            // olds + final are a permutation of init + vals
            std::vector<T> a, b;
            a.push_back(mem);
            for (int i = lo; i < hi; ++i) a.push_back(vals[i]);
            for (int i = lo; i < hi; ++i) b.push_back(olds[i]);
            b.push_back(fin[t]);
            for (size_t i = 0; i < a.size(); ++i)
                for (size_t j = i + 1; j < a.size(); ++j) { if (a[j] < a[i]) { T x = a[i]; a[i] = a[j]; a[j] = x; } if (b[j] < b[i]) { T x = b[i]; b[i] = b[j]; b[j] = x; } }
            ok = ok && a == b;
        } else {
            for (int i = lo; i < hi; ++i) {
                if constexpr (std::is_integral<T>::value) {
                    if (O == AND || O == OR || O == XOR) ref_bits<O, T>(mem, vals[i]); else ref_apply<O, T>(mem, vals[i], T());
                } else if (O != AND && O != OR && O != XOR) ref_apply<O, T>(mem, vals[i], T());
            }
            ok = ok && same(mem, fin[t]);
            // every returned old value must be a possible intermediate: for Add of positive ints: old in [init, final]
        }
    }
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("FAIL atomic%s%s<%s> parallel: final %.17g\n", oname(O), sname(S), tname<T>(), (double)fin[0]); }
    cudaFree(dt); cudaFree(dv); cudaFree(dold);
}

// atomicInc counted by many threads: limit 9 -> value cycles, each of 0..9 returned equally often
template <Scope S>
__global__ void inc_many(unsigned *t, unsigned *olds, unsigned lim)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    olds[g] = apply<S, INC, unsigned>(t, lim);
}
template <Scope S>
__global__ void dec_many(unsigned *t, unsigned *olds, unsigned lim)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    olds[g] = apply<S, DEC, unsigned>(t, lim);
}
template <Scope S>
static void test_incdec_par()
{
    if (S == SYSTEM && !HAVE_SYSTEM_SCOPE) return;
    for (int dec = 0; dec < 2; ++dec) {
        const int N = 64;                             // one block: block scope is defined per block
        unsigned *t, *o;
        OK(cudaMalloc(&t, 4)); OK(cudaMalloc(&o, N * 4));
        unsigned init = dec ? 9 : 0;
        OK(cudaMemcpy(t, &init, 4, cudaMemcpyHostToDevice));
        if (dec) dec_many<S><<<1, N>>>(t, o, 9u); else inc_many<S><<<1, N>>>(t, o, 9u);
        OK(cudaDeviceSynchronize());
        std::vector<unsigned> olds(N);
        unsigned fin;
        OK(cudaMemcpy(olds.data(), o, N * 4, cudaMemcpyDeviceToHost));
        OK(cudaMemcpy(&fin, t, 4, cudaMemcpyDeviceToHost));
        int cnt[10] = {};
        bool ok = true;
        for (auto v : olds) { if (v > 9) ok = false; else ++cnt[v]; }
        // 64 steps over a cycle of 10: values 0..3 appear 7 times, 4..9 six times (inc starting at 0)
        // dec starting at 9: 9,8,...,0,9,...: 9..6 appear 7 times, 5..0 six times
        for (int v = 0; v < 10; ++v) {
            int want = dec ? (v >= 6 ? 7 : 6) : (v <= 3 ? 7 : 6);
            ok = ok && cnt[v] == want;
        }
        unsigned wantfin = dec ? (9u - 64 % 10 + 10) % 10 : 64 % 10;
        ok = ok && fin == wantfin;
        ++g_checks;
        if (!ok) { ++g_fail; std::printf("FAIL atomic%s%s parallel wrap: final %u want %u\n", dec ? "Dec" : "Inc", sname(S), fin, wantfin); }
        cudaFree(t); cudaFree(o);
    }
}

template <Scope S>
static void test_scope()
{
#define SEQ(op, T, init) test_seq<S, op, T>(init)
#define PAR(op, T, init) test_par<S, op, T>(init)
    // documented type sets
    SEQ(ADD, int, 3); SEQ(ADD, unsigned, 3u); SEQ(ADD, unsigned long long, 3ull); SEQ(ADD, float, 1.5f); SEQ(ADD, double, 1.5);
    SEQ(SUB, int, 3); SEQ(SUB, unsigned, 3u);
    SEQ(EXCH, int, 3); SEQ(EXCH, unsigned, 3u); SEQ(EXCH, unsigned long long, 3ull); SEQ(EXCH, float, 1.5f);
    SEQ(MIN, int, 3); SEQ(MIN, unsigned, 3u); SEQ(MIN, unsigned long long, 3ull); SEQ(MIN, long long, 3ll);
    SEQ(MAX, int, 3); SEQ(MAX, unsigned, 3u); SEQ(MAX, unsigned long long, 3ull); SEQ(MAX, long long, 3ll);
    SEQ(AND, int, -1); SEQ(AND, unsigned, 0xffffffffu); SEQ(AND, unsigned long long, ~0ull);
    SEQ(OR, int, 0); SEQ(OR, unsigned, 0u); SEQ(OR, unsigned long long, 0ull);
    SEQ(XOR, int, 0x55); SEQ(XOR, unsigned, 0x55u); SEQ(XOR, unsigned long long, 0x55ull);
    SEQ(CAS, int, 3); SEQ(CAS, unsigned, 3u); SEQ(CAS, unsigned long long, 3ull);
    test_incdec<S, INC>();
    test_incdec<S, DEC>();
    test_incdec_par<S>();

    PAR(ADD, int, 5); PAR(ADD, unsigned, 5u); PAR(ADD, unsigned long long, 5ull); PAR(ADD, float, 1.5f); PAR(ADD, double, 1.5);
    PAR(SUB, int, 5); PAR(SUB, unsigned, 5u);
    PAR(EXCH, int, 9); PAR(EXCH, unsigned, 9u); PAR(EXCH, unsigned long long, 9ull); PAR(EXCH, float, 9.f);
    PAR(MIN, int, 4000); PAR(MIN, unsigned, 4000u); PAR(MIN, unsigned long long, 4000ull); PAR(MIN, long long, 4000ll);
    PAR(MAX, int, -4000); PAR(MAX, unsigned, 7u); PAR(MAX, unsigned long long, 7ull); PAR(MAX, long long, -4000ll);
    PAR(AND, int, -1); PAR(AND, unsigned, 0xffffffffu); PAR(AND, unsigned long long, ~0ull);
    PAR(OR, int, 0); PAR(OR, unsigned, 0u); PAR(OR, unsigned long long, 0ull);
    PAR(XOR, int, 0); PAR(XOR, unsigned, 0u); PAR(XOR, unsigned long long, 0ull);
    PAR(CAS, int, 0); PAR(CAS, unsigned, 0u); PAR(CAS, unsigned long long, 0ull);
#undef SEQ
#undef PAR
}

// the value operand converts to the pointee type, like the plain atomics
__global__ void scoped_convert(int *i, unsigned *u, float *f, unsigned long long *ull)
{
    atomicAdd_block(i, 2u);             // unsigned to int
    atomicAdd_block(f, 1);              // int to float
    atomicExch_block(u, -1);            // int to unsigned: 0xffffffff
    atomicMax_block(ull, 5);
    atomicOr_block(ull, 1);
#if HAVE_SYSTEM_SCOPE
    atomicAdd_system(i, 'a');
    atomicAdd_system(f, 0.5);
    atomicCAS_system(u, 0xffffffffu, 3);
    atomicSub_system(i, 1u);
#endif
}
static void test_scoped_convert()
{
    int *i; unsigned *u; float *f; unsigned long long *ull;
    OK(cudaMalloc(&i, 4)); OK(cudaMalloc(&u, 4)); OK(cudaMalloc(&f, 4)); OK(cudaMalloc(&ull, 8));
    OK(cudaMemset(i, 0, 4)); OK(cudaMemset(u, 0, 4)); OK(cudaMemset(f, 0, 4)); OK(cudaMemset(ull, 0, 8));
    scoped_convert<<<1, 1>>>(i, u, f, ull);
    OK(cudaDeviceSynchronize());
    int hi; unsigned hu; float hf; unsigned long long hl;
    OK(cudaMemcpy(&hi, i, 4, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(&hu, u, 4, cudaMemcpyDeviceToHost));
    OK(cudaMemcpy(&hf, f, 4, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(&hl, ull, 8, cudaMemcpyDeviceToHost));
#if HAVE_SYSTEM_SCOPE
    CHECK(hi == 2 + 'a' - 1);
    CHECK(hf == 1.5f);
    CHECK(hu == 3u);
#else
    CHECK(hi == 2);
    CHECK(hf == 1.0f);
    CHECK(hu == 0xffffffffu);
#endif
    CHECK(hl == 5ull);
    cudaFree(i); cudaFree(u); cudaFree(f); cudaFree(ull);
}

// block-scope atomics on shared memory (their natural use)
__global__ void shared_block(int *out, unsigned *mx)
{
    __shared__ int cnt;
    __shared__ unsigned m;
    if (threadIdx.x == 0) { cnt = 0; m = 0; }
    __syncthreads();
    atomicAdd_block(&cnt, 1);
    atomicMax_block(&m, threadIdx.x * 3u);
    __syncthreads();
    if (threadIdx.x == 0) { out[blockIdx.x] = cnt; mx[blockIdx.x] = m; }
}
static void test_shared_block()
{
    int *o; unsigned *m;
    OK(cudaMalloc(&o, 12)); OK(cudaMalloc(&m, 12));
    shared_block<<<3, 96>>>(o, m);
    OK(cudaDeviceSynchronize());
    int ho[3]; unsigned hm[3];
    OK(cudaMemcpy(ho, o, 12, cudaMemcpyDeviceToHost)); OK(cudaMemcpy(hm, m, 12, cudaMemcpyDeviceToHost));
    for (int b = 0; b < 3; ++b) { CHECK(ho[b] == 96); CHECK(hm[b] == 95u * 3); }
    cudaFree(o); cudaFree(m);
}

// ====================================================================
// Cooperative Groups: meta groups
// ====================================================================
struct Meta { unsigned size[8]; unsigned rank[8]; };

template <unsigned N>
__device__ void rec(Meta &m, int slot, const cg::thread_block_tile<N, cg::thread_block> &t)
{
    m.size[slot] = t.meta_group_size();
    m.rank[slot] = t.meta_group_rank();
}

__global__ void meta_k(Meta *out)
{
    cg::thread_block blk = cg::this_thread_block();
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    Meta m;
    for (int i = 0; i < 8; ++i) m.size[i] = m.rank[i] = 0xdeadbeefu;
    auto t32 = cg::tiled_partition<32>(blk);
    auto t16 = cg::tiled_partition<16>(blk);
    auto t8 = cg::tiled_partition<8>(blk);
    auto t4 = cg::tiled_partition<4>(blk);
    auto t2 = cg::tiled_partition<2>(blk);
    auto t1 = cg::tiled_partition<1>(blk);
    m.size[0] = t32.meta_group_size(); m.rank[0] = t32.meta_group_rank();
    m.size[1] = t16.meta_group_size(); m.rank[1] = t16.meta_group_rank();
    m.size[2] = t8.meta_group_size(); m.rank[2] = t8.meta_group_rank();
    m.size[3] = t4.meta_group_size(); m.rank[3] = t4.meta_group_rank();
    m.size[4] = t2.meta_group_size(); m.rank[4] = t2.meta_group_rank();
    m.size[5] = t1.meta_group_size(); m.rank[5] = t1.meta_group_rank();
    out[lin] = m;
}
static void test_meta_block(dim3 bd)
{
    int n = bd.x * bd.y * bd.z;
    Meta *d;
    OK(cudaMalloc(&d, n * sizeof(Meta)));
    meta_k<<<1, bd>>>(d);
    OK(cudaDeviceSynchronize());
    std::vector<Meta> h(n);
    OK(cudaMemcpy(h.data(), d, n * sizeof(Meta), cudaMemcpyDeviceToHost));
    unsigned sz[6] = {32, 16, 8, 4, 2, 1};
    bool ok = true;
    int bad_t = -1, bad_s = -1;
    for (int t = 0; t < n && ok; ++t)
        for (int s = 0; s < 6; ++s) {
            if (n % sz[s]) continue;
            if (h[t].size[s] != (unsigned)n / sz[s] || h[t].rank[s] != (unsigned)t / sz[s]) { ok = false; bad_t = t; bad_s = s; break; }
        }
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL meta group, block %ux%ux%u: thread %d tile<%u> size=%u rank=%u want %u %u\n", bd.x, bd.y, bd.z, bad_t, sz[bad_s],
                    h[bad_t].size[bad_s], h[bad_t].rank[bad_s], n / sz[bad_s], bad_t / sz[bad_s]);
    }
    cudaFree(d);
}

// nested: tile of tile; and the parent-less type
__global__ void nested_k(Meta *out)
{
    cg::thread_block blk = cg::this_thread_block();
    unsigned lin = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    Meta m;
    for (int i = 0; i < 8; ++i) m.size[i] = m.rank[i] = 0xdeadbeefu;
    auto t32 = cg::tiled_partition<32>(blk);
    auto t8 = cg::tiled_partition<8>(t32);
    auto t4 = cg::tiled_partition<4>(t8);
    auto t2 = cg::tiled_partition<2>(t4);
    auto t16 = cg::tiled_partition<16>(t32);
    m.size[0] = t8.meta_group_size(); m.rank[0] = t8.meta_group_rank();
    m.size[1] = t4.meta_group_size(); m.rank[1] = t4.meta_group_rank();
    m.size[2] = t2.meta_group_size(); m.rank[2] = t2.meta_group_rank();
    m.size[3] = t16.meta_group_size(); m.rank[3] = t16.meta_group_rank();
    // converted to the parent-less tile type: the documented way of storing a tile
    cg::thread_block_tile<8> p8 = t8;
    cg::thread_block_tile<16> p16 = cg::tiled_partition<16>(blk);
    cg::thread_block_tile<32> p32 = cg::tiled_partition<32>(blk);
    cg::thread_block_tile<4> p4 = t4;
    m.size[4] = p8.meta_group_size(); m.rank[4] = p8.meta_group_rank();
    m.size[5] = p16.meta_group_size(); m.rank[5] = p16.meta_group_rank();
    m.size[6] = p32.meta_group_size(); m.rank[6] = p32.meta_group_rank();
    m.size[7] = p4.meta_group_size(); m.rank[7] = p4.meta_group_rank();
    out[lin] = m;
}
static void test_meta_nested(dim3 bd)
{
    int n = bd.x * bd.y * bd.z;
    Meta *d;
    OK(cudaMalloc(&d, n * sizeof(Meta)));
    nested_k<<<1, bd>>>(d);
    OK(cudaDeviceSynchronize());
    std::vector<Meta> h(n);
    OK(cudaMemcpy(h.data(), d, n * sizeof(Meta), cudaMemcpyDeviceToHost));
    bool ok = true;
    int bad = -1, slot = -1;
    for (int t = 0; t < n && ok; ++t) {
        unsigned in32 = t % 32;
        struct { unsigned size, rank; } w[8] = {
            {4, in32 / 8}, {2, (in32 % 8) / 4}, {2, (in32 % 4) / 2}, {2, in32 / 16},
            {4, in32 / 8}, {(unsigned)n / 16, (unsigned)t / 16}, {(unsigned)n / 32, (unsigned)t / 32}, {2, (in32 % 8) / 4}};
        for (int s = 0; s < 8; ++s)
            if (h[t].size[s] != w[s].size || h[t].rank[s] != w[s].rank) { ok = false; bad = t; slot = s; break; }
    }
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::printf("FAIL nested meta group, block %ux%ux%u: thread %d slot %d got size=%u rank=%u\n", bd.x, bd.y, bd.z, bad, slot,
                    h[bad].size[slot], h[bad].rank[slot]);
    }
    cudaFree(d);
}

// cg::reduce works on a tile that has a parent, and the generic function
// takes the tile by template parameter.
template <typename Tile>
__device__ int sum_of(Tile t, int v) { return cg::reduce(t, v, cg::plus<int>()); }
__global__ void reduce_k(int *out)
{
    cg::thread_block blk = cg::this_thread_block();
    auto t8 = cg::tiled_partition<8>(blk);
    auto t16 = cg::tiled_partition<16>(blk);
    int v = threadIdx.x + 1;
    int s8 = sum_of(t8, v);
    int s16 = sum_of(t16, v);
    out[threadIdx.x * 3 + 0] = s8;
    out[threadIdx.x * 3 + 1] = s16;
    out[threadIdx.x * 3 + 2] = (int)(t8.meta_group_rank() * 1000 + t8.meta_group_size());
}
static void test_reduce_tiles()
{
    const int n = 64;
    int *d;
    OK(cudaMalloc(&d, n * 3 * 4));
    reduce_k<<<1, n>>>(d);
    OK(cudaDeviceSynchronize());
    std::vector<int> h(n * 3);
    OK(cudaMemcpy(h.data(), d, n * 3 * 4, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int t = 0; t < n; ++t) {
        int b8 = t / 8 * 8, b16 = t / 16 * 16, s8 = 0, s16 = 0;
        for (int k = 0; k < 8; ++k) s8 += b8 + k + 1;
        for (int k = 0; k < 16; ++k) s16 += b16 + k + 1;
        ok = ok && h[t * 3] == s8 && h[t * 3 + 1] == s16 && h[t * 3 + 2] == t / 8 * 1000 + 8;
    }
    CHECK(ok);
    cudaFree(d);
}

int main()
{
    test_hints();
    test_hint_special();
    test_hint_convert();
    test_ldcv();

    test_scope<PLAIN>();            // control: the unscoped atomics under the same harness
    test_scope<BLOCK>();
    test_scope<SYSTEM>();           // no-op where the target has no system scope
    test_scoped_convert();
    test_shared_block();
    std::printf("note: system-scope atomics %s\n", HAVE_SYSTEM_SCOPE ? "tested" : "not provided on this target");

    dim3 shapes[] = {dim3(32), dim3(64), dim3(128), dim3(256), dim3(1024), dim3(8, 8), dim3(4, 4, 4), dim3(16, 4, 2),
                     dim3(96), dim3(32, 3), dim3(2, 2, 8), dim3(64, 2, 1)};
    for (dim3 s : shapes) test_meta_block(s);
    dim3 nshapes[] = {dim3(32), dim3(64), dim3(128), dim3(512), dim3(8, 8), dim3(4, 4, 4)};
    for (dim3 s : nshapes) test_meta_nested(s);
    test_reduce_tiles();
#ifdef OCU_MUTANT
    CHECK(false);
#endif
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
