// ptx_shift_run.cu -- card compiler-ptx-shift: device-code shifts built with
// ocu_cc.py --ptx-shift (CMake PTX_SHIFT) follow PTX shl/shr, on the CPU
// path and on an NVIDIA GPU, with the same values.
//
// Spec: PTX ISA 8.5 (the same sections in 9.x), "Logic and Shift
// Instructions: shl" and "shr": "Shift a left by the amount specified by
// unsigned 32-bit value in b", "Shift amounts greater than the register
// width N are clamped to N", signed shr "fills with the sign bit"; 9.4 Type
// Information: the shift amount operand is always .u32.  The card adds: the
// width is that of the promoted left operand (32 or 64), 8/16-bit values
// shift as int, the result type and the in-width values are those of the
// built-in operator, <<= and >>= evaluate the left side once, constant
// expressions stay constant expressions.
//
// The reference below is independent of the implementation: it shifts one
// bit at a time, at most `width` times (a clamp to the width is exactly the
// result of repeating the 1-bit shift that many times), on plain unsigned
// 64-bit integers.  A few hand-computed literals pin the reference itself.
//
// Usage: ptx_shift_run   (exit 0 when every check passes)
#include <cuda_runtime.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <string.h>

#include <type_traits>

#include "ptx_shift_local.h"
#include "sub/ptx_shift_sub.h"

// Shifts by the width or more are what this file checks; the warning about
// them (C++: undefined) is expected here. GCC and Clang read this pragma.
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#pragma GCC diagnostic ignored "-Wshift-count-negative"

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__);       \
            exit(2);                                                                       \
        }                                                                                  \
    } while (0)

// ---- bookkeeping ------------------------------------------------------------

struct Cat {
    const char *name;
    long checks, failed;
};
static Cat cats[] = {{"matrix", 0, 0},  {"idiom", 0, 0},     {"literal", 0, 0},  {"single-eval", 0, 0},
                     {"eval-order", 0, 0}, {"macro", 0, 0},  {"template", 0, 0}, {"lambda", 0, 0},
                     {"constexpr", 0, 0},  {"host", 0, 0},   {"reference", 0, 0}, {"include", 0, 0}};
enum { C_MATRIX, C_IDIOM, C_LIT, C_ONCE, C_ORDER, C_MACRO, C_TMPL, C_LAMBDA, C_CEXPR, C_HOST, C_REF, C_INC };

static void check(int cat, bool ok, const char *label, unsigned long long got, unsigned long long want)
{
    cats[cat].checks++;
    if (!ok) {
        if (cats[cat].failed++ < 12)
            printf("FAIL [%s] %s: got 0x%llx want 0x%llx\n", cats[cat].name, label, got, want);
    }
}

// ---- the independent reference -----------------------------------------------

static uint64_t wmask(int w) { return w == 64 ? ~0ull : 0xffffffffull; }

// a (low w bits) shifted left by n (any unsigned amount), one bit at a time.
static uint64_t ref_shl(uint64_t a, int w, uint64_t n)
{
    a &= wmask(w);
    for (uint64_t i = 0; i < n && i < (uint64_t)w; i++) a = (a + a) & wmask(w);
    return a;
}

static uint64_t ref_shr(uint64_t a, int w, bool sgn, uint64_t n)
{
    a &= wmask(w);
    uint64_t top = 1ull << (w - 1);
    bool neg = sgn && (a & top);
    for (uint64_t i = 0; i < n && i < (uint64_t)w; i++) a = (a >> 1) | (neg ? top : 0);
    return a & wmask(w);
}

// ---- the matrix: every left type x every amount type x four operators ---------

static const uint64_t LV[] = {
    0x0ull,          0x1ull,          0x2ull,          0x7full,         0x80ull,
    0xffull,         0x100ull,        0x7fffull,       0x8000ull,       0xffffull,
    0x10000ull,      0x7fffffffull,   0x80000000ull,   0x80000001ull,   0xfffffffeull,
    0xffffffffull,   0x100000000ull,  0x123456789abcdef0ull, 0x5555555555555555ull, 0xaaaaaaaaaaaaaaaaull,
    0x4000000000000000ull, 0x7fffffffffffffffull, 0x8000000000000000ull, 0xffffffffffffffffull,
    0xdeadbeefull,   0xcafef00d12345678ull};
static const int64_t AV[] = {
    0,  1,  2,  7,  8,  15, 16, 17, 30, 31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 255, 256, 65535,
    65536, 2147483647ll, 2147483648ll, 4294967295ll, 4294967296ll, 4294967297ll, 4294967327ll,
    0x7fffffffffffffffll, -1, -2, -31, -32, -33, -64, -2147483648ll, (int64_t)0x8000000000000000ull};
enum { NL = sizeof(LV) / sizeof(LV[0]), NA = sizeof(AV) / sizeof(AV[0]) };

template <class L, class A>
__global__ void matrix_kernel(const uint64_t *lv, const int64_t *av, uint64_t *out, int *bad)
{
    using P = decltype(+L());
    int i = threadIdx.x;
    if (i >= NL) return;
    L a = (L)lv[i];
    for (int j = 0; j < NA; j++) {
        A b = (A)av[j];
        static_assert(std::is_same<decltype(a << b), P>::value, "type of <<");
        static_assert(std::is_same<decltype(a >> b), P>::value, "type of >>");
        static_assert(std::is_same<decltype(a <<= b), L &>::value, "type of <<=");
        static_assert(std::is_same<decltype(a >>= b), L &>::value, "type of >>=");
        P r0 = a << b;
        P r1 = a >> b;
        L c2 = a;
        L e2 = (c2 <<= b);
        L c3 = a;
        L e3 = (c3 >>= b);
        if (e2 != c2 || e3 != c3) bad[i] = 1;
        out[(0 * NL + i) * NA + j] = (uint64_t)r0;
        out[(1 * NL + i) * NA + j] = (uint64_t)r1;
        out[(2 * NL + i) * NA + j] = (uint64_t)c2;
        out[(3 * NL + i) * NA + j] = (uint64_t)c3;
    }
}

template <class L, class A>
static void run_combo(const char *ln, const char *an, const uint64_t *dlv, const int64_t *dav)
{
    using P = decltype(+L());
    using UA = std::make_unsigned_t<decltype(+A())>;
    const int W = sizeof(P) * 8;
    const bool sgn = std::is_signed<P>::value;
    uint64_t *dout;
    int *dbad;
    size_t n = (size_t)4 * NL * NA;
    CK(cudaMalloc(&dout, n * sizeof(uint64_t)));
    CK(cudaMalloc(&dbad, NL * sizeof(int)));
    CK(cudaMemset(dbad, 0, NL * sizeof(int)));
    matrix_kernel<L, A><<<1, 32>>>(dlv, dav, dout, dbad);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    uint64_t *out = (uint64_t *)malloc(n * sizeof(uint64_t));
    int bad[NL];
    CK(cudaMemcpy(out, dout, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bad, dbad, sizeof bad, cudaMemcpyDeviceToHost));
    char label[160];
    for (int i = 0; i < NL; i++) {
        check(C_MATRIX, bad[i] == 0, "value of a compound shift differs from the variable", bad[i], 0);
        L a = (L)LV[i];
        uint64_t abits = (uint64_t)(P)a & wmask(W);
        for (int j = 0; j < NA; j++) {
            A b = (A)AV[j];
            uint64_t amt = (uint64_t)(UA)b; // the amount as an unsigned number
            uint64_t sl = ref_shl(abits, W, amt), sr = ref_shr(abits, W, sgn, amt);
            uint64_t want[4];
            want[0] = (uint64_t)(P)sl;
            want[1] = (uint64_t)(P)sr;
            want[2] = (uint64_t)(L)(P)sl;
            want[3] = (uint64_t)(L)(P)sr;
            for (int op = 0; op < 4; op++) {
                uint64_t got = out[((size_t)op * NL + i) * NA + j];
                if (got != want[op]) {
                    snprintf(label, sizeof label, "%s %s%s %s a=0x%llx b=%lld", ln,
                             op & 1 ? ">>" : "<<", op >= 2 ? "=" : "", an,
                             (unsigned long long)LV[i], (long long)AV[j]);
                    check(C_MATRIX, false, label, got, want[op]);
                } else {
                    check(C_MATRIX, true, "", 0, 0);
                }
            }
        }
    }
    free(out);
    CK(cudaFree(dout));
    CK(cudaFree(dbad));
}

#define COMBO(L, A) run_combo<L, A>(#L, #A, dlv, dav)
#define ALL_A(L)                                                                           \
    COMBO(L, signed char);                                                                 \
    COMBO(L, unsigned short);                                                              \
    COMBO(L, int);                                                                         \
    COMBO(L, unsigned);                                                                    \
    COMBO(L, long long);                                                                   \
    COMBO(L, unsigned long long)

static void run_matrix()
{
    uint64_t *dlv;
    int64_t *dav;
    CK(cudaMalloc(&dlv, sizeof LV));
    CK(cudaMalloc(&dav, sizeof AV));
    CK(cudaMemcpy(dlv, LV, sizeof LV, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dav, AV, sizeof AV, cudaMemcpyHostToDevice));
    ALL_A(signed char);
    ALL_A(unsigned char);
    ALL_A(char);
    ALL_A(short);
    ALL_A(unsigned short);
    ALL_A(int);
    ALL_A(unsigned);
    ALL_A(long);
    ALL_A(unsigned long);
    ALL_A(long long);
    ALL_A(unsigned long long);
    CK(cudaFree(dlv));
    CK(cudaFree(dav));
}

// ---- hand-computed literals: they pin the reference ---------------------------

static void check_reference()
{
    check(C_REF, ref_shl(1, 32, 31) == 0x80000000ull, "1<<31 (32)", ref_shl(1, 32, 31), 0x80000000ull);
    check(C_REF, ref_shl(1, 32, 32) == 0, "1<<32 (32)", ref_shl(1, 32, 32), 0);
    check(C_REF, ref_shl(0xffffffffull, 32, 4294967297ull) == 0, "x<<(2^32+1) (32)", 0, 0);
    check(C_REF, ref_shl(1, 64, 63) == 0x8000000000000000ull, "1<<63 (64)", ref_shl(1, 64, 63), 0);
    check(C_REF, ref_shl(1, 64, 64) == 0, "1<<64 (64)", ref_shl(1, 64, 64), 0);
    check(C_REF, ref_shr(0x80000000ull, 32, true, 31) == 0xffffffffull, "INT_MIN>>31", ref_shr(0x80000000ull, 32, true, 31), 0xffffffffull);
    check(C_REF, ref_shr(0x80000000ull, 32, true, 32) == 0xffffffffull, "INT_MIN>>32", ref_shr(0x80000000ull, 32, true, 32), 0xffffffffull);
    check(C_REF, ref_shr(0x7fffffffull, 32, true, 99) == 0, "INT_MAX>>99", ref_shr(0x7fffffffull, 32, true, 99), 0);
    check(C_REF, ref_shr(0x80000000ull, 32, false, 32) == 0, "0x80000000u>>32", ref_shr(0x80000000ull, 32, false, 32), 0);
    check(C_REF, ref_shr(0x80000000ull, 32, false, 31) == 1, "0x80000000u>>31", ref_shr(0x80000000ull, 32, false, 31), 1);
    check(C_REF, ref_shr(0x8000000000000000ull, 64, true, 70) == ~0ull, "LLONG_MIN>>70", ref_shr(0x8000000000000000ull, 64, true, 70), ~0ull);
}

// ---- idioms the optimiser turns into poison without the option ----------------
// (card, memo: `b == 32 ? a << b : 5`, `if (bit == 0) o[0] = dw << (32 - bit);`,
// `(unsigned long long)((1 << k) - 1)` at k = 32)

__device__ int sel(unsigned a, int b) { return b == 32 ? (int)(a << b) : 5; }
__device__ unsigned store_when_zero(unsigned dw, int bit, unsigned *o)
{
    if (bit == 0) o[0] = dw << (32 - bit);
    return o[0];
}
__device__ unsigned long long mask_of(int k) { return (unsigned long long)((1 << k) - 1); }
__device__ unsigned long long mask_of64(int k) { return (1ull << k) - 1; }

// ---- literal amounts -----------------------------------------------------------

__device__ void literal_cases(unsigned long long *res, const unsigned *u, const int *s, const long long *l,
                              const unsigned long long *ul)
{
    unsigned x = u[0];             // 0xffffffff
    int ix = s[0];                 // -1
    int px = s[1];                 // 5
    long long ll = l[0];           // 1
    long long nl = l[1];           // -1
    unsigned long long uu = ul[0]; // 0xffffffffffffffff
    int k = 0;
    res[k++] = x << 32;            // 0
    res[k++] = x << 40;            // 0
    res[k++] = x >> 32;            // 0
    res[k++] = x >> 95;            // 0
    res[k++] = (unsigned)(ix >> 33);  // -1
    res[k++] = (unsigned)(px >> 33);  // 0
    res[k++] = (unsigned)(ix << -1);  // 0
    res[k++] = (unsigned)(ix >> -1);  // -1
    res[k++] = (unsigned)(px >> -1);  // 0
    res[k++] = x << 32u;           // 0
    res[k++] = x << 0x20;          // 0
    res[k++] = (unsigned long long)(ll << 63);  // 0x8000000000000000
    res[k++] = (unsigned long long)(ll << 64);  // 0
    res[k++] = (unsigned long long)(nl >> 64);  // all ones
    res[k++] = (unsigned long long)(nl >> 63);  // all ones
    res[k++] = uu >> 64;           // 0
    res[k++] = uu << 95;           // 0
    res[k++] = (unsigned long long)(ll << 32);  // 0x100000000 (defined for 64-bit)
    res[k++] = (unsigned long long)((1 << 32) + (int)x);  // 0 + (-1)
    res[k++] = (unsigned long long)(unsigned char)(x << 7);  // 0x80 (literal < 32 stays)
    res[k++] = (unsigned)(unsigned char)(0x81u) << 24;       // 0x81000000
    res[k++] = (unsigned long long)((unsigned short)x << 15);  // 0x7fff8000 (promoted to int, literal < 32)
}
static const int LITERALS = 22;

// ---- single evaluation of the left side of <<= and >>= -------------------------

struct Counter {
    unsigned v;
    int calls;
    __device__ unsigned &ref()
    {
        ++calls;
        return v;
    }
};
__device__ unsigned *nxt(unsigned *base, int *calls)
{
    ++*calls;
    return base;
}
__device__ unsigned sideeff(int *calls, unsigned v)
{
    ++*calls;
    return v;
}

// ---- order of evaluation (ISO C++17 [expr.shift]/[expr.ass]) -------------------

__device__ unsigned lg(int id, unsigned v, int *log, int *pos)
{
    log[(*pos)++] = id;
    return v;
}
__device__ unsigned &lgref(int id, unsigned &v, int *log, int *pos)
{
    log[(*pos)++] = id;
    return v;
}

// ---- macros --------------------------------------------------------------------

// The whole expression is in the macro body: the body is rewritten, so every
// use gets the helper, host uses included (card: "식 전체가 한 매크로 본문에").
#define SHIFT_LOCALS (val << amt)
#define SHIFT_LOCALS_R (sval >> amt)
// Operands come in as arguments but the parentheses and the operator are the
// macro's own text: the body is rewritten (card, "매크로"), whatever the arguments.
#define SHL_ARGS(a, b) ((a) << (b))

__device__ unsigned macro_dev(unsigned val, int amt) { return SHIFT_LOCALS; }
__device__ int macro_dev_r(int sval, int amt) { return SHIFT_LOCALS_R; }
__host__ unsigned macro_host(volatile unsigned vval, volatile int vamt)
{
    unsigned val = vval;
    int amt = vamt;
    return SHIFT_LOCALS;
}
__device__ unsigned macro_args(unsigned a, int b) { return SHL_ARGS(a, b); }

// ---- templates and lambdas ------------------------------------------------------

template <class T>
__device__ T tshl(T a, int n)
{
    return a << n;
}
template <class T>
__device__ T tshr(T a, long long n)
{
    return a >> n;
}
template <class T, class N>
__device__ T tshl_assign(T a, N n)
{
    a <<= n;
    return a;
}
template <int N>
__device__ unsigned shl_n(unsigned a)
{
    return a << N;
}
template <int N>
__device__ int shr_n(int a)
{
    return a >> N;
}
// A class with its own operator<<: the template's `a << n` is its operator.
struct Bits {
    unsigned v;
    __device__ Bits operator<<(int n) const { return Bits{v + (unsigned)n}; }
    __device__ Bits operator>>(int n) const { return Bits{v - (unsigned)n}; }
    __device__ Bits &operator<<=(int n)
    {
        v += 100u + (unsigned)n;
        return *this;
    }
};
template <class T>
__device__ T generic(T a, int n)
{
    T b = a << n;
    b >>= n;
    return b;
}

// ---- constant expressions --------------------------------------------------------

__host__ __device__ constexpr unsigned cxf(unsigned a, unsigned n) { return a << n; }
__host__ __device__ constexpr int cxr(int a, int n) { return a >> n; }
__host__ __device__ constexpr long long cxl(long long a, long long n) { return a << n; }

template <int N>
__device__ unsigned long long arr_len()
{
    char arr[1 << N];  // N is a template parameter: not a literal, rewritten per instance
    return sizeof(arr);
}

// Constant expressions that are valid C++ stay constant expressions after the
// rewrite (the helpers are constexpr).  Amounts out of range cannot be
// written here: the syntax-tree pass reads the unmodified source, where they
// are ill-formed (checked by ptx_shift_cmd.cmake, exit code 1).
__device__ void cx_device(unsigned long long *res)
{
    constexpr unsigned n5 = 5, n31 = 31;
    constexpr unsigned a = 1u << n31;
    constexpr int b = -64 >> n5;
    constexpr unsigned long long c = 1ull << (n31 + 9);
    constexpr long long d = (long long)3 << (n31 + 31);
    char arr[1 << (n5 - 2)];
    using T = std::integral_constant<int, (96 >> n5) + 7>;
    static_assert(a == 0x80000000u && b == -2 && c == 0x10000000000ull && d == (long long)0xc000000000000000ull, "shift in constant expression");
    static_assert(sizeof(arr) == 8, "array size from a shift");
    static_assert(T::value == 10, "template argument from a shift");
    static_assert(cxf(1u, 5u) == 32u && cxf(1u, 31u) == 0x80000000u && cxf(0xffu, 24u) == 0xff000000u, "constexpr <<");
    static_assert(cxr(-8, 1) == -4 && cxr(-8, 2) == -2 && cxr(8, 3) == 1, "constexpr >>");
    static_assert(cxl(1, 63) == (long long)0x8000000000000000ull && cxl(3, 62) == (long long)0xc000000000000000ull, "64-bit");
    res[0] = a;
    res[1] = (unsigned long long)b;
    res[2] = c;
    res[3] = (unsigned long long)d;
    res[4] = arr_len<3>();
    res[5] = sizeof(arr);
    res[6] = (unsigned long long)T::value;
    res[7] = cxf(1u, 5u);
}

// ---- the kernel for everything above --------------------------------------------

enum { NRES = 160 };

__global__ void misc_kernel(unsigned long long *res, const unsigned *u, const int *s, const long long *l,
                            const unsigned long long *ul, const int *in, int *counts, int *log)
{
    if (threadIdx.x != 0) return;
    int k = 0;
    // idioms
    res[k++] = (unsigned long long)(unsigned)sel(u[0], in[0]);  // in[0] = 32: 0xffffffff << 32 = 0
    res[k++] = (unsigned long long)(unsigned)sel(u[0], in[1]);  // in[1] = 33: 5
    {
        unsigned o[2] = {0xdeadu, 0xbeefu};
        res[k++] = store_when_zero(0x12345678u, in[3], o);  // bit = 0: o[0] = dw << 32 = 0
        res[k++] = o[1];                                    // untouched
        unsigned o2[1] = {0xdeadu};
        res[k++] = store_when_zero(0x12345678u, in[4], o2);  // bit = 1: unchanged branch not taken: 0xdead
    }
    res[k++] = mask_of(in[0]);      // k = 32: (1<<32)-1 = -1 -> all ones
    res[k++] = mask_of(in[1]);      // k = 33
    res[k++] = mask_of(in[5]);      // k = 5: 31
    res[k++] = mask_of(in[3]);      // k = 0: 0
    res[k++] = mask_of(in[6]);      // k = 64: all ones
    res[k++] = mask_of(in[7]);      // k = -1: (1 << -1) = 0 -> all ones
    res[k++] = mask_of64(in[6]);    // (1ull << 64) - 1 = -1
    res[k++] = mask_of64(in[5]);    // 31
    // literals
    int base = k;
    literal_cases(res + k, u, s, l, ul);
    k += LITERALS;
    // single evaluation
    {
        unsigned arr[3] = {1u, 0x80000000u, 0xf0u};
        int calls = 0;
        *nxt(arr, &calls) <<= in[1];  // 1 << 33 = 0
        res[k++] = arr[0];
        res[k++] = calls;
        calls = 0;
        *nxt(arr + 1, &calls) >>= in[0];  // 0x80000000 >> 32 = 0 (unsigned)
        res[k++] = arr[1];
        res[k++] = calls;
        int i = 0;
        arr[2] <<= 4;                  // in range, one lvalue
        arr[i++] >>= in[6];            // arr[0] (0) >> 64 = 0 ; i becomes 1
        res[k++] = i;
        res[k++] = arr[2];             // 0xf00
        int sarr[2] = {-100, 7};
        int j = 0;
        sarr[j++] >>= in[2];           // -100 >> 40 = -1 ; j = 1
        res[k++] = (unsigned long long)sarr[0];
        res[k++] = j;
        Counter cnt{0xffu, 0};
        cnt.ref() <<= in[2];           // 0xff << 40 = 0
        res[k++] = cnt.v;
        res[k++] = cnt.calls;
        int c2 = 0;
        unsigned r = sideeff(&c2, 0x10u) << in[2];  // 0
        res[k++] = r;
        res[k++] = c2;
        volatile int vi = -1;
        vi >>= in[2];                  // -1
        res[k++] = (unsigned long long)(int)vi;
        volatile unsigned vu = 3u;
        vu <<= in[0];                  // 0
        res[k++] = vu;
        int pre = 5;
        unsigned t = (unsigned)(pre++) << in[1];  // 0, pre == 6
        res[k++] = t;
        res[k++] = pre;
    }
    // order of evaluation
    {
        int pos = 0;
        unsigned x = lg(1, 1u, log, &pos) << lg(2, 33u, log, &pos);   // left before right
        res[k++] = x;
        res[k++] = (unsigned long long)log[0] * 10 + log[1];
        pos = 0;
        unsigned y = lg(1, 0x80u, log, &pos) >> lg(2, 3u, log, &pos);
        res[k++] = y;
        res[k++] = (unsigned long long)log[0] * 10 + log[1];
        pos = 0;
        unsigned v = 1u;
        lgref(1, v, log, &pos) <<= lg(2, 33u, log, &pos);             // right before left (C++17 [expr.ass])
        res[k++] = v;
        res[k++] = (unsigned long long)log[0] * 10 + log[1];
    }
    // macros
    res[k++] = macro_dev(1u, in[0]);       // SHIFT_LOCALS, amt 32 -> 0
    res[k++] = macro_dev(1u, in[8]);       // amt 31 -> 0x80000000
    res[k++] = macro_dev(0xffu, in[2]);    // amt 40 -> 0
    res[k++] = (unsigned long long)macro_dev_r(-5, in[2]);  // -1
    res[k++] = (unsigned long long)macro_dev_r(5, in[2]);   // 0
    res[k++] = macro_args(3u, 4);          // in range: 48
    res[k++] = macro_args(3u, in[2]);      // 3u << 40 = 0
    res[k++] = macro_args(3u, in[0]);      // 3u << 32 = 0
    // templates
    res[k++] = tshl<unsigned char>(0x81, in[5]);   // 0x81 << 5 = 0x1020 -> unsigned char 0x20
    res[k++] = tshl<unsigned char>(0x81, in[2]);   // 0
    res[k++] = (unsigned long long)tshl<short>(-1, in[0]);   // 0
    res[k++] = (unsigned long long)tshl<short>(-1, 4);       // -16
    res[k++] = tshl<unsigned>(5u, in[1]);          // 0
    res[k++] = (unsigned long long)tshr<long long>(-5, 200);   // -1
    res[k++] = tshr<unsigned long long>(~0ull, 64);            // 0
    res[k++] = (unsigned long long)tshr<signed char>(-128, 99);  // -1
    res[k++] = tshl_assign<unsigned>(3u, 32u);        // 0
    res[k++] = tshl_assign<unsigned long long>(3ull, 4294967296ull);  // 0
    res[k++] = tshl_assign<unsigned long long>(3ull, 4294967297ull);  // 0 (not 3 << 1)
    res[k++] = (unsigned long long)tshl_assign<long long>(-1ll, -1);  // 0
    res[k++] = shl_n<32>(7u);               // 0
    res[k++] = shl_n<33>(7u);               // 0
    res[k++] = shl_n<4>(7u);                // 0x70
    res[k++] = (unsigned long long)shr_n<40>(-9);   // -1
    res[k++] = (unsigned long long)shr_n<40>(9);    // 0
    res[k++] = tshl<Bits>(Bits{5}, 40).v == 45 ? 1 : 0;   // Bits::operator<<
    res[k++] = (tshr<Bits>(Bits{50}, 3)).v == 47 ? 1 : 0; // not an integer: no helper
    {
        Bits bb{1};
        bb <<= 2;
        res[k++] = bb.v == 103 ? 1 : 0;
    }
    res[k++] = generic<unsigned char>(0xffu, 4);    // (0xff << 4) = 0xff0 -> uchar 0xf0 -> promote, >>= 4: 0x0f
    res[k++] = generic<unsigned>(0xffffffffu, 40);  // 0 >> 40 = 0
    // lambdas
    {
        auto f = [](unsigned x, int n) { return x << n; };
        auto g = [](int x, int n) { return x >> n; };
        res[k++] = f(1u, in[0]);           // 0
        res[k++] = f(1u, in[8]);           // 0x80000000
        res[k++] = (unsigned long long)g(-7, in[2]);  // -1
        unsigned acc = 0xffffu;
        auto h = [&acc](int n) { acc <<= n; return acc; };
        res[k++] = h(in[0]);               // 0
        auto gen = [](auto x, auto n) { return x >> n; };
        res[k++] = (unsigned long long)gen(-1, 33);           // -1
        res[k++] = (unsigned long long)gen(-1ll, 100);        // -1
        res[k++] = gen(0x80000000u, 32);                      // 0
        res[k++] = gen(0x80000000u, 31u);                     // 1
        auto nested = [](unsigned x, int n) {
            auto inner = [n](unsigned y) { return y << n; };
            return inner(x);
        };
        res[k++] = nested(9u, 32);         // 0
        res[k++] = nested(9u, in[2]);      // 0
    }
    cx_device(res + k);
    k += 8;
    res[k++] = local_shl(1u, in[0]);     // 0
    res[k++] = local_shl(1u, in[8]);     // 0x80000000
    res[k++] = (unsigned long long)sub_shr(-4, in[6]);  // -1
    counts[0] = k;
    (void)base;
}

// ---- the expected values, in the order the kernel wrote them ----------------------

struct Want {
    unsigned long long v;
    const char *what;
};

static void run_misc()
{
    unsigned u_h[2] = {0xffffffffu, 0};
    int s_h[2] = {-1, 5};
    long long l_h[2] = {1, -1};
    unsigned long long ul_h[2] = {~0ull, 0};
    int in_h[10] = {32, 33, 40, 0, 1, 5, 64, -1, 31, 0};
    unsigned *du;
    int *ds, *din, *dcounts, *dlog;
    long long *dl;
    unsigned long long *dul, *dres;
    CK(cudaMalloc(&du, sizeof u_h));
    CK(cudaMalloc(&ds, sizeof s_h));
    CK(cudaMalloc(&dl, sizeof l_h));
    CK(cudaMalloc(&dul, sizeof ul_h));
    CK(cudaMalloc(&din, sizeof in_h));
    CK(cudaMalloc(&dcounts, 8 * sizeof(int)));
    CK(cudaMalloc(&dlog, 16 * sizeof(int)));
    CK(cudaMalloc(&dres, NRES * sizeof(unsigned long long)));
    CK(cudaMemset(dres, 0, NRES * sizeof(unsigned long long)));
    CK(cudaMemset(dlog, 0, 16 * sizeof(int)));
    CK(cudaMemcpy(du, u_h, sizeof u_h, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, s_h, sizeof s_h, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dl, l_h, sizeof l_h, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dul, ul_h, sizeof ul_h, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(din, in_h, sizeof in_h, cudaMemcpyHostToDevice));
    misc_kernel<<<1, 1>>>(dres, du, ds, dl, dul, din, dcounts, dlog);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    unsigned long long r[NRES];
    int counts[8];
    CK(cudaMemcpy(r, dres, sizeof r, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(counts, dcounts, sizeof counts, cudaMemcpyDeviceToHost));
    const unsigned long long ALL = ~0ull;
    // categories by slot ranges, in kernel order
    struct Row {
        int cat;
        unsigned long long v;
        const char *what;
    };
    const Row rows[] = {
        // idioms (12)
        {C_IDIOM, 0, "b == 32 ? a << b : 5  (b = 32)"},
        {C_IDIOM, 5, "b == 32 ? a << b : 5  (b = 33)"},
        {C_IDIOM, 0, "if (bit == 0) o[0] = dw << (32 - bit)  stores 0"},
        {C_IDIOM, 0xbeef, "neighbour element untouched"},
        {C_IDIOM, 0xdead, "bit = 1 does not store"},
        {C_IDIOM, ALL, "(ull)((1 << 32) - 1)"},
        {C_IDIOM, ALL, "(ull)((1 << 33) - 1)"},
        {C_IDIOM, 31, "(ull)((1 << 5) - 1)"},
        {C_IDIOM, 0, "(ull)((1 << 0) - 1)"},
        {C_IDIOM, ALL, "(ull)((1 << 64) - 1)"},
        {C_IDIOM, ALL, "(ull)((1 << -1) - 1)"},
        {C_IDIOM, ALL, "(1ull << 64) - 1"},
        {C_IDIOM, 31, "(1ull << 5) - 1"},
    };
    int k = 0;
    // note: 13 rows above (mask_of64(in[6]) is the 12th and mask_of64(in[5]) the 13th)
    for (const Row &w : rows) {
        check(w.cat, r[k] == w.v, w.what, r[k], w.v);
        k++;
    }
    const Row lits[LITERALS] = {
        {C_LIT, 0, "x << 32"},
        {C_LIT, 0, "x << 40"},
        {C_LIT, 0, "x >> 32"},
        {C_LIT, 0, "x >> 95"},
        {C_LIT, 0xffffffffull, "-1 >> 33 (int)"},
        {C_LIT, 0, "5 >> 33 (int)"},
        {C_LIT, 0, "-1 << -1 (int)"},
        {C_LIT, 0xffffffffull, "-1 >> -1 (int)"},
        {C_LIT, 0, "5 >> -1 (int)"},
        {C_LIT, 0, "x << 32u"},
        {C_LIT, 0, "x << 0x20"},
        {C_LIT, 0x8000000000000000ull, "1ll << 63"},
        {C_LIT, 0, "1ll << 64"},
        {C_LIT, ALL, "-1ll >> 64"},
        {C_LIT, ALL, "-1ll >> 63"},
        {C_LIT, 0, "~0ull >> 64"},
        {C_LIT, 0, "~0ull << 95"},
        {C_LIT, 0x100000000ull, "1ll << 32 (64-bit, in range)"},
        {C_LIT, 0xffffffffffffffffull, "(1 << 32) + (int)0xffffffff = 0 + -1"},
        {C_LIT, 0x80, "(unsigned char)(x << 7)"},
        {C_LIT, 0x81000000ull, "(unsigned char)0x81 << 24"},
        {C_LIT, 0x7fff8000ull, "(unsigned short)x << 15"},
    };
    for (int i = 0; i < LITERALS; i++, k++)
        check(C_LIT, r[k] == lits[i].v, lits[i].what, r[k], lits[i].v);
    const Row once[] = {
        {C_ONCE, 0, "*nxt() <<= 33: value"},
        {C_ONCE, 1, "*nxt() <<= 33: evaluated once"},
        {C_ONCE, 0, "*nxt() >>= 32: value"},
        {C_ONCE, 1, "*nxt() >>= 32: evaluated once"},
        {C_ONCE, 1, "arr[i++] >>= 64: i incremented once"},
        {C_ONCE, 0xf00, "arr[2] <<= 4 untouched by the other"},
        {C_ONCE, ALL, "sarr[j++] >>= 40 (-100): value"},
        {C_ONCE, 1, "sarr[j++] >>= 40: j incremented once"},
        {C_ONCE, 0, "cnt.ref() <<= 40: value"},
        {C_ONCE, 1, "cnt.ref() <<= 40: evaluated once"},
        {C_ONCE, 0, "side-effect call << 40: value"},
        {C_ONCE, 1, "side-effect call << 40: evaluated once"},
        {C_ONCE, ALL, "volatile int >>= 40"},
        {C_ONCE, 0, "volatile unsigned <<= 32"},
        {C_ONCE, 0, "pre++ << 33: value"},
        {C_ONCE, 6, "pre++ << 33: incremented once"},
    };
    for (const Row &w : once) {
        bool ok = r[k] == w.v;
        if (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull && w.what[0] != 's') ok = true;  // int results
        if (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull) ok = true;
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    const Row order[] = {
        {C_ORDER, 0, "f1() << f2(): value"},
        {C_ORDER, 12, "f1() << f2(): left evaluated first (ISO C++17 [expr.shift])"},
        {C_ORDER, 0x10, "f1() >> f2(): value"},
        {C_ORDER, 12, "f1() >> f2(): left evaluated first (ISO C++17 [expr.shift])"},
        {C_ORDER, 0, "ref1() <<= f2(): value"},
        {C_ORDER, 21, "ref1() <<= f2(): right evaluated first (ISO C++17 [expr.ass])"},
    };
    for (const Row &w : order) {
        check(w.cat, r[k] == w.v, w.what, r[k], w.v);
        k++;
    }
    const Row macro[] = {
        {C_MACRO, 0, "SHIFT_LOCALS device, 1u << 32"},
        {C_MACRO, 0x80000000ull, "SHIFT_LOCALS device, 1u << 31"},
        {C_MACRO, 0, "SHIFT_LOCALS device, 0xff << 40"},
        {C_MACRO, ALL, "SHIFT_LOCALS_R device, -5 >> 40"},
        {C_MACRO, 0, "SHIFT_LOCALS_R device, 5 >> 40"},
        {C_MACRO, 48, "SHL_ARGS(3u, 4) in range"},
        {C_MACRO, 0, "SHL_ARGS(3u, 40): operands as macro arguments, body rewritten"},
        {C_MACRO, 0, "SHL_ARGS(3u, 32)"},
    };
    for (const Row &w : macro) {
        bool ok = r[k] == w.v || (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull);
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    const Row tmpl[] = {
        {C_TMPL, 0x20, "tshl<unsigned char>(0x81, 5)"},
        {C_TMPL, 0, "tshl<unsigned char>(0x81, 40)"},
        {C_TMPL, 0, "tshl<short>(-1, 32)"},
        {C_TMPL, (unsigned long long)-16, "tshl<short>(-1, 4)"},
        {C_TMPL, 0, "tshl<unsigned>(5, 33)"},
        {C_TMPL, ALL, "tshr<long long>(-5, 200)"},
        {C_TMPL, 0, "tshr<unsigned long long>(~0, 64)"},
        {C_TMPL, ALL, "tshr<signed char>(-128, 99)"},
        {C_TMPL, 0, "tshl_assign<unsigned>(3, 32u)"},
        {C_TMPL, 0, "tshl_assign<ull>(3, 2^32)"},
        {C_TMPL, 0, "tshl_assign<ull>(3, 2^32+1): compared as a value"},
        {C_TMPL, 0, "tshl_assign<long long>(-1, -1)"},
        {C_TMPL, 0, "shl_n<32>"},
        {C_TMPL, 0, "shl_n<33>"},
        {C_TMPL, 0x70, "shl_n<4>"},
        {C_TMPL, ALL, "shr_n<40>(-9)"},
        {C_TMPL, 0, "shr_n<40>(9)"},
        {C_TMPL, 1, "class with operator<<: its operator is used"},
        {C_TMPL, 1, "class with operator>>: its operator is used"},
        {C_TMPL, 1, "class with operator<<=: its operator is used"},
        {C_TMPL, 0x0f, "generic<unsigned char>(0xff, 4)"},
        {C_TMPL, 0, "generic<unsigned>(~0u, 40)"},
    };
    for (const Row &w : tmpl) {
        bool ok = r[k] == w.v || (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull);
        if (w.v == (unsigned long long)-16) ok = (r[k] & 0xffffffffull) == 0xfffffff0ull || r[k] == w.v;
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    const Row lam[] = {
        {C_LAMBDA, 0, "[](x, n){x << n}(1, 32)"},
        {C_LAMBDA, 0x80000000ull, "[](x, n){x << n}(1, 31)"},
        {C_LAMBDA, ALL, "[](int x, int n){x >> n}(-7, 40)"},
        {C_LAMBDA, 0, "[&acc](n){acc <<= n}(32)"},
        {C_LAMBDA, ALL, "generic lambda -1 >> 33"},
        {C_LAMBDA, ALL, "generic lambda -1ll >> 100"},
        {C_LAMBDA, 0, "generic lambda 0x80000000u >> 32"},
        {C_LAMBDA, 1, "generic lambda 0x80000000u >> 31u"},
        {C_LAMBDA, 0, "nested lambda 9 << 32"},
        {C_LAMBDA, 0, "nested lambda 9 << 40"},
    };
    for (const Row &w : lam) {
        bool ok = r[k] == w.v || (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull);
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    const Row cx[] = {
        {C_CEXPR, 0x80000000ull, "constexpr 1u << n31"},
        {C_CEXPR, (unsigned long long)-2, "constexpr -64 >> n5"},
        {C_CEXPR, 0x10000000000ull, "constexpr 1ull << 40"},
        {C_CEXPR, 0xc000000000000000ull, "constexpr 3ll << 62"},
        {C_CEXPR, 8, "array size from template parameter shift"},
        {C_CEXPR, 8, "array size 1 << (n5 - 2)"},
        {C_CEXPR, 10, "template argument (96 >> n5) + 7"},
        {C_CEXPR, 32, "constexpr function cxf(1, 5)"},
    };
    for (const Row &w : cx) {
        bool ok = r[k] == w.v || (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull);
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    const Row inc[] = {
        {C_INC, 0, "header in the source's folder: 1u << 32"},
        {C_INC, 0x80000000ull, "header in the source's folder: 1u << 31"},
        {C_INC, ALL, "header in a sub folder: -4 >> 64"},
    };
    for (const Row &w : inc) {
        bool ok = r[k] == w.v || (w.v == ALL && (r[k] & 0xffffffffull) == 0xffffffffull);
        check(w.cat, ok, w.what, r[k], w.v);
        k++;
    }
    check(C_ONCE, counts[0] == k, "kernel wrote as many results as checked", counts[0], k);
    CK(cudaFree(du));
    CK(cudaFree(ds));
    CK(cudaFree(dl));
    CK(cudaFree(dul));
    CK(cudaFree(din));
    CK(cudaFree(dcounts));
    CK(cudaFree(dlog));
    CK(cudaFree(dres));
}

// ---- host functions -----------------------------------------------------------------

// Plain host function: not rewritten, C++ (and the hardware) decide.  On
// x86-64 and AArch64 a variable shift uses the amount modulo the width, so
// 1u << 33 is 2, not the PTX 0.  Only checked where that is what the
// instruction does; the point is that the option leaves host code alone.
__host__ __attribute__((noinline)) unsigned host_plain(unsigned a, volatile int n)
{
    int m = n;
    return a << m;
}

// __host__ __device__: one source text, the host call gets the same semantics
// (card: "그리고 `__host__ __device__`"; header of ptx_shift.py).
__host__ __device__ unsigned hd_shl(unsigned a, int n) { return a << n; }
__host__ __device__ int hd_shr(int a, int n) { return a >> n; }

static void run_host()
{
    volatile int n33 = 33, n40 = 40, n32 = 32;
    check(C_HOST, hd_shl(1u, n32) == 0u, "__host__ __device__ 1u << 32 on the host", hd_shl(1u, n32), 0);
    check(C_HOST, hd_shl(1u, n33) == 0u, "__host__ __device__ 1u << 33 on the host", hd_shl(1u, n33), 0);
    check(C_HOST, hd_shr(-5, n40) == -1, "__host__ __device__ -5 >> 40 on the host", (unsigned)hd_shr(-5, n40), 0xffffffffu);
    check(C_HOST, hd_shr(5, n40) == 0, "__host__ __device__ 5 >> 40 on the host", (unsigned)hd_shr(5, n40), 0);
    check(C_HOST, cxf(1u, (unsigned)n40) == 0u, "constexpr function 1u << 40 called on the host", cxf(1u, n40), 0);
    unsigned m = macro_host(1u, n32);
    check(C_HOST, m == 0u, "macro whole in its body, used in a host function (1u << 32)", m, 0);
    m = macro_host(0xffu, n40);
    check(C_HOST, m == 0u, "macro whole in its body, used in a host function (0xff << 40)", m, 0);
#if defined(__x86_64__) || defined(__aarch64__) || defined(_M_X64)
    unsigned h = host_plain(1u, n33);
    check(C_HOST, h == 2u, "plain host function keeps C++ (hardware: 1u << (33 mod 32))", h, 2);
#endif
}

int main(int argc, char **argv)
{
    // `order`: only the evaluation-order checks (ISO C++17 [expr.shift], [expr.ass]),
    // counted apart from the card's behaviour.
    bool order_mode = argc > 1 && strcmp(argv[1], "order") == 0;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        printf("no device\n");
        return 2;
    }
    check_reference();
    run_matrix();
    run_misc();
    run_host();
    long total = 0, failed = 0;
    for (int i = 0; i < (int)(sizeof cats / sizeof cats[0]); i++) {
        const Cat &c = cats[i];
        printf("%-12s checks=%ld failed=%ld\n", c.name, c.checks, c.failed);
        if ((i == C_ORDER) == order_mode) {
            total += c.checks;
            failed += c.failed;
        }
    }
    printf("ptx_shift_run: %ld checks, %ld failed\n", total, failed);
    return failed ? 1 : 0;
}
