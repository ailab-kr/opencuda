// The BLAS library API (cublas_v2.h) as an ordinary CUDA program uses it,
// graded against a host reference in higher precision.
//
// Reference: products of the (exactly representable) inputs summed in long
// double; also the sum of |products| ("scale"). A result passes when
//   |got - ref| <= 2 (k + 1) u_c scale + u_o |ref| + tiny_o
// where u_c is the unit roundoff the compute type permits (2^-24 for 32F,
// 2^-11 for 16F, TF32 and FAST_16F, 2^-8 for FAST_16BF, 2^-53 for 64F),
// u_o that of the output type and tiny_o its smallest subnormal. That is
// the classical bound for any summation order, plus one rounding of the
// result. NaN must stay NaN, infinities must match.
//
// Also checked: padding between columns of C is never written; alpha == 0
// never reads A and B (they hold inf and NaN); beta == 0 never reads C
// (it holds NaN); batches, strides, pointer arrays, transposes, host and
// device pointer modes, streams, status codes. geam (both scalars, in place),
// getrfBatched (P A = L U with |L| <= 1, info for singular matrices) and the
// Set/Get Matrix/Vector copies (strides, padding untouched, async).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorName(e_)); ++g_fail; std::exit(1); } } while (0)
#define BLAS_IS(call, want) do { ++g_checks; cublasStatus_t s_ = (call); if (s_ != (want)) { std::printf("FAIL %s:%d: %s -> %s, want %s\n", __FILE__, __LINE__, #call, cublasGetStatusName(s_), cublasGetStatusName(want)); ++g_fail; } } while (0)
#define BLAS_OK(call) BLAS_IS(call, CUBLAS_STATUS_SUCCESS)

typedef long double ld;

// ---- element types on the host -------------------------------------------

enum class ty { f32, f64, f16, bf16 };

static size_t esize(ty t) { return t == ty::f64 ? 8 : t == ty::f32 ? 4 : 2; }
static cudaDataType_t dtype(ty t)
{
    return t == ty::f64 ? CUDA_R_64F : t == ty::f32 ? CUDA_R_32F : t == ty::f16 ? CUDA_R_16F : CUDA_R_16BF;
}
static double unit_roundoff(ty t)
{
    return t == ty::f64 ? 0x1p-53 : t == ty::f32 ? 0x1p-24 : t == ty::f16 ? 0x1p-11 : 0x1p-8;
}
static double tiny(ty t)
{
    return t == ty::f64 ? 0x1p-1074 : t == ty::f32 ? 0x1p-149 : t == ty::f16 ? 0x1p-24 : 0x1p-133;
}

static double get(ty t, const std::vector<uint8_t> &b, size_t i)
{
    switch (t) {
    case ty::f64: { double v; std::memcpy(&v, &b[i * 8], 8); return v; }
    case ty::f32: { float v; std::memcpy(&v, &b[i * 4], 4); return v; }
    case ty::f16: { uint16_t h; std::memcpy(&h, &b[i * 2], 2); return __half2float(__ushort_as_half(h)); }
    default: { uint16_t h; std::memcpy(&h, &b[i * 2], 2); uint32_t u = uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f; }
    }
}

// Stores values that are exact in the type (inputs are chosen that way),
// or NaN / inf.
static void put(ty t, std::vector<uint8_t> &b, size_t i, double v)
{
    switch (t) {
    case ty::f64: std::memcpy(&b[i * 8], &v, 8); break;
    case ty::f32: { float f = float(v); std::memcpy(&b[i * 4], &f, 4); break; }
    case ty::f16: { uint16_t h = __half_as_ushort(__float2half(float(v))); std::memcpy(&b[i * 2], &h, 2); break; }
    default: {
        float f = float(v); uint32_t u; std::memcpy(&u, &f, 4);
        uint16_t h = uint16_t(u >> 16); // exact for the values used
        if (std::isnan(f)) h = 0x7fc0;
        std::memcpy(&b[i * 2], &h, 2);
        break;
    }
    }
}

static std::mt19937 g_rng(12345);

// A value exact in type t: general for f32/f64, a small multiple of 1/32
// for the 16-bit types (exact in both binary16 and bfloat16).
static double rnd(ty t)
{
    if (t == ty::f16 || t == ty::bf16) return double(int(g_rng() % 129) - 64) / 32.0;
    std::uniform_real_distribution<double> d(-2.0, 2.0);
    double v = d(g_rng);
    return t == ty::f32 ? double(float(v)) : v;
}

struct mat {
    ty t;
    int rows, cols, ld;
    std::vector<uint8_t> h;
    void *d = nullptr;
    mat(ty t_, int r, int c, int ld_, double pad = NAN) : t(t_), rows(r), cols(c), ld(ld_)
    {
        size_t n = size_t(ld) * (c > 0 ? c : 1);
        h.assign(n * esize(t), 0);
        for (size_t i = 0; i < n; ++i) put(t, h, i, pad);
        CUDA_OK(cudaMalloc(&d, h.size()));
    }
    ~mat() { cudaFree(d); }
    mat(const mat &) = delete;
    void fill() { for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) put(t, h, i + size_t(j) * ld, rnd(t)); }
    void fill_value(double v) { for (int j = 0; j < cols; ++j) for (int i = 0; i < rows; ++i) put(t, h, i + size_t(j) * ld, v); }
    double at(int i, int j) const { return get(t, h, i + size_t(j) * ld); }
    void upload() { CUDA_OK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); }
    std::vector<uint8_t> download() const
    {
        std::vector<uint8_t> out(h.size());
        CUDA_OK(cudaMemcpy(out.data(), d, out.size(), cudaMemcpyDeviceToHost));
        return out;
    }
};

// ---- grading ---------------------------------------------------------------

struct tally { int n = 0, bad = 0; double worst = 0; std::string worst_where, sample; };
static std::map<std::string, tally> g_tally;

static bool same_bits(ty t, const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, size_t i)
{
    return std::memcmp(&a[i * esize(t)], &b[i * esize(t)], esize(t)) == 0;
}

// Grades C (after the call) against the reference computed from A, B and
// the C before the call. `poison_ab`: alpha == 0 case, the result is beta*C.
static void grade(const char *group, const std::string &what, char ta, char tb, int m, int n,
                  int k, double alpha, const mat &a, const mat &b, double beta, const mat &c0,
                  const std::vector<uint8_t> &got, double u_c)
{
    tally &t = g_tally[group];
    const ty ct = c0.t;
    int bad_here = 0;
    for (int j = 0; j < c0.cols; ++j) {
        for (int i = 0; i < c0.ld; ++i) {
            const size_t idx = i + size_t(j) * c0.ld;
            if (i >= m || j >= n) { // padding: untouched
                if (!same_bits(ct, got, c0.h, idx)) { ++bad_here; if (bad_here < 3) std::printf("  %s %s: padding (%d,%d) written\n", group, what.c_str(), i, j); }
                continue;
            }
            ld s = 0, abs_s = 0;
            if (alpha != 0) {
                for (int p = 0; p < k; ++p) {
                    ld x = ta == 'N' ? a.at(i, p) : a.at(p, i);
                    ld y = tb == 'N' ? b.at(p, j) : b.at(j, p);
                    s += x * y;
                    abs_s += std::fabs(x * y);
                }
            }
            const double cv = beta != 0 ? c0.at(i, j) : 0.0;
            const ld ref = ld(alpha) * s + (beta != 0 ? ld(beta) * cv : 0);
            const ld scale = std::fabs(ld(alpha)) * abs_s + std::fabs(ld(beta) * cv);
            const double g = get(ct, got, idx);
            const double bound = double(2.0L * (k + 1) * u_c * scale + unit_roundoff(ct) * std::fabs(ref)) + tiny(ct);
            double ratio;
            if (std::isnan(double(ref))) ratio = std::isnan(g) ? 0 : INFINITY;
            else if (std::isinf(double(ref))) ratio = g == double(ref) ? 0 : INFINITY;
            else ratio = std::isnan(g) ? INFINITY : double(std::fabs(ld(g) - ref)) / bound;
            ++t.n;
            if (!(ratio <= 1.0)) {
                ++bad_here;
                if (bad_here < 3) std::printf("  %s %s: C(%d,%d) = %.9g, want %.9g (bound %.3g)\n", group, what.c_str(), i, j, g, double(ref), bound);
            }
            if (ratio > t.worst || std::isnan(ratio)) { t.worst = ratio; t.worst_where = what; }
            if (t.sample.empty() && m * n > 100 && i == m / 2 && j == n / 2) {
                char buf[160];
                std::snprintf(buf, sizeof buf, "%s C(%d,%d) got %.9g want %.9g", what.c_str(), i, j, g, double(ref));
                t.sample = buf;
            }
        }
    }
    ++g_checks;
    if (bad_here) { ++g_fail; t.bad += bad_here; std::printf("FAIL %s %s: %d elements\n", group, what.c_str(), bad_here); }
}

static cublasOperation_t op_of(char c) { return c == 'N' ? CUBLAS_OP_N : c == 'T' ? CUBLAS_OP_T : CUBLAS_OP_C; }

static cublasHandle_t H;

// ---- GEMM through cublasGemmEx (and the typed entry points) ----------------

struct combo {
    const char *name;
    ty ab, c;
    cublasComputeType_t compute;
    double u_c;
    ty scale; // type of alpha/beta
};

static const combo COMBOS[] = {
    {"ex_32F_s", ty::f32, ty::f32, CUBLAS_COMPUTE_32F, 0x1p-24, ty::f32},
    {"ex_32F_pedantic_s", ty::f32, ty::f32, CUBLAS_COMPUTE_32F_PEDANTIC, 0x1p-24, ty::f32},
    {"ex_32F_h_h", ty::f16, ty::f16, CUBLAS_COMPUTE_32F, 0x1p-24, ty::f32},
    {"ex_32F_h_s", ty::f16, ty::f32, CUBLAS_COMPUTE_32F, 0x1p-24, ty::f32},
    {"ex_32F_bf_bf", ty::bf16, ty::bf16, CUBLAS_COMPUTE_32F, 0x1p-24, ty::f32},
    {"ex_32F_bf_s", ty::bf16, ty::f32, CUBLAS_COMPUTE_32F, 0x1p-24, ty::f32},
    {"ex_16F_h_h", ty::f16, ty::f16, CUBLAS_COMPUTE_16F, 0x1p-11, ty::f16},
    {"ex_FAST_16F_s", ty::f32, ty::f32, CUBLAS_COMPUTE_32F_FAST_16F, 0x1p-11, ty::f32},
    {"ex_FAST_16BF_s", ty::f32, ty::f32, CUBLAS_COMPUTE_32F_FAST_16BF, 0x1p-8, ty::f32},
    {"ex_FAST_TF32_s", ty::f32, ty::f32, CUBLAS_COMPUTE_32F_FAST_TF32, 0x1p-11, ty::f32},
    {"ex_64F_d", ty::f64, ty::f64, CUBLAS_COMPUTE_64F, 0x1p-53, ty::f64},
    {"ex_legacy_R_32F", ty::f32, ty::f32, (cublasComputeType_t)CUDA_R_32F, 0x1p-24, ty::f32},
};

// Host scalar of the given type, as bytes.
struct scalar {
    uint8_t b[8];
    scalar(ty t, double v) { std::vector<uint8_t> tmp(8); put(t, tmp, 0, v); std::memcpy(b, tmp.data(), 8); }
};

enum class api { ex, typed };

static void run_gemm(const char *group, const combo &cb, api how, char ta, char tb, int m, int n,
                     int k, double alpha, double beta, bool poison_ab = false)
{
    const int arows = ta == 'N' ? m : k, acols = ta == 'N' ? k : m;
    const int brows = tb == 'N' ? k : n, bcols = tb == 'N' ? n : k;
    mat a(cb.ab, arows, acols, arows + 2), b(cb.ab, brows, bcols, brows + 1), c(cb.c, m, n, m + 3, 3.5);
    if (poison_ab) {
        a.fill_value(INFINITY); b.fill_value(NAN);
        for (size_t i = 0; i < a.h.size() / esize(a.t); ++i) put(a.t, a.h, i, (i % 2) ? NAN : -INFINITY);
    } else {
        a.fill(); b.fill();
    }
    if (beta == 0) c.fill_value(NAN); else c.fill();
    a.upload(); b.upload(); c.upload();
    scalar al(cb.scale, alpha), be(cb.scale, beta);
    cublasStatus_t st;
    if (how == api::ex) {
        st = cublasGemmEx(H, op_of(ta), op_of(tb), m, n, k, al.b, a.d, dtype(cb.ab), a.ld, b.d,
                          dtype(cb.ab), b.ld, be.b, c.d, dtype(cb.c), c.ld, cb.compute,
                          CUBLAS_GEMM_DEFAULT);
    } else if (cb.ab == ty::f32) {
        st = cublasSgemm(H, op_of(ta), op_of(tb), m, n, k, (const float *)al.b, (const float *)a.d, a.ld,
                         (const float *)b.d, b.ld, (const float *)be.b, (float *)c.d, c.ld);
    } else if (cb.ab == ty::f64) {
        st = cublasDgemm(H, op_of(ta), op_of(tb), m, n, k, (const double *)al.b, (const double *)a.d, a.ld,
                         (const double *)b.d, b.ld, (const double *)be.b, (double *)c.d, c.ld);
    } else {
        st = cublasHgemm(H, op_of(ta), op_of(tb), m, n, k, (const __half *)al.b, (const __half *)a.d, a.ld,
                         (const __half *)b.d, b.ld, (const __half *)be.b, (__half *)c.d, c.ld);
    }
    char what[160];
    std::snprintf(what, sizeof what, "%c%c m%d n%d k%d a%g b%g%s", ta, tb, m, n, k, alpha, beta, poison_ab ? " poisonAB" : "");
    ++g_checks;
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::printf("FAIL %s %s: status %s\n", group, what, cublasGetStatusName(st));
        ++g_fail;
        return;
    }
    CUDA_OK(cudaDeviceSynchronize());
    grade(group, what, ta == 'C' ? 'T' : ta, tb == 'C' ? 'T' : tb, m, n, k, alpha, a, b, beta, c, c.download(), cb.u_c);
}

static const int SHAPES[][3] = {
    {1, 1, 1}, {2, 3, 4}, {16, 16, 16}, {17, 33, 9}, {64, 64, 64}, {65, 129, 31},
    {128, 128, 128}, {129, 130, 131}, {200, 257, 70}, {1, 300, 5}, {300, 1, 7}, {33, 47, 0},
};

static void test_sgemm_dgemm_hgemm()
{
    const char tr[][2] = {{'N', 'N'}, {'T', 'N'}, {'N', 'T'}, {'T', 'T'}};
    const double ab[][2] = {{1.0, 0.0}, {0.5, -0.75}, {-2.0, 1.0}};
    for (auto &s : SHAPES)
        for (auto &t : tr)
            for (auto &x : ab) run_gemm("sgemm", COMBOS[0], api::typed, t[0], t[1], s[0], s[1], s[2], x[0], x[1]);
    run_gemm("sgemm", COMBOS[0], api::typed, 'C', 'C', 37, 21, 19, 1.0, 0.5); // OP_C == OP_T for reals
    // alpha == 0: A and B are inf/NaN and must not be read.
    run_gemm("sgemm", COMBOS[0], api::typed, 'N', 'N', 20, 30, 10, 0.0, 0.5, true);
    run_gemm("sgemm", COMBOS[0], api::typed, 'T', 'N', 20, 30, 10, 0.0, 0.0, true);
    for (int i : {3, 5, 7, 8})
        for (auto &t : tr) run_gemm("dgemm", COMBOS[10], api::typed, t[0], t[1], SHAPES[i][0], SHAPES[i][1], SHAPES[i][2], 0.5, i % 2 ? 0.0 : -1.0);
    run_gemm("dgemm", COMBOS[10], api::typed, 'N', 'T', 20, 30, 10, 0.0, 2.0, true);
    for (int i : {3, 5, 7, 8})
        for (auto &t : tr) run_gemm("hgemm", COMBOS[6], api::typed, t[0], t[1], SHAPES[i][0], SHAPES[i][1], SHAPES[i][2], 0.5, i % 2 ? 0.0 : -1.0);
    run_gemm("hgemm", COMBOS[6], api::typed, 'N', 'N', 20, 30, 10, 0.0, 0.5, true);
}

static void test_gemm_ex()
{
    const char tr[][2] = {{'N', 'N'}, {'T', 'N'}, {'N', 'T'}, {'T', 'T'}};
    for (const combo &cb : COMBOS)
        for (int i : {3, 5, 7})
            for (auto &t : tr)
                for (double beta : {0.0, 0.5}) run_gemm(cb.name, cb, api::ex, t[0], t[1], SHAPES[i][0], SHAPES[i][1], SHAPES[i][2], 1.5, beta);
    for (const combo &cb : COMBOS) run_gemm(cb.name, cb, api::ex, 'T', 'N', 9, 11, 13, 0.0, 0.0, true);
}

// ---- batches ---------------------------------------------------------------

// count batches of m x n x k; strided (stride >= size, or 0 for a shared B)
// or through device arrays of pointers.
static void run_batched(const char *group, const combo &cb, bool arrays, bool typed, char ta, char tb,
                        int m, int n, int k, int count, bool share_b, double alpha, double beta)
{
    const int arows = ta == 'N' ? m : k, acols = ta == 'N' ? k : m;
    const int brows = tb == 'N' ? k : n, bcols = tb == 'N' ? n : k;
    // One allocation per operand holding every batch (strided), matrices
    // spaced by more than their size.
    const int lda = arows + 1, ldb = brows, ldc = m + 1;
    const long long sa = (long long)lda * acols + 5, sb = share_b ? 0 : (long long)ldb * bcols + 3,
                    sc = (long long)ldc * n + 7;
    std::vector<mat *> as, bs, cs;
    mat abig(cb.ab, int(sa), count, int(sa)), bbig(cb.ab, int(sb ? sb : (long long)ldb * bcols), share_b ? 1 : count, int(sb ? sb : (long long)ldb * bcols)),
        cbig(cb.c, int(sc), count, int(sc), 3.5);
    for (int q = 0; q < count; ++q) {
        as.push_back(new mat(cb.ab, arows, acols, lda));
        bs.push_back(new mat(cb.ab, brows, bcols, ldb));
        cs.push_back(new mat(cb.c, m, n, ldc, 3.5));
        as[q]->fill(); bs[q]->fill();
        if (beta == 0) cs[q]->fill_value(NAN); else cs[q]->fill();
        if (share_b && q > 0) bs[q]->h = bs[0]->h;
        std::memcpy(&abig.h[q * sa * esize(cb.ab)], as[q]->h.data(), as[q]->h.size());
        if (!share_b || q == 0) std::memcpy(&bbig.h[q * sb * esize(cb.ab)], bs[q]->h.data(), bs[q]->h.size());
        std::memcpy(&cbig.h[q * sc * esize(cb.c)], cs[q]->h.data(), cs[q]->h.size());
        as[q]->upload(); bs[q]->upload(); cs[q]->upload();
    }
    abig.upload(); bbig.upload(); cbig.upload();
    scalar al(cb.scale, alpha), be(cb.scale, beta);
    std::vector<const void *> pa, pb;
    std::vector<void *> pc;
    for (int q = 0; q < count; ++q) { pa.push_back(as[q]->d); pb.push_back(bs[share_b ? 0 : q]->d); pc.push_back(cs[q]->d); }
    void **dpa = nullptr, **dpb = nullptr, **dpc = nullptr;
    CUDA_OK(cudaMalloc(&dpa, count * sizeof(void *)));
    CUDA_OK(cudaMalloc(&dpb, count * sizeof(void *)));
    CUDA_OK(cudaMalloc(&dpc, count * sizeof(void *)));
    CUDA_OK(cudaMemcpy(dpa, pa.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dpb, pb.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dpc, pc.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
    cublasStatus_t st;
    const cublasOperation_t oa = op_of(ta), ob = op_of(tb);
    if (arrays && typed && cb.ab == ty::f32)
        st = cublasSgemmBatched(H, oa, ob, m, n, k, (const float *)al.b, (const float *const *)dpa, lda,
                                (const float *const *)dpb, ldb, (const float *)be.b, (float *const *)dpc, ldc, count);
    else if (arrays && typed)
        st = cublasHgemmBatched(H, oa, ob, m, n, k, (const __half *)al.b, (const __half *const *)dpa, lda,
                                (const __half *const *)dpb, ldb, (const __half *)be.b, (__half *const *)dpc, ldc, count);
    else if (arrays)
        st = cublasGemmBatchedEx(H, oa, ob, m, n, k, al.b, (const void *const *)dpa, dtype(cb.ab), lda,
                                 (const void *const *)dpb, dtype(cb.ab), ldb, be.b, dpc, dtype(cb.c), ldc,
                                 count, cb.compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    else if (typed && cb.ab == ty::f32)
        st = cublasSgemmStridedBatched(H, oa, ob, m, n, k, (const float *)al.b, (const float *)abig.d, lda, sa,
                                       (const float *)bbig.d, ldb, sb, (const float *)be.b, (float *)cbig.d, ldc, sc, count);
    else if (typed)
        st = cublasHgemmStridedBatched(H, oa, ob, m, n, k, (const __half *)al.b, (const __half *)abig.d, lda, sa,
                                       (const __half *)bbig.d, ldb, sb, (const __half *)be.b, (__half *)cbig.d, ldc, sc, count);
    else
        st = cublasGemmStridedBatchedEx(H, oa, ob, m, n, k, al.b, abig.d, dtype(cb.ab), lda, sa, bbig.d,
                                        dtype(cb.ab), ldb, sb, be.b, cbig.d, dtype(cb.c), ldc, sc, count,
                                        cb.compute, CUBLAS_GEMM_DEFAULT);
    char what[160];
    std::snprintf(what, sizeof what, "%s %c%c m%d n%d k%d x%d%s b%g", arrays ? "arrays" : "strided", ta, tb, m, n, k, count, share_b ? " sharedB" : "", beta);
    ++g_checks;
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::printf("FAIL %s %s: status %s\n", group, what, cublasGetStatusName(st));
        ++g_fail;
    } else {
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint8_t> all = cbig.download();
        for (int q = 0; q < count; ++q) {
            std::vector<uint8_t> got;
            if (arrays) got = cs[q]->download();
            else got.assign(all.begin() + q * sc * esize(cb.c), all.begin() + (q * sc * esize(cb.c) + cs[q]->h.size()));
            grade(group, what + std::string(" #") + std::to_string(q), ta, tb, m, n, k, alpha, *as[q], *bs[q], beta, *cs[q], got, cb.u_c);
        }
        if (!arrays) { // the gaps between strided batches stay untouched
            int bad = 0;
            for (int q = 0; q < count; ++q)
                for (long long i = (long long)ldc * n; i < sc; ++i)
                    if (!same_bits(cb.c, all, cbig.h, q * sc + i)) ++bad;
            CHECK(bad == 0);
        }
    }
    cudaFree(dpa); cudaFree(dpb); cudaFree(dpc);
    for (int q = 0; q < count; ++q) { delete as[q]; delete bs[q]; delete cs[q]; }
}

static void test_batched()
{
    const combo &s = COMBOS[0], &hs = COMBOS[3], &h16 = COMBOS[6], &hh = COMBOS[2];
    run_batched("sgemm_strided_batched", s, false, true, 'N', 'N', 33, 17, 20, 4, false, 1.0, 0.0);
    run_batched("sgemm_strided_batched", s, false, true, 'T', 'N', 65, 70, 9, 3, true, 0.5, 1.0);
    run_batched("sgemm_batched", s, true, true, 'N', 'T', 31, 66, 12, 5, false, 1.0, -1.0);
    run_batched("hgemm_strided_batched", h16, false, true, 'T', 'N', 40, 20, 24, 3, false, 1.0, 0.0);
    run_batched("hgemm_batched", h16, true, true, 'N', 'N', 18, 19, 21, 3, false, 0.5, 0.5);
    run_batched("gemm_strided_batched_ex", hs, false, false, 'T', 'N', 64, 33, 40, 6, false, 1.0, 0.0);
    run_batched("gemm_strided_batched_ex", hh, false, false, 'N', 'T', 70, 65, 30, 2, true, 1.0, 0.5);
    run_batched("gemm_batched_ex", hs, true, false, 'T', 'N', 50, 7, 64, 4, false, 1.0, 0.0);
    run_batched("gemm_batched_ex", COMBOS[4], true, false, 'N', 'N', 20, 30, 40, 3, false, 2.0, 0.0);

    // More batches than one launch's z dimension holds.
    const int count = 65537;
    float *a, *b, *c;
    CUDA_OK(cudaMalloc(&a, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&b, count * sizeof(float)));
    CUDA_OK(cudaMalloc(&c, count * sizeof(float)));
    std::vector<float> ha(count), hb(count), hc(count, NAN);
    for (int q = 0; q < count; ++q) { ha[q] = float(q % 1000); hb[q] = float(q % 7) - 3.0f; }
    CUDA_OK(cudaMemcpy(a, ha.data(), count * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b, hb.data(), count * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(c, hc.data(), count * 4, cudaMemcpyHostToDevice));
    const float one = 1.0f, zero = 0.0f;
    BLAS_OK(cublasSgemmStridedBatched(H, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, 1, 1, b, 1, 1, &zero, c, 1, 1, count));
    CUDA_OK(cudaMemcpy(hc.data(), c, count * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int q = 0; q < count; ++q) bad += hc[q] != ha[q] * hb[q];
    CHECK(bad == 0);
    cudaFree(a); cudaFree(b); cudaFree(c);
}

// ---- pointer mode, streams ---------------------------------------------------

static void test_device_pointer_mode()
{
    const int m = 70, n = 33, k = 45;
    mat a(ty::f32, m, k, m), b(ty::f32, k, n, k), c(ty::f32, m, n, m);
    a.fill(); b.fill(); c.fill();
    a.upload(); b.upload(); c.upload();
    float *scal;
    CUDA_OK(cudaMalloc(&scal, 4 * sizeof(float)));
    const float hs[4] = {1.25f, -0.5f, 0.0f, 0.0f};
    CUDA_OK(cudaMemcpy(scal, hs, sizeof hs, cudaMemcpyHostToDevice));
    BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_DEVICE));
    cublasPointerMode_t pm;
    BLAS_OK(cublasGetPointerMode(H, &pm));
    CHECK(pm == CUBLAS_POINTER_MODE_DEVICE);
    BLAS_OK(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, scal, (float *)a.d, m, (float *)b.d, k, scal + 1, (float *)c.d, m));
    CUDA_OK(cudaDeviceSynchronize());
    grade("pointer_mode_device", "sgemm NN", 'N', 'N', m, n, k, 1.25, a, b, -0.5, c, c.download(), 0x1p-24);

    // alpha = 0 and beta = 0 from device memory: C = 0, NaN in C not read.
    c.fill_value(NAN); c.upload();
    BLAS_OK(cublasSgemm(H, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, scal + 2, (float *)a.d, k, (float *)b.d, n, scal + 3, (float *)c.d, m));
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> got = c.download();
    int nonzero = 0;
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) nonzero += get(ty::f32, got, i + j * m) != 0.0;
    CHECK(nonzero == 0);

    // Half scalars in device memory for COMPUTE_16F.
    mat ha(ty::f16, m, k, m), hb(ty::f16, k, n, k), hc(ty::f16, m, n, m);
    ha.fill(); hb.fill(); hc.fill(); ha.upload(); hb.upload(); hc.upload();
    __half hsc[2] = {__float2half(0.5f), __float2half(2.0f)};
    __half *dh;
    CUDA_OK(cudaMalloc(&dh, sizeof hsc));
    CUDA_OK(cudaMemcpy(dh, hsc, sizeof hsc, cudaMemcpyHostToDevice));
    BLAS_OK(cublasGemmEx(H, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, dh, ha.d, CUDA_R_16F, m, hb.d, CUDA_R_16F, k, dh + 1, hc.d, CUDA_R_16F, m, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CUDA_OK(cudaDeviceSynchronize());
    grade("pointer_mode_device", "gemmex 16F NN", 'N', 'N', m, n, k, 0.5, ha, hb, 2.0, hc, hc.download(), 0x1p-11);

    // Level 1 with device scalars and results.
    std::vector<float> x(100), y(100);
    for (int i = 0; i < 100; ++i) { x[i] = float(i % 9) - 4.0f; y[i] = float(i % 5); }
    float *dx, *dy, *dres;
    CUDA_OK(cudaMalloc(&dx, 400)); CUDA_OK(cudaMalloc(&dy, 400)); CUDA_OK(cudaMalloc(&dres, 8));
    CUDA_OK(cudaMemcpy(dx, x.data(), 400, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dy, y.data(), 400, cudaMemcpyHostToDevice));
    BLAS_OK(cublasSdot(H, 100, dx, 1, dy, 1, dres));
    BLAS_OK(cublasSnrm2(H, 100, dx, 1, dres + 1));
    BLAS_OK(cublasSaxpy(H, 100, scal, dx, 1, dy, 1)); // y += 1.25 x
    float res[2];
    CUDA_OK(cudaMemcpy(res, dres, 8, cudaMemcpyDeviceToHost));
    std::vector<float> yo(100);
    CUDA_OK(cudaMemcpy(yo.data(), dy, 400, cudaMemcpyDeviceToHost));
    double dot = 0, ss = 0;
    int bad = 0;
    for (int i = 0; i < 100; ++i) { dot += double(x[i]) * y[i]; ss += double(x[i]) * x[i]; bad += yo[i] != y[i] + 1.25f * x[i]; }
    CHECK(res[0] == float(dot)); // small integers: exact
    CHECK(std::fabs(res[1] - std::sqrt(ss)) <= 1e-5 * std::sqrt(ss));
    CHECK(bad == 0);
    BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_HOST));
    cudaFree(dx); cudaFree(dy); cudaFree(dres); cudaFree(scal); cudaFree(dh);
}

static void test_streams()
{
    cudaStream_t s, got = (cudaStream_t)0x1234;
    CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    BLAS_OK(cublasGetStream(H, &got));
    CHECK(got == nullptr);
    BLAS_OK(cublasSetStream(H, s));
    BLAS_OK(cublasGetStream(H, &got));
    CHECK(got == s);
    // Several dependent calls queued on the stream, one wait at the end.
    const int m = 96, n = 80, k = 50;
    mat a(ty::f32, m, k, m), b(ty::f32, k, n, k), c(ty::f32, m, n, m), c2(ty::f32, m, n, m);
    a.fill(); b.fill(); c.fill_value(NAN);
    a.upload(); b.upload(); c.upload();
    const float one = 1.0f, zero = 0.0f, two = 2.0f;
    BLAS_OK(cublasSgemm(H, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, (float *)a.d, m, (float *)b.d, k, &zero, (float *)c.d, m));
    BLAS_OK(cublasSscal(H, m * n, &two, (float *)c.d, 1));
    CUDA_OK(cudaStreamSynchronize(s));
    grade("streams", "sgemm then sscal", 'N', 'N', m, n, k, 2.0, a, b, 0.0, c, c.download(), 0x1p-24);
    float r = -1;
    BLAS_OK(cublasSdot(H, 3, (float *)a.d, 1, (float *)a.d, 1, &r)); // host result: waits
    CHECK(r == float(double(a.at(0, 0)) * a.at(0, 0) + double(a.at(1, 0)) * a.at(1, 0) + double(a.at(2, 0)) * a.at(2, 0)) ||
          std::fabs(r - (a.at(0, 0) * a.at(0, 0) + a.at(1, 0) * a.at(1, 0) + a.at(2, 0) * a.at(2, 0))) < 1e-5);
    BLAS_OK(cublasSetStream(H, nullptr));
    CUDA_OK(cudaStreamDestroy(s));
}

// ---- level 1 and 2 -----------------------------------------------------------

static long long vidx(int i, int n, int inc) { return inc >= 0 ? (long long)i * inc : (long long)(n - 1 - i) * -inc; }

static void test_level1_2()
{
    for (int n : {1, 7, 255, 256, 1000, 5000}) {
        for (int incx : {1, 2, -3}) {
            const int incy = incx == 1 ? 1 : -1;
            const size_t lx = 1 + size_t(n - 1) * std::abs(incx), ly = 1 + size_t(n - 1) * std::abs(incy);
            std::vector<float> x(lx), y(ly);
            for (auto &v : x) v = float(rnd(ty::f32));
            for (auto &v : y) v = float(rnd(ty::f32));
            float *dx, *dy;
            CUDA_OK(cudaMalloc(&dx, lx * 4)); CUDA_OK(cudaMalloc(&dy, ly * 4));
            CUDA_OK(cudaMemcpy(dx, x.data(), lx * 4, cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemcpy(dy, y.data(), ly * 4, cudaMemcpyHostToDevice));
            // dot
            float r;
            BLAS_OK(cublasSdot(H, n, dx, incx, dy, incy, &r));
            ld s = 0, as = 0;
            for (int i = 0; i < n; ++i) { ld p = ld(x[vidx(i, n, incx)]) * y[vidx(i, n, incy)]; s += p; as += std::fabs(p); }
            tally &td = g_tally["sdot"]; ++td.n;
            double rd = double(std::fabs(r - s) / (2 * (n + 1) * 0x1p-24 * as + 0x1p-149));
            if (rd > td.worst) td.worst = rd;
            CHECK(rd <= 1);
            // nrm2 (positive increments only; others give 0)
            BLAS_OK(cublasSnrm2(H, n, dx, incx, &r));
            if (incx > 0) {
                ld q = 0;
                for (int i = 0; i < n; ++i) q += ld(x[size_t(i) * incx]) * x[size_t(i) * incx];
                double rn = double(std::fabs(r - std::sqrt(q)) / (4 * (n + 1) * 0x1p-24 * std::sqrt(q)));
                tally &tn = g_tally["snrm2"]; ++tn.n;
                if (rn > tn.worst) tn.worst = rn;
                CHECK(rn <= 1);
            } else {
                CHECK(r == 0.0f);
            }
            // axpy: one rounding of an fma or of a product and a sum
            const float alpha = -1.5f;
            BLAS_OK(cublasSaxpy(H, n, &alpha, dx, incx, dy, incy));
            std::vector<float> yo(ly);
            CUDA_OK(cudaMemcpy(yo.data(), dy, ly * 4, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int i = 0; i < n; ++i) {
                ld want = ld(alpha) * x[vidx(i, n, incx)] + y[vidx(i, n, incy)];
                ld bound = 2 * 0x1p-24 * (std::fabs(ld(alpha) * x[vidx(i, n, incx)]) + std::fabs(ld(y[vidx(i, n, incy)])));
                if (std::fabs(yo[vidx(i, n, incy)] - want) > bound) ++bad;
            }
            ++g_tally["saxpy"].n;
            CHECK(bad == 0);
            // scal: exact for a power of two
            const float four = 4.0f;
            BLAS_OK(cublasSscal(H, n, &four, dx, std::abs(incx)));
            std::vector<float> xo(lx);
            CUDA_OK(cudaMemcpy(xo.data(), dx, lx * 4, cudaMemcpyDeviceToHost));
            bad = 0;
            for (size_t i = 0; i < lx; ++i) bad += xo[i] != ((i % std::abs(incx)) == 0 ? 4.0f * x[i] : x[i]);
            ++g_tally["sscal"].n;
            CHECK(bad == 0);
            cudaFree(dx); cudaFree(dy);
        }
    }
    // nrm2 neither overflows nor underflows
    {
        float hx[3] = {3e30f, 4e30f, 0.0f}, r = 0;
        float *dx;
        CUDA_OK(cudaMalloc(&dx, sizeof hx));
        CUDA_OK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
        BLAS_OK(cublasSnrm2(H, 3, dx, 1, &r));
        CHECK(std::fabs(r - 5e30f) <= 5e30f * 1e-6f);
        hx[0] = 3e-30f; hx[1] = 4e-30f;
        CUDA_OK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
        BLAS_OK(cublasSnrm2(H, 3, dx, 1, &r));
        CHECK(std::fabs(r - 5e-30f) <= 5e-30f * 1e-6f);
        hx[2] = INFINITY;
        CUDA_OK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
        BLAS_OK(cublasSnrm2(H, 3, dx, 1, &r));
        CHECK(std::isinf(r));
        hx[1] = NAN;
        CUDA_OK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));
        BLAS_OK(cublasSnrm2(H, 3, dx, 1, &r));
        CHECK(std::isnan(r));
        cudaFree(dx);
    }
    // gemv
    for (char t : {'N', 'T'}) {
        for (auto mn : {std::pair<int, int>{1, 1}, {37, 19}, {300, 257}, {5, 600}}) {
            for (double beta : {0.0, 0.5}) {
                const int m = mn.first, n = mn.second, lda = m + 2;
                const int xl = t == 'N' ? n : m, yl = t == 'N' ? m : n;
                const int incx = 2, incy = -1;
                mat a(ty::f32, m, n, lda);
                a.fill(); a.upload();
                std::vector<float> x(size_t(xl) * 2), y(yl);
                for (auto &v : x) v = float(rnd(ty::f32));
                for (auto &v : y) v = beta == 0 ? NAN : float(rnd(ty::f32));
                float *dx, *dy;
                CUDA_OK(cudaMalloc(&dx, x.size() * 4)); CUDA_OK(cudaMalloc(&dy, y.size() * 4));
                CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
                const float al = 1.25f, be = float(beta);
                BLAS_OK(cublasSgemv(H, t == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, m, n, &al, (float *)a.d, lda, dx, incx, &be, dy, incy));
                std::vector<float> yo(yl);
                CUDA_OK(cudaMemcpy(yo.data(), dy, yl * 4, cudaMemcpyDeviceToHost));
                int bad = 0;
                tally &tg = g_tally["sgemv"];
                for (int i = 0; i < yl; ++i) {
                    ld s = 0, as = 0;
                    for (int j = 0; j < xl; ++j) {
                        ld p = ld(t == 'N' ? a.at(i, j) : a.at(j, i)) * x[vidx(j, xl, incx)];
                        s += p; as += std::fabs(p);
                    }
                    const ld yv = beta != 0 ? ld(y[vidx(i, yl, incy)]) : 0;
                    const ld want = ld(al) * s + ld(be) * yv;
                    const ld bound = 2 * (xl + 1) * 0x1p-24 * (ld(al) * as + std::fabs(ld(be) * yv)) + 0x1p-149;
                    const double r = std::isnan(yo[vidx(i, yl, incy)]) ? INFINITY : double(std::fabs(yo[vidx(i, yl, incy)] - want) / bound);
                    if (r > tg.worst) tg.worst = r;
                    ++tg.n;
                    bad += !(r <= 1);
                }
                CHECK(bad == 0);
                cudaFree(dx); cudaFree(dy);
            }
        }
    }
}

// ---- handles, modes, status codes ------------------------------------------

static void test_api_rules()
{
    cublasHandle_t h2 = nullptr;
    BLAS_OK(cublasCreate(&h2));
    CHECK(h2 != nullptr);
    int v = 0;
    BLAS_OK(cublasGetVersion(h2, &v));
    CHECK(v == CUBLAS_VERSION);
    BLAS_OK(cublasGetProperty(MAJOR_VERSION, &v));
    CHECK(v == CUBLAS_VER_MAJOR);
    cublasMath_t mm;
    BLAS_OK(cublasGetMathMode(h2, &mm));
    CHECK(mm == CUBLAS_DEFAULT_MATH);
    for (cublasMath_t m : {CUBLAS_TF32_TENSOR_OP_MATH, CUBLAS_PEDANTIC_MATH, CUBLAS_TENSOR_OP_MATH,
                           (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)}) {
        BLAS_OK(cublasSetMathMode(h2, m));
        BLAS_OK(cublasGetMathMode(h2, &mm));
        CHECK(mm == m);
    }
    BLAS_IS(cublasSetMathMode(h2, (cublasMath_t)9), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSetPointerMode(h2, (cublasPointerMode_t)5), CUBLAS_STATUS_INVALID_VALUE);
    void *ws;
    CUDA_OK(cudaMalloc(&ws, 1 << 20));
    BLAS_OK(cublasSetWorkspace(h2, ws, 1 << 20));
    BLAS_OK(cublasSetAtomicsMode(h2, CUBLAS_ATOMICS_ALLOWED));
    cublasAtomicsMode_t am;
    BLAS_OK(cublasGetAtomicsMode(h2, &am));
    CHECK(am == CUBLAS_ATOMICS_ALLOWED);

    // With TF32 math mode and a workspace, results are still within the bound.
    float *a, *c;
    CUDA_OK(cudaMalloc(&a, 64 * 4)); CUDA_OK(cudaMalloc(&c, 64 * 4));
    std::vector<float> ha(64);
    for (int i = 0; i < 64; ++i) ha[i] = float(i) / 8.0f;
    CUDA_OK(cudaMemcpy(a, ha.data(), 256, cudaMemcpyHostToDevice));
    const float one = 1.0f, zero = 0.0f;
    BLAS_OK(cublasSetMathMode(h2, CUBLAS_TF32_TENSOR_OP_MATH));
    BLAS_OK(cublasSgemm(h2, CUBLAS_OP_N, CUBLAS_OP_T, 8, 8, 8, &one, a, 8, a, 8, &zero, c, 8));
    std::vector<float> hc(64);
    CUDA_OK(cudaMemcpy(hc.data(), c, 256, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) {
            double s = 0;
            for (int p = 0; p < 8; ++p) s += double(ha[i + p * 8]) * ha[j + p * 8];
            bad += hc[i + j * 8] != float(s); // multiples of 1/64, exact in binary32
        }
    CHECK(bad == 0);

    // Argument errors: nothing runs, the documented status comes back.
    BLAS_IS(cublasSgemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, 1, a, 1, &zero, c, 1), CUBLAS_STATUS_NOT_INITIALIZED);
    BLAS_IS(cublasSgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, -1, 1, 1, &one, a, 1, a, 1, &zero, c, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, 7, a, 8, &zero, c, 8), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N, 8, 8, 9, &one, a, 8, a, 8, &zero, c, 8), CUBLAS_STATUS_INVALID_VALUE); // lda < k
    BLAS_IS(cublasSgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &one, a, 8, a, 8, &zero, c, 7), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgemm(h2, (cublasOperation_t)7, CUBLAS_OP_N, 1, 1, 1, &one, a, 1, a, 1, &zero, c, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgemmStridedBatched(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, 1, 1, a, 1, 1, &zero, c, 1, 1, -1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgemv(h2, CUBLAS_OP_N, 8, 8, &one, a, 8, a, 0, &zero, c, 1), CUBLAS_STATUS_INVALID_VALUE);
    // Empty problems succeed and touch nothing.
    BLAS_OK(cublasSgemm(h2, CUBLAS_OP_N, CUBLAS_OP_N, 0, 8, 8, &one, a, 1, a, 8, &zero, c, 1));
    BLAS_OK(cublasSgemmStridedBatched(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, 1, 1, a, 1, 1, &zero, c, 1, 1, 0));
    // Combinations not computed here: "not supported", never a result.
    const cublasStatus_t NS = CUBLAS_STATUS_NOT_SUPPORTED;
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_C_32F, 1, a, CUDA_C_32F, 1, &zero, c, CUDA_C_32F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), NS);
    // 8-bit integers into 32-bit ones: computed (test_gemm_i8); other integer outputs are not
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_8I, 1, a, CUDA_R_8I, 1, &zero, c, CUDA_R_32F, 1, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_16F, 1, a, CUDA_R_16F, 1, &zero, c, CUDA_R_16F, 1, CUBLAS_COMPUTE_16F_PEDANTIC, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_16F, 1, a, CUDA_R_32F, 1, &zero, c, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_32F, 1, a, CUDA_R_32F, 1, &zero, c, CUDA_R_16F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_16F, 1, a, CUDA_R_16F, 1, &zero, c, CUDA_R_32F, 1, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_8F_E4M3, 1, a, CUDA_R_8F_E4M3, 1, &zero, c, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), NS);
    BLAS_IS(cublasGemmEx(h2, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &one, a, CUDA_R_32F, 1, a, CUDA_R_32F, 1, &zero, c, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t)50), CUBLAS_STATUS_INVALID_VALUE);

    // Status names and strings exist for every status.
    for (cublasStatus_t s : {CUBLAS_STATUS_SUCCESS, CUBLAS_STATUS_NOT_INITIALIZED, CUBLAS_STATUS_ALLOC_FAILED,
                             CUBLAS_STATUS_INVALID_VALUE, CUBLAS_STATUS_ARCH_MISMATCH, CUBLAS_STATUS_MAPPING_ERROR,
                             CUBLAS_STATUS_EXECUTION_FAILED, CUBLAS_STATUS_INTERNAL_ERROR, CUBLAS_STATUS_NOT_SUPPORTED,
                             CUBLAS_STATUS_LICENSE_ERROR}) {
        CHECK(cublasGetStatusString(s) != nullptr && cublasGetStatusString(s)[0] != 0);
        CHECK(cublasGetStatusName(s) != nullptr && std::strncmp(cublasGetStatusName(s), "CUBLAS_STATUS_", 14) == 0);
    }
    CHECK(std::strcmp(cublasGetStatusName(CUBLAS_STATUS_NOT_SUPPORTED), "CUBLAS_STATUS_NOT_SUPPORTED") == 0);
    CHECK(CUBLAS_STATUS_NOT_SUPPORTED == 15 && CUBLAS_COMPUTE_32F == 68 && CUDA_R_16BF == 14);

    BLAS_OK(cublasDestroy(h2));
    BLAS_IS(cublasDestroy(nullptr), CUBLAS_STATUS_NOT_INITIALIZED);
    cudaFree(a); cudaFree(c); cudaFree(ws);
}

// ---- triangular solves (batched) -------------------------------------------
//
// Graded by the residual, which a backward-stable substitution keeps small
// whatever the conditioning: |op(A) X - alpha B| <= 2 (n + 1) u (|op(A)| |X| + |alpha B|)
// (left side; the right side likewise with X op(A)). The triangle is made
// diagonally dominant so X stays moderate. The other triangle holds NaN:
// reading it would show.
static void test_trsm_batched()
{
    const int count = 3;
    tally &t = g_tally["strsm_batched"];
    for (char side : {'L', 'R'})
        for (char uplo : {'U', 'L'})
            for (char tr : {'N', 'T'})
                for (char dg : {'N', 'U'})
                    for (auto mn : {std::pair<int, int>{1, 1}, {13, 7}, {40, 65}}) {
                        const int m = mn.first, n = mn.second, order = side == 'L' ? m : n;
                        const int lda = order + 1, ldb = m + 2;
                        const float alpha = side == 'L' ? 1.5f : -0.5f;
                        std::vector<mat *> as, bs;
                        std::vector<void *> pa, pb;
                        for (int q = 0; q < count; ++q) {
                            mat *a = new mat(ty::f32, order, order, lda), *b = new mat(ty::f32, m, n, ldb, 3.5);
                            for (int j = 0; j < order; ++j)
                                for (int i = 0; i < order; ++i) {
                                    const bool in = uplo == 'U' ? i <= j : i >= j;
                                    double v = !in ? NAN : i == j ? (g_rng() % 2 ? 1 : -1) * (order + 1.0 + rnd(ty::f32)) : rnd(ty::f32) / order;
                                    if (i == j && dg == 'U') v = NAN; // unit diagonal: never read
                                    put(ty::f32, a->h, i + size_t(j) * lda, v);
                                }
                            b->fill();
                            a->upload(); b->upload();
                            as.push_back(a); bs.push_back(b);
                            pa.push_back(a->d); pb.push_back(b->d);
                        }
                        void **dpa, **dpb;
                        CUDA_OK(cudaMalloc(&dpa, count * sizeof(void *)));
                        CUDA_OK(cudaMalloc(&dpb, count * sizeof(void *)));
                        CUDA_OK(cudaMemcpy(dpa, pa.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
                        CUDA_OK(cudaMemcpy(dpb, pb.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
                        BLAS_OK(cublasStrsmBatched(H, side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                                   uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                                                   tr == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                                                   dg == 'U' ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                                                   (const float *const *)dpa, lda, (float *const *)dpb, ldb, count));
                        CUDA_OK(cudaDeviceSynchronize());
                        int bad = 0;
                        for (int q = 0; q < count; ++q) {
                            std::vector<uint8_t> got = bs[q]->download();
                            auto T = [&](int i, int j) -> ld { // op(A), unit diagonal applied
                                const int r = tr == 'N' ? i : j, c = tr == 'N' ? j : i;
                                const bool in = uplo == 'U' ? r <= c : r >= c;
                                if (!in) return 0;
                                if (r == c && dg == 'U') return 1;
                                return as[q]->at(r, c);
                            };
                            auto X = [&](int i, int j) -> ld { return get(ty::f32, got, i + size_t(j) * ldb); };
                            for (int j = 0; j < n; ++j)
                                for (int i = 0; i < m; ++i) {
                                    ld s = 0, as_ = 0;
                                    for (int p = 0; p < order; ++p) {
                                        const ld v = side == 'L' ? T(i, p) * X(p, j) : X(i, p) * T(p, j);
                                        s += v; as_ += std::fabs(v);
                                    }
                                    const ld rhs = ld(alpha) * bs[q]->at(i, j);
                                    const double r = double(std::fabs(s - rhs) / (2 * (order + 1) * 0x1p-24 * (as_ + std::fabs(rhs)) + 0x1p-149));
                                    ++t.n;
                                    if (!(r <= 1)) ++bad;
                                    if (r > t.worst || std::isnan(r)) { t.worst = r; t.worst_where = std::string(1, side) + uplo + tr + dg; }
                                }
                            for (int j = 0; j < n; ++j) // padding rows untouched
                                for (int i = m; i < ldb; ++i) bad += !same_bits(ty::f32, got, bs[q]->h, i + size_t(j) * ldb);
                        }
                        ++g_checks;
                        if (bad) { ++g_fail; t.bad += bad; std::printf("FAIL strsm_batched %c%c%c%c m%d n%d: %d\n", side, uplo, tr, dg, m, n, bad); }
                        cudaFree(dpa); cudaFree(dpb);
                        for (int q = 0; q < count; ++q) { delete as[q]; delete bs[q]; }
                    }
    // alpha == 0: B = 0, A (all NaN here) not read.
    mat a(ty::f32, 4, 4, 4), b(ty::f32, 4, 3, 4);
    b.fill(); a.upload(); b.upload();
    void *pa = a.d, *pb = b.d, **dp;
    CUDA_OK(cudaMalloc(&dp, 2 * sizeof(void *)));
    CUDA_OK(cudaMemcpy(dp, &pa, sizeof(void *), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dp + 1, &pb, sizeof(void *), cudaMemcpyHostToDevice));
    const float zero = 0.0f;
    BLAS_OK(cublasStrsmBatched(H, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 3, &zero,
                               (const float *const *)dp, 4, (float *const *)dp + 1, 4, 1));
    std::vector<uint8_t> got = b.download();
    int nonzero = 0;
    for (int i = 0; i < 12; ++i) nonzero += get(ty::f32, got, i) != 0.0;
    CHECK(nonzero == 0);
    BLAS_IS(cublasStrsmBatched(H, (cublasSideMode_t)2, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 3, &zero,
                               (const float *const *)dp, 4, (float *const *)dp + 1, 4, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasStrsmBatched(H, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 4, 5, &zero,
                               (const float *const *)dp, 4, (float *const *)dp + 1, 4, 1), CUBLAS_STATUS_INVALID_VALUE); // lda < n
    cudaFree(dp);
}

// ---- geam: C = alpha op(A) + beta op(B) ----------------------------------

template <class T>
static cublasStatus_t geam(cublasOperation_t ta, cublasOperation_t tb, int m, int n, const T *al, const T *A, int lda,
                           const T *be, const T *B, int ldb, T *C, int ldc);
template <>
cublasStatus_t geam<float>(cublasOperation_t ta, cublasOperation_t tb, int m, int n, const float *al, const float *A,
                           int lda, const float *be, const float *B, int ldb, float *C, int ldc)
{
    return cublasSgeam(H, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc);
}
template <>
cublasStatus_t geam<double>(cublasOperation_t ta, cublasOperation_t tb, int m, int n, const double *al,
                            const double *A, int lda, const double *be, const double *B, int ldb, double *C, int ldc)
{
    return cublasDgeam(H, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc);
}

template <class T>
static void run_geam(ty t, const char *group)
{
    tally &tl = g_tally[group];
    const double u = unit_roundoff(t);
    const double scal[][2] = {{1.5, -0.5}, {0.0, 2.0}, {-2.0, 0.0}, {1.0, 1.0}};
    for (char ta : {'N', 'T'})
        for (char tb : {'N', 'T'})
            for (auto mn : {std::pair<int, int>{1, 1}, {13, 7}, {40, 65}})
                for (auto &sc : scal) {
                    const int m = mn.first, n = mn.second;
                    const int ar = ta == 'N' ? m : n, ac = ta == 'N' ? n : m;
                    const int br = tb == 'N' ? m : n, bc = tb == 'N' ? n : m;
                    mat a(t, ar, ac, ar + 1), b(t, br, bc, br + 3), c(t, m, n, m + 2, 7.25);
                    // alpha == 0 never reads A, beta == 0 never reads B: they hold NaN then.
                    if (sc[0] == 0) a.fill_value(NAN); else a.fill();
                    if (sc[1] == 0) b.fill_value(NAN); else b.fill();
                    c.fill_value(NAN); // C is output only
                    a.upload(); b.upload(); c.upload();
                    const T al = T(sc[0]), be = T(sc[1]);
                    BLAS_OK(geam<T>(op_of(ta), op_of(tb), m, n, &al, (const T *)a.d, a.ld, &be, (const T *)b.d, b.ld,
                                    (T *)c.d, c.ld));
                    std::vector<uint8_t> got = c.download();
                    int bad = 0;
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < m; ++i) {
                            const ld x = sc[0] == 0 ? 0 : ld(sc[0]) * (ta == 'N' ? a.at(i, j) : a.at(j, i));
                            const ld y = sc[1] == 0 ? 0 : ld(sc[1]) * (tb == 'N' ? b.at(i, j) : b.at(j, i));
                            const ld ref = x + y;
                            const double r = double(std::fabs(get(t, got, i + size_t(j) * c.ld) - ref) /
                                                    (3 * u * (std::fabs(x) + std::fabs(y)) + tiny(t)));
                            ++tl.n;
                            if (!(r <= 1)) ++bad;
                            if (r > tl.worst || std::isnan(r)) { tl.worst = r; tl.worst_where = std::string(1, ta) + tb; }
                        }
                    for (int j = 0; j < n; ++j) // padding rows untouched
                        for (int i = m; i < c.ld; ++i) bad += !same_bits(t, got, c.h, i + size_t(j) * c.ld);
                    ++g_checks;
                    if (bad) { ++g_fail; tl.bad += bad; std::printf("FAIL %s %c%c m%d n%d a%g b%g: %d\n", group, ta, tb, m, n, sc[0], sc[1], bad); }
                }
    // In place: C = A (op N, same ld), and C = B.
    const int m = 9, n = 6;
    mat a(t, m, n, m + 1), b(t, m, n, m + 1);
    a.fill(); b.fill(); a.upload(); b.upload();
    const T al = T(2), be = T(-1);
    BLAS_OK(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, (const T *)a.d, a.ld, &be, (const T *)b.d, b.ld, (T *)a.d, a.ld));
    std::vector<uint8_t> ga = a.download();
    BLAS_OK(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, (const T *)a.d, a.ld, &be, (const T *)b.d, b.ld, (T *)b.d, b.ld));
    std::vector<uint8_t> gb = b.download();
    int bad = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            const size_t k = i + size_t(j) * a.ld;
            const double c1 = 2 * a.at(i, j) - b.at(i, j), c2 = 2 * c1 - b.at(i, j);
            bad += std::fabs(get(t, ga, k) - c1) > 4 * u * (std::fabs(c1) + 1);
            bad += std::fabs(get(t, gb, k) - c2) > 8 * u * (std::fabs(c2) + 1);
        }
    CHECK(bad == 0);
    // Device pointer mode for the scalars.
    T *dsc;
    CUDA_OK(cudaMalloc(&dsc, 2 * sizeof(T)));
    const T hs[2] = {T(0.5), T(0.25)};
    CUDA_OK(cudaMemcpy(dsc, hs, sizeof hs, cudaMemcpyHostToDevice));
    mat x(t, 4, 3, 4), y(t, 4, 3, 4), z(t, 4, 3, 4);
    x.fill(); y.fill(); x.upload(); y.upload();
    BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_DEVICE));
    BLAS_OK(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, 4, 3, dsc, (const T *)x.d, 4, dsc + 1, (const T *)y.d, 4, (T *)z.d, 4));
    BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_HOST));
    std::vector<uint8_t> gz = z.download();
    bad = 0;
    for (int i = 0; i < 12; ++i) {
        const double want = 0.5 * get(t, x.h, i) + 0.25 * get(t, y.h, i);
        bad += std::fabs(get(t, gz, i) - want) > 3 * u * (std::fabs(want) + 1);
    }
    CHECK(bad == 0);
    cudaFree(dsc);
    // Status codes.
    BLAS_IS(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, -1, 3, &al, (const T *)x.d, 4, &be, (const T *)y.d, 4, (T *)z.d, 4), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, 4, 3, &al, (const T *)x.d, 3, &be, (const T *)y.d, 4, (T *)z.d, 4), CUBLAS_STATUS_INVALID_VALUE); // lda < m
    BLAS_IS(geam<T>(CUBLAS_OP_T, CUBLAS_OP_N, 4, 5, &al, (const T *)x.d, 4, &be, (const T *)y.d, 4, (T *)z.d, 4), CUBLAS_STATUS_INVALID_VALUE); // lda < n for op T
    BLAS_IS(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, 4, 3, &al, (const T *)x.d, 4, &be, (const T *)y.d, 4, (T *)z.d, 3), CUBLAS_STATUS_INVALID_VALUE); // ldc < m
    BLAS_OK(geam<T>(CUBLAS_OP_N, CUBLAS_OP_N, 0, 3, &al, (const T *)x.d, 4, &be, (const T *)y.d, 4, (T *)z.d, 4));
}

static void test_geam()
{
    run_geam<float>(ty::f32, "sgeam");
#ifndef OCU_CUBLAS_NO_FP64
    run_geam<double>(ty::f64, "dgeam");
#endif
}

// ---- getrfBatched: P A = L U, partial pivoting ----------------------------

static void test_getrf_batched()
{
    tally &tl = g_tally["sgetrf_batched"];
    const int count = 3;
    for (int n : {1, 2, 5, 32, 33})
        for (bool pivot : {true, false}) {
            const int lda = n + 1;
            std::vector<mat *> as;
            std::vector<void *> pa;
            for (int q = 0; q < count; ++q) {
                mat *a = new mat(ty::f32, n, n, lda, 3.5);
                a->fill();
                if (!pivot) // no pivoting: make it diagonally dominant
                    for (int i = 0; i < n; ++i) put(ty::f32, a->h, i + size_t(i) * lda, float(n + 2.0 + rnd(ty::f32)));
                a->upload();
                as.push_back(a);
                pa.push_back(a->d);
            }
            void **dpa;
            int *dpiv, *dinfo;
            CUDA_OK(cudaMalloc(&dpa, count * sizeof(void *)));
            CUDA_OK(cudaMalloc(&dpiv, count * n * sizeof(int)));
            CUDA_OK(cudaMalloc(&dinfo, count * sizeof(int)));
            CUDA_OK(cudaMemcpy(dpa, pa.data(), count * sizeof(void *), cudaMemcpyHostToDevice));
            CUDA_OK(cudaMemset(dinfo, 0x55, count * sizeof(int)));
            BLAS_OK(cublasSgetrfBatched(H, n, (float *const *)dpa, lda, pivot ? dpiv : nullptr, dinfo, count));
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<int> piv(count * n, 0), info(count);
            if (pivot) CUDA_OK(cudaMemcpy(piv.data(), dpiv, piv.size() * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(info.data(), dinfo, info.size() * sizeof(int), cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int q = 0; q < count; ++q) {
                bad += info[q] != 0;
                std::vector<uint8_t> got = as[q]->download();
                auto F = [&](int i, int j) -> ld { return get(ty::f32, got, i + size_t(j) * lda); };
                // P A: apply the interchanges in order to the original rows.
                std::vector<int> row(n);
                for (int i = 0; i < n; ++i) row[i] = i;
                for (int i = 0; i < n && pivot; ++i) {
                    const int p = piv[q * n + i] - 1;
                    if (p < i || p >= n) { ++bad; continue; } // 1-based, never above the current row
                    std::swap(row[i], row[p]);
                }
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        ld s = 0, as_ = 0;
                        for (int k = 0; k <= (i < j ? i : j); ++k) {
                            const ld l = k == i ? 1 : F(i, k), uu = F(k, j);
                            s += l * uu; as_ += std::fabs(l * uu);
                        }
                        const ld ref = as[q]->at(row[i], j);
                        const double r = double(std::fabs(s - ref) / (2 * (n + 1) * 0x1p-24 * (as_ + std::fabs(ref)) + 0x1p-149));
                        ++tl.n;
                        if (!(r <= 1)) ++bad;
                        if (r > tl.worst || std::isnan(r)) { tl.worst = r; tl.worst_where = "n" + std::to_string(n); }
                        if (pivot && i > j && !(std::fabs(F(i, j)) <= 1)) ++bad; // partial pivoting: |L| <= 1
                    }
                for (int j = 0; j < n; ++j) // padding row untouched
                    bad += !same_bits(ty::f32, got, as[q]->h, n + size_t(j) * lda);
            }
            ++g_checks;
            if (bad) { ++g_fail; tl.bad += bad; std::printf("FAIL sgetrf_batched n%d pivot%d: %d\n", n, pivot, bad); }
            cudaFree(dpa); cudaFree(dpiv); cudaFree(dinfo);
            for (auto *a : as) delete a;
        }
    // Singular: column 2 (1-based 3) is zero below and on the diagonal after
    // elimination -> info = 3; a zero first column -> info = 1.
    for (int zero_col : {0, 2}) {
        const int n = 4;
        mat a(ty::f32, n, n, n);
        a.fill();
        for (int i = 0; i < n; ++i) put(ty::f32, a.h, i + size_t(zero_col) * n, zero_col == 0 ? 0.0 : a.at(i, 0));
        if (zero_col == 2) // column 2 = column 0: dependent
            for (int i = 0; i < n; ++i) put(ty::f32, a.h, i + size_t(2) * n, a.at(i, 0));
        a.upload();
        void *pa = a.d, **dpa;
        int *dpiv, *dinfo, info = -1;
        CUDA_OK(cudaMalloc(&dpa, sizeof(void *)));
        CUDA_OK(cudaMalloc(&dpiv, n * sizeof(int)));
        CUDA_OK(cudaMalloc(&dinfo, sizeof(int)));
        CUDA_OK(cudaMemcpy(dpa, &pa, sizeof(void *), cudaMemcpyHostToDevice));
        BLAS_OK(cublasSgetrfBatched(H, n, (float *const *)dpa, n, dpiv, dinfo, 1));
        CUDA_OK(cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (zero_col == 0) CHECK(info == 1);
        else CHECK(info == 3 || info == 0); // exact zero only if the rounding cancels exactly
        cudaFree(dpa); cudaFree(dpiv); cudaFree(dinfo);
    }
    int *dinfo;
    CUDA_OK(cudaMalloc(&dinfo, sizeof(int)));
    BLAS_IS(cublasSgetrfBatched(H, -1, nullptr, 1, nullptr, dinfo, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSgetrfBatched(H, 4, nullptr, 3, nullptr, dinfo, 1), CUBLAS_STATUS_INVALID_VALUE); // lda < n
    BLAS_IS(cublasSgetrfBatched(H, 4, nullptr, 4, nullptr, dinfo, -1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_OK(cublasSgetrfBatched(H, 4, nullptr, 4, nullptr, dinfo, 0));
    cudaFree(dinfo);
}

// ---- Set/Get Matrix and Vector --------------------------------------------

static void test_set_get()
{
    int bad = 0;
    for (int es : {1, 2, 4, 8}) {
        const int rows = 5, cols = 3, lda = 7, ldb = 6, ldc = 8;
        std::vector<uint8_t> A(size_t(lda) * cols * es), C(size_t(ldc) * cols * es, 0xEE), dev_seen(size_t(ldb) * cols * es);
        for (size_t i = 0; i < A.size(); ++i) A[i] = uint8_t(g_rng());
        uint8_t *dB;
        CUDA_OK(cudaMalloc(&dB, dev_seen.size()));
        CUDA_OK(cudaMemset(dB, 0xCD, dev_seen.size()));
        BLAS_OK(cublasSetMatrix(rows, cols, es, A.data(), lda, dB, ldb));
        CUDA_OK(cudaMemcpy(dev_seen.data(), dB, dev_seen.size(), cudaMemcpyDeviceToHost));
        BLAS_OK(cublasGetMatrix(rows, cols, es, dB, ldb, C.data(), ldc));
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < ldc; ++i)
                for (int b = 0; b < es; ++b) {
                    const uint8_t in = A[(i + size_t(j) * lda) * es + b];
                    if (i < ldb) bad += dev_seen[(i + size_t(j) * ldb) * es + b] != (i < rows ? in : 0xCD);
                    bad += C[(i + size_t(j) * ldc) * es + b] != (i < rows ? in : 0xEE);
                }
        // Vectors: n elements at strides incx (host) and incy (device).
        const int n = 6, incx = 2, incy = 3;
        std::vector<uint8_t> x(size_t(n) * incx * es), y(size_t(n) * incx * es, 0x11), dy_seen(size_t(n) * incy * es);
        for (auto &v : x) v = uint8_t(g_rng());
        uint8_t *dy;
        CUDA_OK(cudaMalloc(&dy, dy_seen.size()));
        CUDA_OK(cudaMemset(dy, 0xAB, dy_seen.size()));
        cudaStream_t st;
        CUDA_OK(cudaStreamCreate(&st));
        BLAS_OK(cublasSetVectorAsync(n, es, x.data(), incx, dy, incy, st));
        CUDA_OK(cudaStreamSynchronize(st));
        CUDA_OK(cudaMemcpy(dy_seen.data(), dy, dy_seen.size(), cudaMemcpyDeviceToHost));
        BLAS_OK(cublasGetVector(n, es, dy, incy, y.data(), incx));
        for (int i = 0; i < n * incy; ++i)
            for (int b = 0; b < es; ++b)
                bad += dy_seen[size_t(i) * es + b] != (i % incy == 0 ? x[size_t(i / incy) * incx * es + b] : 0xAB);
        for (int i = 0; i < n * incx; ++i)
            for (int b = 0; b < es; ++b)
                bad += y[size_t(i) * es + b] != (i % incx == 0 ? x[size_t(i) * es + b] : 0x11);
        // Async matrix copies, then the plain vector set.
        std::vector<uint8_t> C2(C.size(), 0xEE);
        BLAS_OK(cublasSetMatrixAsync(rows, cols, es, A.data(), lda, dB, ldb, st));
        BLAS_OK(cublasGetMatrixAsync(rows, cols, es, dB, ldb, C2.data(), ldc, st));
        CUDA_OK(cudaStreamSynchronize(st));
        bad += C2 != C;
        std::vector<uint8_t> y2(y.size(), 0x11);
        BLAS_OK(cublasSetVector(n, es, x.data(), incx, dy, incy));
        BLAS_OK(cublasGetVectorAsync(n, es, dy, incy, y2.data(), incx, st));
        CUDA_OK(cudaStreamSynchronize(st));
        bad += y2 != y;
        cudaStreamDestroy(st);
        cudaFree(dB); cudaFree(dy);
    }
    CHECK(bad == 0);
    float h[4] = {}, *d;
    CUDA_OK(cudaMalloc(&d, sizeof h));
    BLAS_IS(cublasSetMatrix(-1, 1, 4, h, 1, d, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSetMatrix(1, 1, 0, h, 1, d, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasGetMatrix(1, 1, 4, d, 0, h, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_IS(cublasSetVector(2, 4, h, 0, d, 1), CUBLAS_STATUS_INVALID_VALUE);
    BLAS_OK(cublasSetMatrix(0, 3, 4, h, 1, d, 1));
    cudaFree(d);
}

// ---- 8-bit integer GEMM ---------------------------------------------------
// cublasGemmEx with A, B CUDA_R_8I, C CUDA_R_32I, CUBLAS_COMPUTE_32I (and the
// older CUDA_R_32I compute value): int32 alpha and beta, exact int32 sums.
// Exact comparison with a host reference in int64 (inputs and k keep every
// value inside int32). Shapes off the tile sizes, the four transposes, beta
// 0 (C holds garbage and must not be read) and not, alpha 0 (A and B are not
// needed), padded leading dimensions (the padding must stay untouched),
// device pointer mode, and a strided batch.
static void test_gemm_i8()
{
    std::mt19937 rng(77);
    const int shapes[][3] = {{1, 1, 1}, {7, 5, 3}, {64, 64, 16}, {65, 130, 33}, {129, 63, 257}, {200, 17, 1000}, {3, 300, 64}};
    const char tr[][2] = {{'N', 'N'}, {'T', 'N'}, {'N', 'T'}, {'T', 'T'}};
    int32_t *dscal = nullptr;
    CUDA_OK(cudaMalloc(&dscal, 2 * sizeof(int32_t)));
    long long bad_total = 0, n_total = 0;
    for (auto &sh : shapes)
        for (auto &t : tr)
            for (int variant = 0; variant < 4; ++variant) {
                const int m = sh[0], n = sh[1], k = sh[2];
                const bool ta = t[0] == 'T', tb = t[1] == 'T';
                const int arows = ta ? k : m, acols = ta ? m : k, brows = tb ? n : k, bcols = tb ? k : n;
                const int lda = arows + 3, ldb = brows + 1, ldc = m + 2;
                const int32_t alpha = variant == 2 ? 0 : (variant == 3 ? -3 : 2), beta = variant == 0 ? 0 : 5;
                const int compute = variant == 1 ? (int)CUDA_R_32I : (int)CUBLAS_COMPUTE_32I;
                std::vector<int8_t> A((size_t)lda * acols), B((size_t)ldb * bcols);
                std::vector<int32_t> C((size_t)ldc * n), C0;
                for (auto &v : A) v = (int8_t)((int)(rng() % 256) - 128);
                for (auto &v : B) v = (int8_t)((int)(rng() % 256) - 128);
                for (auto &v : C) v = (int32_t)(rng() % 2001) - 1000;
                if (beta == 0) for (auto &v : C) v = (int32_t)0x7f7f7f7f; // garbage that must not be read
                C0 = C;
                int8_t *dA, *dB; int32_t *dC;
                CUDA_OK(cudaMalloc(&dA, A.size())); CUDA_OK(cudaMalloc(&dB, B.size())); CUDA_OK(cudaMalloc(&dC, C.size() * 4));
                CUDA_OK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
                const bool devmode = (variant == 3);
                if (devmode) {
                    int32_t s[2] = {alpha, beta};
                    CUDA_OK(cudaMemcpy(dscal, s, sizeof s, cudaMemcpyHostToDevice));
                    BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_DEVICE));
                }
                BLAS_OK(cublasGemmEx(H, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k,
                                     devmode ? (const void *)dscal : (const void *)&alpha, dA, CUDA_R_8I, lda, dB, CUDA_R_8I, ldb,
                                     devmode ? (const void *)(dscal + 1) : (const void *)&beta, dC, CUDA_R_32I, ldc,
                                     (cublasComputeType_t)compute, CUBLAS_GEMM_DEFAULT));
                if (devmode) BLAS_OK(cublasSetPointerMode(H, CUBLAS_POINTER_MODE_HOST));
                CUDA_OK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
                long long bad = 0;
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < m; ++i) {
                        long long s = 0;
                        for (int p = 0; p < k; ++p) {
                            int a = ta ? A[(size_t)p + (size_t)i * lda] : A[(size_t)i + (size_t)p * lda];
                            int b = tb ? B[(size_t)j + (size_t)p * ldb] : B[(size_t)p + (size_t)j * ldb];
                            s += (long long)a * b;
                        }
                        long long want = (long long)alpha * s + (beta ? (long long)beta * C0[(size_t)i + (size_t)j * ldc] : 0);
                        bad += C[(size_t)i + (size_t)j * ldc] != (int32_t)want;
                    }
                    for (int i = m; i < ldc; ++i) bad += C[(size_t)i + (size_t)j * ldc] != C0[(size_t)i + (size_t)j * ldc]; // padding
                }
                ++g_checks;
                n_total += (long long)m * n;
                if (bad) { ++g_fail; std::printf("FAIL gemm i8 %dx%dx%d %c%c variant %d: %lld wrong\n", m, n, k, t[0], t[1], variant, bad); }
                bad_total += bad;
                CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
            }
    // a strided batch of 5, B shared (stride 0)
    {
        const int m = 33, n = 20, k = 70, batch = 5;
        const long long sa = (long long)m * k + 11, sc = (long long)m * n + 7;
        std::vector<int8_t> A((size_t)sa * batch), B((size_t)k * n);
        std::vector<int32_t> C((size_t)sc * batch, 0);
        for (auto &v : A) v = (int8_t)((int)(rng() % 256) - 128);
        for (auto &v : B) v = (int8_t)((int)(rng() % 256) - 128);
        int8_t *dA, *dB; int32_t *dC;
        CUDA_OK(cudaMalloc(&dA, A.size())); CUDA_OK(cudaMalloc(&dB, B.size())); CUDA_OK(cudaMalloc(&dC, C.size() * 4));
        CUDA_OK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
        const int32_t one = 1, zero = 0;
        BLAS_OK(cublasGemmStridedBatchedEx(H, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, dA, CUDA_R_8I, m, sa, dB, CUDA_R_8I, k, 0,
                                           &zero, dC, CUDA_R_32I, m, sc, batch, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        CUDA_OK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (int q = 0; q < batch; ++q)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    long long s = 0;
                    for (int p = 0; p < k; ++p) s += (long long)A[(size_t)q * sa + i + (size_t)p * m] * B[(size_t)p + (size_t)j * k];
                    bad += C[(size_t)q * sc + i + (size_t)j * m] != (int32_t)s;
                }
        ++g_checks;
        if (bad) { ++g_fail; std::printf("FAIL gemm i8 strided batch: %lld wrong\n", bad); }
        CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
    }
    CUDA_OK(cudaFree(dscal));
    std::printf("gemm i8: %lld results exact checks, %lld wrong\n", n_total, bad_total);
}

int main()
{
    BLAS_OK(cublasCreate(&H));
    test_api_rules();
    test_gemm_i8();
    test_sgemm_dgemm_hgemm();
    test_gemm_ex();
    test_batched();
    test_device_pointer_mode();
    test_streams();
    test_level1_2();
    test_trsm_batched();
    test_geam();
    test_getrf_batched();
    test_set_get();
    BLAS_OK(cublasDestroy(H));

    std::printf("per group: elements checked, failures, worst error / bound\n");
    for (auto &kv : g_tally) {
        std::printf("  %-26s %7d %3d  %.3f%s%s\n", kv.first.c_str(), kv.second.n, kv.second.bad,
                    kv.second.worst, kv.second.worst_where.empty() ? "" : "  at ", kv.second.worst_where.c_str());
    }
    std::printf("samples:\n");
    int shown = 0;
    for (auto &kv : g_tally)
        if (!kv.second.sample.empty() && shown++ < 6) std::printf("  %s: %s\n", kv.first.c_str(), kv.second.sample.c_str());
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
