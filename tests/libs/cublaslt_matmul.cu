// cublasLtMatmul against a double-precision host reference: D = alpha op(A)
// op(B) + beta C for column- and row-major layouts, the four transpose
// combinations, C and D at different addresses (C unchanged), strided
// batches, FP32 / FP16 / BF16 storage with FP32 compute (FP16 / BF16 D
// within one rounding of the reference), device pointer mode, and the
// epilogues RELU, BIAS, RELU_BIAS, GELU (tanh form), GELU_BIAS with the
// bias one value per row of D (FP32 bias on FP16 D through BIAS_DATA_TYPE).
// Also the object API: attribute get/set round trips and sizes, the
// heuristic (one algorithm, no workspace), and the documented errors
// (wrong dimensions, missing bias, a scale pointer: not supported).
//   ocu-nvcc [--ocu-target=spirv] tests/libs/cublaslt_matmul.cu -lcublasLt -o cublaslt_matmul
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 25) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

static unsigned g_seed = 12345;
static float rnd() { g_seed = g_seed * 1103515245u + 12345u; return (float)((g_seed >> 9) & 0x3fff) / 8192.0f - 1.0f; }

static uint16_t f2bf(float f) { uint32_t x; std::memcpy(&x, &f, 4); return (uint16_t)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16); }
static float bf2f(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; std::memcpy(&f, &x, 4); return f; }

enum Kind { F32, F16, BF16 };
static size_t esize(Kind k) { return k == F32 ? 4 : 2; }
static cudaDataType_t dtype(Kind k) { return k == F32 ? CUDA_R_32F : (k == F16 ? CUDA_R_16F : CUDA_R_16BF); }
// value stored as kind k (rounded), returned as float; and packing
static float quant(Kind k, float v) { return k == F32 ? v : (k == F16 ? __half2float(__float2half(v)) : bf2f(f2bf(v))); }
static void put(Kind k, std::vector<unsigned char> &buf, size_t i, float v)
{
    if (k == F32) std::memcpy(&buf[i * 4], &v, 4);
    else if (k == F16) { __half h = __float2half(v); std::memcpy(&buf[i * 2], &h, 2); }
    else { uint16_t b = f2bf(v); std::memcpy(&buf[i * 2], &b, 2); }
}
static float get(Kind k, const std::vector<unsigned char> &buf, size_t i)
{
    if (k == F32) { float v; std::memcpy(&v, &buf[i * 4], 4); return v; }
    if (k == F16) { __half h; std::memcpy(&h, &buf[i * 2], 2); return __half2float(h); }
    uint16_t b; std::memcpy(&b, &buf[i * 2], 2); return bf2f(b);
}

struct Case {
    const char *name;
    Kind kind;
    bool row, ta, tb, separate_c, device_scalars;
    int m, n, k, batch;
    cublasLtEpilogue_t epi;
    bool f32_bias;
};

static double gelu(double x) { return 0.5 * x * (1 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

static void run(cublasLtHandle_t lt, const Case &c)
{
    const int m = c.m, n = c.n, k = c.k, nb = c.batch;
    const int ar = c.ta ? k : m, ac = c.ta ? m : k, br = c.tb ? n : k, bc = c.tb ? k : n;
    // leading dimensions with padding; element (r, col) at col-major r + col*ld, row-major r*ld + col
    const int lda = (c.row ? ac : ar) + 3, ldb = (c.row ? bc : br) + 2, ldc = (c.row ? n : m) + 1, ldd = (c.row ? n : m) + 4;
    const long long sa = (long long)lda * (c.row ? ar : ac) + 7, sb = (long long)ldb * (c.row ? br : bc) + 5;
    const long long sc = (long long)ldc * (c.row ? m : n) + 3, sd = (long long)ldd * (c.row ? m : n) + 9;
    auto idx = [&](int r, int col, int ld) { return c.row ? (size_t)r * ld + col : (size_t)r + (size_t)col * ld; };
    std::vector<unsigned char> A(sa * nb * esize(c.kind)), B(sb * nb * esize(c.kind)), C(sc * nb * esize(c.kind)),
        D(sd * nb * esize(c.kind));
    std::vector<float> fa(sa * nb), fb(sb * nb), fc(sc * nb);
    for (size_t i = 0; i < fa.size(); ++i) { fa[i] = quant(c.kind, rnd()); put(c.kind, A, i, fa[i]); }
    for (size_t i = 0; i < fb.size(); ++i) { fb[i] = quant(c.kind, rnd()); put(c.kind, B, i, fb[i]); }
    for (size_t i = 0; i < fc.size(); ++i) { fc[i] = quant(c.kind, rnd()); put(c.kind, C, i, fc[i]); }
    std::vector<float> bias(m * nb);
    for (auto &b : bias) b = quant(c.f32_bias ? F32 : c.kind, rnd());
    std::vector<unsigned char> biasbuf(bias.size() * (c.f32_bias ? 4 : esize(c.kind)));
    for (size_t i = 0; i < bias.size(); ++i) put(c.f32_bias ? F32 : c.kind, biasbuf, i, bias[i]);
    const float alpha = 0.75f, beta = c.epi == CUBLASLT_EPILOGUE_DEFAULT ? -0.5f : 0.25f;

    void *dA, *dB, *dC, *dD, *dbias, *dscal;
    cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dC, C.size()); cudaMalloc(&dD, D.size());
    cudaMalloc(&dbias, biasbuf.size()); cudaMalloc(&dscal, 8);
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
    if (!c.separate_c) cudaMemcpy(dD, C.data(), std::min(C.size(), D.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(dbias, biasbuf.data(), biasbuf.size(), cudaMemcpyHostToDevice);
    float scal[2] = {alpha, beta};
    cudaMemcpy(dscal, scal, 8, cudaMemcpyHostToDevice);

    cublasLtMatmulDesc_t op;
    CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS, "%s desc", c.name);
    cublasOperation_t ta = c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb = c.tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    if (c.device_scalars) {
        cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof pm);
    }
    cublasLtEpilogue_t epi = c.epi;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
    const bool has_bias = c.epi == CUBLASLT_EPILOGUE_BIAS || c.epi == CUBLASLT_EPILOGUE_RELU_BIAS || c.epi == CUBLASLT_EPILOGUE_GELU_BIAS;
    if (has_bias) {
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias, sizeof dbias);
        long long bstride = m;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &bstride, sizeof bstride);
        if (c.f32_bias) {
            cudaDataType_t bt = CUDA_R_32F;
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof bt);
        }
    }
    cublasLtMatrixLayout_t la, lb, lc, ld;
    cublasLtMatrixLayoutCreate(&la, dtype(c.kind), ar, ac, lda);
    cublasLtMatrixLayoutCreate(&lb, dtype(c.kind), br, bc, ldb);
    cublasLtMatrixLayoutCreate(&lc, dtype(c.kind), m, n, ldc);
    cublasLtMatrixLayoutCreate(&ld, dtype(c.kind), m, n, c.separate_c ? ldd : ldc);
    const long long strides[4] = {sa, sb, sc, c.separate_c ? sd : sc};
    cublasLtMatrixLayout_t ls[4] = {la, lb, lc, ld};
    for (int i = 0; i < 4; ++i) {
        cublasLtOrder_t ord = c.row ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL;
        cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_ORDER, &ord, sizeof ord);
        cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &nb, sizeof nb);
        cublasLtMatrixLayoutSetAttribute(ls[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof strides[i]);
    }
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 1 << 20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    cublasLtMatmulHeuristicResult_t heur[4];
    int got = 0;
    CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, ld, pref, 4, heur, &got) == CUBLAS_STATUS_SUCCESS && got == 1 &&
          heur[0].workspaceSize == 0 && heur[0].state == CUBLAS_STATUS_SUCCESS, "%s heuristic (%d)", c.name, got);
    void *Dout = c.separate_c ? dD : dC;
    const void *pa = c.device_scalars ? dscal : (const void *)&alpha, *pb = c.device_scalars ? (const void *)((char *)dscal + 4) : (const void *)&beta;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cublasStatus_t st = cublasLtMatmul(lt, op, pa, dA, la, dB, lb, pb, dC, lc, Dout, ld, &heur[0].algo, nullptr, 0, s);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "%s matmul status %d", c.name, (int)st);
    cudaStreamSynchronize(s);
    std::vector<unsigned char> out(c.separate_c ? D.size() : C.size()), cback(C.size());
    cudaMemcpy(out.data(), Dout, out.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(cback.data(), dC, C.size(), cudaMemcpyDeviceToHost);
    const int ldo = c.separate_c ? ldd : ldc;
    const long long so = c.separate_c ? sd : sc;
    int bad = 0;
    double worst = 0;
    for (int b = 0; b < nb; ++b)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double acc = 0;
                for (int p = 0; p < k; ++p) {
                    double x = c.ta ? fa[b * sa + idx(p, i, lda)] : fa[b * sa + idx(i, p, lda)];
                    double y = c.tb ? fb[b * sb + idx(j, p, ldb)] : fb[b * sb + idx(p, j, ldb)];
                    acc += x * y;
                }
                double want = alpha * acc + beta * fc[b * sc + idx(i, j, ldc)];
                if (has_bias) want += bias[b * m + i];
                if (c.epi == CUBLASLT_EPILOGUE_RELU || c.epi == CUBLASLT_EPILOGUE_RELU_BIAS) want = want > 0 ? want : 0;
                if (c.epi == CUBLASLT_EPILOGUE_GELU || c.epi == CUBLASLT_EPILOGUE_GELU_BIAS) want = gelu(want);
                float v = get(c.kind, out, b * so + idx(i, j, ldo));
                // FP32: a k-term FP32 sum; FP16 / BF16 D: also one rounding of D (and of the pre-epilogue value)
                double tol = 1e-5 * (k + 4) + (c.kind == F16 ? 2.0 / 1024 : (c.kind == BF16 ? 2.0 / 128 : 0)) * (std::fabs(want) + 1);
                double err = std::fabs(v - want);
                if (err > worst) worst = err;
                if (!(err <= tol) && bad++ < 2) std::printf("  %s batch %d (%d,%d): %g want %g\n", c.name, b, i, j, v, want);
            }
    CHECK(bad == 0, "%s: %d wrong (worst error %g)", c.name, bad, worst);
    if (c.separate_c) CHECK(std::memcmp(cback.data(), C.data(), C.size()) == 0, "%s: C changed", c.name);
    cublasLtMatmulPreferenceDestroy(pref);
    for (auto l : ls) cublasLtMatrixLayoutDestroy(l);
    cublasLtMatmulDescDestroy(op);
    cudaStreamDestroy(s);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dbias); cudaFree(dscal);
}

int main()
{
    cublasLtHandle_t lt;
    CHECK(cublasLtCreate(&lt) == CUBLAS_STATUS_SUCCESS, "create");
    const Case cases[] = {
        {"f32 col NN", F32, false, false, false, false, false, 37, 29, 41, 1, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 col TN", F32, false, true, false, true, false, 64, 48, 33, 1, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 col NT", F32, false, false, true, true, false, 20, 70, 16, 3, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 col TT batched", F32, false, true, true, true, false, 33, 17, 50, 4, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 row NN", F32, true, false, false, true, false, 45, 31, 27, 2, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 row TN", F32, true, true, false, true, false, 19, 64, 40, 1, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 row NT batched", F32, true, false, true, false, false, 32, 32, 32, 3, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 device scalars", F32, false, false, true, true, true, 40, 24, 36, 2, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f32 relu", F32, false, false, false, true, false, 50, 30, 20, 1, CUBLASLT_EPILOGUE_RELU, false},
        {"f32 bias", F32, false, true, false, true, false, 50, 30, 20, 2, CUBLASLT_EPILOGUE_BIAS, false},
        {"f32 relu_bias row", F32, true, false, true, true, false, 28, 44, 24, 2, CUBLASLT_EPILOGUE_RELU_BIAS, false},
        {"f32 gelu", F32, false, false, false, true, false, 33, 33, 33, 1, CUBLASLT_EPILOGUE_GELU, false},
        {"f32 gelu_bias", F32, false, false, true, true, false, 64, 16, 48, 2, CUBLASLT_EPILOGUE_GELU_BIAS, false},
        {"f16 col NN", F16, false, false, false, true, false, 64, 64, 64, 1, CUBLASLT_EPILOGUE_DEFAULT, false},
        {"f16 row TN bias(f32)", F16, true, true, false, true, false, 48, 40, 32, 2, CUBLASLT_EPILOGUE_BIAS, true},
        {"f16 gelu_bias", F16, false, true, true, true, false, 31, 47, 29, 1, CUBLASLT_EPILOGUE_GELU_BIAS, false},
        {"bf16 col NT relu", BF16, false, false, true, true, false, 40, 56, 64, 2, CUBLASLT_EPILOGUE_RELU, false},
        {"bf16 row NN bias", BF16, true, false, false, true, false, 36, 28, 40, 1, CUBLASLT_EPILOGUE_BIAS, false},
    };
    for (const Case &c : cases) run(lt, c);

    // objects in the caller's storage (the Init functions), as programs that
    // keep them on the stack use them: D = A B, 24 x 20 x 16, column-major
    {
        const int m = 24, n = 20, k = 16;
        std::vector<float> a(m * k), b(k * n), d(m * n, -7.0f);
        for (auto &x : a) x = rnd();
        for (auto &x : b) x = rnd();
        float *da, *db, *dd;
        cudaMalloc(&da, a.size() * 4); cudaMalloc(&db, b.size() * 4); cudaMalloc(&dd, d.size() * 4);
        cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice);
        cublasLtMatmulDescOpaque_t opq = {};
        cublasLtMatrixLayoutOpaque_t aq = {}, bq = {}, dq = {};
        cublasLtMatmulPreferenceOpaque_t pq = {};
        CHECK(cublasLtMatmulDescInit(&opq, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutInit(&aq, CUDA_R_32F, m, k, m) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutInit(&bq, CUDA_R_32F, k, n, k) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutInit(&dq, CUDA_R_32F, m, n, m) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulPreferenceInit(&pq) == CUBLAS_STATUS_SUCCESS, "Init functions");
        cublasLtMatmulHeuristicResult_t h;
        int nh = 0;
        CHECK(cublasLtMatmulAlgoGetHeuristic(lt, &opq, &aq, &bq, &dq, &dq, &pq, 1, &h, &nh) == CUBLAS_STATUS_SUCCESS && nh == 1, "Init heuristic");
        float one = 1, zero = 0;
        CHECK(cublasLtMatmul(lt, &opq, &one, da, &aq, db, &bq, &zero, dd, &dq, dd, &dq, &h.algo, nullptr, 0, 0) == CUBLAS_STATUS_SUCCESS,
              "Init matmul");
        cudaMemcpy(d.data(), dd, d.size() * 4, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double w = 0;
                for (int p = 0; p < k; ++p) w += (double)a[i + p * m] * b[p + j * k];
                bad += std::fabs(d[i + j * m] - w) > 1e-4;
            }
        CHECK(bad == 0, "Init objects: %d wrong", bad);
        cudaFree(da); cudaFree(db); cudaFree(dd);
    }

    // object API: get / set round trips, sizes, errors
    cublasLtMatmulDesc_t op;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t t = CUBLAS_OP_T, back = CUBLAS_OP_N;
    size_t written = 0;
    CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t) == CUBLAS_STATUS_SUCCESS, "set transa");
    CHECK(cublasLtMatmulDescGetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &back, sizeof back, &written) == CUBLAS_STATUS_SUCCESS &&
          back == CUBLAS_OP_T && written == sizeof back, "get transa");
    CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, 2) == CUBLAS_STATUS_INVALID_VALUE, "wrong size");
    cublasLtMatrixLayout_t l;
    cublasLtMatrixLayoutCreate(&l, CUDA_R_32F, 7, 9, 11);
    uint64_t rows = 0;
    CHECK(cublasLtMatrixLayoutGetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof rows, &written) == CUBLAS_STATUS_SUCCESS &&
          rows == 7 && written == 8, "layout rows");
    // dimensions that do not fit: A 7x9 times B 7x9 into 7x9
    float one = 1, zero = 0;
    void *dm;
    cudaMalloc(&dm, 1024 * sizeof(float));
    CHECK(cublasLtMatmul(lt, op, &one, dm, l, dm, l, &zero, dm, l, dm, l, nullptr, nullptr, 0, 0) == CUBLAS_STATUS_INVALID_VALUE,
          "mismatched dimensions");
    cublasOperation_t nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nn, sizeof nn);
    cublasLtMatrixLayout_t sq;
    cublasLtMatrixLayoutCreate(&sq, CUDA_R_32F, 8, 8, 8);
    cublasLtEpilogue_t eb = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &eb, sizeof eb);
    CHECK(cublasLtMatmul(lt, op, &one, dm, sq, dm, sq, &zero, dm, sq, dm, sq, nullptr, nullptr, 0, 0) == CUBLAS_STATUS_INVALID_VALUE,
          "bias epilogue without a bias");
    void *scale = dm;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale, sizeof scale);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &scale, sizeof scale);
    CHECK(cublasLtMatmul(lt, op, &one, dm, sq, dm, sq, &zero, dm, sq, dm, sq, nullptr, nullptr, 0, 0) == CUBLAS_STATUS_NOT_SUPPORTED,
          "scale pointer: not supported");
    cudaFree(dm);
    cublasLtMatrixLayoutDestroy(l);
    cublasLtMatrixLayoutDestroy(sq);
    cublasLtMatmulDescDestroy(op);
    CHECK(cublasLtDestroy(lt) == CUBLAS_STATUS_SUCCESS, "destroy");
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
