// The real-type level 1 and 2 functions against the reference BLAS loops
// computed on the host: copy, swap, asum, iamax / iamin (S and D), axpy,
// scal, dot, nrm2, gemv (D), with positive and negative increments, n = 0,
// the reference rules for incx <= 0 (asum / nrm2 / iamax give 0), iamax /
// iamin ties (first index), a NaN first (1) or later (ignored), nrm2 without
// overflow (values near 1e200) or underflow (near 1e-200), device pointer
// mode; cublasDotEx for 32F, 64F, 16F (16F or 32F result) and 16BF.
//   ocu-nvcc [--ocu-target=spirv] tests/libs/cublas_level1.cu -lcublas -o cublas_level1
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { if (g_fail++ < 25) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

static unsigned g_seed = 777;
static double rnd() { g_seed = g_seed * 1103515245u + 12345u; return (double)((g_seed >> 8) & 0xffff) / 32768.0 - 1.0; }

static long idx(long i, int n, int inc) { return inc >= 0 ? i * inc : (long)(n - 1 - i) * -inc; }
static size_t span(int n, int inc) { return n <= 0 ? 1 : (size_t)(n - 1) * (inc < 0 ? -inc : inc) + 1; }

template <class T> static T *up(const std::vector<T> &v) { T *p; cudaMalloc(&p, v.size() * sizeof(T) + 16); cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return p; }
template <class T> static std::vector<T> down(const T *p, size_t n) { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }

template <class T>
static int ref_iamax(const std::vector<T> &x, int n, int inc, bool mx)
{
    if (n <= 0 || inc <= 0) return 0;
    int r = 1;
    T best = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        T a = std::fabs(x[(size_t)i * inc]);
        if (mx ? a > best : a < best) { best = a; r = i + 1; }
    }
    return r;
}

template <class T>
static void level1(cublasHandle_t h, const char *tn)
{
    const int ns[] = {0, 1, 7, 300, 5000};
    const int incs[] = {1, 3, -2};
    for (int n : ns)
        for (int ix : incs)
            for (int iy : incs) {
                std::vector<T> x(span(n, ix)), y(span(n, iy));
                for (auto &v : x) v = (T)rnd();
                for (auto &v : y) v = (T)rnd();
                T *dx = up(x), *dy = up(y);
                // copy
                std::vector<T> want = y;
                for (int i = 0; i < n; ++i) want[idx(i, n, iy)] = x[idx(i, n, ix)];
                cublasStatus_t st = sizeof(T) == 4 ? cublasScopy(h, n, (const float *)dx, ix, (float *)dy, iy)
                                                   : cublasDcopy(h, n, (const double *)dx, ix, (double *)dy, iy);
                CHECK(st == CUBLAS_STATUS_SUCCESS && down(dy, y.size()) == want, "%scopy n=%d inc %d %d", tn, n, ix, iy);
                // swap
                cudaMemcpy(dy, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice);
                std::vector<T> wx = x, wy = y;
                for (int i = 0; i < n; ++i) std::swap(wx[idx(i, n, ix)], wy[idx(i, n, iy)]);
                st = sizeof(T) == 4 ? cublasSswap(h, n, (float *)dx, ix, (float *)dy, iy) : cublasDswap(h, n, (double *)dx, ix, (double *)dy, iy);
                CHECK(st == CUBLAS_STATUS_SUCCESS && down(dx, x.size()) == wx && down(dy, y.size()) == wy, "%sswap n=%d", tn, n);
                cudaMemcpy(dx, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice);
                cudaMemcpy(dy, y.data(), y.size() * sizeof(T), cudaMemcpyHostToDevice);
                if (ix == iy) {
                    // asum, iamax, iamin (incx > 0 only meaningful)
                    T as = 0, wantas = 0;
                    if (ix > 0) for (int i = 0; i < n; ++i) wantas += std::fabs(x[(size_t)i * ix]);
                    st = sizeof(T) == 4 ? cublasSasum(h, n, (const float *)dx, ix, (float *)&as) : cublasDasum(h, n, (const double *)dx, ix, (double *)&as);
                    CHECK(st == CUBLAS_STATUS_SUCCESS && std::fabs(as - wantas) <= 1e-5 * (std::fabs(wantas) + 1) * (sizeof(T) == 4 ? 10 : 1e-6), "%sasum n=%d inc %d: %g want %g", tn, n, ix, (double)as, (double)wantas);
                    for (int mx = 0; mx < 2; ++mx) {
                        int r = -1;
                        if (sizeof(T) == 4) st = mx ? cublasIsamax(h, n, (const float *)dx, ix, &r) : cublasIsamin(h, n, (const float *)dx, ix, &r);
                        else st = mx ? cublasIdamax(h, n, (const double *)dx, ix, &r) : cublasIdamin(h, n, (const double *)dx, ix, &r);
                        int w = ref_iamax(x, n, ix, mx);
                        CHECK(st == CUBLAS_STATUS_SUCCESS && r == w, "i%sa%s n=%d inc %d: %d want %d", tn, mx ? "max" : "min", n, ix, r, w);
                    }
                }
                cudaFree(dx);
                cudaFree(dy);
            }
    // iamax rules: ties, NaN first, NaN later
    std::vector<T> t = {(T)1, (T)-3, (T)2, (T)3, (T)-3, (T)0.5};
    T *dt = up(t);
    int r = 0;
    if (sizeof(T) == 4) cublasIsamax(h, 6, (const float *)dt, 1, &r); else cublasIdamax(h, 6, (const double *)dt, 1, &r);
    CHECK(r == 2, "%s iamax tie: %d", tn, r);
    if (sizeof(T) == 4) cublasIsamin(h, 6, (const float *)dt, 1, &r); else cublasIdamin(h, 6, (const double *)dt, 1, &r);
    CHECK(r == 6, "%s iamin: %d", tn, r);
    t[0] = (T)NAN;
    cudaMemcpy(dt, t.data(), t.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (sizeof(T) == 4) cublasIsamax(h, 6, (const float *)dt, 1, &r); else cublasIdamax(h, 6, (const double *)dt, 1, &r);
    CHECK(r == 1, "%s iamax NaN first: %d", tn, r);
    t[0] = 1; t[2] = (T)NAN;
    cudaMemcpy(dt, t.data(), t.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (sizeof(T) == 4) cublasIsamax(h, 6, (const float *)dt, 1, &r); else cublasIdamax(h, 6, (const double *)dt, 1, &r);
    CHECK(r == 2, "%s iamax NaN later: %d", tn, r);
    if (sizeof(T) == 4) cublasIsamax(h, 6, (const float *)dt, -1, &r); else cublasIdamax(h, 6, (const double *)dt, -1, &r);
    CHECK(r == 0, "%s iamax incx < 0: %d", tn, r);
    cudaFree(dt);
}

int main()
{
    cublasHandle_t h;
    CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "create");
    level1<float>(h, "S");
    level1<double>(h, "D");

    // D axpy, scal, dot, nrm2, gemv
    const int n = 3000;
    std::vector<double> x(n), y(n);
    for (auto &v : x) v = rnd();
    for (auto &v : y) v = rnd();
    double *dx = up(x), *dy = up(y);
    double a = 0.375, dotr = 0, nrm = 0;
    cublasDaxpy(h, n, &a, dx, 1, dy, 1);
    std::vector<double> got = down(dy, n);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += std::fabs(got[i] - (a * x[i] + y[i])) > 1e-15;
    CHECK(bad == 0, "Daxpy: %d wrong", bad);
    cudaMemcpy(dy, y.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    double w = 0;
    for (int i = 0; i < n; ++i) w += x[i] * y[n - 1 - i];
    CHECK(cublasDdot(h, n, dx, 1, dy, -1, &dotr) == CUBLAS_STATUS_SUCCESS && std::fabs(dotr - w) < 1e-12, "Ddot negative inc: %.17g want %.17g", dotr, w);
    std::vector<double> big(64);
    for (int i = 0; i < 64; ++i) big[i] = (i % 2 ? 1e200 : -3e200) * (1 + i / 64.0);
    double *db = up(big);
    long double ws = 0;
    for (double v : big) ws += (long double)(v / 1e200) * (v / 1e200);
    CHECK(cublasDnrm2(h, 64, db, 1, &nrm) == CUBLAS_STATUS_SUCCESS && std::isfinite(nrm) && std::fabs(nrm / (double)(std::sqrt(ws) * 1e200) - 1) < 1e-14,
          "Dnrm2 near 1e200: %g", nrm);
    for (int i = 0; i < 64; ++i) big[i] = (i % 2 ? 1e-200 : -3e-200);
    cudaMemcpy(db, big.data(), 64 * sizeof(double), cudaMemcpyHostToDevice);
    CHECK(cublasDnrm2(h, 64, db, 1, &nrm) == CUBLAS_STATUS_SUCCESS && nrm > 0 && std::fabs(nrm / (std::sqrt(32 * 1.0 + 32 * 9.0) * 1e-200) - 1) < 1e-14,
          "Dnrm2 near 1e-200: %g", nrm);
    double sc = -2;
    cublasDscal(h, n, &sc, dx, 1);
    got = down(dx, n);
    bad = 0;
    for (int i = 0; i < n; ++i) bad += got[i] != -2 * x[i];
    CHECK(bad == 0, "Dscal: %d wrong", bad);
    // device pointer mode: Ddot result and alpha in device memory
    double *dres;
    cudaMalloc(&dres, 2 * sizeof(double));
    cudaMemcpy(dres, &a, sizeof(double), cudaMemcpyHostToDevice);
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    CHECK(cublasDdot(h, n, dx, 1, dx, 1, dres + 1) == CUBLAS_STATUS_SUCCESS, "Ddot device mode");
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    double dd = 0;
    cudaMemcpy(&dd, dres + 1, sizeof(double), cudaMemcpyDeviceToHost);
    w = 0;
    for (int i = 0; i < n; ++i) w += 4 * x[i] * x[i];
    CHECK(std::fabs(dd - w) < 1e-10, "Ddot to device memory: %g want %g", dd, w);
    // Dgemv N and T, with beta
    const int m = 37, k = 53, lda = 41;
    std::vector<double> A(lda * k), vx(k > m ? k : m), vy(k > m ? k : m);
    for (auto &v : A) v = rnd();
    for (auto &v : vx) v = rnd();
    for (auto &v : vy) v = rnd();
    double *dA = up(A), *dvx = up(vx), *dvy = up(vy);
    for (int t = 0; t < 2; ++t) {
        cudaMemcpy(dvy, vy.data(), vy.size() * sizeof(double), cudaMemcpyHostToDevice);
        double al = 1.5, be = -0.5;
        CHECK(cublasDgemv(h, t ? CUBLAS_OP_T : CUBLAS_OP_N, m, k, &al, dA, lda, dvx, 1, &be, dvy, 1) == CUBLAS_STATUS_SUCCESS, "Dgemv %d", t);
        std::vector<double> g = down(dvy, vy.size());
        int rows = t ? k : m, cols = t ? m : k;
        bad = 0;
        for (int i = 0; i < rows; ++i) {
            double s = 0;
            for (int j = 0; j < cols; ++j) s += (t ? A[j + i * lda] : A[i + j * lda]) * vx[j];
            bad += std::fabs(g[i] - (al * s + be * vy[i])) > 1e-12;
        }
        CHECK(bad == 0, "Dgemv %s: %d wrong", t ? "T" : "N", bad);
    }
    double one = 1, zero = 0;
    CHECK(cublasDgemv(h, CUBLAS_OP_N, m, k, &one, dA, m - 1, dvx, 1, &zero, dvy, 1) == CUBLAS_STATUS_INVALID_VALUE, "Dgemv lda < m");

    // cublasDotEx
    const int nd = 777;
    std::vector<float> fx(nd), fy(nd);
    std::vector<__half> hx(nd), hy(nd);
    std::vector<__nv_bfloat16> bx(nd), by(nd);
    double wf = 0, wh = 0, wb = 0;
    for (int i = 0; i < nd; ++i) {
        fx[i] = (float)rnd(); fy[i] = (float)rnd();
        hx[i] = __float2half(fx[i]); hy[i] = __float2half(fy[i]);
        bx[i] = __float2bfloat16(fx[i]); by[i] = __float2bfloat16(fy[i]);
        wf += (double)fx[i] * fy[i];
        wh += (double)__half2float(hx[i]) * __half2float(hy[i]);
        wb += (double)__bfloat162float(bx[i]) * __bfloat162float(by[i]);
    }
    float *dfx = up(fx), *dfy = up(fy);
    __half *dhx = up(hx), *dhy = up(hy);
    __nv_bfloat16 *dbx = up(bx), *dby = up(by);
    float rf = 0;
    CHECK(cublasDotEx(h, nd, dfx, CUDA_R_32F, 1, dfy, CUDA_R_32F, 1, &rf, CUDA_R_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS && std::fabs(rf - wf) < 1e-3, "DotEx 32F: %g want %g", rf, wf);
    CHECK(cublasDotEx(h, nd, dhx, CUDA_R_16F, 1, dhy, CUDA_R_16F, 1, &rf, CUDA_R_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS && std::fabs(rf - wh) < 1e-3, "DotEx 16F -> 32F: %g want %g", rf, wh);
    __half rh;
    CHECK(cublasDotEx(h, nd, dhx, CUDA_R_16F, 1, dhy, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
          std::fabs(__half2float(rh) - wh) <= std::fabs(wh) / 512 + 1e-3, "DotEx 16F -> 16F: %g want %g", __half2float(rh), wh);
    CHECK(cublasDotEx(h, nd, dbx, CUDA_R_16BF, 1, dby, CUDA_R_16BF, 1, &rf, CUDA_R_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS && std::fabs(rf - wb) < 1e-3, "DotEx 16BF: %g want %g", rf, wb);
    double rd = 0;
    CHECK(cublasDotEx(h, n, dy, CUDA_R_64F, 1, dy, CUDA_R_64F, 1, &rd, CUDA_R_64F, CUDA_R_64F) == CUBLAS_STATUS_SUCCESS, "DotEx 64F");
    w = 0;
    for (int i = 0; i < n; ++i) w += y[i] * y[i];
    CHECK(std::fabs(rd - w) < 1e-11, "DotEx 64F: %.17g want %.17g", rd, w);
    CHECK(cublasDotEx(h, nd, dfx, CUDA_R_32F, 1, dhy, CUDA_R_16F, 1, &rf, CUDA_R_32F, CUDA_R_32F) == CUBLAS_STATUS_NOT_SUPPORTED, "DotEx mixed types");
    rf = 5;
    CHECK(cublasDotEx(h, 0, dfx, CUDA_R_32F, 1, dfy, CUDA_R_32F, 1, &rf, CUDA_R_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS && rf == 0, "DotEx n=0");

    cublasDestroy(h);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
