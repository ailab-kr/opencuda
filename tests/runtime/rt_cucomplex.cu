// cuComplex.h on host and device: the types are float2 / double2 (x real,
// y imaginary); make, real/imag, conj, add, sub, mul are exact per the
// formulas; div and abs within a few ulps of a long double reference and
// without the intermediate overflow / underflow of the naive formulas
// (|1e30 + 1e30 i| is finite, (1e-30 + 1e-30 i) / (1e-30 + 2e-30 i) is
// right); fma is x * y + d; float <-> double conversions.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_cucomplex.cu -o rt_cucomplex
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <type_traits>

static int g_fail, g_checks;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static_assert(std::is_same<cuFloatComplex, float2>::value && std::is_same<cuDoubleComplex, double2>::value &&
              std::is_same<cuComplex, cuFloatComplex>::value, "types");

struct inputs { float fx, fy, gx, gy, hx, hy; double dx, dy, ex, ey, kx, ky; };
struct outputs { cuFloatComplex add, sub, mul, div, conj, fma; float abs, re, im;
                 cuDoubleComplex dadd, dsub, dmul, ddiv, dconj, dfma; double dabs, dre, dim; cuDoubleComplex up; cuFloatComplex down; };

__host__ __device__ void eval(const inputs &in, outputs &o)
{
    cuFloatComplex a = make_cuFloatComplex(in.fx, in.fy), b = make_cuComplex(in.gx, in.gy), c = make_cuFloatComplex(in.hx, in.hy);
    o.add = cuCaddf(a, b); o.sub = cuCsubf(a, b); o.mul = cuCmulf(a, b); o.div = cuCdivf(a, b);
    o.conj = cuConjf(a); o.fma = cuCfmaf(a, b, c); o.abs = cuCabsf(a); o.re = cuCrealf(a); o.im = cuCimagf(a);
    cuDoubleComplex x = make_cuDoubleComplex(in.dx, in.dy), y = make_cuDoubleComplex(in.ex, in.ey), z = make_cuDoubleComplex(in.kx, in.ky);
    o.dadd = cuCadd(x, y); o.dsub = cuCsub(x, y); o.dmul = cuCmul(x, y); o.ddiv = cuCdiv(x, y);
    o.dconj = cuConj(x); o.dfma = cuCfma(x, y, z); o.dabs = cuCabs(x); o.dre = cuCreal(x); o.dim = cuCimag(x);
    o.up = cuComplexFloatToDouble(a); o.down = cuComplexDoubleToFloat(x);
}

__global__ void k(const inputs *in, outputs *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) eval(in[i], out[i]);
}

static bool near(long double got, long double want, long double rel)
{
    if (std::isnan(want)) return std::isnan(got);
    if (std::isinf(want)) return got == want;
    return std::fabs(got - want) <= rel * std::fabs(want) + 1e-300L;
}

static void check(const inputs &in, const outputs &o, const char *where)
{
    typedef std::complex<long double> C;
    C a(in.fx, in.fy), b(in.gx, in.gy), x(in.dx, in.dy), y(in.ex, in.ey);
    CHECK(o.add.x == in.fx + in.gx && o.add.y == in.fy + in.gy, "%s add", where);
    CHECK(o.sub.x == in.fx - in.gx && o.sub.y == in.fy - in.gy, "%s sub", where);
    CHECK(o.conj.x == in.fx && o.conj.y == -in.fy && o.re == in.fx && o.im == in.fy, "%s conj/re/im", where);
    C m = a * b, q = a / b, dm = x * y, dq = x / y;
    // products: componentwise error within a few ulps of |a| |b| (the formula's own
    // bound; a component may cancel), checked where |a| |b| is inside the range
    long double sf = std::abs(a) * std::abs(b), sd = std::abs(x) * std::abs(y);
    auto close = [](long double got, long double want, long double scale, long double eps) {
        return std::fabs(got - want) <= eps * scale;
    };
    bool fin_f = sf < 1e37L && sf > 1e-36L, fin_d = sd < 1e307L && sd > 1e-306L;
    if (fin_f) CHECK(close(o.mul.x, m.real(), sf, 3e-7L) && close(o.mul.y, m.imag(), sf, 3e-7L), "%s mul", where);
    CHECK(near(o.div.x, q.real(), 4e-6L) && near(o.div.y, q.imag(), 4e-6L), "%s div %a %a / %a %a = %a %a", where, in.fx, in.fy, in.gx, in.gy, o.div.x, o.div.y);
    CHECK(near(o.abs, std::abs(a), 3e-7L), "%s abs %a %a = %a", where, in.fx, in.fy, o.abs);
    if (fin_d) CHECK(close(o.dmul.x, dm.real(), sd, 5e-16L) && close(o.dmul.y, dm.imag(), sd, 5e-16L), "%s dmul", where);
    CHECK(near(o.ddiv.x, dq.real(), 8e-15L) && near(o.ddiv.y, dq.imag(), 8e-15L), "%s ddiv", where);
    CHECK(near(o.dabs, std::abs(x), 5e-16L), "%s dabs", where);
    C f = a * b + C(in.hx, in.hy), df = x * y + C(in.kx, in.ky);
    long double tf = sf + std::abs(C(in.hx, in.hy)), td = sd + std::abs(C(in.kx, in.ky));
    if (fin_f) CHECK(close(o.fma.x, f.real(), tf, 3e-7L) && close(o.fma.y, f.imag(), tf, 3e-7L), "%s fma", where);
    if (fin_d) CHECK(close(o.dfma.x, df.real(), td, 5e-16L) && close(o.dfma.y, df.imag(), td, 5e-16L), "%s dfma", where);
    CHECK(o.up.x == (double)in.fx && o.up.y == (double)in.fy && o.down.x == (float)in.dx && o.down.y == (float)in.dy, "%s conversions", where);
}

int main()
{
    inputs in[] = {
        {1.5f, -2.0f, 0.25f, 4.0f, 1.0f, 1.0f, 1.5, -2.0, 0.25, 4.0, 1.0, 1.0},
        {3.0f, 4.0f, -1.0f, 0.5f, 0.0f, -2.0f, 3.0, 4.0, -1.0, 0.5, 0.0, -2.0},
        {1e30f, 1e30f, 1e30f, 2e30f, 0, 0, 1e300, 1e300, 1e300, 2e300, 0, 0},            // no overflow of squares
        {1e-30f, 1e-30f, 1e-30f, 2e-30f, 0, 0, 1e-300, 1e-300, 1e-300, 2e-300, 0, 0},     // no underflow
        {7.0f, 0.0f, 0.0f, 3.0f, 2.0f, 2.0f, 7.0, 0.0, 0.0, 3.0, 2.0, 2.0},              // |y.y| > |y.x| branch
        {-5.5f, 1e-3f, 2.0f, -2.0f, 1.0f, 0.0f, -5.5, 1e-3, 2.0, -2.0, 1.0, 0.0},
    };
    const int n = sizeof in / sizeof in[0];
    outputs host[n], dev[n];
    for (int i = 0; i < n; ++i) eval(in[i], host[i]);
    inputs *din; outputs *dout;
    cudaMalloc(&din, sizeof in); cudaMalloc(&dout, sizeof dev);
    cudaMemcpy(din, in, sizeof in, cudaMemcpyHostToDevice);
    k<<<1, 32>>>(din, dout, n);
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "kernel");
    cudaMemcpy(dev, dout, sizeof dev, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) { check(in[i], host[i], "host"); check(in[i], dev[i], "device"); }
    CHECK(std::isfinite(host[2].abs) && std::isfinite(host[2].dabs), "abs of 1e30 + 1e30 i is finite");
    CHECK(cuCabsf(make_cuFloatComplex(INFINITY, NAN)) == INFINITY && std::isnan(cuCabsf(make_cuFloatComplex(1.0f, NAN))), "abs specials");
    cudaFree(din); cudaFree(dout);
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
