// __half conversions and arithmetic in device code give the same bits as
// host code: all 65536 half -> float, float -> half (round to nearest even)
// at every half boundary and midpoint and around them, from double, and
// __hadd / __hmul / __hdiv / __hfma on many pairs and triples.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_half_convert.cu -o rt_half_convert
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int NH = 65536;
__global__ void to_float(float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NH) out[i] = __half2float(__ushort_as_half((unsigned short)i));
}
__global__ void from_float(const float *in, unsigned short *out, unsigned short *ctor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half_as_ushort(__float2half(in[i]));
        __half h = in[i]; // the constructor
        ctor[i] = __half_as_ushort(h);
    }
}
__global__ void from_double(const double *in, unsigned short *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half_as_ushort(__double2half(in[i]));
}
__global__ void arith(const unsigned short *a, const unsigned short *b, const unsigned short *c, unsigned short *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __half x = __ushort_as_half(a[i]), y = __ushort_as_half(b[i]), z = __ushort_as_half(c[i]);
    out[4 * i + 0] = __half_as_ushort(__hadd(x, y));
    out[4 * i + 1] = __half_as_ushort(x * y);
    out[4 * i + 2] = __half_as_ushort(__hdiv(x, y));
    out[4 * i + 3] = __half_as_ushort(__hfma(x, y, z));
}

// __half fields at addresses that are 2 mod 4 (blocks of 110 and 210 bytes,
// as in llama.cpp's q3_K / q6_K), read one by one and as part of a sum
struct blk110 { signed char q[108]; __half d; };
struct blk210 { signed char q[208]; __half d; };
static_assert(sizeof(blk110) == 110 && sizeof(blk210) == 210, "packed layout");
template <typename B>
__global__ void read_fields(const B *b, float *out, float *sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = b[i].d; // operator float
    out[i] = d;
    float s = 0;
    for (int k = 0; k < 16; ++k) s += d * b[i].q[k];
    sum[i] = s;
}

static unsigned rng = 12345;
static unsigned next() { return rng = rng * 1664525u + 1013904223u; }
static bool same_half(unsigned short a, unsigned short b)
{
    bool na = (a & 0x7c00) == 0x7c00 && (a & 0x3ff), nb = (b & 0x7c00) == 0x7c00 && (b & 0x3ff);
    return na || nb ? na == nb : a == b; // any NaN for NaN
}

int main()
{
    int checks = 0, failed = 0;
    auto check = [&](bool ok, const char *what, long i) {
        ++checks;
        if (!ok && failed++ < 10) std::printf("FAIL: %s at %ld\n", what, i);
    };
    // half -> float, every pattern
    std::vector<float> f(NH);
    float *df;
    cudaMalloc(&df, NH * 4);
    to_float<<<NH / 256, 256>>>(df);
    cudaMemcpy(f.data(), df, NH * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < NH; ++i) {
        float r = __half2float(__ushort_as_half((unsigned short)i));
        bool nan = r != r;
        check(nan ? f[i] != f[i] : std::memcmp(&f[i], &r, 4) == 0, "half -> float", i);
    }
    // float -> half: each finite half value, the midpoints, one float ulp around both, overflow edge
    std::vector<float> in;
    for (int i = 0; i < 0x7c00; ++i) {
        float v = __half2float(__ushort_as_half((unsigned short)i));
        float w = __half2float(__ushort_as_half((unsigned short)(i + 1)));
        float mid = (v + w) / 2; // exact in float
        for (float x : {v, mid, nextafterf(mid, 0.0f), nextafterf(mid, 1e9f), nextafterf(v, 1e9f)}) {
            in.push_back(x);
            in.push_back(-x);
        }
    }
    for (float x : {65504.0f, 65519.0f, 65520.0f, 70000.0f, 1e-8f, 2.98e-8f, 2.9802322e-8f, 1.0f / 0.0f}) {
        in.push_back(x);
        in.push_back(-x);
    }
    in.push_back(__builtin_nanf(""));
    for (int i = 0; i < 100000; ++i) {
        unsigned u = next();
        float x;
        std::memcpy(&x, &u, 4);
        in.push_back(x);
    }
    int n = (int)in.size();
    float *din;
    unsigned short *dout, *dctor;
    cudaMalloc(&din, n * 4);
    cudaMalloc(&dout, n * 2);
    cudaMalloc(&dctor, n * 2);
    cudaMemcpy(din, in.data(), n * 4, cudaMemcpyHostToDevice);
    from_float<<<(n + 255) / 256, 256>>>(din, dout, dctor, n);
    std::vector<unsigned short> out(n), ctor(n);
    cudaMemcpy(out.data(), dout, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(ctor.data(), dctor, n * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        unsigned short r = __half_as_ushort(__float2half(in[i]));
        check(same_half(out[i], r), "float -> half", i);
        check(same_half(ctor[i], r), "__half(float)", i);
    }
    // double -> half: values a float cannot hold (one rounding, not two)
    std::vector<double> din_h;
    for (int i = 0; i < 0x7bff; i += 7) {
        double v = __half2float(__ushort_as_half((unsigned short)i)), w = __half2float(__ushort_as_half((unsigned short)(i + 1)));
        double mid = (v + w) / 2;
        for (double x : {mid, mid + mid * 1e-12, mid - mid * 1e-12, v + (w - v) * 0.4999999999}) din_h.push_back(x);
    }
    int nd = (int)din_h.size();
    double *dd;
    unsigned short *ddo;
    cudaMalloc(&dd, nd * 8);
    cudaMalloc(&ddo, nd * 2);
    cudaMemcpy(dd, din_h.data(), nd * 8, cudaMemcpyHostToDevice);
    from_double<<<(nd + 255) / 256, 256>>>(dd, ddo, nd);
    std::vector<unsigned short> dout_h(nd);
    cudaMemcpy(dout_h.data(), ddo, nd * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < nd; ++i) check(same_half(dout_h[i], __half_as_ushort(__double2half(din_h[i]))), "double -> half", i);
    // arithmetic
    const int na = 1 << 18;
    std::vector<unsigned short> a(na), b(na), c(na), r(4 * na);
    for (int i = 0; i < na; ++i) {
        a[i] = (unsigned short)next();
        b[i] = (unsigned short)(next() >> 7);
        c[i] = (unsigned short)(next() >> 3);
        if (i & 1) b[i] = (unsigned short)((a[i] & 0x8000) | ((a[i] + (next() & 0x1ff) - 0xff) & 0x7fff)); // close magnitudes
    }
    unsigned short *da, *db, *dc, *dr;
    cudaMalloc(&da, na * 2);
    cudaMalloc(&db, na * 2);
    cudaMalloc(&dc, na * 2);
    cudaMalloc(&dr, na * 8);
    cudaMemcpy(da, a.data(), na * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), na * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c.data(), na * 2, cudaMemcpyHostToDevice);
    arith<<<na / 256, 256>>>(da, db, dc, dr, na);
    cudaMemcpy(r.data(), dr, na * 8, cudaMemcpyDeviceToHost);
    const char *names[4] = {"__hadd", "operator*", "__hdiv", "__hfma"};
    for (int i = 0; i < na; ++i) {
        __half x = __ushort_as_half(a[i]), y = __ushort_as_half(b[i]), z = __ushort_as_half(c[i]);
        unsigned short ref[4] = {__half_as_ushort(__hadd(x, y)), __half_as_ushort(x * y), __half_as_ushort(__hdiv(x, y)),
                                 __half_as_ushort(__hfma(x, y, z))};
        for (int k = 0; k < 4; ++k) check(same_half(r[4 * i + k], ref[k]), names[k], i);
    }
    // fields of packed blocks
    auto blocks = [&](auto tag, const char *what) {
        using B = decltype(tag);
        const int nb = 4096;
        std::vector<B> hb(nb);
        for (int i = 0; i < nb; ++i) {
            for (auto &q : hb[i].q) q = (signed char)(next() >> 24);
            hb[i].d = __ushort_as_half((unsigned short)(next() & 0x3fff)); // finite, small and subnormal
        }
        B *db2;
        float *dout2, *dsum;
        cudaMalloc(&db2, nb * sizeof(B));
        cudaMalloc(&dout2, nb * 4);
        cudaMalloc(&dsum, nb * 4);
        cudaMemcpy(db2, hb.data(), nb * sizeof(B), cudaMemcpyHostToDevice);
        read_fields<<<nb / 256, 256>>>(db2, dout2, dsum, nb);
        std::vector<float> o(nb), s(nb);
        cudaMemcpy(o.data(), dout2, nb * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(s.data(), dsum, nb * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nb; ++i) {
            float d = hb[i].d, ref = 0;
            for (int k = 0; k < 16; ++k) ref += d * hb[i].q[k];
            check(o[i] == d, what, i);
            check(std::fabs(s[i] - ref) <= 1e-5f * (std::fabs(ref) + 1e-30f) + 1e-35f, what, i);
        }
        cudaFree(db2);
        cudaFree(dout2);
        cudaFree(dsum);
    };
    blocks(blk110{}, "__half field of a 110-byte block");
    blocks(blk210{}, "__half field of a 210-byte block");
    cudaError_t e = cudaGetLastError();
    check(e == cudaSuccess, "no error", 0);
    std::printf("%s: %d checks, %d failed\n", failed ? "FAIL" : "PASS", checks, failed);
    return failed ? 1 : 0;
}
