// 64-bit integer division and remainder in kernels: an index split into
// coordinates by / and % with int64_t sizes passed as arguments, in a
// grid-stride loop (as HeCBench lrn-cuda does), and a table of signed and
// unsigned / and % over small, large (> 2^32) and negative operands,
// against the host. The SPIR-V path takes a 32-bit division when both
// operands are small.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_div64.cu -o rt_div64
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

__global__ void split(int64_t *out, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, int64_t total)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int64_t n = (idx / (C * D * H * W)) % N;
        int64_t c = (idx / (D * H * W)) % C;
        int64_t d = (idx / (H * W)) % D;
        int64_t h = (idx / W) % H;
        int64_t w = idx % W;
        out[idx] = (((n * C + c) * D + d) * H + h) * W + w; // the index again
    }
}

__global__ void table(const int64_t *a, const int64_t *b, int64_t *q, int64_t *r, uint64_t *uq, uint64_t *ur, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
    uq[i] = (uint64_t)a[i] / (uint64_t)b[i];
    ur[i] = (uint64_t)a[i] % (uint64_t)b[i];
}

int main()
{
    int checks = 0, failed = 0;
    auto check = [&](bool ok, const char *what, long i) {
        ++checks;
        if (!ok && failed++ < 10) std::printf("FAIL: %s at %ld\n", what, i);
    };
    const int64_t N = 2, C = 15, D = 1, H = 224, W = 224, total = N * C * D * H * W;
    int64_t *dout;
    cudaMalloc(&dout, total * 8);
    cudaMemset(dout, 0xff, total * 8);
    split<<<4096, 256>>>(dout, N, C, D, H, W, total);
    std::vector<int64_t> out(total);
    cudaMemcpy(out.data(), dout, total * 8, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < total; ++i) check(out[i] == i, "index split", (long)i);

    std::vector<int64_t> a, b;
    const int64_t vals[] = {0, 1, 2, 3, 7, 100, 65535, 65536, 2147483647LL, 2147483648LL, 4294967295LL, 4294967296LL,
                            4294967297LL, 123456789012345LL, 0x7fffffffffffffffLL, -1, -2, -7, -2147483648LL,
                            -2147483649LL, -4294967296LL, -123456789012345LL};
    for (int64_t x : vals)
        for (int64_t y : vals)
            if (y != 0 && !(x == INT64_MIN && y == -1)) { a.push_back(x); b.push_back(y); }
    unsigned s = 9;
    for (int i = 0; i < 20000; ++i) {
        s = s * 1664525u + 1013904223u;
        int64_t x = (int64_t)(((uint64_t)s << (s % 33)) & 0x7fffffffffffffffULL), y = (int64_t)((s >> 7) | 1) >> (s % 13);
        if (s & 0x100) x = -x;
        if (s & 0x200) y = -y;
        if (y == 0) y = 3;
        a.push_back(x);
        b.push_back(y);
    }
    int n = (int)a.size();
    int64_t *da, *db, *dq, *dr;
    uint64_t *duq, *dur;
    cudaMalloc(&da, n * 8); cudaMalloc(&db, n * 8); cudaMalloc(&dq, n * 8); cudaMalloc(&dr, n * 8);
    cudaMalloc(&duq, n * 8); cudaMalloc(&dur, n * 8);
    cudaMemcpy(da, a.data(), n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 8, cudaMemcpyHostToDevice);
    table<<<(n + 255) / 256, 256>>>(da, db, dq, dr, duq, dur, n);
    std::vector<int64_t> q(n), r(n);
    std::vector<uint64_t> uq(n), ur(n);
    cudaMemcpy(q.data(), dq, n * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data(), dr, n * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(uq.data(), duq, n * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(ur.data(), dur, n * 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        check(q[i] == a[i] / b[i] && r[i] == a[i] % b[i], "signed / %", i);
        check(uq[i] == (uint64_t)a[i] / (uint64_t)b[i] && ur[i] == (uint64_t)a[i] % (uint64_t)b[i], "unsigned / %", i);
    }
    check(cudaGetLastError() == cudaSuccess, "no error", 0);
    std::printf("%s: %d checks, %d failed\n", failed ? "FAIL" : "PASS", checks, failed);
    return failed ? 1 : 0;
}
