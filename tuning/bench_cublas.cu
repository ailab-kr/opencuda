// bench_cublas: GEMM throughput of the BLAS library API (cublas_v2.h) on
// the active device, by the measurement rules of PLAN 6.4
// (tuning/ocu_bench.h): 3 untimed runs, then timed runs of one call plus
// cudaDeviceSynchronize. Prints one JSON object per line: median and
// 90th-percentile time, GFLOP/s at the median, and whether the spread
// between repeats was small enough to count.
//
//   bench_cublas [sizes...]          (default 1024 2048 4096)
//   OCU_BENCH_REPEATS=1 bench_cublas (one sample of 10 runs, quicker)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "ocu_bench.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct job {
    const char *kind;
    cublasHandle_t h;
    int n;
    cublasOperation_t ta;
    void *a, *b, *c;
};

int run(void *p)
{
    job *j = static_cast<job *>(p);
    const float one = 1.0f, zero = 0.0f;
    const __half hone = __float2half(1.0f), hzero = __float2half(0.0f);
    cublasStatus_t st;
    const int n = j->n;
    if (std::strcmp(j->kind, "sgemm") == 0)
        st = cublasSgemm(j->h, j->ta, CUBLAS_OP_N, n, n, n, &one, (float *)j->a, n, (float *)j->b, n,
                         &zero, (float *)j->c, n);
    else if (std::strcmp(j->kind, "hgemm") == 0)
        st = cublasHgemm(j->h, j->ta, CUBLAS_OP_N, n, n, n, &hone, (__half *)j->a, n, (__half *)j->b, n,
                         &hzero, (__half *)j->c, n);
    else /* gemmex_h_32F: fp16 in, fp16 or fp32 out (gemmex_h_32F_out32), fp32 compute */
        st = cublasGemmEx(j->h, j->ta, CUBLAS_OP_N, n, n, n, &one, j->a, CUDA_R_16F, n, j->b, CUDA_R_16F,
                          n, &zero, j->c, std::strstr(j->kind, "out32") ? CUDA_R_32F : CUDA_R_16F, n,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return int(st);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

} // namespace

int main(int argc, char **argv)
{
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 2048, 4096};
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "bench_cublas: no device\n");
        return 1;
    }
    ocu_bench_config cfg;
    ocu_bench_default_config(&cfg);
    if (const char *r = std::getenv("OCU_BENCH_REPEATS")) cfg.repeats = std::atoi(r);
    for (int n : sizes) {
        const size_t count = size_t(n) * n;
        std::vector<float> hf(count);
        std::vector<__half> hh(count);
        for (size_t i = 0; i < count; ++i) {
            hf[i] = float(int(i % 17) - 8) * 0.125f;
            hh[i] = __float2half(hf[i]);
        }
        void *a, *b, *c;
        if (cudaMalloc(&a, count * 4) != cudaSuccess || cudaMalloc(&b, count * 4) != cudaSuccess ||
            cudaMalloc(&c, count * 4) != cudaSuccess) {
            std::fprintf(stderr, "bench_cublas: out of memory at %d\n", n);
            return 1;
        }
        for (const char *kind : {"sgemm", "hgemm", "gemmex_h_32F", "gemmex_h_32F_out32"}) {
            const bool half = std::strcmp(kind, "sgemm") != 0;
            const void *src = half ? static_cast<const void *>(hh.data()) : static_cast<const void *>(hf.data());
            cudaMemcpy(a, src, count * (half ? 2 : 4), cudaMemcpyHostToDevice);
            cudaMemcpy(b, src, count * (half ? 2 : 4), cudaMemcpyHostToDevice);
            for (cublasOperation_t ta : {CUBLAS_OP_N, CUBLAS_OP_T}) {
                job j{kind, h, n, ta, a, b, c};
                ocu_bench_result r;
                if (ocu_bench_run(&cfg, run, &j, nullptr, nullptr, &r) != 0 || r.status != 0) {
                    std::printf("{\"op\":\"%s\",\"n\":%d,\"error\":%d}\n", kind, n, r.status);
                    continue;
                }
                const double gflops = 2.0 * n * double(n) * n / (r.median_ms * 1e-3) / 1e9;
                std::printf("{\"op\":\"%s\",\"trans\":\"%cN\",\"n\":%d,\"median_ms\":%.3f,\"p90_ms\":%.3f,"
                            "\"gflops\":%.0f,\"spread\":%.3f,\"valid\":%s}\n",
                            kind, ta == CUBLAS_OP_N ? 'N' : 'T', n, r.median_ms, r.p90_ms, gflops,
                            r.spread, r.valid ? "true" : "false");
                std::fflush(stdout);
            }
        }
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    cublasDestroy(h);
    return 0;
}
