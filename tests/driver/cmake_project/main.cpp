// Host code compiled by the C++ compiler; the runtime comes from CUDA::cudart.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

extern "C" int run_scale_add(int n, const float *x, float *y);

int main()
{
    const int n = 1000;
    std::vector<float> hx(n), hy(n, 1.0f);
    for (int i = 0; i < n; ++i) hx[i] = float(i);
    float *x = nullptr, *y = nullptr;
    if (cudaMalloc(&x, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&y, n * sizeof(float)) != cudaSuccess) return 1;
    cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (run_scale_add(n, x, y) != 0) return 1;
    cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += hy[i] != 1.0f + 2.0f * float(i);
    cudaFree(x);
    cudaFree(y);
    std::printf("host_main: %s (%d bad)\n", bad ? "FAIL" : "ok", bad);
    return bad != 0;
}
