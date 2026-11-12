// Calls run() from the unmodified probe_kernel.cu, as an ordinary CUDA
// program: built by ocu_cc.py --target cpu, linked with ocu_cudart.
#include <cstdio>
#include <vector>

void run(int n, float a, const float *x, float *y);

int main()
{
    const int n = 4096;
    std::vector<float> hx(n), hy(n, 1.0f);
    for (int i = 0; i < n; ++i) hx[i] = float(i);
    float *x = nullptr, *y = nullptr;
    if (cudaMalloc(&x, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&y, n * sizeof(float)) != cudaSuccess) {
        return 1;
    }
    cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    run(n, 0.5f, x, y);
    int bad = cudaGetLastError() != cudaSuccess;
    cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) bad += hy[i] != 0.5f * float(i) + 1.0f;
    cudaFree(x);
    cudaFree(y);
    std::printf("main_saxpy: %s (%d bad)\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}
