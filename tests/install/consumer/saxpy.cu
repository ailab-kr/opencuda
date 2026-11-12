// Built against the installed package only.
#include <cstdio>

__global__ void saxpy(int n, float a, const float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main()
{
    const int n = 4096;
    float *x, *y;
    if (cudaMallocHost(&x, n * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMallocHost(&y, n * sizeof(float)) != cudaSuccess) return 1;
    for (int i = 0; i < n; ++i) { x[i] = float(i); y[i] = 1.0f; }
    float *dx, *dy;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, dx, dy);
    cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += y[i] != 2.0f * float(i) + 1.0f;
    std::printf("consumer_saxpy: %s\n", bad ? "FAIL" : "ok");
    return bad != 0 || cudaGetLastError() != cudaSuccess;
}
