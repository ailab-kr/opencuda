// What a CUDA compile provides without the program including it, as
// programs that build with nvcc rely on: assert() in host and device code
// and std::chrono, with no #include of <assert.h> or <chrono> here (only
// <cstdio>; cuda_runtime.h comes from the compile itself).
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_implicit_headers.cu -o rt_implicit_headers
#include <cstdio>

__global__ void k(int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(n > 0); // holds: compiles and does not stop the kernel
    if (i < n) out[i] = 2 * i;
}

int main()
{
    auto t0 = std::chrono::steady_clock::now();
    const int n = 1000;
    int *d = nullptr;
    cudaMalloc(&d, n * sizeof(int));
    k<<<(n + 127) / 128, 128>>>(d, n);
    cudaError_t e = cudaDeviceSynchronize();
    static int h[n];
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += h[i] != 2 * i;
    assert(e == cudaSuccess);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cudaFree(d);
    const bool ok = e == cudaSuccess && bad == 0 && ms >= 0.0;
    std::printf("%s: kernel %s, %d wrong, chrono %s\n", ok ? "PASS" : "FAIL", cudaGetErrorString(e), bad, ms >= 0.0 ? "ok" : "bad");
    return ok ? 0 : 1;
}
