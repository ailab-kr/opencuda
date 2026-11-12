// A whole program in one .cu file: shared memory, a reduction, a template.
#include <cstdio>

template <int BLOCK>
__global__ void block_sum(const int *in, int *out, int n)
{
    __shared__ int part[BLOCK];
    int i = blockIdx.x * BLOCK + threadIdx.x;
    part[threadIdx.x] = i < n ? in[i] : 0;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, part[0]);
}

int main()
{
    const int n = 10000;
    int host[n];
    for (int i = 0; i < n; ++i) host[i] = i % 7;
    int *in = nullptr, *out = nullptr;
    cudaMalloc(&in, n * sizeof(int));
    cudaMalloc(&out, sizeof(int));
    cudaMemcpy(in, host, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(out, 0, sizeof(int));
    block_sum<256><<<(n + 255) / 256, 256>>>(in, out, n);
    int got = 0, want = 0;
    cudaMemcpy(&got, out, sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) want += i % 7;
    cudaError_t err = cudaGetLastError();
    cudaFree(in);
    cudaFree(out);
    bool ok = got == want && err == cudaSuccess;
    std::printf("cu_main: %s (%d, expected %d, %s)\n", ok ? "ok" : "FAIL", got, want,
                cudaGetErrorString(err));
    return ok ? 0 : 1;
}
