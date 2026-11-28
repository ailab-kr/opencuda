// A work-item writes shared memory through a generic pointer in a function
// that is not inlined, then reads the same place as shared memory, no barrier.
#include <cuda_runtime.h>
#include <cstdio>

__device__ __noinline__ void put(int *p, int lane, int v)
{
    if (lane < 14) p[lane] = v;
}
__device__ __noinline__ void scan(const int *in, int *out, int n)
{
    int lane = threadIdx.x & 31;
    int data = (lane > 0 && lane < n) ? in[lane - 1] : 0;
    for (int i = 1; i < 32; i *= 2) {
        int other = __shfl_up_sync(0xffffffffu, data, i);
        if (lane >= i) data += other;
    }
    if (lane < n) out[lane] = data;
}

__global__ void k(int *out, int mode)
{
    __shared__ int h[16];
    int tid = threadIdx.x;
    if (tid < 16) h[tid] = tid + 1;
    __syncthreads();
    if (tid < 32) {
        if (mode == 0) put(h, tid, 100 + tid);
        else scan(h, h, 14);
        if (tid < 14) out[blockIdx.x * 16 + tid] = h[tid];
    }
}

int main()
{
    const int B = 1024;
    int *d, h[B * 16];
    cudaMalloc(&d, sizeof h);
    int bad[2] = {0, 0};
    for (int mode = 0; mode < 2; ++mode) {
        cudaMemset(d, 0, sizeof h);
        k<<<B, 64>>>(d, mode);
        cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
        for (int b = 0; b < B; ++b) {
            int s = 0;
            for (int i = 0; i < 14; ++i) {
                int ref = mode == 0 ? 100 + i : s;
                s += i + 1;
                if (h[b * 16 + i] != ref) bad[mode]++;
            }
        }
    }
    printf("generic write then shared read: put %d wrong, scan %d wrong (of %d)\n", bad[0], bad[1], B * 14);
    return bad[0] + bad[1] != 0;
}
