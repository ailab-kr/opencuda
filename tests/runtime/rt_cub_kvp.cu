// cub::KeyValuePair with cub::ArgMax / cub::ArgMin through cub::BlockReduce,
// as top-k / routing kernels (HeCBench moe-cuda) use them: per block, the
// (index, value) of the largest and smallest value, the lower index when
// values tie (blocks are built with ties), checked against a host loop; the
// pair in __shared__ memory and its operator!=.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_cub_kvp.cu -o rt_cub_kvp
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

constexpr int TPB = 128;
using kvp = cub::KeyValuePair<int, float>;

__global__ void arg_reduce(const float *v, int n, kvp *out_max, kvp *out_min, int *ne)
{
    using BlockReduce = cub::BlockReduce<kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp;
    __shared__ kvp first;
    const int i = blockIdx.x * TPB + threadIdx.x;
    kvp x(i, i < n ? v[i] : -1e30f);
    kvp mx = BlockReduce(tmp).Reduce(x, cub::ArgMax());
    __syncthreads();
    kvp y(i, i < n ? v[i] : 1e30f);
    kvp mn = BlockReduce(tmp).Reduce(y, cub::ArgMin());
    if (threadIdx.x == 0) {
        first = x;
        out_max[blockIdx.x] = mx;
        out_min[blockIdx.x] = mn;
    }
    __syncthreads();
    if (threadIdx.x == 1) ne[blockIdx.x] = (first != x) ? 1 : 0; /* thread 1's pair differs from thread 0's */
}

int main()
{
    const int blocks = 40, n = blocks * TPB - 17;
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = (float)((i * 37) % 101) - 50.0f; // many equal values: ties
    float *dv;
    kvp *dmax, *dmin;
    int *dne;
    cudaMalloc(&dv, n * sizeof(float));
    cudaMalloc(&dmax, blocks * sizeof(kvp));
    cudaMalloc(&dmin, blocks * sizeof(kvp));
    cudaMalloc(&dne, blocks * sizeof(int));
    cudaMemcpy(dv, v.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    arg_reduce<<<blocks, TPB>>>(dv, n, dmax, dmin, dne);
    cudaError_t e = cudaDeviceSynchronize();
    std::vector<kvp> gmax(blocks), gmin(blocks);
    std::vector<int> ne(blocks);
    cudaMemcpy(gmax.data(), dmax, blocks * sizeof(kvp), cudaMemcpyDeviceToHost);
    cudaMemcpy(gmin.data(), dmin, blocks * sizeof(kvp), cudaMemcpyDeviceToHost);
    cudaMemcpy(ne.data(), dne, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    int bad = 0, checks = 0;
    for (int b = 0; b < blocks; ++b) {
        int imax = -1, imin = -1;
        for (int t = 0; t < TPB; ++t) {
            int i = b * TPB + t;
            if (i >= n) break;
            if (imax < 0 || v[i] > v[imax]) imax = i; // strictly larger: the first (lowest) index on a tie
            if (imin < 0 || v[i] < v[imin]) imin = i;
        }
        checks += 3;
        if (gmax[b].key != imax || gmax[b].value != v[imax]) { if (bad++ < 3) std::printf("  block %d max: (%d, %g) want (%d, %g)\n", b, gmax[b].key, gmax[b].value, imax, v[imax]); }
        if (gmin[b].key != imin || gmin[b].value != v[imin]) { if (bad++ < 3) std::printf("  block %d min: (%d, %g) want (%d, %g)\n", b, gmin[b].key, gmin[b].value, imin, v[imin]); }
        bad += ne[b] != 1;
    }
    std::printf("%s: %d checks, %d failed (%s)\n", e == cudaSuccess && bad == 0 ? "PASS" : "FAIL", checks, bad, cudaGetErrorString(e));
    return e == cudaSuccess && bad == 0 ? 0 : 1;
}
