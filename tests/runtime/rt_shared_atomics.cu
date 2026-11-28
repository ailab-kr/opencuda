// Atomics on __shared__ memory: a block histogram with atomicAdd / atomicMax
// / atomicMin / atomicSub / atomicCAS on shared counters (the pointer known
// to be shared memory, and passed through a function), the same with each
// bin a constant index into a shared array (h2[3], h2[13]: LLVM 23's SPIR-V
// back end placed those 4x too far), then an in-place warp scan of the
// counters, as HeCBench segsort's bin kernel does; compared with the same
// computed on the host.
//   ocu-nvcc [--ocu-target=spirv] tests/runtime/rt_shared_atomics.cu -o rt_shared_atomics
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

constexpr int NB = 14, BS = 256, NO = NB + 6 + NB;

__device__ __noinline__ void bump(int *c, int v) { atomicAdd(c, v); }

__global__ void histo(const int *sizes, int n, int *out)
{
    __shared__ int h[NB], h2[NB];
    __shared__ int mx, mn, sub, cas_count;
    __shared__ unsigned long long sum64;
    int tid = threadIdx.x, gid = blockIdx.x * blockDim.x + tid;
    if (tid < NB) { h[tid] = 0; h2[tid] = 0; }
    if (tid == 0) { mx = -1; mn = 1 << 30; sub = 1000000; cas_count = 0; sum64 = 0; }
    __syncthreads();
    if (gid < n) {
        int s = sizes[gid];
        int b = s <= 1 ? 0 : 32 - __clz(s - 1); // bin of ceil(log2(s))
        if (b > NB - 2) b = NB - 2;
        atomicAdd(&h[b], 1);
        bump(&h[NB - 1], 1);
        atomicMax(&mx, s);
        atomicMin(&mn, s);
        atomicSub(&sub, 1);
        atomicAdd(&sum64, (unsigned long long)s);
        int old = cas_count, assumed;
        do { assumed = old; old = atomicCAS(&cas_count, assumed, assumed + 1); } while (old != assumed);
        // constant indices, as segsort's bins
        if (s <= 1) atomicAdd(&h2[0], 1);
        if (1 < s && s <= 2) atomicAdd(&h2[1], 1);
        if (2 < s && s <= 4) atomicAdd(&h2[2], 1);
        if (4 < s && s <= 8) atomicAdd(&h2[3], 1);
        if (8 < s && s <= 16) atomicAdd(&h2[4], 1);
        if (16 < s && s <= 32) atomicAdd(&h2[5], 1);
        if (32 < s && s <= 64) atomicAdd(&h2[6], 1);
        if (64 < s && s <= 128) atomicAdd(&h2[7], 1);
        if (128 < s && s <= 256) atomicAdd(&h2[8], 1);
        if (256 < s && s <= 512) atomicAdd(&h2[9], 1);
        if (512 < s && s <= 1024) atomicAdd(&h2[10], 1);
        if (1024 < s && s <= 2048) atomicAdd(&h2[11], 1);
        if (2048 < s) atomicAdd(&h2[12], 1);
        atomicMax(&h2[13], s);
    }
    __syncthreads();
    if (tid < 32) { // in-place exclusive scan of h by the first warp
        int lane = tid;
        int data = (lane > 0 && lane < NB) ? h[lane - 1] : 0;
        for (int i = 1; i < 32; i *= 2) {
            int other = __shfl_up_sync(0xffffffffu, data, i);
            if (lane >= i) data += other;
        }
        if (lane < NB) h[lane] = data;
    }
    __syncthreads();
    int *o = out + blockIdx.x * NO;
    if (tid < NB) { o[tid] = h[tid]; o[NB + 6 + tid] = h2[tid]; }
    if (tid == 0) { o[NB] = mx; o[NB + 1] = mn; o[NB + 2] = sub; o[NB + 3] = cas_count;
                    o[NB + 4] = (int)(sum64 & 0x7fffffff); o[NB + 5] = (int)(sum64 >> 31); }
}

int main()
{
    const int blocks = 512, n = blocks * BS - 37;
    std::vector<int> sizes(n);
    unsigned r = 7;
    for (int &s : sizes) { r = r * 1664525u + 1013904223u; s = 1 + (int)((r >> 8) % 5000) / (1 + (int)((r >> 4) & 15)); }
    int *ds, *dout;
    cudaMalloc(&ds, n * 4);
    cudaMalloc(&dout, blocks * NO * 4);
    cudaMemcpy(ds, sizes.data(), n * 4, cudaMemcpyHostToDevice);
    histo<<<blocks, BS>>>(ds, n, dout);
    std::vector<int> out(blocks * NO);
    cudaError_t e = cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    int checks = 0, failed = 0;
    for (int b = 0; b < blocks; ++b) {
        int h[NB] = {}, h2[NB] = {}, mx = -1, mn = 1 << 30, cnt = 0;
        unsigned long long sum = 0;
        for (int t = 0; t < BS; ++t) {
            int g = b * BS + t;
            if (g >= n) break;
            int s = sizes[g], k = s <= 1 ? 0 : 32 - __builtin_clz(s - 1);
            if (k > NB - 2) k = NB - 2;
            h[k]++; h[NB - 1]++; mx = s > mx ? s : mx; mn = s < mn ? s : mn; cnt++; sum += s;
            h2[k > 12 ? 12 : k]++;
            h2[13] = s > h2[13] ? s : h2[13];
        }
        int scan = 0, ref[NO];
        for (int i = 0; i < NB; ++i) { ref[i] = scan; scan += h[i]; ref[NB + 6 + i] = h2[i]; }
        ref[NB] = mx; ref[NB + 1] = mn; ref[NB + 2] = 1000000 - cnt; ref[NB + 3] = cnt;
        ref[NB + 4] = (int)(sum & 0x7fffffff); ref[NB + 5] = (int)(sum >> 31);
        for (int i = 0; i < NO; ++i) {
            ++checks;
            if (out[b * NO + i] != ref[i] && failed++ < 8)
                std::printf("FAIL: block %d item %d: %d, expected %d\n", b, i, out[b * NO + i], ref[i]);
        }
    }
    if (e != cudaSuccess) failed++;
    std::printf("%s: %d checks, %d failed\n", failed ? "FAIL" : "PASS", checks, failed);
    return failed ? 1 : 0;
}
