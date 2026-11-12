// Probe input: warp-level operations as real code uses them.
// Written for the probe; every path compiles it unchanged.
#include "probe_defs.h"

// Sum within each warp with a shuffle-down tree, lane 0 writes it.
__global__ void warp_sum(const float *in, float *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[i];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    int warp = i / warpSize;
    if (threadIdx.x % warpSize == 0) out[warp] = v;
}

// Every lane gets the maximum of its 32-lane group (explicit width 32).
__global__ void group_max(const int *in, int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    for (int m = 16; m > 0; m >>= 1) {
        int o = __shfl_xor_sync(0xffffffffu, v, m, 32);
        v = o > v ? o : v;
    }
    out[i] = v;
}

// Broadcast lane 3, inclusive prefix sum with shuffle-up, vote counts.
__global__ void misc(const int *in, int *bcast, int *scan, unsigned *ballots,
                     int *any_all)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int v = in[i];
    bcast[i] = __shfl_sync(0xffffffffu, v, 3, 32);
    int s = v;
    for (int d = 1; d < 32; d *= 2) {
        int up = __shfl_up_sync(0xffffffffu, s, d, 32);
        if (lane >= d) s += up;
    }
    scan[i] = s;
    unsigned b = __ballot_sync(0xffffffffu, v % 3 == 0);
    int a = __any_sync(0xffffffffu, v == 7);
    int l = __all_sync(0xffffffffu, v >= 0);
    __syncwarp();
    if (lane == 0) {
        ballots[i / 32] = b;
        any_all[2 * (i / 32)] = a;
        any_all[2 * (i / 32) + 1] = l;
    }
}

void run_warp_sum(const float *in, float *out, int blocks, int threads)
{
    warp_sum<<<blocks, threads>>>(in, out);
}

void run_group_max(const int *in, int *out, int blocks, int threads)
{
    group_max<<<blocks, threads>>>(in, out);
}

void run_misc(const int *in, int *bcast, int *scan, unsigned *ballots,
              int *any_all, int blocks, int threads)
{
    misc<<<blocks, threads>>>(in, bcast, scan, ballots, any_all);
}
