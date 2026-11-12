// Programs that end the process or the launch, run in their own process by
// trap_test.cmake (card runtime-llama-gaps, items 25, 26 and 30).
//
// Usage: rt_trap trap | nodevcode | gridsync
//
// CPU path: the kernel stops the process (abort) after the runtime has
// flushed the output written so far and printed a message with the block
// and thread position. NVIDIA path: the launch fails; the next
// synchronization returns cudaErrorLaunchFailure (719) and the trapping
// thread writes nothing after the trap.
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdio>
#include <cstring>

namespace cg = cooperative_groups;

// The pattern used by a real project: a function that never returns, and a
// macro that is empty in host code and calls it in device code.
[[noreturn]] static __device__ void no_device_code(const char *file, int line)
{
    printf("%s:%d: ERROR: CUDA kernel was compiled without support for this device\n", file, line);
    __trap();
}
#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__)
#else
#define NO_DEVICE_CODE
#endif

__global__ void k_trap(int *flags)
{
    printf("before trap block %d thread %d\n", (int)blockIdx.x, (int)threadIdx.x);
    if (blockIdx.x == 1 && threadIdx.x == 2) {
        __trap();
        flags[0] = 1; // must never happen
    }
}

__global__ void k_nodevcode(int *flags)
{
    if (blockIdx.x == 1 && threadIdx.x == 2) {
        NO_DEVICE_CODE;
        flags[0] = 1; // reached only when the macro is empty
    }
}

__global__ void k_gridsync(int *flags)
{
    cg::grid_group g = cg::this_grid();
    if (g.is_valid()) flags[1] = 1;
    g.sync();
    if (threadIdx.x == 0 && blockIdx.x == 1) flags[0] = 1; // must never happen
}

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    int *d = nullptr;
    if (cudaMalloc(&d, 2 * sizeof(int)) != cudaSuccess) return 3;
    cudaMemset(d, 0, 2 * sizeof(int));
    if (!std::strcmp(argv[1], "trap")) k_trap<<<2, 4>>>(d);
    else if (!std::strcmp(argv[1], "nodevcode")) k_nodevcode<<<2, 4>>>(d);
    else if (!std::strcmp(argv[1], "gridsync")) k_gridsync<<<2, 4>>>(d);
    else return 2;
    cudaError_t e = cudaDeviceSynchronize();
    int flags[2] = {-1, -1};
    cudaMemcpy(flags, d, sizeof flags, cudaMemcpyDeviceToHost);
    // only reached when the kernel did not stop the process
    std::printf("sync=%d after_trap_written=%d grid_valid=%d\n", (int)e, flags[0], flags[1]);
    return 0;
}
