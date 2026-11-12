// Compile-time cases of card runtime-llama-gaps (items 15, 20 and 31),
// selected with -DCASE=<n> by compile_case.cmake.
//   0        control: must compile
//   1..7     must NOT compile (names or sizes that the documentation does
//            not allow, or that this project does not provide)
//   8        must NOT compile (NVIDIA path only: a device-only function
//            called from host code)
//   20..22   must compile
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#ifndef CASE
#define CASE 0
#endif

__global__ void k(int *out)
{
    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<8> t8 = cg::tiled_partition<8>(blk);
    cg::thread_block_tile<32> t32 = cg::tiled_partition<32>(blk);
    out[0] = (int)(t8.size() + t32.size());
    out[1] = cg::reduce(t8, 1, cg::plus<int>());
#if CASE == 1
    cg::thread_block_tile<3> bad; // not a power of two
    out[2] = (int)bad.size();
#elif CASE == 2
    cg::thread_block_tile<64> bad; // wider than a warp
    out[2] = (int)bad.size();
#elif CASE == 3
    out[2] = cg::reduce(blk, 1, cg::plus<int>()); // reduce over a block: tiles only
#elif CASE == 4
    cg::thread_block_tile<16> bad = cg::tiled_partition<16>(t8); // larger than its parent
    out[2] = (int)bad.size();
#elif CASE == 5
    auto bad = cg::tiled_partition(blk, 16); // run-time size
    out[2] = (int)bad.size();
#elif CASE == 6
    auto bad = cg::coalesced_threads();
    out[2] = (int)bad.size();
#elif CASE == 7
    cg::memcpy_async(blk, out, out + 1, sizeof(int));
#elif CASE == 22
    out[2] = __double2int_rd(1.5) + __double2int_ru(1.5) + __double2int_rn(1.5) + __double2int_rz(1.5);
#elif CASE == 20
    cg::thread_block_tile<8> ok = cg::tiled_partition<8>(t32); // a smaller tile cut from a tile
    out[2] = (int)ok.size();
#endif
}

#if CASE == 8
void host_function(unsigned *p) { *p = __vsub4(1u, 2u); }
#endif

#if CASE == 21
// The C++ overload that takes the kernel itself, as the documentation shows.
cudaError_t launch_by_function()
{
    int *p = nullptr;
    void *args[] = {&p};
    return cudaLaunchCooperativeKernel(k, dim3(1), dim3(32), args, 0, 0);
}
#endif
