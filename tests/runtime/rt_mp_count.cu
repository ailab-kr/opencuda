// cudaDeviceProp::multiProcessorCount is the backend's count of the units
// blocks run on (an SM; an Intel Xe core = sub-slice), 1 where the backend
// does not know it, and OCU_MULTIPROCESSOR_COUNT=N overrides it (N >= 1;
// other values are ignored). cudaDevAttrMultiProcessorCount says the same.
// Runs on every path:
//   ocu-nvcc tests/runtime/rt_mp_count.cu -o rt_mp_count && ./rt_mp_count
// On an Intel GPU (OpenCL) the default is slices x sub-slices per slice
// (clinfo: "Slices (Intel)" x "Sub-slices per slice (Intel)"); print it with
// OCU_MP_PRINT=1 to compare.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static int g_fail, g_checks;
#define CHECK(c) do { ++g_checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static int props_count()
{
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return -1;
    int v = -2;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess) return -3;
    return v == p.multiProcessorCount ? v : -4;
}

int main()
{
    unsetenv("OCU_MULTIPROCESSOR_COUNT");
    const int base = props_count();
    CHECK(base >= 1);
    if (std::getenv("OCU_MP_PRINT")) std::printf("multiProcessorCount %d\n", base);

    setenv("OCU_MULTIPROCESSOR_COUNT", "7", 1);
    CHECK(props_count() == 7);
    setenv("OCU_MULTIPROCESSOR_COUNT", "1", 1);
    CHECK(props_count() == 1);
    for (const char *bad : {"0", "-3", "x", ""}) {
        setenv("OCU_MULTIPROCESSOR_COUNT", bad, 1);
        CHECK(props_count() == base);
    }
    unsetenv("OCU_MULTIPROCESSOR_COUNT");
    CHECK(props_count() == base);

    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
